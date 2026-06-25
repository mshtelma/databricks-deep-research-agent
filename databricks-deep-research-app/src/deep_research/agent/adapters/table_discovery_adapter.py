"""App-side adapter implementing the framework's :class:`TableDiscoveryProvider`.

The framework's ``table_discovery`` tool calls
:meth:`TableDiscoveryProvider.list_tables` to populate a
:class:`TableBindingRegistry` with ``BindingInfo`` records sourced from
upstream catalogs (Unity Catalog, in production). The framework intentionally
ships *no* default provider — it cannot reach into Databricks SDK without an
OBO token bound to a request, which is an app concern.

This adapter is the app's bridge:

1. It enumerates tables by either (a) wrapping a configured set of
   ``(catalog, schema)`` Unity Catalog scopes via
   ``WorkspaceClient.tables.list`` or (b) passing through a caller-supplied
   list of pre-curated ``BindingInfo`` records (e.g. derived from the
   Designer's user-selected ``DesignerAsset`` payload).
2. It scopes each call with the supplied OBO ``user_token`` by constructing
   a fresh per-call ``WorkspaceClient`` via the injected factory; the
   plaintext token is never logged or stored in caches.
3. It does **NOT** auto-infer roles. The discovery tool's contract is to
   surface candidates; ``BindingInfo.roles`` is left ``None`` for
   DISCOVERED entries. Role inference happens lazily inside
   ``table_load`` / ``table_search`` (via
   :func:`databricks_deep_research.tools.builtins.text_table.role_inference.infer_roles`)
   when the user actually exercises the binding.

The adapter is intentionally generic — it carries no app-domain corpus
names or per-corpus hard-coded schemas.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, Protocol

from databricks_deep_research.tools.builtins.text_table import (
    BindingInfo,
    BindingSource,
)

if TYPE_CHECKING:
    from databricks.sdk import WorkspaceClient

logger = logging.getLogger(__name__)


__all__ = [
    "CatalogSchemaScope",
    "DesignerTableDiscoveryProvider",
    "WorkspaceClientFactory",
]


class WorkspaceClientFactory(Protocol):
    """Per-call factory returning an OBO-authenticated ``WorkspaceClient``.

    The framework's ``TableDiscoveryProvider`` protocol passes a plaintext
    ``user_token``; this factory translates the token into a workspace
    client. Implementations MUST NOT log the token. Returning ``None`` is
    permitted — the adapter then falls back to listing only the
    pre-supplied static bindings, if any.
    """

    def __call__(self, *, user_token: str) -> WorkspaceClient | None:
        ...


class CatalogSchemaScope(Protocol):
    """A (catalog, schema) pair the adapter will enumerate via the SDK.

    This is a structural protocol so callers may pass either a tuple, a
    Pydantic model, or a custom dataclass — anything with ``catalog`` and
    ``schema`` attributes. Both fields must be non-empty strings.
    """

    @property
    def catalog(self) -> str: ...

    @property
    def schema(self) -> str: ...


class _Scope:
    """Minimal concrete CatalogSchemaScope used by ``from_pairs``."""

    __slots__ = ("catalog", "schema")

    def __init__(self, catalog: str, schema: str) -> None:
        self.catalog = catalog
        self.schema = schema


class DesignerTableDiscoveryProvider:
    """App adapter implementing :class:`TableDiscoveryProvider`.

    Two enumeration sources, combined and de-duplicated by ``name``:

    - **Static bindings** — supplied at construction time. Useful when the
      Designer or a workflow surface has already determined an explicit set
      of tables to expose (e.g. user-selected ``DesignerAsset`` records).
      These are returned as ``BindingInfo(source=DISCOVERED)`` regardless
      of how the caller labelled them.

    - **Catalog scopes** — Unity Catalog ``(catalog, schema)`` pairs. When
      ``user_token`` is supplied to :meth:`list_tables`, the adapter
      constructs an OBO ``WorkspaceClient`` via ``client_factory`` and
      iterates ``client.tables.list(...)`` for each scope. Each table is
      returned with ``fqn = "<catalog>.<schema>.<name>"`` and a short
      description if the SDK exposes one.

    Token safety
    ------------

    The ``user_token`` is forwarded only to ``client_factory``. It is never
    logged, never cached, never echoed in error messages, and never stored
    on the adapter instance. If ``client_factory`` returns ``None`` the
    adapter logs (without the token) and proceeds with whatever static
    bindings it has.
    """

    def __init__(
        self,
        *,
        client_factory: WorkspaceClientFactory | None = None,
        scopes: Sequence[CatalogSchemaScope] | None = None,
        static_bindings: Iterable[BindingInfo] | None = None,
    ) -> None:
        self._client_factory = client_factory
        self._scopes: tuple[CatalogSchemaScope, ...] = tuple(scopes or ())
        self._static: tuple[BindingInfo, ...] = tuple(
            self._coerce_to_discovered(b) for b in (static_bindings or ())
        )

    # -- construction helpers ------------------------------------------------

    @classmethod
    def from_pairs(
        cls,
        *,
        client_factory: WorkspaceClientFactory | None = None,
        scopes: Iterable[tuple[str, str]] | None = None,
        static_bindings: Iterable[BindingInfo] | None = None,
    ) -> DesignerTableDiscoveryProvider:
        """Convenience constructor accepting raw ``(catalog, schema)`` tuples."""
        scope_objs = [
            _Scope(catalog=str(c).strip(), schema=str(s).strip())
            for c, s in (scopes or ())
            if str(c).strip() and str(s).strip()
        ]
        return cls(
            client_factory=client_factory,
            scopes=scope_objs,
            static_bindings=static_bindings,
        )

    # -- TableDiscoveryProvider Protocol -----------------------------------

    async def list_tables(
        self,
        *,
        user_token: str,
        name_pattern: str | None = None,
    ) -> list[BindingInfo]:
        """Return ``BindingInfo`` records for matching tables.

        Substring filter on ``name`` (case-insensitive). Static bindings
        are surfaced even when ``user_token`` is empty / no client factory
        is configured. Catalog-scoped enumeration is best-effort — failures
        per-scope are logged and the remaining scopes still run.
        """
        results: dict[str, BindingInfo] = {}
        # Static bindings first — they take precedence on name collisions.
        for info in self._static:
            results[info.name] = info

        client = None
        if self._scopes and self._client_factory is not None:
            try:
                client = self._client_factory(user_token=user_token)
            except Exception:  # noqa: BLE001 — never crash discovery on auth
                logger.warning(
                    "TABLE_DISCOVERY_CLIENT_FACTORY_FAILED scopes=%d",
                    len(self._scopes),
                    exc_info=True,
                )
                client = None

        if client is not None:
            for scope in self._scopes:
                catalog = str(getattr(scope, "catalog", "")).strip()
                schema = str(getattr(scope, "schema", "")).strip()
                if not catalog or not schema:
                    continue
                try:
                    listing = client.tables.list(
                        catalog_name=catalog, schema_name=schema
                    )
                except Exception:  # noqa: BLE001 — partial results are fine
                    logger.warning(
                        "TABLE_DISCOVERY_LIST_FAILED catalog=%s schema=%s",
                        catalog,
                        schema,
                        exc_info=True,
                    )
                    continue
                for entry in self._iter_entries(listing):
                    binding = self._entry_to_binding(entry, catalog, schema)
                    if binding is None:
                        continue
                    # Static bindings take precedence — only fill if missing.
                    results.setdefault(binding.name, binding)

        # Apply substring filter, if any. Case-insensitive on name.
        out = list(results.values())
        if name_pattern:
            needle = name_pattern.casefold()
            out = [
                info
                for info in out
                if needle in info.name.casefold()
                or needle in info.fqn.casefold()
            ]
        # Stable ordering for deterministic prompt output.
        out.sort(key=lambda info: info.fqn)
        return out

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _coerce_to_discovered(info: BindingInfo) -> BindingInfo:
        """Force ``source=DISCOVERED`` regardless of the caller's labelling.

        ``BindingInfo`` is frozen so we rebuild it. Roles are deliberately
        cleared — the discovery tool's contract is "surface candidates";
        role inference runs lazily on the first ``table_load`` /
        ``table_search`` call.
        """
        if info.source is BindingSource.DISCOVERED and info.roles is None:
            return info
        return BindingInfo(
            name=info.name,
            fqn=info.fqn,
            source=BindingSource.DISCOVERED,
            description=info.description,
            roles=None,
            numeric_columns=info.numeric_columns,
            structured_passages=info.structured_passages,
            verbose=info.verbose,
        )

    @staticmethod
    def _iter_entries(listing: Any) -> Iterable[Any]:
        """Iterate the SDK's ``tables.list`` return value safely.

        The SDK returns an ``Iterator[TableInfo]``. We treat anything
        non-iterable as empty rather than crashing.
        """
        try:
            iterator: Iterable[Any] = iter(listing)
        except TypeError:
            return ()
        return iterator

    @staticmethod
    def _entry_to_binding(
        entry: Any, catalog: str, schema: str
    ) -> BindingInfo | None:
        """Convert one ``TableInfo`` SDK record into a DISCOVERED ``BindingInfo``.

        Returns ``None`` if the entry is missing a name. Description is
        pulled from the SDK ``comment`` field when present.
        """
        name = getattr(entry, "name", None)
        if not isinstance(name, str) or not name:
            return None
        full_name = getattr(entry, "full_name", None)
        if isinstance(full_name, str) and full_name:
            fqn = full_name
        else:
            fqn = f"{catalog}.{schema}.{name}"
        comment = getattr(entry, "comment", None)
        description = (
            comment.strip() if isinstance(comment, str) and comment.strip() else None
        )
        return BindingInfo(
            name=name,
            fqn=fqn,
            source=BindingSource.DISCOVERED,
            description=description,
            roles=None,
        )


# ---------------------------------------------------------------------------
# Convenience factory: build from a stable WorkspaceClient instance
# ---------------------------------------------------------------------------


def workspace_client_factory_from(
    workspace_client: WorkspaceClient | None,
) -> WorkspaceClientFactory:
    """Wrap a stable ``WorkspaceClient`` as a per-call factory.

    The returned factory ignores ``user_token`` and reuses the supplied
    client for every call. Use this when the app constructs an
    OBO-authenticated client upstream (the standard
    :func:`build_app_workflow_runner` path) and just wants to forward it
    into the discovery adapter without re-authenticating per request.
    """

    def _call(*, user_token: str) -> WorkspaceClient | None:
        # ``user_token`` deliberately ignored — auth already baked in.
        del user_token
        return workspace_client

    return _call
