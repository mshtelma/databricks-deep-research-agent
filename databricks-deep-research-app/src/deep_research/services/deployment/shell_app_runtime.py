"""Single source of truth for a shell-app's Databricks runtime requirements.

``ShellAppExporter.translate`` (``shell_app.py``) is the ONLY place that decides
what runtime resources a generated shell app needs (web search, an explicit Brave
provider, text-table SQL) and resolves their locations (Brave secret scope/key,
SQL warehouse id, the built-in web-search endpoint). It records that decision in
the deployment ``Artifact.metadata`` via
:meth:`ShellAppRuntimeBindings.to_metadata`.

The deploy side (``shell_app_apps_api.py``) reconstructs the decision with
:meth:`ShellAppRuntimeBindings.from_metadata` and binds App resources/env from it
— it never re-derives the decision from settings/defaults. One source of truth is
what prevents the class of bug where a default-provider (e.g. Gemini) agent was
still bound a Brave secret resource it never needed, breaking deploys on
workspaces without a Brave secret scope.

This module is a dependency-free leaf (stdlib only) imported by both ``shell_app``
and ``shell_app_apps_api`` — no import cycle.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

# Resource binding names + Brave secret defaults. Single home — these were
# previously duplicated across shell_app.py and shell_app_apps_api.py.
BRAVE_SECRET_RESOURCE_NAME = "brave-api-key"
SQL_WAREHOUSE_RESOURCE_NAME = "text-table-sql-warehouse"
DEFAULT_BRAVE_SECRET_SCOPE = "deep-research-secrets"
DEFAULT_BRAVE_SECRET_KEY = "BRAVE_API_KEY"

_TRUE_STRINGS: frozenset[str] = frozenset({"1", "true", "yes", "y"})


def _parse_bool(value: str | None) -> bool:
    """Parse a metadata string ("true"/"false"/…) to bool; None/"" → False."""
    return (value or "").strip().lower() in _TRUE_STRINGS


def _bool_str(value: bool) -> str:
    return "true" if value else "false"


def _clean(value: str | None) -> str | None:
    """Trim a resource identifier; empty/whitespace → None."""
    if value is None:
        return None
    trimmed = str(value).strip()
    return trimmed or None


@dataclass(frozen=True)
class ShellAppRuntimeBindings:
    """The resolved Databricks runtime requirements for a generated shell app.

    Decided once by ``translate`` and serialized into ``Artifact.metadata``; read
    back unchanged by the deploy path. Construct via :meth:`build` (or
    :meth:`from_metadata`) so these invariants always hold:

    * ``brave_secret_scope``/``brave_secret_key`` are non-None **only** when
      ``uses_brave`` (a web tool explicitly pinned ``provider: brave``).
    * ``storage_warehouse_id`` is non-None **only** when ``requires_sql_warehouse``.
    """

    requires_web_search: bool
    brave_secret_scope: str | None
    brave_secret_key: str | None
    uses_brave: bool = False
    requires_sql_warehouse: bool = False
    storage_warehouse_id: str | None = None
    databricks_web_search_endpoint: str = ""
    brave_secret_resource_name: str = BRAVE_SECRET_RESOURCE_NAME
    sql_warehouse_resource_name: str = SQL_WAREHOUSE_RESOURCE_NAME

    @classmethod
    def build(
        cls,
        *,
        requires_web_search: bool,
        uses_brave: bool,
        requires_sql_warehouse: bool,
        brave_secret_scope: str | None,
        brave_secret_key: str | None,
        storage_warehouse_id: str | None,
        databricks_web_search_endpoint: str = "",
        brave_secret_resource_name: str = BRAVE_SECRET_RESOURCE_NAME,
        sql_warehouse_resource_name: str = SQL_WAREHOUSE_RESOURCE_NAME,
    ) -> ShellAppRuntimeBindings:
        """Construct with invariants normalized (never raises).

        Brave scope/key are dropped unless ``uses_brave``; the warehouse id is
        dropped unless ``requires_sql_warehouse``. This makes the Brave gate
        impossible to bypass via an inconsistent caller or stale metadata.
        """
        if not uses_brave:
            brave_secret_scope = None
            brave_secret_key = None
        if not requires_sql_warehouse:
            storage_warehouse_id = None
        return cls(
            requires_web_search=requires_web_search,
            brave_secret_scope=_clean(brave_secret_scope),
            brave_secret_key=_clean(brave_secret_key),
            uses_brave=uses_brave,
            requires_sql_warehouse=requires_sql_warehouse,
            storage_warehouse_id=_clean(storage_warehouse_id),
            databricks_web_search_endpoint=(databricks_web_search_endpoint or "").strip(),
            brave_secret_resource_name=(brave_secret_resource_name or BRAVE_SECRET_RESOURCE_NAME),
            sql_warehouse_resource_name=(
                sql_warehouse_resource_name or SQL_WAREHOUSE_RESOURCE_NAME
            ),
        )

    def to_metadata(self) -> dict[str, str]:
        """Serialize to the ``Artifact.metadata`` string dict.

        Also emits the legacy ``*_configured`` boolean keys for back-compat with
        existing structured logs and tests.
        """
        return {
            "requires_web_search": _bool_str(self.requires_web_search),
            "uses_brave": _bool_str(self.uses_brave),
            "requires_sql_warehouse": _bool_str(self.requires_sql_warehouse),
            "brave_secret_scope": self.brave_secret_scope or "",
            "brave_secret_key": self.brave_secret_key or "",
            "storage_warehouse_id": self.storage_warehouse_id or "",
            "databricks_web_search_endpoint": self.databricks_web_search_endpoint or "",
            "brave_secret_resource_name": self.brave_secret_resource_name,
            "sql_warehouse_resource_name": self.sql_warehouse_resource_name,
            "brave_secret_scope_configured": _bool_str(bool(self.brave_secret_scope)),
            "brave_secret_key_configured": _bool_str(bool(self.brave_secret_key)),
            "storage_warehouse_id_configured": _bool_str(bool(self.storage_warehouse_id)),
        }

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, str] | None) -> ShellAppRuntimeBindings:
        """Reconstruct from ``Artifact.metadata`` with safe defaults + invariants.

        Missing keys default to False/None/"" — so an artifact produced before
        ``uses_brave`` existed yields ``uses_brave=False`` (no Brave binding), the
        safe direction. ``build`` re-applies the invariants on the read path.
        """
        md: Mapping[str, str] = metadata or {}
        return cls.build(
            requires_web_search=_parse_bool(md.get("requires_web_search")),
            uses_brave=_parse_bool(md.get("uses_brave")),
            requires_sql_warehouse=_parse_bool(md.get("requires_sql_warehouse")),
            brave_secret_scope=md.get("brave_secret_scope"),
            brave_secret_key=md.get("brave_secret_key"),
            storage_warehouse_id=md.get("storage_warehouse_id"),
            databricks_web_search_endpoint=md.get("databricks_web_search_endpoint") or "",
            brave_secret_resource_name=(
                md.get("brave_secret_resource_name") or BRAVE_SECRET_RESOURCE_NAME
            ),
            sql_warehouse_resource_name=(
                md.get("sql_warehouse_resource_name") or SQL_WAREHOUSE_RESOURCE_NAME
            ),
        )
