"""TableBindingRegistry — keyed lookup of BOUND + DISCOVERED bindings.

Two registration paths:

- ``register_bound`` — declared in YAML at boot. Duplicate name within
  ``BOUND`` is a configuration error (``INVALID_BINDING``).
- ``register_discovered`` — added at runtime. If a name collides with an
  existing ``BOUND`` entry, the BOUND wins and the DISCOVERED entry is
  stored under ``discovered.<name>``; the caller receives a warning
  ``ToolError(error_code=DUPLICATE_BINDING)``.

The registry exposes both a frozen ``metadata_snapshot`` (used to hydrate
the in-compute ``bindings`` namespace at compute-turn entry — does NOT
reflect mid-turn mutations) and a live ``metadata_view`` (opt-in for the
``bindings_live`` namespace). Both views are read-only mappings.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from types import MappingProxyType

from .binding import BindingInfo, BindingSource, RoleMap
from .error_codes import ErrorCode, ToolError, ToolErrorException


class _LiveView(Mapping[str, BindingInfo]):
    """Read-only live view onto a mutable backing dict."""

    def __init__(self, src: dict[str, BindingInfo]) -> None:
        self._src = src

    def __getitem__(self, key: str) -> BindingInfo:
        return self._src[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._src)

    def __len__(self) -> int:
        return len(self._src)


class TableBindingRegistry:
    """In-memory registry of text-table bindings."""

    def __init__(self) -> None:
        self._items: dict[str, BindingInfo] = {}

    def register_bound(self, info: BindingInfo) -> None:
        """Register a BOUND binding. Raises on wrong source / duplicate name."""
        if info.source is not BindingSource.BOUND:
            raise ValueError(
                f"register_bound requires source=BOUND, got {info.source!r}"
            )
        if info.name in self._items:
            existing = self._items[info.name]
            raise ToolErrorException(
                ToolError(
                    error_code=ErrorCode.INVALID_BINDING,
                    message=(
                        f"duplicate BOUND binding name {info.name!r} "
                        f"(existing source={existing.source.value})"
                    ),
                    binding=info.name,
                    details={
                        "name": info.name,
                        "existing_fqn": existing.fqn,
                        "new_fqn": info.fqn,
                    },
                )
            )
        self._items[info.name] = info

    def register_discovered(
        self, info: BindingInfo
    ) -> tuple[str, ToolError | None]:
        """Register a DISCOVERED binding. Returns (canonical_name, optional_warning).

        If a BOUND entry already owns ``info.name``, BOUND wins and the new
        info is stored under ``discovered.<name>`` with a warning ToolError.
        """
        if info.source is not BindingSource.DISCOVERED:
            raise ValueError(
                f"register_discovered requires source=DISCOVERED, got {info.source!r}"
            )
        existing = self._items.get(info.name)
        if existing is not None and existing.source is BindingSource.BOUND:
            ns_name = f"discovered.{info.name}"
            self._items[ns_name] = info
            warning = ToolError(
                error_code=ErrorCode.DUPLICATE_BINDING,
                message=(
                    f"BOUND binding {info.name!r} already exists; "
                    f"DISCOVERED entry namespaced as {ns_name!r}"
                ),
                binding=ns_name,
                details={
                    "requested_name": info.name,
                    "registered_as": ns_name,
                    "bound_fqn": existing.fqn,
                    "discovered_fqn": info.fqn,
                },
            )
            return ns_name, warning
        self._items[info.name] = info
        return info.name, None

    def get(self, name: str) -> BindingInfo:
        if name not in self._items:
            raise ToolErrorException(
                ToolError(
                    error_code=ErrorCode.INVALID_BINDING,
                    message=f"binding {name!r} not registered",
                    binding=name,
                    details={"available": sorted(self._items)},
                )
            )
        return self._items[name]

    def update_roles(
        self,
        name: str,
        roles: RoleMap,
        *,
        numeric_columns: tuple[str, ...] | None = None,
    ) -> BindingInfo:
        """Store inferred/explicit roles for an existing binding.

        DISCOVERED bindings start role-less by design. First-use inference or a
        caller-provided ``roles={...}`` override updates the binding in-place so
        later tool calls in the same agent step can reuse the validated role
        map without re-probing the table.
        """
        info = self.get(name)
        updated = BindingInfo(
            name=info.name,
            fqn=info.fqn,
            source=info.source,
            description=info.description,
            roles=roles,
            numeric_columns=(
                tuple(numeric_columns)
                if numeric_columns is not None
                else info.numeric_columns
            ),
            structured_passages=info.structured_passages,
            verbose=info.verbose,
        )
        self._items[name] = updated
        return updated

    def __contains__(self, name: object) -> bool:
        return name in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def names(self) -> tuple[str, ...]:
        return tuple(self._items)

    def metadata_snapshot(self) -> Mapping[str, BindingInfo]:
        """Return an immutable snapshot of the registry state at call time."""
        return MappingProxyType(dict(self._items))

    def metadata_view(self) -> Mapping[str, BindingInfo]:
        """Return a live read-only view that reflects subsequent mutations."""
        return _LiveView(self._items)
