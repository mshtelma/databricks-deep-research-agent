"""TableDiscoveryProvider — protocol for runtime table discovery.

Implementations of this protocol enumerate tables from an external source
(Unity Catalog, Designer registry, etc.) and return ``BindingInfo`` records
that the ``TableDiscoveryTool`` can register into a ``TableBindingRegistry``.

The protocol is intentionally minimal:
- A single async method, ``list_tables``.
- ``user_token`` is the OBO token to use when reaching the upstream catalog.
- ``name_pattern`` is an optional substring filter applied by the implementation.

Returned ``BindingInfo`` instances should have ``source = BindingSource.DISCOVERED``;
the ``TableDiscoveryTool`` will pass them through ``register_discovered`` which
enforces this contract.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .binding import BindingInfo


@runtime_checkable
class TableDiscoveryProvider(Protocol):
    """Discover tables at runtime, returning ``BindingInfo`` records."""

    async def list_tables(
        self,
        *,
        user_token: str,
        name_pattern: str | None = None,
    ) -> list[BindingInfo]:
        """Return discovered tables matching ``name_pattern``.

        Implementations must:
        - Use ``user_token`` for any upstream auth.
        - Filter by ``name_pattern`` (substring match, case-insensitive)
          when provided.
        - Return ``BindingInfo`` instances with ``source = BindingSource.DISCOVERED``.
        """
        ...
