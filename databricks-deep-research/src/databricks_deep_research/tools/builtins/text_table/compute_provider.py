"""ComputeCallableProvider — Protocol for tools that expose a callable form.

The 6 ``table_*`` tools each have two surfaces:

1. The external **ReAct** path — ``execute(arguments, context) -> ToolResult``
   with a JSON envelope and ``success=False`` on errors.
2. The internal **compute** path — a plain Python callable injected directly
   into the ``PythonComputeTool`` namespace. Errors raise
   ``ToolErrorException`` instead of returning an envelope; results are
   native Python dicts/lists.

The single canonical implementation lives in the tool class. Both surfaces
delegate to it via ``to_compute_callable`` (compute) or ``execute`` (ReAct).

The compute callable's return type is tool-specific:
- 5 read-only tools (discovery / search / read / neighbors / aggregate)
  return ``list[dict[str, Any]]`` or ``dict[str, Any]``.
- ``table_load`` mutates the namespace via the resolver passed at
  callable-creation time AND returns the loaded rows.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ComputeCallableProvider(Protocol):
    """A tool that exposes a callable for direct in-compute use.

    Implementations MUST:

    * Return a callable from ``to_compute_callable`` that bypasses the JSON
      envelope. Errors should raise ``ToolErrorException`` (not return a
      ``ToolResult`` with ``success=False``).
    * Expose a ``compute_name`` property — the kwarg name under which the
      callable is injected into the compute namespace.
    """

    @property
    def compute_name(self) -> str:
        """Identifier used as the compute-namespace key for this callable."""
        ...

    def to_compute_callable(self, *, compute: Any) -> Callable[..., Any]:
        """Return a plain Python callable bound to ``compute``.

        ``compute`` is the hosting :class:`PythonComputeTool`; the
        implementation may use it to mutate the user namespace (e.g.
        ``table_load``) or treat it as a contextual handle.
        """
        ...


__all__ = ["ComputeCallableProvider"]
