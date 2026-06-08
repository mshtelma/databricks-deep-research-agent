"""Protocol surface for objects that can drive the HITL approval gate.

Defines :class:`ReactLoopHook`, the typed surface that
:func:`databricks_deep_research.agents.react_loop_hitl.run_hitl_gate`
depends on. Decoupling the HITL gate from the concrete
:class:`databricks_deep_research.agents.react_loop.ReactLoop` class removes
the runtime-introspection (``getattr`` / ``hasattr`` / ``isinstance`` /
``callable``) that previously violated Constitution #4.

Constitution #4 carve-out: ``@runtime_checkable`` is a Protocol-blessed
exception. ``isinstance(x, ReactLoopHook)`` is permitted in test code
only; production code must accept :class:`ReactLoopHook` by parameter
type and rely on ``mypy --strict`` for verification. A lint test
(``test_no_isinstance_reactloophook_in_production``) enforces this
boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ReactLoopHook(Protocol):
    """Typed surface required by the HITL approval gate.

    Members:
        node_id: Identifier for the agent node (read-only view).
        extras: Read-only mapping of framework extras (e.g. approval
            broker, owner user_id).
        emit_event: Best-effort emission of a HITL event.
    """

    @property
    def node_id(self) -> str:
        ...

    @property
    def extras(self) -> Mapping[str, Any]:
        ...

    def emit_event(self, event: Any) -> None:
        ...


__all__ = ["ReactLoopHook"]
