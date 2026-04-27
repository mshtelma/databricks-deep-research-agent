"""Protocol-satisfaction tests for :class:`ReactLoopHook`.

Verifies that:
- The concrete :class:`ReactLoop` satisfies the Protocol structurally.
- A minimal dataclass-based fake also satisfies the Protocol.
- :func:`run_hitl_gate` accepts a fake hook and returns ``None`` when no
  approval broker is wired.
- :meth:`emit_event` produces a structured ``REACT_HITL_EVENT`` log line.
- Owner ``user_id`` propagates from extras into ``broker.request``.
- Production source code does not contain
  ``isinstance(..., ReactLoopHook)`` (Constitution #4 carve-out).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from databricks_deep_research.agents.protocol import ReactLoopHook
from databricks_deep_research.agents.react_loop import ReactLoop
from databricks_deep_research.agents.react_loop_hitl import run_hitl_gate
from databricks_deep_research.tools.protocol import ToolDefinition


class _StubLLMClient:
    """Tiny FrameworkLLMClient stand-in (constructor-level only)."""


class _StubTool:
    def __init__(self, *, name: str = "test_tool", reason: str = "") -> None:
        meta: dict[str, Any] = {}
        if reason:
            meta["approval_reason"] = reason
        self.definition = ToolDefinition(
            name=name,
            description="",
            parameters={"type": "object"},
            metadata=meta,
        )


@dataclass
class _DataclassFake:
    """Minimal Protocol-satisfying fake."""

    _node_id: str = "fake-node"
    _extras: dict[str, Any] = field(default_factory=dict)
    captured: list[Any] = field(default_factory=list)

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def extras(self) -> Mapping[str, Any]:
        return self._extras

    def emit_event(self, event: Any) -> None:
        self.captured.append(event)


def test_react_loop_satisfies_hook_protocol() -> None:
    """ReactLoop satisfies ReactLoopHook structurally (runtime check)."""
    loop = ReactLoop(
        llm_client=_StubLLMClient(),  # type: ignore[arg-type]
        tools=[],
        node_id="real-loop",
    )
    assert isinstance(loop, ReactLoopHook)
    assert loop.node_id == "real-loop"
    # extras is a Mapping view of the underlying ToolContext extras.
    assert isinstance(loop.extras, Mapping)


def test_dataclass_fake_satisfies_hook_protocol() -> None:
    """A minimal dataclass with extras/node_id/emit_event satisfies the Protocol."""
    fake = _DataclassFake()
    assert isinstance(fake, ReactLoopHook)


@pytest.mark.asyncio
async def test_run_hitl_gate_with_fake_hook() -> None:
    """run_hitl_gate accepts a Protocol-conforming fake; no broker -> None."""
    fake = _DataclassFake()
    tool = _StubTool()
    result = await run_hitl_gate(fake, tool, {"k": "v"})  # type: ignore[arg-type]
    assert result is None
    assert fake.captured == []


def test_emit_event_logs(caplog: pytest.LogCaptureFixture) -> None:
    """ReactLoop.emit_event emits a structured REACT_HITL_EVENT log line."""

    @dataclass
    class _Evt:
        event_type: str = "gate_waiting"

    loop = ReactLoop(
        llm_client=_StubLLMClient(),  # type: ignore[arg-type]
        tools=[],
        node_id="log-node",
    )
    with caplog.at_level(logging.INFO, logger="databricks_deep_research.agents.react_loop"):
        loop.emit_event(_Evt())
    matches = [
        r for r in caplog.records
        if "REACT_HITL_EVENT" in r.getMessage()
        and "node_id=log-node" in r.getMessage()
        and "event_type=gate_waiting" in r.getMessage()
    ]
    assert matches, f"expected REACT_HITL_EVENT log line, got: {[r.getMessage() for r in caplog.records]}"


@pytest.mark.asyncio
async def test_run_hitl_gate_user_id_propagates() -> None:
    """extras['_framework_user_id'] propagates as owner_user_id to broker.request."""
    captured: dict[str, Any] = {}

    @dataclass
    class _Decision:
        approved: bool = True
        reason: str | None = None
        approver: str | None = "alice"

    class _CapturingBroker:
        async def request(
            self,
            request_id: str,
            tool_name: str,
            arguments: dict[str, Any],
            *,
            reason: str = "",
            owner_user_id: str | None = None,
        ) -> _Decision:
            captured["owner_user_id"] = owner_user_id
            captured["request_id"] = request_id
            return _Decision()

    broker = _CapturingBroker()
    fake = _DataclassFake(
        _extras={
            "_framework_approval_broker": broker,
            "_framework_user_id": "alice",
        },
    )
    tool = _StubTool(reason="destructive")

    result = await run_hitl_gate(fake, tool, {"k": "v"})  # type: ignore[arg-type]
    # Approved path returns None.
    assert result is None
    assert captured["owner_user_id"] == "alice"


def test_no_isinstance_reactloophook_in_production() -> None:
    """Constitution #4 carve-out: isinstance(..., ReactLoopHook) is permitted in tests only.

    Walks every .py file under databricks-deep-research/src/ and rejects any
    occurrence of the pattern ``isinstance(...ReactLoopHook)``.
    """
    src_root = (
        Path(__file__).resolve().parent.parent.parent / "src" / "databricks_deep_research"
    )
    assert src_root.is_dir(), f"expected src tree at {src_root}"

    # Match real code only: ignore comments, docstring prose ("``..``"
    # references, double-backtick markup), and string literals containing
    # the example. The strict form requires `isinstance(` followed by a
    # bare identifier and `, ReactLoopHook)` on the same line, with no
    # surrounding backticks/quotes/hash on that line.
    pattern = re.compile(r"\bisinstance\([A-Za-z_][\w.]*,\s*ReactLoopHook\)")
    offenders: list[tuple[Path, int, str]] = []
    for path in src_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "``" in line:
                # Docstring rST cross-reference (e.g. ``isinstance(x, ReactLoopHook)``).
                continue
            if pattern.search(line):
                offenders.append((path, lineno, stripped))

    assert offenders == [], (
        "isinstance(..., ReactLoopHook) is permitted in tests only; "
        f"found in production code: {offenders}"
    )
