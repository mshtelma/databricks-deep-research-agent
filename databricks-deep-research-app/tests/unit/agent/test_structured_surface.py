"""Unit tests for the agent-surface structuring pass (fake LLM, fail-soft)."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from deep_research.agent import structured_surface as ss
from deep_research.agent.orchestration_config import OrchestrationConfig
from deep_research.agent.structured_surface import load_agent_surface
from deep_research.services.llm.types import LLMResponse
from deep_research.surface.output_schema import SlotSpec, collect_output_slots

pytestmark = pytest.mark.unit


@dataclass
class _Claim:
    claim_text: str
    citation_key: str | None = None
    citation_keys: list[str] | None = None
    verification_verdict: str | None = "supported"


def _slots() -> dict[str, SlotSpec]:
    surface = {
        "version": 1,
        "components": [
            {"id": "root", "component": "Column", "props": {},
             "children": ["findings", "tbl"]},
            {"id": "findings", "component": "KeyFindings",
             "props": {"source": {"path": "/results/run/data/key_findings"}},
             "children": []},
            {"id": "tbl", "component": "Table",
             "props": {
                 "source": {"path": "/results/run/data/comparison"},
                 "columns": [
                     {"key": "item", "label": "Item", "type": "string"},
                     {"key": "score", "label": "Score", "type": "number"},
                 ],
             },
             "children": []},
        ],
        "data_model": {"query": ""},
        "bindings": [
            {"action": "run", "kind": "run_agent",
             "inputs": {"query": {"path": "/query"}}, "options": {},
             "output": {"target": "/results/run", "mode": "report"},
             "concurrency": "replace"},
        ],
    }
    return collect_output_slots(surface).slots_for("run")


_REPORT = "R" * 400  # comfortably above MIN_REPORT_CHARS


def _response(content: str, *, validate_with: Any = None) -> LLMResponse:
    structured = None
    if validate_with is not None:
        try:
            structured = validate_with.model_validate_json(content)
        except Exception:  # noqa: BLE001 — mimic the client's fail-open
            structured = None
    return LLMResponse(
        content=content, usage={}, endpoint_id="fake", duration_ms=1.0,
        structured=structured,
    )


async def test_load_agent_surface_guard_paths() -> None:
    assert await load_agent_surface(OrchestrationConfig(), "user", None) is None
    assert (
        await load_agent_surface(
            OrchestrationConfig(agent_id="not-a-uuid"), "user", None
        )
        is None
    )
    assert (
        await load_agent_surface(
            OrchestrationConfig(agent_id="57a439d4-7590-4e47-b621-a620287ea3d2"),
            None,
            None,
        )
        is None
    )


# ---------------------------------------------------------------------------
# Schema-aware synthesis: contract injection into synthesizer prompts
# ---------------------------------------------------------------------------


def _slotted_surface_dict() -> dict[str, Any]:
    return {
        "version": 1,
        "components": [
            {"id": "root", "component": "Column", "props": {},
             "children": ["findings"]},
            {"id": "findings", "component": "KeyFindings",
             "props": {"source": {"path": "/results/run/data/key_findings"}},
             "children": []},
        ],
        "data_model": {"query": ""},
        "bindings": [
            {"action": "run", "kind": "run_agent",
             "inputs": {"query": {"path": "/query"}}, "options": {},
             "output": {"target": "/results/run", "mode": "report"},
             "concurrency": "replace"},
        ],
    }


async def test_contract_injected_into_synthesizer_prompts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deep_research.agent import framework_orchestrator as fo

    wf = fo.load_workflow_from_dict({
        "id": "wf", "name": "t", "version": 1,
        "required_inputs": ["query"],
        "root": {
            "id": "root", "type": "sequence", "label": "seq", "config": {},
            "children": [
                {"id": "res", "type": "agent", "label": "res",
                 "config": {"subtype": "researcher",
                            "user_prompt_template": "Research (query)."},
                 "children": []},
                {"id": "synth", "type": "agent", "label": "synth",
                 "config": {"subtype": "synthesizer",
                            "user_prompt_template": "Write the report."},
                 "children": []},
            ],
        },
    })

    async def fake_load(*_args: Any, **_kwargs: Any) -> Any:
        return (_slotted_surface_dict(), "etag-x")

    monkeypatch.setattr(ss, "load_agent_surface", fake_load)

    config = OrchestrationConfig(
        agent_id="57a439d4-7590-4e47-b621-a620287ea3d2"
    )
    await fo._inject_structured_output_contract(wf, config, "user", None)

    res_cfg = wf.root.children[0].config
    synth_cfg = wf.root.children[1].config

    # Researcher gets the research-targeting contract (same slots), NOT the
    # synthesizer's report-coverage one — this drives evidence gathering.
    res_template = res_cfg["user_prompt_template"]
    assert res_template.startswith("Research (query).")
    assert "Structured results coverage" not in res_template
    assert "Required result sections" in res_template
    assert "key_findings" in res_template
    assert "{" not in res_template.split("Required result sections")[1]

    # Synthesizer gets the report-coverage contract, not the research one.
    template = synth_cfg["user_prompt_template"]
    assert template.startswith("Write the report.")
    assert "Structured results coverage" in template
    assert "Required result sections" not in template
    assert "key_findings" in template
    # Brace-sanitized so designer labels can never become placeholders.
    assert "{" not in template.split("Structured results coverage")[1]

    # Idempotent on a second pass (both contracts).
    await fo._inject_structured_output_contract(wf, config, "user", None)
    assert (
        wf.root.children[1].config["user_prompt_template"].count(
            "Structured results coverage"
        )
        == 1
    )
    assert (
        wf.root.children[0].config["user_prompt_template"].count(
            "Required result sections"
        )
        == 1
    )


async def test_contract_injection_skips_without_agent_or_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deep_research.agent import framework_orchestrator as fo

    wf = fo.load_workflow_from_dict({
        "id": "wf", "name": "t", "version": 1,
        "required_inputs": ["query"],
        "root": {
            "id": "root", "type": "sequence", "label": "seq", "config": {},
            "children": [
                {"id": "synth", "type": "agent", "label": "synth",
                 "config": {"subtype": "synthesizer",
                            "user_prompt_template": "Write the report."},
                 "children": []},
            ],
        },
    })

    # No agent_id → untouched (loader must not even be called).
    called = False

    async def fake_load(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal called
        called = True
        return None

    monkeypatch.setattr(ss, "load_agent_surface", fake_load)
    await fo._inject_structured_output_contract(
        wf, OrchestrationConfig(), "user", None
    )
    assert called is False

    # Loader returning None (no surface) → untouched.
    await fo._inject_structured_output_contract(
        wf,
        OrchestrationConfig(agent_id="57a439d4-7590-4e47-b621-a620287ea3d2"),
        "user",
        None,
    )
    assert (
        wf.root.children[0].config["user_prompt_template"]
        == "Write the report."
    )


# ===========================================================================
# v2 — per-slot wires (run_slot_wires / guards / envelopes / structure_and_update)
# ===========================================================================

from deep_research.agent.structured_evidence import EvidenceItem  # noqa: E402
from deep_research.agent.structured_surface import (  # noqa: E402
    apply_slot_guards,
    build_envelope_v2,
    build_pending_envelope,
    run_slot_wires,
    structure_and_update,
)

_EVIDENCE = [
    EvidenceItem(ref="1", url="https://e1", title="E1", snippet="s1"),
    EvidenceItem(ref="2", url="https://e2", title="E2", snippet="s2"),
]


@dataclass
class _FakeWireLLM:
    """Per-slot queue-driven fake (parallel wires need slot-keyed content).

    Slot is derived from the wire model name (``Wire_<slot>``); delays are
    consumed per call BEFORE the content pop, so a timed-out call leaves its
    content for the retry.
    """

    by_slot: dict[str, list[str]] = field(default_factory=dict)
    delays: dict[str, list[float]] = field(default_factory=dict)
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def complete(self, **kwargs: Any) -> LLMResponse:
        self.calls.append(kwargs)
        model = kwargs["structured_output"]
        slot = model.__name__.removeprefix("Wire_")
        delay_queue = self.delays.get(slot)
        delay = delay_queue.pop(0) if delay_queue else 0.0
        if delay:
            await asyncio.sleep(delay)
        content = self.by_slot[slot].pop(0)
        return _response(content, validate_with=model)

    def calls_for(self, slot: str) -> list[dict[str, Any]]:
        return [
            c for c in self.calls
            if c["structured_output"].__name__ == f"Wire_{slot}"
        ]


def _wire_contents(
    comparison: str | None = None, key_findings: str | None = None
) -> dict[str, list[str]]:
    return {
        "comparison": [comparison or json.dumps({"comparison": [
            {"item": "Option A [K1]", "score": 8.1, "source_refs": ["1"]},
        ]})],
        "key_findings": [key_findings or json.dumps({"key_findings": [
            {"text": "Finding one [K1].", "source_refs": ["2"]},
        ]})],
    }


async def test_wires_fill_all_slots_with_refs() -> None:
    llm = _FakeWireLLM(by_slot=_wire_contents())
    results = await run_slot_wires(
        slots=_slots(),
        evidence=_EVIDENCE,
        claims=[_Claim("c", citation_key="K1")],
        report=_REPORT,
        llm=llm,  # type: ignore[arg-type]
    )
    assert results.outcomes["comparison"].status == "ok"
    assert results.outcomes["key_findings"].status == "ok"
    assert results.outcomes["comparison"].items[0]["source_refs"] == ["1"]
    assert results.used_refs == {"1", "2"}


async def test_wire_guards_drop_unsourced_rows() -> None:
    content = json.dumps({"comparison": [
        {"item": "Good", "score": 1.0, "source_refs": ["S1"]},  # coerced → 1
        {"item": "Bad", "score": 2.0, "source_refs": ["99"]},   # unresolvable
        {"item": "None", "score": 3.0},                          # missing refs
    ]})
    llm = _FakeWireLLM(by_slot=_wire_contents(comparison=content))
    results = await run_slot_wires(
        slots=_slots(),
        evidence=_EVIDENCE,
        claims=[],
        report=_REPORT,
        llm=llm,  # type: ignore[arg-type]
    )
    outcome = results.outcomes["comparison"]
    assert outcome.status == "ok"
    assert len(outcome.items) == 1
    assert outcome.items[0]["source_refs"] == ["1"]
    assert outcome.dropped_unsourced == 2


async def test_one_wire_failure_leaves_siblings_ok() -> None:
    llm = _FakeWireLLM(by_slot={
        "comparison": ["not json at all", "still not json"],
        "key_findings": _wire_contents()["key_findings"],
    })
    results = await run_slot_wires(
        slots=_slots(),
        evidence=_EVIDENCE,
        claims=[],
        report=_REPORT,
        llm=llm,  # type: ignore[arg-type]
    )
    assert results.outcomes["comparison"].status == "failed"
    assert results.outcomes["comparison"].error is not None
    assert results.outcomes["comparison"].attempts == 2
    assert results.outcomes["key_findings"].status == "ok"


async def test_wire_timeout_then_trimmed_retry() -> None:
    contents = _wire_contents()
    llm = _FakeWireLLM(
        by_slot=contents,
        delays={"comparison": [5.0, 0.0]},
    )
    results = await run_slot_wires(
        slots=_slots(),
        evidence=_EVIDENCE,
        claims=[],
        report=_REPORT,
        llm=llm,  # type: ignore[arg-type]
        wire_timeout_s=0.05,
    )
    outcome = results.outcomes["comparison"]
    assert outcome.status == "ok"
    assert outcome.attempts == 2
    calls = llm.calls_for("comparison")
    assert len(calls) == 2
    first_user = calls[0]["messages"][-1]["content"]
    second_user = calls[1]["messages"][-1]["content"]
    assert len(second_user) <= len(first_user)  # trimmed inputs


async def test_wire_validation_feedback_retry() -> None:
    good = _wire_contents()["comparison"][0]
    llm = _FakeWireLLM(by_slot={
        "comparison": [json.dumps({"wrong_key": []}), good],
        "key_findings": _wire_contents()["key_findings"],
    })
    results = await run_slot_wires(
        slots=_slots(),
        evidence=_EVIDENCE,
        claims=[],
        report=_REPORT,
        llm=llm,  # type: ignore[arg-type]
    )
    outcome = results.outcomes["comparison"]
    assert outcome.status == "ok"
    assert outcome.attempts == 2
    retry_messages = llm.calls_for("comparison")[1]["messages"]
    assert any(
        "failed validation" in str(m.get("content", ""))
        for m in retry_messages
    )


async def test_wire_lenient_recovery_of_extra_key() -> None:
    content = json.dumps({"comparison": [
        {"item": "A", "score": 1.0, "source_refs": ["1"], "bogus": "x"},
    ]})
    llm = _FakeWireLLM(by_slot=_wire_contents(comparison=content))
    results = await run_slot_wires(
        slots=_slots(),
        evidence=_EVIDENCE,
        claims=[],
        report=_REPORT,
        llm=llm,  # type: ignore[arg-type]
    )
    outcome = results.outcomes["comparison"]
    assert outcome.status == "ok"
    assert outcome.items[0]["item"] == "A"
    assert any(w["code"] == "lenient_validation" for w in outcome.warnings)


async def test_guards_fail_open_without_evidence() -> None:
    llm = _FakeWireLLM(by_slot=_wire_contents())
    results = await run_slot_wires(
        slots=_slots(),
        evidence=[],
        claims=[],
        report=_REPORT,
        llm=llm,  # type: ignore[arg-type]
    )
    outcome = results.outcomes["comparison"]
    assert outcome.status == "ok"
    assert outcome.items[0]["source_refs"] == []  # cleared, not dropped
    envelope = build_envelope_v2(
        binding="run", agent_id="a", surface_etag=None,
        results=results, evidence=[],
    )
    assert envelope["meta"]["evidence"] == "report_only"


async def test_wires_strip_unknown_markers() -> None:
    content = json.dumps({"key_findings": [
        {"text": "Known [K1] and bogus [Nope].", "source_refs": ["1"]},
    ]})
    llm = _FakeWireLLM(by_slot=_wire_contents(key_findings=content))
    results = await run_slot_wires(
        slots=_slots(),
        evidence=_EVIDENCE,
        claims=[_Claim("c", citation_key="K1")],
        report=_REPORT,
        llm=llm,  # type: ignore[arg-type]
    )
    text = results.outcomes["key_findings"].items[0]["text"]
    assert "[K1]" in text
    assert "[Nope]" not in text
    assert results.stripped_citation_keys == ["Nope"]


def test_apply_slot_guards_metrics_never_dropped() -> None:
    items = [
        {"label": "A", "value": "1", "source_refs": ["src 1", "77"]},
        {"label": "B", "value": "2", "source_refs": []},
    ]
    kept, dropped, used = apply_slot_guards("metrics", items, {"1", "2"})
    assert len(kept) == 2
    assert dropped == 0
    assert kept[0]["source_refs"] == ["1"]  # coerced + filtered
    assert used == {"1"}


def test_apply_slot_guards_enforce_false_untouched() -> None:
    items = [{"source_refs": "a raw column value"}]
    kept, dropped, used = apply_slot_guards(
        "table", items, {"1"}, enforce=False
    )
    assert kept == items
    assert dropped == 0
    assert used == set()


def test_build_pending_envelope_shape() -> None:
    envelope = build_pending_envelope(
        binding="run", agent_id="agent-1", surface_etag="W/x",
        slot_names=["a", "b"],
    )
    assert envelope["version"] == 2
    assert envelope["agent_id"] == "agent-1"
    assert envelope["data"] == {}
    assert envelope["meta"]["slots"] == {
        "a": {"status": "pending"}, "b": {"status": "pending"},
    }


async def test_build_envelope_v2_excludes_failed_slot_data() -> None:
    llm = _FakeWireLLM(by_slot={
        "comparison": ["garbage", "garbage"],
        "key_findings": _wire_contents()["key_findings"],
    })
    results = await run_slot_wires(
        slots=_slots(), evidence=_EVIDENCE, claims=[], report=_REPORT,
        llm=llm,  # type: ignore[arg-type]
    )
    envelope = build_envelope_v2(
        binding="run", agent_id="a", surface_etag="W/x",
        results=results, evidence=_EVIDENCE,
    )
    assert envelope["version"] == 2
    assert "comparison" not in envelope["data"]
    assert envelope["meta"]["slots"]["comparison"]["status"] == "failed"
    assert "error" in envelope["meta"]["slots"]["comparison"]
    assert envelope["meta"]["slots"]["key_findings"]["status"] == "ok"
    # Legend carries only refs actually used by kept items.
    assert [s["ref"] for s in envelope["meta"]["sources"]] == ["2"]


async def test_structure_and_update_persists_via_targeted_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def _fake_update(**kwargs: Any) -> bool:
        captured.update(kwargs)
        return True

    monkeypatch.setattr(
        "deep_research.agent.persistence.update_structured_output_independent",
        _fake_update,
    )
    from uuid import uuid4

    session_id = uuid4()
    llm = _FakeWireLLM(by_slot=_wire_contents())
    envelope = await structure_and_update(
        binding="run",
        agent_id="agent-1",
        surface_etag="W/x",
        slots=_slots(),
        report=_REPORT,
        claims=[_Claim("c", citation_key="K1")],
        sources=[{"url": "https://e1", "title": "E1"},
                 {"url": "https://e2", "title": "E2"}],
        chat_id=None,
        research_session_id=session_id,
        llm=llm,  # type: ignore[arg-type]
    )
    assert captured["research_session_id"] == session_id
    assert captured["envelope"] is envelope
    assert envelope["binding"] == "run"
    assert envelope["meta"]["slots"]["comparison"]["status"] == "ok"


async def test_structure_and_update_partial_rerun_merges_prior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_update(**kwargs: Any) -> bool:
        return True

    monkeypatch.setattr(
        "deep_research.agent.persistence.update_structured_output_independent",
        _fake_update,
    )
    from uuid import uuid4

    prior = build_pending_envelope(
        binding="run", agent_id="a", surface_etag=None,
        slot_names=["comparison", "key_findings"],
    )
    prior["data"]["key_findings"] = [{"text": "old", "source_refs": []}]
    prior["meta"]["slots"]["key_findings"] = {"status": "ok"}

    llm = _FakeWireLLM(by_slot=_wire_contents())
    envelope = await structure_and_update(
        binding="run",
        agent_id="a",
        surface_etag=None,
        slots=_slots(),
        report=_REPORT,
        claims=[],
        sources=[{"url": "https://e1", "title": "E1"}],
        chat_id=None,
        research_session_id=uuid4(),
        llm=llm,  # type: ignore[arg-type]
        only_slots={"comparison"},
        prior_envelope=prior,
    )
    # Only the rerun slot hit the LLM; the sibling's data was carried over.
    assert [c["structured_output"].__name__ for c in llm.calls] == [
        "Wire_comparison"
    ]
    assert envelope["data"]["key_findings"] == [
        {"text": "old", "source_refs": []}
    ]
    assert envelope["meta"]["slots"]["comparison"]["status"] == "ok"
    assert envelope["meta"]["slots"]["key_findings"]["status"] == "ok"
