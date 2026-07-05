"""Unit tests for the framework structured-output generation core.

Covers the pure envelope helpers (moved here from the app) plus an end-to-end
``build_structured_envelope`` over a fake ``StructuredCompletionClient`` — the
seam both the app ``LLMClient`` and the framework ``FrameworkLLMClient``
satisfy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from databricks_deep_research.surface.contract import (
    RESEARCH_CONTRACT_MARKER,
    STRUCTURED_CONTRACT_MARKER,
    build_contracts,
)
from databricks_deep_research.surface.generation import (
    SlotOutcome,
    SlotWireResults,
    _merge_envelopes,
    _truncate_payload,
    apply_slot_guards,
    build_envelope_v2,
    build_pending_envelope,
    build_structured_envelope,
)
from databricks_deep_research.surface.output_schema import (
    ColumnSpec,
    SlotSpec,
    collect_output_slots,
    resolve_binding_for_run,
)

# ---------------------------------------------------------------------------
# Fake LLM client (satisfies StructuredCompletionClient structurally)
# ---------------------------------------------------------------------------


@dataclass
class _FakeResponse:
    structured: Any
    content: str = ""


class FakeClient:
    """Returns a valid parse of the per-slot wire model for each call."""

    def __init__(self, items_by_slot: dict[str, list[dict[str, Any]]] | None = None):
        self.calls = 0
        self._items = items_by_slot or {}

    async def complete(
        self,
        *,
        messages: list[dict[str, str]],
        tier: Any,
        structured_output: type[Any] | None = None,
    ) -> _FakeResponse:
        self.calls += 1
        assert structured_output is not None
        field_name = next(iter(structured_output.model_fields))
        items = self._items.get(
            field_name, [{"text": "A grounded finding.", "source_refs": ["1"]}]
        )
        model = structured_output.model_validate({field_name: items})
        return _FakeResponse(structured=model, content="")


def _findings_surface() -> dict[str, Any]:
    return {
        "version": 1,
        "components": [
            {"id": "root", "component": "Column", "props": {}, "children": ["kf"]},
            {
                "id": "kf",
                "component": "KeyFindings",
                "props": {"source": {"path": "/results/data/findings"}},
                "children": [],
            },
        ],
        "data_model": {},
        "bindings": [
            {
                "action": "run",
                "kind": "run_agent",
                "inputs": {},
                "options": {},
                "output": {"target": "/results", "mode": "report"},
                "concurrency": "replace",
            }
        ],
    }


# ---------------------------------------------------------------------------
# Pure envelope helpers
# ---------------------------------------------------------------------------


def test_truncate_payload_halves_largest_list() -> None:
    big = {"rows": [{"text": "x" * 400} for _ in range(600)], "small": ["keep"]}
    truncated: list[str] = []
    out = _truncate_payload(big, truncated)
    assert truncated == ["rows"]
    assert len(out["rows"]) < 600
    assert out["small"] == ["keep"]


def test_merge_envelopes_remaps_refs_by_url() -> None:
    prior = {
        "version": 2, "binding": "run", "agent_id": "a",
        "generated_at": "2026-01-01T00:00:00+00:00",
        "data": {"slot_a": [{"text": "kept", "source_refs": ["1"]}]},
        "meta": {
            "slots": {"slot_a": {"status": "ok"}},
            "sources": [{"ref": "1", "url": "https://old", "title": "Old"}],
        },
    }
    fresh = {
        "version": 2, "binding": "run", "agent_id": "a",
        "generated_at": "2026-01-02T00:00:00+00:00",
        "data": {"slot_b": [{"text": "new", "source_refs": ["1", "2"]}]},
        "meta": {
            "slots": {"slot_b": {"status": "ok"}},
            "sources": [
                {"ref": "1", "url": "https://new", "title": "New"},
                {"ref": "2", "url": "https://old", "title": "Old"},
            ],
        },
    }
    merged = _merge_envelopes(prior, fresh)
    assert merged["data"]["slot_a"][0]["source_refs"] == ["1"]
    # fresh "1" (new URL) → "2"; fresh "2" (known URL) → prior "1"
    assert merged["data"]["slot_b"][0]["source_refs"] == ["2", "1"]
    legend = {entry["url"]: entry["ref"] for entry in merged["meta"]["sources"]}
    assert legend == {"https://old": "1", "https://new": "2"}
    assert set(merged["meta"]["slots"]) == {"slot_a", "slot_b"}


def test_apply_slot_guards_drops_unsourced_and_fails_open() -> None:
    items = [
        {"text": "sourced", "source_refs": ["1"]},
        {"text": "unsourced", "source_refs": []},
    ]
    kept, dropped, used = apply_slot_guards("strings", list(items), {"1", "2"})
    assert dropped == 1
    assert [i["text"] for i in kept] == ["sourced"]
    assert used == {"1"}

    # Fail-open: no valid refs → nothing dropped, refs cleared.
    kept2, dropped2, used2 = apply_slot_guards("strings", list(items), set())
    assert dropped2 == 0
    assert len(kept2) == 2
    assert used2 == set()


def test_build_pending_and_envelope_v2_shapes() -> None:
    pending = build_pending_envelope(
        binding="run", agent_id="a", surface_etag=None,
        slot_names=["findings"],
    )
    assert pending["version"] == 2
    assert pending["meta"]["slots"]["findings"]["status"] == "pending"

    results = SlotWireResults(
        outcomes={
            "findings": SlotOutcome(
                status="ok",
                items=[{"text": "f", "source_refs": ["1"]}],
                used_refs={"1"},
            )
        },
        used_refs={"1"},
    )
    from databricks_deep_research.surface.evidence import EvidenceItem

    env = build_envelope_v2(
        binding="run", agent_id="a", surface_etag=None, results=results,
        evidence=[EvidenceItem(ref="1", url="https://a", title="A")],
    )
    assert env["data"]["findings"][0]["text"] == "f"
    assert env["meta"]["slots"]["findings"]["status"] == "ok"
    assert env["meta"]["sources"] == [
        {"ref": "1", "url": "https://a", "title": "A"}
    ]


# ---------------------------------------------------------------------------
# End-to-end generation over the fake client
# ---------------------------------------------------------------------------


async def test_build_structured_envelope_end_to_end() -> None:
    resolved = resolve_binding_for_run(
        collect_output_slots(_findings_surface()), None
    )
    assert resolved is not None
    assert set(resolved.slots) == {"findings"}

    client = FakeClient()
    env = await build_structured_envelope(
        binding=resolved.action,
        agent_id="agent-x",
        surface_etag="etag-1",
        slots=resolved.slots,
        report="r" * 300,
        claims=[],
        sources=[
            {"url": "https://a", "title": "A", "snippet": "s", "relevance_score": 0.9}
        ],
        llm=client,
    )
    assert client.calls == 1
    assert env["binding"] == "run"
    item = env["data"]["findings"][0]
    assert item["text"] == "A grounded finding."
    assert item["source_refs"] == ["1"]
    assert env["meta"]["slots"]["findings"]["status"] == "ok"
    assert env["meta"]["sources"][0]["url"] == "https://a"


# ---------------------------------------------------------------------------
# Contract text (Part A)
# ---------------------------------------------------------------------------


def test_build_contracts_brace_sanitized_with_markers() -> None:
    slots = {
        "competitors": SlotSpec(
            slot="competitors",
            kind="table",
            columns=(ColumnSpec(key="name", label="Company {name}", type="string"),),
        )
    }
    research, synth = build_contracts(slots)
    assert RESEARCH_CONTRACT_MARKER in research
    assert STRUCTURED_CONTRACT_MARKER in synth
    # Designer-authored braces must never survive into a prompt template.
    assert "{" not in research and "}" not in research
    assert "{" not in synth and "}" not in synth
