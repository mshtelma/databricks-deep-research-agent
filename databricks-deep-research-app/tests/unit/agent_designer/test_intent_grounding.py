"""Plan v2.2 intent-grounding tests.

Two layers under test:

* :func:`match_text_to_resources` / :func:`extract_fqn_candidates` —
  pure-Python helpers that surface free-text workspace-resource mentions
  in user_intent against the catalog enumerated by ``discover_sources``.
* :class:`EmitGroundedAssetsTool` — builtin tool that the intent-grounding
  agent calls once per turn to merge resolved assets into
  ``state.designer_assets``.

The goal is a first-principles fix to the general problem that LLMs fail
to pick the right runtime tools whenever the user names a workspace
resource in free text but does not UI-select it. Grounding the mention
against the actual catalog is the missing layer (b) — *(a) intent +
(b) workspace state + (c) tool catalog* drives correct tool selection.
"""
from __future__ import annotations

import json
from typing import Any

import pytest
from databricks_deep_research.tools.protocol import ToolContext

from deep_research.agent_designer.discovery import (
    DiscoveredResource,
    extract_fqn_candidates,
    match_text_to_resources,
)
from deep_research.agent_designer.framework_tools import EmitGroundedAssetsTool

# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def _resource(
    kind: str,
    name: str,
    full_name: str | None = None,
    source_id: str | None = None,
) -> DiscoveredResource:
    return DiscoveredResource(
        kind=kind,  # type: ignore[arg-type]
        name=name,
        full_name=full_name,
        source_id=source_id,
    )


def test_extract_fqn_candidates_returns_three_part_names() -> None:
    intent = (
        "Use main.officeqa_benchmark.treasury_chunks_vs_index and "
        "delta.sales.orders_2024 to answer the question."
    )
    assert extract_fqn_candidates(intent) == [
        "main.officeqa_benchmark.treasury_chunks_vs_index",
        "delta.sales.orders_2024",
    ]


def test_extract_fqn_candidates_ignores_two_part_and_dotted_filenames() -> None:
    intent = "Open file.py and connect to schema.table but use catalog.schema.real_name."
    assert extract_fqn_candidates(intent) == ["catalog.schema.real_name"]


def test_extract_fqn_candidates_dedupes() -> None:
    intent = "main.a.b is the index. Don't forget main.a.b."
    assert extract_fqn_candidates(intent) == ["main.a.b"]


def test_extract_fqn_candidates_handles_empty_input() -> None:
    assert extract_fqn_candidates("") == []
    assert extract_fqn_candidates("no fqn here") == []


def test_match_text_to_resources_exact_fqn_match() -> None:
    intent = "Use main.officeqa_benchmark.treasury_chunks_vs_index to answer."
    resources = [
        _resource(
            "vector_index",
            "treasury_chunks_vs_index",
            full_name="main.officeqa_benchmark.treasury_chunks_vs_index",
        ),
        _resource("genie_space", "other", full_name="other.space.id"),
    ]
    matches = match_text_to_resources(intent, resources)
    assert len(matches) == 1
    assert matches[0].score == 100
    assert matches[0].matched_via == "fqn_exact"
    assert matches[0].resource.full_name == "main.officeqa_benchmark.treasury_chunks_vs_index"


def test_match_text_to_resources_case_insensitive_fqn_match() -> None:
    intent = "Use MAIN.OFFICEQA_BENCHMARK.TREASURY_CHUNKS_VS_INDEX please."
    resources = [
        _resource(
            "vector_index",
            "x",
            full_name="main.officeqa_benchmark.treasury_chunks_vs_index",
        ),
    ]
    matches = match_text_to_resources(intent, resources)
    assert len(matches) == 1
    assert matches[0].score == 90
    assert matches[0].matched_via == "fqn_ci"


def test_match_text_to_resources_name_substring_match() -> None:
    intent = "Use the treasury_chunks_vs_index for the queries."
    resources = [
        _resource(
            "vector_index",
            "treasury_chunks_vs_index",
            full_name="main.officeqa_benchmark.treasury_chunks_vs_index",
        ),
    ]
    matches = match_text_to_resources(intent, resources)
    assert len(matches) == 1
    assert matches[0].score == 80
    assert matches[0].matched_via == "name_exact"


def test_match_text_to_resources_skips_below_min_score() -> None:
    intent = "Random question about something."
    resources = [
        _resource(
            "vector_index", "treasury_index", full_name="main.x.treasury_index",
        ),
    ]
    assert match_text_to_resources(intent, resources) == []


def test_match_text_to_resources_returns_sorted_by_score() -> None:
    intent = "Use main.a.b and also random_name in the answer."
    resources = [
        # 60 — case-insensitive substring "random_name" in intent (lowercased)
        _resource("genie_space", "Random_Name", full_name="random.name.id"),
        # 100 — exact FQN match
        _resource("vector_index", "b", full_name="main.a.b"),
    ]
    matches = match_text_to_resources(intent, resources)
    assert [m.score for m in matches] == [100, 60]


def test_match_text_to_resources_empty_inputs() -> None:
    assert match_text_to_resources("", []) == []
    assert match_text_to_resources("anything", []) == []
    assert match_text_to_resources("", [_resource("vector_index", "x")]) == []


def test_match_text_to_resources_generic_across_kinds() -> None:
    """The matcher must not hardcode any kind-specific behavior."""
    intent = (
        "Index main.vs.alpha, table delta.warehouse.beta, "
        "and genie space gen.team.gamma."
    )
    resources = [
        _resource("vector_index", "alpha", full_name="main.vs.alpha"),
        _resource("delta_table", "beta", full_name="delta.warehouse.beta"),
        _resource("genie_space", "gamma", full_name="gen.team.gamma"),
    ]
    matches = match_text_to_resources(intent, resources)
    kinds = {m.resource.kind for m in matches}
    assert kinds == {"vector_index", "delta_table", "genie_space"}


# ---------------------------------------------------------------------------
# EmitGroundedAssetsTool
# ---------------------------------------------------------------------------


@pytest.fixture
def ctx() -> ToolContext:
    return ToolContext(query="")


async def test_emit_grounded_assets_merges_with_existing(ctx: ToolContext) -> None:
    """Grounded assets are merged with the existing designer_assets payload."""
    existing = {
        "assets": [
            {
                "kind": "delta_table",
                "full_name": "user.selected.table",
                "usage": "required",
            }
        ],
        "count": 1,
    }
    written: list[Any] = []
    tool = EmitGroundedAssetsTool(
        asset_getter=lambda: existing,
        designer_assets_setter=written.append,
    )
    args = {
        "matches": [
            {
                "kind": "vector_index",
                "full_name": "main.officeqa_benchmark.treasury_chunks_vs_index",
                "matched_text": "main.officeqa_benchmark.treasury_chunks_vs_index",
            }
        ],
        "unresolved": [],
    }
    result = await tool.execute(args, ctx)
    body = json.loads(result.content)
    assert body["merged_count"] == 2
    assert body["added_count"] == 1
    assert body["added"][0]["kind"] == "vector_index"
    # State setter received the merged payload.
    assert written
    merged_payload = written[0]
    full_names = [a["full_name"] for a in merged_payload["assets"]]
    assert "user.selected.table" in full_names
    assert "main.officeqa_benchmark.treasury_chunks_vs_index" in full_names


async def test_emit_grounded_assets_dedupes_against_existing(ctx: ToolContext) -> None:
    existing = {
        "assets": [
            {
                "kind": "vector_index",
                "full_name": "main.cat.idx",
                "usage": "required",
            }
        ],
        "count": 1,
    }
    written: list[Any] = []
    tool = EmitGroundedAssetsTool(
        asset_getter=lambda: existing,
        designer_assets_setter=written.append,
    )
    # Caller emits the same identity (case-insensitive) — no duplicate added.
    args = {
        "matches": [
            {"kind": "vector_index", "full_name": "MAIN.CAT.IDX"},
        ],
    }
    result = await tool.execute(args, ctx)
    body = json.loads(result.content)
    assert body["added_count"] == 0
    # Existing required asset is preserved (not overwritten by grounded copy).
    assert not written  # no write because nothing was added


async def test_emit_grounded_assets_empty_matches_writes_nothing(
    ctx: ToolContext,
) -> None:
    """matches=[] is a valid emission (says: nothing to ground)."""
    written: list[Any] = []
    tool = EmitGroundedAssetsTool(
        asset_getter=lambda: {"assets": [], "count": 0},
        designer_assets_setter=written.append,
    )
    result = await tool.execute({"matches": []}, ctx)
    body = json.loads(result.content)
    assert body["merged_count"] == 0
    assert body["added_count"] == 0
    assert not written


async def test_emit_grounded_assets_drops_entries_missing_identity(
    ctx: ToolContext,
) -> None:
    written: list[Any] = []
    tool = EmitGroundedAssetsTool(
        asset_getter=lambda: {"assets": [], "count": 0},
        designer_assets_setter=written.append,
    )
    args = {
        "matches": [
            {"kind": "vector_index"},  # missing full_name/source_id/name
            {"kind": "vector_index", "full_name": "ok.cat.idx"},
        ],
    }
    result = await tool.execute(args, ctx)
    body = json.loads(result.content)
    assert body["added_count"] == 1
    assert body["rejected"]
    assert body["added"][0]["full_name"] == "ok.cat.idx"


async def test_emit_grounded_assets_rejects_unknown_kind(ctx: ToolContext) -> None:
    """Invalid kinds are rejected via DesignerAsset validation, not silently
    coerced. The good entries still land."""
    written: list[Any] = []
    tool = EmitGroundedAssetsTool(
        asset_getter=lambda: {"assets": [], "count": 0},
        designer_assets_setter=written.append,
    )
    args = {
        "matches": [
            {"kind": "totally_fake_kind", "full_name": "bad.kind.entry"},
            {"kind": "delta_table", "full_name": "good.kind.entry"},
        ],
    }
    result = await tool.execute(args, ctx)
    body = json.loads(result.content)
    assert body["added_count"] == 1
    assert body["added"][0]["full_name"] == "good.kind.entry"
    assert body["rejected"] and "totally_fake_kind" in body["rejected"][0]["entry"]


async def test_emit_grounded_assets_carries_unresolved(ctx: ToolContext) -> None:
    """The LLM's ``unresolved`` list rides through to the result data so
    downstream nodes can surface the gap."""
    written: list[Any] = []
    tool = EmitGroundedAssetsTool(
        asset_getter=lambda: {"assets": [], "count": 0},
        designer_assets_setter=written.append,
    )
    args = {
        "matches": [],
        "unresolved": ["main.unknown.index", "  ", "main.other.thing"],
    }
    result = await tool.execute(args, ctx)
    body = json.loads(result.content)
    assert body["unresolved"] == ["main.unknown.index", "main.other.thing"]


async def test_emit_grounded_assets_no_setter_still_returns_payload(
    ctx: ToolContext,
) -> None:
    """Used in unit tests without a state setter — payload still returned."""
    tool = EmitGroundedAssetsTool(asset_getter=lambda: None)
    args = {
        "matches": [{"kind": "vector_index", "full_name": "ok.cat.idx"}],
    }
    result = await tool.execute(args, ctx)
    body = json.loads(result.content)
    assert body["added_count"] == 1


def test_emit_grounded_assets_tool_definition_shape() -> None:
    tool = EmitGroundedAssetsTool()
    definition = tool.definition
    assert definition.name == "emit_grounded_assets"
    assert definition.source_type == "builtin"
    properties = (definition.parameters or {}).get("properties") or {}
    assert "matches" in properties
    assert "unresolved" in properties
