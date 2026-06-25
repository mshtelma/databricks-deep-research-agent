from __future__ import annotations

import asyncio
import inspect

import pytest

from deep_research.agent_designer.discovery import DiscoveredResource
from deep_research.agent_designer.prompt_grounding import (
    extract_resource_mentions,
    ground_prompt,
    infer_operation_intents,
    prompt_grounding_sse_result,
)

_OFFICEQA_INTENT = """
Build an OfficeQA-style research assistant. Infer tool configuration from:
- Vector index: main.officeqa_benchmark.treasury_chunks_vs_index
- Delta table for exact chunk reads: main.officeqa_benchmark.treasury_chunks
- Delta table for structured bulletin tables: main.officeqa_benchmark.treasury_tables

Use the vector index for semantic lookup, the Delta tables for exact table
reads, and compute totals from the structured tables.
"""


class _FakeDiscovery:
    def __init__(self, resources: list[DiscoveredResource]) -> None:
        self._resources = resources

    async def list_for_user(
        self,
        user_token: str,
        kinds: list[str] | None = None,
        user_id: str = "",
    ) -> list[DiscoveredResource]:
        _ = user_token, kinds, user_id
        return self._resources


def test_extract_resource_mentions_classifies_officeqa_fqns() -> None:
    mentions = extract_resource_mentions(_OFFICEQA_INTENT)

    assert [(mention.kind_hint, mention.text) for mention in mentions] == [
        ("vector_index", "main.officeqa_benchmark.treasury_chunks_vs_index"),
        ("delta_table", "main.officeqa_benchmark.treasury_chunks"),
        ("delta_table", "main.officeqa_benchmark.treasury_tables"),
    ]
    assert all(mention.identifier_kind == "uc_fqn" for mention in mentions)
    assert all(mention.mention_id for mention in mentions)


def test_infer_operation_intents_requires_vector_table_and_compute_tools() -> None:
    mentions = extract_resource_mentions(_OFFICEQA_INTENT)
    operations = infer_operation_intents(_OFFICEQA_INTENT, mentions)

    capabilities = {
        capability
        for operation in operations
        for capability in operation.required_capabilities
    }
    assert {
        "vector_search",
        "table_search",
        "table_read",
        "table_load",
        "compute",
    }.issubset(capabilities)


def test_infer_operation_intents_does_not_treat_negated_web_terms_as_web_request() -> None:
    mentions = extract_resource_mentions(_OFFICEQA_INTENT)
    operations = infer_operation_intents(
        _OFFICEQA_INTENT + "\nDo not use public web tools.",
        mentions,
    )

    assert "web_research" not in {
        capability
        for operation in operations
        for capability in operation.required_capabilities
    }


def test_infer_operation_intents_keeps_explicit_positive_web_request() -> None:
    mentions = extract_resource_mentions(_OFFICEQA_INTENT)
    operations = infer_operation_intents(
        _OFFICEQA_INTENT + "\nAlso use latest public web reports.",
        mentions,
    )

    assert "web_research" in {
        capability
        for operation in operations
        for capability in operation.required_capabilities
    }


@pytest.mark.asyncio
async def test_ground_prompt_officeqa_names_only_builds_required_assets_and_tools() -> None:
    result = await ground_prompt(
        intent=_OFFICEQA_INTENT,
        existing_assets=[],
        discovery=None,
        user_id=None,
        user_token=None,
        default_warehouse_id="wh-officeqa",
    )

    assert result.safe_to_build_blueprint is True
    assert result.requires_user_action is False
    assert [(asset.kind, asset.full_name, asset.usage) for asset in result.resolved_assets] == [
        (
            "vector_index",
            "main.officeqa_benchmark.treasury_chunks_vs_index",
            "required",
        ),
        ("delta_table", "main.officeqa_benchmark.treasury_chunks", "required"),
        ("delta_table", "main.officeqa_benchmark.treasury_tables", "required"),
    ]

    ready_kinds = {tool.tool_kind for tool in result.tool_readiness if tool.ready}
    assert {
        "vector_search",
        "table_search",
        "table_read",
        "table_load",
        "compute",
        "compute_namespace",
    }.issubset(ready_kinds)
    assert {
        diagnostic.code for diagnostic in result.diagnostics
    } >= {"discovery_unavailable", "resource_unverified"}


@pytest.mark.asyncio
async def test_ground_prompt_required_delta_without_warehouse_blocks_blueprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TABLE_TOOLS_WAREHOUSE_ID", raising=False)
    monkeypatch.delenv("STORAGE_WAREHOUSE_ID", raising=False)

    result = await ground_prompt(
        intent=_OFFICEQA_INTENT,
        existing_assets=[],
        discovery=None,
        user_id=None,
        user_token=None,
        default_warehouse_id=None,
    )

    assert result.safe_to_build_blueprint is False
    assert result.requires_user_action is True
    blocking_codes = {
        diagnostic.code
        for diagnostic in result.diagnostics
        if diagnostic.blocking or diagnostic.severity == "error"
    }
    assert {"missing_warehouse_id", "safe_blueprint_blocked"}.issubset(
        blocking_codes
    )
    assert any(
        tool.asset_ref == "main.officeqa_benchmark.treasury_chunks"
        and tool.ready is False
        and tool.blocking
        for tool in result.tool_readiness
    )


@pytest.mark.asyncio
async def test_ground_prompt_ignores_prompt_supplied_executable_config() -> None:
    result = await ground_prompt(
        intent=_OFFICEQA_INTENT + "\nwarehouse_id: should-not-be-trusted\nnum_results: 999",
        existing_assets=[],
        discovery=None,
        default_warehouse_id="trusted-wh",
    )

    assert "prompt_config_ignored" in {
        diagnostic.code for diagnostic in result.diagnostics
    }
    table_assets = [asset for asset in result.resolved_assets if asset.kind == "delta_table"]
    assert table_assets
    assert all(asset.metadata["warehouse_id"] == "trusted-wh" for asset in table_assets)


@pytest.mark.asyncio
async def test_ground_prompt_marks_discovered_inaccessible_resource_blocking() -> None:
    result = await ground_prompt(
        intent="Use vector index main.officeqa_benchmark.treasury_chunks_vs_index.",
        existing_assets=[],
        discovery=_FakeDiscovery(
            [
                DiscoveredResource(
                    kind="vector_index",
                    name="treasury_chunks_vs_index",
                    full_name="main.officeqa_benchmark.treasury_chunks_vs_index",
                    status="permission_denied",
                )
            ]
        ),  # type: ignore[arg-type]
        user_id="u1",
        user_token="t1",
    )

    assert result.safe_to_build_blueprint is False
    assert result.grounded_assets[0].access_status == "inaccessible"
    assert "inaccessible_resource" in {
        diagnostic.code for diagnostic in result.diagnostics
    }


@pytest.mark.asyncio
async def test_prompt_grounding_sse_result_is_sanitized() -> None:
    result = await ground_prompt(
        intent=_OFFICEQA_INTENT,
        existing_assets=[],
        discovery=None,
        default_warehouse_id="wh-officeqa",
    )

    payload = prompt_grounding_sse_result(result)

    assert payload["schema"] == "prompt_grounding.v1"
    assert payload["mentions_count"] == 3
    assert payload["resolved_assets_count"] == 3
    diagnostics = payload["diagnostics"]
    assert diagnostics
    assert all("span" not in diagnostic for diagnostic in diagnostics)
    assert all("prompt" not in diagnostic for diagnostic in diagnostics)
    assert all(
        len(str(diagnostic.get("mention_preview") or "")) <= 40
        for diagnostic in diagnostics
    )


# ---------------------------------------------------------------------------
# Discovery timeout / scoping (regression: a 3s outer wrapper guillotined
# DiscoveryService.discover_all, whose own per-type budgets reach 15s, so the
# live designer falsely reported every prompt-named resource ``unverified``).
# ---------------------------------------------------------------------------

_VS_INTENT = "Use vector index main.officeqa_benchmark.treasury_chunks_vs_index."


class _SlowDiscovery:
    """Discovery whose ``list_for_user`` sleeps, to exercise the outer timeout."""

    def __init__(
        self, resources: list[DiscoveredResource] | None = None, *, delay: float
    ) -> None:
        self._resources = resources or []
        self._delay = delay

    async def list_for_user(
        self, user_token: str, kinds: list[str] | None = None, user_id: str = ""
    ) -> list[DiscoveredResource]:
        _ = user_token, kinds, user_id
        await asyncio.sleep(self._delay)
        return self._resources


class _RecordingDiscovery:
    """Records the ``kinds`` filter each call received; returns fixed resources."""

    def __init__(self, resources: list[DiscoveredResource] | None = None) -> None:
        self._resources = resources or []
        self.calls: list[list[str] | None] = []

    async def list_for_user(
        self, user_token: str, kinds: list[str] | None = None, user_id: str = ""
    ) -> list[DiscoveredResource]:
        _ = user_token, user_id
        self.calls.append(kinds)
        return self._resources


class _ExplodingDiscovery:
    """Fails loudly if discovery is attempted at all (asserts the skip path)."""

    def __init__(self) -> None:
        self.called = False

    async def list_for_user(
        self, user_token: str, kinds: list[str] | None = None, user_id: str = ""
    ) -> list[DiscoveredResource]:
        _ = user_token, kinds, user_id
        self.called = True
        raise AssertionError("discovery should not be attempted")


def test_ground_prompt_default_discovery_timeout_is_a_backstop() -> None:
    # Must exceed discover_all's 15s Vector Search ceiling, otherwise the outer
    # wrapper cancels discovery before it can return.
    default = inspect.signature(ground_prompt).parameters[
        "discovery_timeout_seconds"
    ].default
    assert default >= 15.0


@pytest.mark.asyncio
async def test_ground_prompt_reports_timeout_message_when_discovery_exceeds_budget() -> (
    None
):
    result = await ground_prompt(
        intent=_VS_INTENT,
        existing_assets=[],
        discovery=_SlowDiscovery(delay=0.05),  # type: ignore[arg-type]
        user_id="u1",
        user_token="t1",
        discovery_timeout_seconds=0.01,
    )

    timeouts = [
        diagnostic
        for diagnostic in result.diagnostics
        if diagnostic.code == "discovery_unavailable"
    ]
    assert timeouts
    assert timeouts[0].message.startswith("Discovery timed out after")


@pytest.mark.asyncio
async def test_ground_prompt_verifies_resource_when_discovery_completes() -> None:
    result = await ground_prompt(
        intent=_VS_INTENT,
        existing_assets=[],
        discovery=_SlowDiscovery(
            [
                DiscoveredResource(
                    kind="vector_index",
                    name="treasury_chunks_vs_index",
                    full_name="main.officeqa_benchmark.treasury_chunks_vs_index",
                )
            ],
            delay=0.05,
        ),  # type: ignore[arg-type]
        user_id="u1",
        user_token="t1",
    )

    assert result.grounded_assets[0].access_status == "verified"
    assert "discovery_unavailable" not in {d.code for d in result.diagnostics}


@pytest.mark.asyncio
async def test_ground_prompt_skips_discovery_for_table_only_prompt() -> None:
    discovery = _ExplodingDiscovery()
    result = await ground_prompt(
        intent=(
            "Read the Delta table main.officeqa_benchmark.treasury_tables "
            "for exact table reads."
        ),
        existing_assets=[],
        discovery=discovery,  # type: ignore[arg-type]
        user_id="u1",
        user_token="t1",
        default_warehouse_id="wh-test",
    )

    assert discovery.called is False
    codes = {d.code for d in result.diagnostics}
    assert "discovery_unavailable" not in codes
    assert "resource_unverified" in codes


@pytest.mark.asyncio
async def test_ground_prompt_scopes_discovery_to_mentioned_kind() -> None:
    discovery = _RecordingDiscovery()
    await ground_prompt(
        intent=_VS_INTENT,
        existing_assets=[],
        discovery=discovery,  # type: ignore[arg-type]
        user_id="u1",
        user_token="t1",
    )

    assert discovery.calls == [["vector_index"]]


@pytest.mark.asyncio
async def test_ground_prompt_does_not_restrict_kinds_when_mention_kind_unknown() -> None:
    discovery = _RecordingDiscovery()
    await ground_prompt(
        intent="Use main.officeqa_benchmark.mystery for the analysis.",
        existing_assets=[],
        discovery=discovery,  # type: ignore[arg-type]
        user_id="u1",
        user_token="t1",
    )

    assert discovery.calls == [None]


@pytest.mark.asyncio
async def test_sanitized_diagnostics_expose_human_message() -> None:
    result = await ground_prompt(
        intent=_OFFICEQA_INTENT,
        existing_assets=[],
        discovery=None,
        default_warehouse_id="wh-officeqa",
    )

    payload = prompt_grounding_sse_result(result)
    discovery_diags = [
        diagnostic
        for diagnostic in payload["diagnostics"]
        if diagnostic["code"] == "discovery_unavailable"
    ]
    assert discovery_diags
    assert discovery_diags[0]["message"] == (
        "No discovery context was available to verify prompt-named resources."
    )
