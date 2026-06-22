from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import UTC, datetime

import pytest
from databricks_deep_research.tools.catalog_types import CatalogCard, ProbeSample, SafeProbe
from databricks_deep_research.tools.protocol import ToolContext
from databricks_deep_research.workflow.definition import ToolDeclaration

from deep_research.agent_designer.tool_probe import (
    ProbeConfig,
    ProbeOrchestrator,
    sanitize_probe_output,
)

# Synthetic, AWS-key-shaped token used to exercise the sanitizer's redaction.
# Assembled from fragments so the literal never appears in source and trips
# secret scanners; the runtime value still matches tool_probe._AWS_KEY_RE.
_FAKE_AWS_KEY = "AKIA" + "1234567890ABCDEF"


class _Provider:
    catalog_cards: Mapping[str, CatalogCard] = {}

    def __init__(self, probe: SafeProbe | None) -> None:
        self.safe_probes = {"safe": probe, "none": None}


class _ProviderMap:
    catalog_cards: Mapping[str, CatalogCard] = {}

    def __init__(self, probes: Mapping[str, SafeProbe | None]) -> None:
        self.safe_probes = probes


class _OkProbe:
    async def run(
        self,
        *,
        config: dict[str, object],
        ctx: ToolContext,
        user_query: str | None,
    ) -> ProbeSample:
        assert ctx.read_only is True
        return ProbeSample(
            sample_input={"query": user_query or ""},
            sample_output=f"email user@example.com and key {_FAKE_AWS_KEY}",
            probed_at=datetime.now(UTC),
            status="ok",
        )


class _FailProbe:
    async def run(
        self,
        *,
        config: dict[str, object],
        ctx: ToolContext,
        user_query: str | None,
    ) -> ProbeSample:
        raise RuntimeError("boom")


class _SlowProbe:
    async def run(
        self,
        *,
        config: dict[str, object],
        ctx: ToolContext,
        user_query: str | None,
    ) -> ProbeSample:
        await asyncio.sleep(1)
        return ProbeSample(
            sample_input={},
            sample_output="late",
            probed_at=datetime.now(UTC),
            status="ok",
        )


@pytest.mark.asyncio
async def test_probe_skips_when_no_safe_probe() -> None:
    orchestrator = ProbeOrchestrator(providers=[_Provider(None)])
    samples = await orchestrator.probe(
        [ToolDeclaration(name="t", kind="none")],
        ctx=ToolContext(),
        user_query=None,
    )

    assert samples[0].status == "skipped"
    assert samples[0].reason == "no SafeProbe declared"


@pytest.mark.asyncio
async def test_probe_sanitizes_and_isolates_failures() -> None:
    orchestrator = ProbeOrchestrator(
        providers=[_ProviderMap({"ok": _OkProbe(), "fail": _FailProbe(), "none": None})],
        config=ProbeConfig(max_output_chars=200),
    )
    samples = await orchestrator.probe(
        [
            ToolDeclaration(name="ok", kind="ok"),
            ToolDeclaration(name="fail", kind="fail"),
            ToolDeclaration(name="missing", kind="none"),
        ],
        ctx=ToolContext(),
        user_query="hello",
    )

    assert samples[0].status == "ok"
    assert "[redacted-email]" in samples[0].sample_output
    assert "[redacted-aws-key]" in samples[0].sample_output
    assert samples[1].status == "error"
    assert samples[1].reason == "boom"
    assert samples[2].status == "skipped"


@pytest.mark.asyncio
async def test_probe_timeout_is_per_tool() -> None:
    orchestrator = ProbeOrchestrator(
        providers=[_Provider(_SlowProbe())],
        config=ProbeConfig(timeout_seconds=0.01),
    )
    samples = await orchestrator.probe(
        [ToolDeclaration(name="slow", kind="safe")],
        ctx=ToolContext(),
        user_query=None,
    )

    assert samples[0].status == "error"
    assert samples[0].reason == "timeout"


def test_sanitizer_redacts_pii_patterns_before_truncation() -> None:
    output = sanitize_probe_output(
        "a@example.com 123-45-6789 "
        f"eyJaaaaaaaaaaa.bbbbbbbbbbb.ccccccccccc {_FAKE_AWS_KEY} tail",
        max_chars=200,
    )

    assert "[redacted-email]" in output
    assert "[redacted-ssn]" in output
    assert "[redacted-jwt]" in output
    assert "[redacted-aws-key]" in output
