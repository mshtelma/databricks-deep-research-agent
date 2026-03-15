"""Integration tests for classical_lite grounded enterprise workflows."""

from __future__ import annotations

from pathlib import Path

import pytest

from databricks_deep_research.llm.client import FrameworkLLMClient
from tests.integration.conftest import (
    RealVectorSearchTool,
    requires_databricks,
)
from tests.integration.enterprise_helpers import (
    EnterpriseDatasetProfile,
    assert_classical_enterprise_output,
    assert_enterprise_baseline,
    build_enterprise_registry,
    print_enterprise_case_diagnostics,
    run_enterprise_case,
)


@pytest.mark.integration
class TestClassicalEnterpriseWorkflow:
    """Real enterprise workflows using classical draft + post-synthesis grounding."""

    @requires_databricks
    @pytest.mark.asyncio
    @pytest.mark.timeout(600)
    async def test_classical_transcript_enterprise_research(
        self,
        llm_client: FrameworkLLMClient,
        real_transcript_vector_search_tool: RealVectorSearchTool,
        examples_dir: Path,
    ) -> None:
        """Transcript-based vector search should work through the classical synthesizer."""
        profile = EnterpriseDatasetProfile(
            label="Classical transcript VS",
            mode="classical",
            query=(
                "What did Kroger management emphasize in recent earnings call "
                "transcripts about guidance, digital growth, and business performance?"
            ),
            expected_tool_name="vector_search",
            required_keywords=("kroger", "guidance", "digital", "management", "earnings"),
            required_term_groups=(
                ("kroger",),
                ("guidance", "outlook"),
                ("digital", "ecommerce", "loyalty"),
            ),
            min_report_length=200,
        )
        result = await run_enterprise_case(
            workflow_path=examples_dir / "classical_enterprise_vector_search.yaml",
            llm_client=llm_client,
            profile=profile,
            enterprise_tools=[real_transcript_vector_search_tool],
            tool_registry=build_enterprise_registry(real_transcript_vector_search_tool),
        )

        assert_enterprise_baseline(result)
        assert_classical_enterprise_output(result)
        print_enterprise_case_diagnostics(result)

    @requires_databricks
    @pytest.mark.asyncio
    @pytest.mark.timeout(600)
    async def test_classical_genie_enterprise_research(
        self,
        llm_client: FrameworkLLMClient,
        real_genie_tool: object,
        examples_dir: Path,
    ) -> None:
        """Genie-backed enterprise research should produce a classical report with enterprise sources."""
        profile = EnterpriseDatasetProfile(
            label="Classical Genie FSI",
            mode="classical",
            query=(
                "Analyze the FSI portfolio by industry and market capitalization, "
                "and highlight notable companies represented in the portfolio."
            ),
            expected_tool_name="genie",
            required_keywords=("portfolio", "industry", "market", "company", "capital"),
            required_term_groups=(
                ("portfolio",),
                ("industry", "sector"),
                ("market", "capital", "capitalization"),
            ),
            min_report_length=200,
        )
        result = await run_enterprise_case(
            workflow_path=examples_dir / "genie_enterprise_research.yaml",
            llm_client=llm_client,
            profile=profile,
            enterprise_tools=[real_genie_tool],
            tool_registry=build_enterprise_registry(real_genie_tool),
        )

        assert_enterprise_baseline(result)
        assert_classical_enterprise_output(result)
        print_enterprise_case_diagnostics(result)

    @requires_databricks
    @pytest.mark.asyncio
    @pytest.mark.timeout(600)
    async def test_classical_knowledge_base_vector_search_smoke(
        self,
        llm_client: FrameworkLLMClient,
        real_knowledge_base_vector_search_tool: RealVectorSearchTool,
        examples_dir: Path,
    ) -> None:
        """Optional smoke test for a structurally different vector-search dataset."""
        profile = EnterpriseDatasetProfile(
            label="Classical knowledge-base VS",
            mode="classical",
            query=(
                "Summarize the network performance optimization model and hotspot "
                "device guidance from the available manuals and product documents."
            ),
            expected_tool_name="vector_search",
            required_keywords=("network", "optimization", "device", "hotspot", "manual"),
            required_term_groups=(
                ("network",),
                ("optimization", "performance"),
                ("device", "hotspot", "manual"),
            ),
            min_report_length=200,
        )
        result = await run_enterprise_case(
            workflow_path=examples_dir / "classical_enterprise_vector_search.yaml",
            llm_client=llm_client,
            profile=profile,
            enterprise_tools=[real_knowledge_base_vector_search_tool],
            tool_registry=build_enterprise_registry(real_knowledge_base_vector_search_tool),
        )

        assert_enterprise_baseline(result)
        assert_classical_enterprise_output(result)
        print_enterprise_case_diagnostics(result)
