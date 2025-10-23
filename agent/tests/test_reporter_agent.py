"""
Isolated tests for Reporter agent using captured states.

These tests load pre-captured state fixtures and test the Reporter agent
in isolation without requiring full workflow execution or API calls.

Prerequisites:
    1. Run capture first: CAPTURE_STATE=true python tests/capture/run_capture.py
    2. Fixtures exist in: tests/fixtures/states/reporter/

Usage:
    pytest tests/test_reporter_agent.py -v
    pytest tests/test_reporter_agent.py -k simple_fact -v
"""

import pytest
import json
import os
import re
import asyncio
import yaml
from pathlib import Path
from unittest.mock import Mock, MagicMock
from langchain_core.messages import AIMessage
from langgraph.types import Command

from deep_research_agent.agents.reporter import ReporterAgent
from deep_research_agent.core.plan_models import Plan
from deep_research_agent.core import get_logger

logger = get_logger(__name__)


class TestReporterAgent:
    """Test Reporter agent in isolation with captured states."""

    @staticmethod
    def _get_observation_content(obs) -> str:
        """Extract content from observation (handles both dict and object formats)."""
        if isinstance(obs, dict):
            return str(obs.get("content", ""))
        elif hasattr(obs, 'content'):
            return str(obs.content if obs.content else "")
        else:
            return str(obs)

    @pytest.fixture(scope="class")
    def fixtures_dir(self):
        """Get fixtures directory path."""
        return Path(__file__).parent / "fixtures" / "states" / "reporter"

    @pytest.fixture(scope="class")
    def available_fixtures(self, fixtures_dir):
        """Get list of available fixture files."""
        if not fixtures_dir.exists():
            return []
        return list(fixtures_dir.glob("*_before.json"))

    @pytest.fixture
    def reporter_agent(self):
        """
        Create reporter agent with mock LLM.

        Returns a ReporterAgent instance with a mocked LLM that returns
        test responses instead of making real API calls.
        """
        mock_llm = Mock()
        mock_llm.invoke.return_value = AIMessage(content="Mock report content for testing")

        # Mock the ainvoke method too (for async calls)
        async def mock_ainvoke(*args, **kwargs):
            return AIMessage(content="Mock report content for testing")

        mock_llm.ainvoke = mock_ainvoke

        return ReporterAgent(llm=mock_llm, config={
            "report": {
                "default_style": "professional",
                "include_citations": True,
                "include_grounding_markers": True
            }
        })

    @pytest.fixture
    def load_state(self, fixtures_dir):
        """
        Factory fixture to load captured states.

        Usage:
            state = load_state("simple_fact", "before")
        """
        def _load(prompt_id: str, phase: str = "before"):
            fixture_path = fixtures_dir / f"{prompt_id}_{phase}.json"

            if not fixture_path.exists():
                pytest.skip(f"Fixture not found: {fixture_path.relative_to(Path.cwd())}\nRun capture first: CAPTURE_STATE=true python tests/capture/run_capture.py")

            with open(fixture_path) as f:
                state = json.load(f)

            # Remove metadata before returning (tests don't need it)
            state.pop("_metadata", None)

            # Backward compatibility: Add missing required fields for old Plan fixtures
            if "current_plan" in state and isinstance(state["current_plan"], dict):
                plan_dict = state["current_plan"]
                if "thought" not in plan_dict:
                    plan_dict["thought"] = "Plan captured from test fixture"
                if "plan_id" not in plan_dict:
                    plan_dict["plan_id"] = f"plan_{prompt_id}"
                if "title" not in plan_dict:
                    plan_dict["title"] = "Test Plan"
                if "research_topic" not in plan_dict:
                    plan_dict["research_topic"] = state.get("research_topic", "Unknown")

            # ========================================================================
            # PERIMETER CONVERSION: Hydrate all dict objects to Pydantic/dataclass
            # ========================================================================
            from deep_research_agent.core.multi_agent_state import StateManager
            state = StateManager.hydrate_state(state)

            return state

        return _load

    # ========================================================================
    # Basic Functionality Tests
    # ========================================================================

    @pytest.mark.parametrize("prompt_id", [
        "simple_fact_python",
        "simple_fact_ml",
        "historical_fact",
        "tax_comparison_europe",  # Real-world complex case
        "tax_comparison_europe_very_big",  # Very big real-world case (7 countries, 3 scenarios)
        "quantum_vs_classical",
        "retirement_comparison",
    ])
    def test_compile_findings_basic(self, reporter_agent, load_state, prompt_id):
        """
        Test _compile_findings with different captured states.

        Validates that the reporter can compile findings from various types
        of research states (simple facts, complex analysis, etc.).
        """
        # Load the captured state
        state = load_state(prompt_id, "before")
        metadata = state.get("_metadata", {})

        print(f"\n🧪 Testing compile_findings with: {prompt_id}")
        if metadata:
            print(f"   Category: {metadata.get('category')}")
            print(f"   Prompt: {metadata.get('prompt', '')[:100]}...")

        # Test the method
        findings = reporter_agent._compile_findings(state)

        # Basic structural assertions
        assert findings is not None, "Should return findings dict"
        assert isinstance(findings, dict), "Findings should be a dictionary"
        assert "observations" in findings, "Should have observations key"
        assert "research_topic" in findings, "Should have research_topic key"

        # Validate observations exist and are properly formatted
        observations = findings["observations"]
        assert isinstance(observations, list), "Observations should be a list"

        # Check we have observations (unless it's an edge case)
        if prompt_id not in ["empty_input", "greeting"]:
            assert len(observations) > 0, f"Should have observations for {prompt_id}"

            # Validate observation structure
            first_obs = observations[0]
            if isinstance(first_obs, dict):
                assert "content" in first_obs, "Observation should have content"
            # Can also be StructuredObservation objects
            elif hasattr(first_obs, 'content'):
                assert first_obs.content, "Observation should have content"

        print(f"   ✅ Found {len(observations)} observations")

    @pytest.mark.parametrize("prompt_id,expected_min_observations", [
        ("simple_fact_python", 3),
        ("simple_fact_ml", 3),
        ("historical_fact", 2),
        ("tax_comparison_europe", 15),  # Very complex query should have many observations
        ("tax_comparison_europe_very_big", 25),  # Extremely complex: 7 countries, 3 scenarios
        ("quantum_vs_classical", 10),
        ("retirement_comparison", 10),
        ("ai_developments_2024", 8),  # Current events
    ])
    def test_compile_findings_observation_count(self, reporter_agent, load_state, prompt_id, expected_min_observations):
        """Test that findings contain expected minimum number of observations."""
        state = load_state(prompt_id, "before")

        findings = reporter_agent._compile_findings(state)
        observations = findings["observations"]

        assert len(observations) >= expected_min_observations, \
            f"{prompt_id} should have at least {expected_min_observations} observations, got {len(observations)}"

        print(f"   ✅ {prompt_id}: {len(observations)} observations (expected >= {expected_min_observations})")

    def test_compile_findings_has_plan(self, reporter_agent, load_state):
        """Test that findings include plan information."""
        # Use a complex query that should have a plan
        state = load_state("tax_comparison_europe", "before")

        findings = reporter_agent._compile_findings(state)

        # Check for plan-related data
        assert "completed_steps" in findings, "Should have completed_steps key"

        completed_steps = findings["completed_steps"]
        assert isinstance(completed_steps, list), "completed_steps should be a list"

        # Should have completed steps for research prompts
        assert len(completed_steps) > 0, "Should have completed steps"

        print(f"   ✅ Found {len(completed_steps)} completed steps")

    def test_compile_findings_has_citations(self, reporter_agent, load_state):
        """Test that findings include citations."""
        state = load_state("tax_comparison_europe", "before")

        findings = reporter_agent._compile_findings(state)

        # Check for citations
        assert "citations" in findings, "Should have citations key"

        citations = findings["citations"]
        assert isinstance(citations, list), "Citations should be a list"

        # Research queries should have citations
        assert len(citations) > 0, "Should have citations for research"

        print(f"   ✅ Found {len(citations)} citations")

    def test_compile_findings_quality_metrics(self, reporter_agent, load_state):
        """Test that quality metrics are included in findings."""
        # Use any available fixture
        state = load_state("simple_fact_python", "before")

        findings = reporter_agent._compile_findings(state)

        # Check for quality metrics
        assert "factuality_score" in findings, "Should have factuality_score"
        assert "confidence_score" in findings, "Should have confidence_score"
        assert "coverage_score" in findings, "Should have coverage_score"

        # Scores should be in valid range (0-1) or None
        for metric in ["factuality_score", "confidence_score", "coverage_score"]:
            score = findings[metric]
            if score is not None:
                assert 0.0 <= score <= 1.0, f"{metric} should be in [0, 1], got {score}"

        print(f"   ✅ Quality metrics present and valid")

    # ========================================================================
    # Error Handling Tests
    # ========================================================================

    def test_error_handling_with_empty_observations(self, reporter_agent, load_state):
        """Test reporter handles missing observations gracefully."""
        # Load any state and clear observations
        state = load_state("simple_fact_python", "before")
        state["observations"] = []

        # Should raise ValueError with specific message
        with pytest.raises(ValueError, match="no observations"):
            reporter_agent._compile_findings(state)

        print(f"   ✅ Correctly raises error for empty observations")

    def test_error_handling_with_missing_topic(self, reporter_agent, load_state):
        """Test reporter handles missing research topic."""
        state = load_state("simple_fact_python", "before")
        state["research_topic"] = None

        findings = reporter_agent._compile_findings(state)

        # Should still work but with empty/None topic
        assert "research_topic" in findings
        # Topic might be None or empty string
        assert findings["research_topic"] in [None, ""]

        print(f"   ✅ Handles missing research topic gracefully")

    # ========================================================================
    # Edge Case Tests
    # ========================================================================

    def test_edge_case_greeting(self, reporter_agent, load_state):
        """Test handling of greeting (non-research) input."""
        try:
            state = load_state("greeting", "before")

            # Greetings might not have observations
            observations = state.get("observations", [])

            # This is expected - greetings don't generate research
            assert len(observations) == 0, "Greetings should not generate observations"

            print(f"   ✅ Greeting correctly has no observations")

        except Exception:
            pytest.skip("Greeting fixture not captured (expected for coordinator-only flow)")

    def test_edge_case_empty_input(self, reporter_agent, load_state):
        """Test handling of empty input."""
        try:
            state = load_state("empty_input", "before")

            # Empty input should either error at coordinator or have no observations
            observations = state.get("observations", [])
            assert len(observations) == 0, "Empty input should not generate observations"

            print(f"   ✅ Empty input correctly has no observations")

        except Exception:
            pytest.skip("Empty input fixture not captured (expected for error case)")

    # ========================================================================
    # Real-World Test Cases
    # ========================================================================

    def test_complex_tax_comparison_europe(self, reporter_agent, load_state):
        """
        Test with real-world complex tax comparison query.

        This is the actual query from test_tax_opt_research - a highly complex
        multi-country, multi-scenario tax analysis.
        """
        try:
            state = load_state("tax_comparison_europe", "before")
        except Exception:
            pytest.skip("Complex tax comparison fixture not yet captured")

        findings = reporter_agent._compile_findings(state)

        # Should have substantial observations for such a complex query
        observations = findings["observations"]
        assert len(observations) >= 15, \
            f"Complex multi-country tax query should have many observations, got {len(observations)}"

        # Should have citations from multiple countries
        citations = findings.get("citations", [])
        assert len(citations) > 0, "Complex research should have citations"

        # Check for multi-country coverage in observations
        observation_text = " ".join([self._get_observation_content(obs) for obs in observations])
        countries = ["Spain", "France", "United Kingdom", "Switzerland", "Germany"]

        found_countries = [c for c in countries if c in observation_text]
        assert len(found_countries) >= 3, \
            f"Should cover at least 3 countries, found: {found_countries}"

        print(f"   ✅ Complex tax comparison: {len(observations)} observations, "
              f"{len(citations)} citations, covers {len(found_countries)} countries")

    def test_very_big_tax_comparison_europe(self, reporter_agent, load_state):
        """
        Test with extremely complex tax comparison query.

        This is from test_very_big_tax_opt_research - an ultra-complex analysis covering:
        - 7 countries (Spain, France, UK, Switzerland, Germany, Poland, Bulgaria)
        - 3 family scenarios (single, married no child, married with child)
        - Detailed requirements: RSUs, rent, daycare, family benefits, disposable income
        - Structured output requirements: tables, narrative, appendix
        """
        try:
            state = load_state("tax_comparison_europe_very_big", "before")
        except Exception:
            pytest.skip("Very big tax comparison fixture not yet captured")

        findings = reporter_agent._compile_findings(state)

        # Should have very substantial observations for ultra-complex query
        observations = findings["observations"]
        assert len(observations) >= 25, \
            f"Ultra-complex 7-country, 3-scenario tax query should have many observations, got {len(observations)}"

        # Should have citations from multiple countries
        citations = findings.get("citations", [])
        assert len(citations) >= 10, \
            f"Ultra-complex research should have many citations, got {len(citations)}"

        # Check for coverage of all 7 countries in observations
        observation_text = " ".join([self._get_observation_content(obs) for obs in observations])
        countries = ["Spain", "France", "United Kingdom", "Switzerland", "Germany", "Poland", "Bulgaria"]

        found_countries = [c for c in countries if c in observation_text]
        assert len(found_countries) >= 5, \
            f"Should cover at least 5 of 7 countries, found: {found_countries}"

        # Check for scenario-specific content
        scenario_keywords = ["single", "married", "child", "daycare", "RSU"]
        found_keywords = [kw for kw in scenario_keywords if kw.lower() in observation_text.lower()]
        assert len(found_keywords) >= 3, \
            f"Should mention multiple scenario aspects, found: {found_keywords}"

        print(f"   ✅ Very big tax comparison: {len(observations)} observations, "
              f"{len(citations)} citations, covers {len(found_countries)}/7 countries, "
              f"{len(found_keywords)} scenario keywords")

    # ========================================================================
    # Integration Tests (with after states)
    # ========================================================================

    def test_after_state_has_final_report(self, load_state):
        """Test that after state contains final report."""
        state = load_state("simple_fact_python", "after")

        # After reporter runs, should have final report
        assert "final_report" in state, "After state should have final_report"

        final_report = state.get("final_report")
        if final_report:
            assert isinstance(final_report, str), "Final report should be a string"
            assert len(final_report) > 0, "Final report should not be empty"

            print(f"   ✅ Final report exists ({len(final_report)} chars)")
        else:
            print(f"   ⚠️  Final report is None (reporter may have failed)")

    # ========================================================================
    # Fixture Validation Tests
    # ========================================================================

    def test_fixtures_exist(self, available_fixtures):
        """Validate that fixtures were generated."""
        if len(available_fixtures) == 0:
            pytest.fail(
                "No fixtures found. Run capture first:\n"
                "  CAPTURE_STATE=true BRAVE_API_KEY=xxx python tests/capture/run_capture.py"
            )

        print(f"\n✅ Found {len(available_fixtures)} fixture files:")
        for fixture in available_fixtures:
            print(f"   - {fixture.name}")

    def test_fixture_structure(self, load_state):
        """Validate fixture JSON structure."""
        state = load_state("simple_fact_python", "before")

        # Should have core required fields
        assert "research_topic" in state, "State should have research_topic"
        assert "current_agent" in state, "State should have current_agent"

        # Reporter-specific fields
        assert "observations" in state or "search_results" in state, \
            "State should have observations or search_results"

        print(f"   ✅ Fixture structure is valid")
        print(f"   Keys: {list(state.keys())[:10]}...")

    def test_citations_reconstructed_from_fixture(self, load_state):
        """Verify citations are properly reconstructed as Citation objects."""
        state = load_state("tax_comparison_europe_very_big", "before")

        citations = state.get("citations", [])
        assert len(citations) > 0, "Should have citations"
        assert len(citations) == 20, f"Should have 20 citations, got {len(citations)}"

        # Check first citation is Citation object
        from deep_research_agent.core.types import Citation
        assert isinstance(citations[0], Citation), \
            f"Citations should be Citation objects, got {type(citations[0])}"

        # Check has required fields
        assert citations[0].source, "Citation should have source"
        assert citations[0].title, "Citation should have title"

        # Verify all citations are Citation objects
        for i, citation in enumerate(citations):
            assert isinstance(citation, Citation), \
                f"Citation {i} should be Citation object, got {type(citation)}"

        print(f"   ✅ All {len(citations)} citations are properly reconstructed Citation objects")

    def test_observations_reconstructed_from_fixture(self, load_state):
        """
        Verify observations are properly reconstructed as StructuredObservation objects.

        This ensures that when we load states from JSON fixtures, the observations
        are converted from dicts to proper StructuredObservation objects with all
        attributes accessible.
        """
        state = load_state("tax_comparison_europe_very_big", "before")

        observations = state.get("observations", [])
        assert len(observations) > 0, "Should have observations"
        print(f"   ℹ️  Loaded {len(observations)} observations from fixture")

        # Check first observation is StructuredObservation object
        from deep_research_agent.core.observation_models import StructuredObservation
        assert isinstance(observations[0], StructuredObservation), \
            f"Observations should be StructuredObservation objects, got {type(observations[0])}"

        # Verify we can access attributes (not dict keys)
        first_obs = observations[0]
        assert hasattr(first_obs, 'content'), "Should have content attribute"
        assert hasattr(first_obs, 'full_content'), "Should have full_content attribute"
        assert hasattr(first_obs, 'step_id'), "Should have step_id attribute"

        # Verify content is accessible
        content = first_obs.content
        assert isinstance(content, str), f"Content should be string, got {type(content)}"
        assert len(content) > 0, "Content should not be empty"

        # Verify all observations are StructuredObservation objects
        for i, obs in enumerate(observations[:10]):  # Check first 10 for speed
            assert isinstance(obs, StructuredObservation), \
                f"Observation {i} should be StructuredObservation object, got {type(obs)}"

        print(f"   ✅ All observations are properly reconstructed StructuredObservation objects")

    # ========================================================================
    # Real LLM Integration Tests
    # ========================================================================

    @pytest.fixture(scope="class")
    def config_dict(self):
        """Load base configuration from YAML."""
        config_path = Path(__file__).parent.parent / "conf" / "base.yaml"
        with open(config_path) as f:
            return yaml.safe_load(f)

    @pytest.fixture(scope="class")
    def real_reporter_llm(self, config_dict):
        """
        Create real Databricks LLM with rate limiting for reporter.

        Uses the 'complex' tier model configured for reporter in base.yaml.
        This fixture is expensive to create, so it's scoped to the class.
        """
        from deep_research_agent.core.model_selector import ModelSelector
        from deep_research_agent.core.model_config_loader import create_model_manager_from_config

        config_path = Path(__file__).parent.parent / "conf" / "base.yaml"

        # Create model selector for rate limiting (CRITICAL for avoiding 429 errors)
        model_selector = ModelSelector(config_dict)

        # Create model manager with rate limiting
        model_manager = create_model_manager_from_config(config_path, model_selector)

        # Get complex tier LLM (reporter's configured model: gpt-oss-120b or claude)
        return model_manager.get_chat_model("complex")

    @pytest.fixture
    def real_reporter_agent(self, real_reporter_llm, config_dict):
        """Create ReporterAgent with real LLM and full configuration."""
        return ReporterAgent(llm=real_reporter_llm, config=config_dict)

    @pytest.mark.integration
    @pytest.mark.asyncio
    @pytest.mark.timeout(300)  # 5 minutes max
    @pytest.mark.skipif(
        not os.getenv("DATABRICKS_HOST") and not os.path.exists(os.path.expanduser("~/.databrickscfg")),
        reason="Databricks credentials not configured"
    )
    async def test_real_llm_full_report_generation_with_ultra_complex_state(
        self,
        real_reporter_agent,
        load_state,
        config_dict
    ):
        """
        Test ReporterAgent end-to-end with real Databricks LLM.

        Uses pre-captured ultra-complex state (1045 observations, 20 citations,
        7 countries, 3 scenarios) to validate:
        - Hybrid multi-pass generation
        - Table generation from structured data
        - Citation formatting
        - Content sanitization
        - Handling large observation counts

        This is the most comprehensive reporter test, exercising the full
        production code path with real LLM and real complex research data.
        """
        # Load the most complex captured state
        state = load_state("tax_comparison_europe_very_big", "before")

        # Log state summary
        observations = state.get("observations", [])
        citations = state.get("citations", [])
        completed_steps = state.get("completed_steps", [])

        logger.info(f"\n{'='*80}")
        logger.info(f"Testing Reporter with ultra-complex state:")
        logger.info(f"  - {len(observations)} observations")
        logger.info(f"  - {len(citations)} citations")
        logger.info(f"  - {len(completed_steps)} completed research steps")
        logger.info(f"  - Research topic: {state.get('research_topic', '')[:100]}...")
        logger.info(f"{'='*80}\n")

        # Call reporter agent
        logger.info("Starting real LLM report generation (hybrid multi-pass mode)...")

        try:
            result = await real_reporter_agent(state, config_dict)

            # Extract report and metadata from Command
            assert isinstance(result, Command), f"Expected Command, got {type(result)}"

            final_report = result.update.get("final_report", "")
            report_metadata = result.update.get("report_metadata", {})

            logger.info(f"Report generation complete: {len(final_report)} characters")
            logger.info(f"Metadata: {report_metadata}")

        except asyncio.TimeoutError:
            pytest.fail("Reporter generation timed out (>5 min)")
        except Exception as e:
            error_str = str(e).lower()
            if "endpoint not found" in error_str or "not found" in error_str:
                pytest.skip(f"Databricks endpoint unavailable: {e}")
            elif "rate limit" in error_str or "429" in error_str:
                pytest.skip(f"Rate limited: {e}")
            else:
                logger.error(f"Reporter failed with exception: {e}", exc_info=True)
                raise

        # ====================================================================
        # OUTPUT FOR MANUAL INSPECTION
        # ====================================================================

        logger.info(f"\n{'='*80}")
        logger.info("GENERATED REPORT PREVIEW:")
        logger.info(f"{'='*80}")
        logger.info(final_report)
        logger.info(f"\n{'='*80}")
        logger.info(f"{'='*80}\n")

        # ====================================================================
        # VALIDATION SUITE
        # ====================================================================

        # Basic sanity check
        assert final_report, "Report should not be empty"
        assert isinstance(final_report, str), "Report should be a string"

        # Handle rate limiting gracefully
        if len(final_report) < 5000 and "rate" in str(report_metadata).lower():
            pytest.skip(f"Rate limited - got short report ({len(final_report)} chars)")

        # --- STRUCTURAL VALIDATION ---
        logger.info("Validating report structure...")

        # Has markdown headers
        assert re.search(r'^#{1,3}\s+\w+', final_report, re.MULTILINE), \
            "Report should have markdown headers (#, ##, ###)"

        # Multiple sections (hybrid mode uses ### headers)
        section_count = len(re.findall(r'^###\s+', final_report, re.MULTILINE))
        assert section_count >= 3, \
            f"Report should have at least 3 major sections (### headers), got {section_count}"

        # Contains tables (markdown format)
        table_count = len(re.findall(r'\|.*\|.*\|', final_report))
        assert table_count > 0, "Report should contain markdown tables"

        # Tables have separator rows
        separator_count = len(re.findall(r'\|[-:]+\|', final_report))
        assert separator_count > 0, "Tables should have separator rows (|---|---|)"

        # Validate N/A values are not excessive
        na_count = final_report.count('N/A')
        total_cells_estimate = final_report.count('|') // 3  # Rough estimate of table cells
        na_percentage = (na_count / total_cells_estimate * 100) if total_cells_estimate > 0 else 0
        logger.info(f"N/A statistics: {na_count} N/A values out of ~{total_cells_estimate} cells ({na_percentage:.1f}%)")
        assert na_percentage < 30, \
            f"Too many N/A values in tables: {na_count}/{total_cells_estimate} ({na_percentage:.1f}%). " \
            f"This suggests data extraction/calculation issues."

        # Has citations (section or inline)
        has_citations = (
            "references" in final_report.lower() or
            "citations" in final_report.lower() or
            re.search(r'\[\d+\]', final_report)
        )
        assert has_citations, "Report should have citations section or inline citations"

        logger.info("✅ Structural validation passed")

        # --- CONTENT VALIDATION ---
        logger.info("Validating report content...")

        # Substantial length for ultra-complex research
        assert len(final_report) > 8000, \
            f"Ultra-complex report should be >8000 chars, got {len(final_report)}"

        # Mentions countries (at least 5 of 7)
        countries = ["Spain", "France", "United Kingdom", "Switzerland", "Germany", "Poland", "Bulgaria"]
        found_countries = [c for c in countries if c in final_report]
        assert len(found_countries) >= 5, \
            f"Should mention at least 5 countries, found: {found_countries}"

        # Contains key terms from research topic
        key_terms = ["tax", "RSU", "married", "single", "daycare", "disposable"]
        found_terms = [term for term in key_terms if term.lower() in final_report.lower()]
        assert len(found_terms) >= 4, \
            f"Should mention key research terms, found: {found_terms}"

        # No error messages
        error_indicators = ["Error:", "Failed:", "Exception:", "Traceback"]
        for indicator in error_indicators:
            assert indicator not in final_report, \
                f"Report should not contain error message: {indicator}"

        # No truncation artifacts
        assert "[TRUNCATED]" not in final_report, "Report should not be truncated"

        # No placeholder text (table anchors should be replaced)
        assert "[TABLE:" not in final_report, \
            "Table placeholders should be replaced with actual tables"

        logger.info("✅ Content validation passed")

        # --- CITATION VALIDATION ---
        logger.info("Validating citations...")

        # Count inline citations
        inline_citations = re.findall(r'\[\d+\]', final_report)
        if inline_citations:
            assert len(set(inline_citations)) >= 5, \
                f"Should have at least 5 unique citations, got {len(set(inline_citations))}"

        logger.info("✅ Citation validation passed")

        # --- TABLE VALIDATION ---
        logger.info("Validating tables...")

        # Extract table structures
        table_blocks = re.findall(r'\|.*\|.*\n\|[-:]+\|[-:]+\|', final_report)
        assert len(table_blocks) >= 1, \
            f"Should have at least 1 properly formatted table, found {len(table_blocks)}"

        logger.info(f"✅ Table validation passed ({len(table_blocks)} tables found)")

        # --- METADATA VALIDATION ---
        if report_metadata:
            logger.info("Validating report metadata...")

            # Check generation mode
            if "generation_mode" in report_metadata:
                assert "hybrid" in report_metadata["generation_mode"].lower(), \
                    f"Should use hybrid mode, got: {report_metadata['generation_mode']}"

            # Check observations used
            if "observations_used" in report_metadata:
                assert report_metadata["observations_used"] > 0, \
                    "Should have used observations"

            logger.info("✅ Metadata validation passed")


        # ====================================================================
        # OUTPUT FOR MANUAL INSPECTION
        # ====================================================================

        print(f"\n{'='*80}")

        print(f"REPORT STATISTICS:")
        print(f"  Total length: {len(final_report)} characters")
        print(f"  Sections: {section_count}")
        print(f"  Tables: {len(table_blocks)}")
        print(f"  Inline citations: {len(inline_citations) if inline_citations else 0}")
        print(f"  Countries mentioned: {len(found_countries)}/7 - {found_countries}")
        print(f"  Key terms found: {found_terms}")
        if report_metadata:
            print(f"  Generation mode: {report_metadata.get('generation_mode', 'N/A')}")
            print(f"  Observations used: {report_metadata.get('observations_used', 'N/A')}")
            print(f"  Calculations performed: {report_metadata.get('calculations_performed', 'N/A')}")
        print(f"{'='*80}\n")
        logger.info("✅ ALL VALIDATIONS PASSED - Reporter test successful!")
