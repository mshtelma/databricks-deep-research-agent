"""End-to-end tests for the research orchestrator with real LLM and Brave Search.

These tests use REAL API calls to:
- Databricks LLM endpoints (via OpenAI SDK)
- Brave Search API

Requirements:
- .env file with DATABRICKS_TOKEN or DATABRICKS_CONFIG_PROFILE
- .env file with BRAVE_API_KEY

Run with:
    uv run pytest backend/tests/integration/test_e2e_research.py -v -s
"""

import os
from uuid import uuid4

import pytest

# Note: .env is loaded by conftest.py
from src.agent.orchestrator import (
    OrchestrationConfig,
    run_research,
    stream_research,
)
from src.agent.tools.web_crawler import WebCrawler
from src.schemas.streaming import (
    ResearchCompletedEvent,
    SynthesisProgressEvent,
)
from src.services.llm.client import LLMClient
from src.services.search.brave import BraveSearchClient

# ---------------------------------------------------------------------------
# Credential Checks
# ---------------------------------------------------------------------------


def _has_databricks_creds() -> bool:
    """Check if Databricks credentials are available."""
    return bool(os.getenv("DATABRICKS_TOKEN") or os.getenv("DATABRICKS_CONFIG_PROFILE"))


def _has_brave_key() -> bool:
    """Check if Brave API key is available."""
    return bool(os.getenv("BRAVE_API_KEY"))


# Skip markers for tests that require real credentials
requires_databricks = pytest.mark.skipif(
    not _has_databricks_creds(),
    reason="Databricks credentials not configured (check .env for DATABRICKS_TOKEN or DATABRICKS_CONFIG_PROFILE)",
)
requires_brave = pytest.mark.skipif(
    not _has_brave_key(),
    reason="Brave API key not configured (check .env for BRAVE_API_KEY)",
)

# Combined marker for tests that need both
requires_all_credentials = pytest.mark.skipif(
    not (_has_databricks_creds() and _has_brave_key()),
    reason="Both Databricks and Brave credentials required (check .env)",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def llm_client() -> LLMClient:
    """Create a real LLMClient with Databricks endpoints."""
    client = LLMClient()
    yield client
    await client.close()


@pytest.fixture
async def brave_client() -> BraveSearchClient:
    """Create a real BraveSearchClient."""
    client = BraveSearchClient()
    yield client
    await client.close()


@pytest.fixture
async def web_crawler() -> WebCrawler:
    """Create a WebCrawler for fetching pages."""
    crawler = WebCrawler(max_concurrent=3)
    yield crawler
    await crawler.close()


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------


@requires_all_credentials
@pytest.mark.asyncio
async def test_deep_research_initial_query(
    llm_client: LLMClient,
    brave_client: BraveSearchClient,
    web_crawler: WebCrawler,
) -> None:
    """Test full research flow with a real query.

    This test performs a complete deep research cycle:
    1. Coordinator classifies the query
    2. Background investigator gathers context
    3. Planner creates a research plan
    4. Researcher executes steps
    5. Reflector evaluates progress
    6. Synthesizer generates final report
    """
    result = await run_research(
        query="What are the latest breakthroughs in quantum computing in 2024?",
        llm=llm_client,
        brave_client=brave_client,
        crawler=web_crawler,
        session_id=uuid4(),
        config=OrchestrationConfig(
            max_plan_iterations=2,
            max_steps_per_plan=5,
            timeout_seconds=300,
        ),
    )

    print("\nFinal report:")
    print(result.state.final_report)

    # Verify research completed successfully
    assert result.state.final_report, "Should have generated a final report"
    assert len(result.state.final_report) > 100, "Report should have meaningful content"

    # Verify sources were found
    assert len(result.state.sources) > 0, "Should have found at least one source"

    # Verify research steps executed
    assert result.steps_executed > 0, "Should have executed at least one research step"

    # Verify timing
    assert result.total_duration_ms > 0, "Duration should be positive"

    # Verify no errors
    assert not result.state.is_cancelled, "Research should not be cancelled"

    # Verify query was classified
    assert result.state.query_classification is not None, "Query should be classified"

    print("\n✅ Research completed successfully!")
    print(f"   - Duration: {result.total_duration_ms / 1000:.1f}s")
    print(f"   - Steps executed: {result.steps_executed}")
    print(f"   - Sources found: {len(result.state.sources)}")
    print(f"   - Report length: {len(result.state.final_report)} chars")


@requires_all_credentials
@pytest.mark.asyncio
async def test_followup_with_conversation_history(
    llm_client: LLMClient,
    brave_client: BraveSearchClient,
    web_crawler: WebCrawler,
) -> None:
    """Test follow-up question uses conversation context.

    This test:
    1. Runs an initial research query about a topic
    2. Builds conversation history from the result
    3. Sends a follow-up question that depends on the first
    4. Verifies the follow-up leverages the context
    """
    session_id = uuid4()

    # Step 1: Run initial research
    print("\n📚 Step 1: Running initial research query...")
    initial_result = await run_research(
        query="What is quantum entanglement?",
        llm=llm_client,
        brave_client=brave_client,
        crawler=web_crawler,
        session_id=session_id,
        config=OrchestrationConfig(
            max_plan_iterations=1,
            max_steps_per_plan=3,
            timeout_seconds=180,
        ),
    )

    assert initial_result.state.final_report, "Initial research should produce a report"
    print(f"   ✅ Initial research complete ({len(initial_result.state.final_report)} chars)")

    print("\nFinal report:")
    print(initial_result.state.final_report)

    # Step 2: Build conversation history
    conversation_history = [
        {"role": "user", "content": "What is quantum entanglement?"},
        {"role": "assistant", "content": initial_result.state.final_report},
    ]

    # Step 3: Send follow-up question
    print("\n📚 Step 2: Sending follow-up question with conversation context...")
    followup_result = await run_research(
        query="How is this used in quantum computers?",
        llm=llm_client,
        brave_client=brave_client,
        crawler=web_crawler,
        conversation_history=conversation_history,
        session_id=session_id,
        config=OrchestrationConfig(
            max_plan_iterations=1,
            max_steps_per_plan=3,
            timeout_seconds=180,
        ),
    )

    print("\nFollow-up report:")
    print(followup_result.state.final_report)

    # Verify follow-up completed
    assert followup_result.state.final_report, "Follow-up should produce a report"
    assert len(followup_result.state.final_report) > 100, "Follow-up report should have content"

    # Verify conversation history was passed
    assert (
        followup_result.state.conversation_history == conversation_history
    ), "Conversation history should be preserved"

    # The follow-up should reference quantum concepts (shows it understood context)
    report_lower = followup_result.state.final_report.lower()
    assert any(
        term in report_lower for term in ["quantum", "qubit", "entangle", "superposition"]
    ), "Follow-up should reference quantum concepts from context"

    print(f"   ✅ Follow-up complete ({len(followup_result.state.final_report)} chars)")
    print("\n✅ Conversation flow test passed!")
    print(f"   - Initial query sources: {len(initial_result.state.sources)}")
    print(f"   - Follow-up sources: {len(followup_result.state.sources)}")


@requires_all_credentials
@pytest.mark.asyncio
async def test_simple_query_detection(
    llm_client: LLMClient,
    brave_client: BraveSearchClient,
    web_crawler: WebCrawler,
) -> None:
    """Test that simple queries are handled directly without full research.

    Simple queries like "What is 2+2?" should be classified as simple
    and answered directly without web searches.
    """
    result = await run_research(
        query="What is 2+2?",
        llm=llm_client,
        brave_client=brave_client,
        crawler=web_crawler,
        session_id=uuid4(),
    )

    # Verify it was classified as simple
    assert result.state.is_simple_query, "Simple math query should be classified as simple"

    # Verify response exists - simple queries use direct_response or final_report
    response = result.state.direct_response or result.state.final_report
    assert response, "Should have a response (direct_response or final_report)"

    # Simple queries typically don't need full research steps
    assert result.steps_executed == 0, "Simple query should not need research steps"

    print("\n✅ Simple query test passed!")
    print(f"   - Is simple: {result.state.is_simple_query}")
    print(f"   - Response: {response}")
    print(f"   - Steps executed: {result.steps_executed}")
    print(f"   - Duration: {result.total_duration_ms / 1000:.1f}s")


@requires_all_credentials
@pytest.mark.asyncio
async def test_streaming_research_events(
    llm_client: LLMClient,
    brave_client: BraveSearchClient,
    web_crawler: WebCrawler,
) -> None:
    """Test streaming produces expected event sequence.

    Verifies that the streaming API emits:
    - AgentStartedEvent for each agent
    - AgentCompletedEvent for each agent
    - SynthesisProgressEvent for content chunks
    - ResearchCompletedEvent at the end
    """
    events: list = []
    synthesis_chunks: list[str] = []

    async for event in stream_research(
        query="Explain how CRISPR gene editing works briefly",
        llm=llm_client,
        brave_client=brave_client,
        crawler=web_crawler,
        session_id=uuid4(),
        config=OrchestrationConfig(
            max_plan_iterations=1,
            max_steps_per_plan=2,
            timeout_seconds=180,
        ),
    ):
        events.append(event)

        # Collect synthesis chunks
        if isinstance(event, SynthesisProgressEvent):
            synthesis_chunks.append(event.content_chunk)

    # Verify we got events
    assert len(events) > 0, "Should emit events"

    # Check for key event types
    event_types = [type(e).__name__ for e in events]

    # Should have agent start/complete events
    assert "AgentStartedEvent" in event_types, "Should have AgentStartedEvent"
    assert "AgentCompletedEvent" in event_types, "Should have AgentCompletedEvent"

    # Should have research completed event
    assert "ResearchCompletedEvent" in event_types, "Should have ResearchCompletedEvent"

    # Verify synthesis content was streamed
    full_synthesis = "".join(synthesis_chunks)
    assert len(full_synthesis) > 0, "Should have streamed synthesis content"

    # Find the completion event and check stats
    completion_events = [e for e in events if isinstance(e, ResearchCompletedEvent)]
    assert len(completion_events) == 1, "Should have exactly one completion event"

    completion = completion_events[0]
    assert completion.total_duration_ms > 0, "Should track duration"

    print("\n✅ Streaming test passed!")
    print(f"   - Total events: {len(events)}")
    print(f"   - Synthesis chunks: {len(synthesis_chunks)}")
    print(f"   - Synthesis length: {len(full_synthesis)} chars")
    print(f"   - Duration: {completion.total_duration_ms / 1000:.1f}s")


@requires_all_credentials
@pytest.mark.asyncio
async def test_research_with_complex_query(
    llm_client: LLMClient,
    brave_client: BraveSearchClient,
    web_crawler: WebCrawler,
) -> None:
    """Test research with a complex multi-part query.

    This tests the system's ability to handle queries that
    require multiple research angles.
    """
    result = await run_research(
        query="Compare the environmental impact and efficiency of solar vs wind energy for residential use",
        llm=llm_client,
        brave_client=brave_client,
        crawler=web_crawler,
        session_id=uuid4(),
        config=OrchestrationConfig(
            max_plan_iterations=2,
            max_steps_per_plan=4,
            timeout_seconds=300,
        ),
    )

    # Verify comprehensive research
    assert result.state.final_report, "Should produce a report"
    assert len(result.state.final_report) > 200, "Complex query should produce detailed report"

    # Verify multiple sources for comparison
    assert len(result.state.sources) >= 2, "Comparison should use multiple sources"

    # Report should mention both topics
    report_lower = result.state.final_report.lower()
    assert "solar" in report_lower, "Report should discuss solar energy"
    assert "wind" in report_lower, "Report should discuss wind energy"

    print("\n✅ Complex query test passed!")
    print(f"   - Report length: {len(result.state.final_report)} chars")
    print(f"   - Sources used: {len(result.state.sources)}")
    print(f"   - Steps executed: {result.steps_executed}")
