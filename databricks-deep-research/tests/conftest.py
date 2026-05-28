"""Shared test fixtures for the framework test suite."""
from __future__ import annotations

import asyncio
from collections.abc import Generator
from pathlib import Path
from typing import Any, TypeVar
from unittest.mock import AsyncMock, MagicMock

import pytest
from dotenv import load_dotenv

import databricks_deep_research._fips_compat  # noqa: F401  # FIPS md5 patch
from databricks_deep_research.events.types import StreamEvent
from databricks_deep_research.llm.client import FrameworkLLMClient, LLMResponse
from databricks_deep_research.tools.protocol import (
    SourceInfo,
    ToolContext,
    ToolDefinition,
    ToolResult,
)
from databricks_deep_research.tools.registry import ToolRegistry

# Auto-load .env.test from the framework root (won't override existing env vars).
# This runs before integration/complex conftest modules evaluate skip conditions.
_ENV_TEST = Path(__file__).resolve().parent.parent / ".env.test"
if _ENV_TEST.exists():
    load_dotenv(_ENV_TEST, override=False)


@pytest.fixture(autouse=True)
def _legacy_sync_event_loop(request: pytest.FixtureRequest) -> Generator[None, None, None]:
    """Provide a current event loop for older sync tests on Python 3.11+."""
    if request.node.get_closest_marker("asyncio") is not None:
        yield
        return

    created_loop: asyncio.AbstractEventLoop | None = None
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        created_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(created_loop)

    try:
        yield
    finally:
        if created_loop is not None:
            created_loop.close()
            asyncio.set_event_loop(None)


# ---------------------------------------------------------------------------
# Trace collection CLI options
# ---------------------------------------------------------------------------


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--collect-traces",
        action="store_true",
        default=False,
        help="Download MLflow traces after test session",
    )
    parser.addoption(
        "--trace-output",
        default="test-traces",
        help="Output directory for trace reports (default: test-traces)",
    )


# ---------------------------------------------------------------------------
# Session-scoped trace collection
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def _trace_session(request: pytest.FixtureRequest) -> Generator[None, None, None]:
    """Auto-configure MLflow and collect traces if --collect-traces is set."""
    if not request.config.getoption("--collect-traces"):
        yield
        return

    from tests.trace_collector import TraceCollector

    collector = TraceCollector(
        output_dir=request.config.getoption("--trace-output"),
    )

    # Setup MLflow tracking
    if not collector.setup_mlflow():
        print("\nWARNING: MLflow setup failed — trace collection disabled")
        yield
        return

    collector.start()
    yield

    # Collect and report
    report = collector.collect()
    if report:
        collector.print_terminal_summary(report)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_query() -> str:
    return "What are the latest advances in quantum computing?"


# ---------------------------------------------------------------------------
# Mock enterprise tools — realistic canned responses for integration tests
# ---------------------------------------------------------------------------

# -- Genie (SQL / BI) --------------------------------------------------------

_GENIE_RESPONSES: dict[str, str] = {
    "default": (
        "Query: SELECT product_line, fiscal_year, revenue_usd, yoy_growth_pct "
        "FROM finance.quarterly_revenue ORDER BY fiscal_year DESC LIMIT 10\n\n"
        "Results:\n"
        "| product_line | fiscal_year | revenue_usd   | yoy_growth_pct |\n"
        "|--------------|-------------|---------------|----------------|\n"
        "| Cloud        | 2025        | 4,200,000,000 | 23.5           |\n"
        "| Cloud        | 2024        | 3,400,000,000 | 31.2           |\n"
        "| Platform     | 2025        | 1,800,000,000 | 12.8           |\n"
        "| Platform     | 2024        | 1,595,000,000 | 18.1           |\n"
        "| Services     | 2025        | 950,000,000   | 8.3            |\n"
        "| Services     | 2024        | 877,000,000   | 5.7            |\n\n"
        "Summary: Total revenue across all product lines reached $6.95B in FY2025, "
        "representing a combined 18.4% year-over-year growth. Cloud segment led "
        "growth at 23.5% YoY, driven by enterprise AI workload adoption."
    ),
}


class MockGenieTool:
    """Mock Genie tool returning realistic SQL query results."""

    def __init__(self, responses: dict[str, str] | None = None) -> None:
        self._responses = responses or _GENIE_RESPONSES

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="genie",
            description=(
                "Query enterprise data warehouse using natural language. "
                "Returns SQL results from internal financial and operational databases."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Natural-language question about enterprise data.",
                    },
                },
                "required": ["question"],
            },
            source_type="enterprise",
        )

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if "question" not in arguments:
            raise ValueError("Missing required argument: question")
        return arguments

    async def execute(
        self, arguments: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        question = arguments["question"].lower()
        # Pick the best matching canned response
        response = self._responses.get("default", "No data available.")
        for key, value in self._responses.items():
            if key != "default" and key in question:
                response = value
                break

        return ToolResult(
            content=response,
            sources=[
                SourceInfo(
                    url="enterprise://genie/finance_warehouse",
                    title="Enterprise Data Warehouse — Finance",
                    snippet="SQL query results from Genie BI",
                    source_type="genie",
                ),
            ],
        )


# -- Vector Search -----------------------------------------------------------

_VECTOR_SEARCH_RESPONSES: dict[str, str] = {
    "default": (
        "Document 1 (score: 0.94): 'Technical Architecture Review — Microservices '\n"
        "The platform uses an event-driven microservices architecture with Apache Kafka "
        "for inter-service communication. Each service maintains its own database "
        "(database-per-service pattern) and exposes gRPC APIs. The API gateway handles "
        "authentication, rate limiting, and request routing. Key services include: "
        "UserService, OrderService, InventoryService, and PaymentService.\n\n"
        "Document 2 (score: 0.89): 'Scaling Strategy Q4 2024'\n"
        "Auto-scaling is configured using custom HPA metrics based on request latency "
        "p99 and queue depth. The system handles 50,000 RPS at peak with sub-100ms "
        "p95 latency. Kubernetes cluster spans 3 availability zones with 120 nodes.\n\n"
        "Document 3 (score: 0.82): 'Security Compliance Report'\n"
        "All services implement mTLS for service-to-service communication. Data at rest "
        "is encrypted using AES-256. Access control follows RBAC with OIDC integration. "
        "SOC 2 Type II and ISO 27001 certifications were renewed in November 2024."
    ),
}


class MockVectorSearchTool:
    """Mock Vector Search returning realistic document chunks with metadata."""

    def __init__(self, responses: dict[str, str] | None = None) -> None:
        self._responses = responses or _VECTOR_SEARCH_RESPONSES

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="vector_search",
            description=(
                "Search internal document index using semantic similarity. "
                "Returns relevant document chunks from engineering docs, "
                "architecture reviews, and technical specifications."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Semantic search query.",
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return.",
                        "default": 3,
                    },
                },
                "required": ["query"],
            },
            source_type="enterprise",
        )

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if "query" not in arguments:
            raise ValueError("Missing required argument: query")
        return arguments

    async def execute(
        self, arguments: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        query = arguments["query"].lower()
        response = self._responses.get("default", "No documents found.")
        for key, value in self._responses.items():
            if key != "default" and key in query:
                response = value
                break

        return ToolResult(
            content=response,
            sources=[
                SourceInfo(
                    url="enterprise://vector_search/tech_docs",
                    title="Internal Technical Documentation",
                    snippet="Semantic search results from document index",
                    source_type="vector_search",
                ),
                SourceInfo(
                    url="enterprise://vector_search/arch_reviews",
                    title="Architecture Review Documents",
                    snippet="Architecture and design documents",
                    source_type="vector_search",
                ),
            ],
        )


# -- Knowledge Assistant -----------------------------------------------------

_KNOWLEDGE_ASSISTANT_RESPONSES: dict[str, str] = {
    "default": (
        "Based on the internal knowledge base:\n\n"
        "The deployment pipeline follows a blue-green deployment strategy with "
        "automated canary analysis. Each release goes through the following stages:\n"
        "1. Build & unit tests (CI, ~5 min)\n"
        "2. Integration tests against staging (~15 min)\n"
        "3. Canary deployment to 5% of production traffic (~30 min observation)\n"
        "4. Full rollout with automated rollback triggers\n\n"
        "Key metrics monitored during canary: error rate (threshold: <0.1%), "
        "p99 latency (threshold: <200ms), CPU utilization (threshold: <70%).\n\n"
        "Confidence: HIGH. This information is sourced from the Platform Engineering "
        "runbook (last updated: January 2025) and the SRE deployment guidelines."
    ),
}


class MockKnowledgeAssistantTool:
    """Mock Knowledge Assistant returning QA responses with confidence."""

    def __init__(self, responses: dict[str, str] | None = None) -> None:
        self._responses = responses or _KNOWLEDGE_ASSISTANT_RESPONSES

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="knowledge_assistant",
            description=(
                "Ask questions to an AI-powered knowledge assistant trained on "
                "internal documentation, runbooks, and engineering guidelines. "
                "Returns answers with confidence levels and source references."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Question to ask the knowledge assistant.",
                    },
                },
                "required": ["question"],
            },
            source_type="enterprise",
        )

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if "question" not in arguments:
            raise ValueError("Missing required argument: question")
        return arguments

    async def execute(
        self, arguments: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        question = arguments["question"].lower()
        response = self._responses.get("default", "No answer available.")
        for key, value in self._responses.items():
            if key != "default" and key in question:
                response = value
                break

        return ToolResult(
            content=response,
            sources=[
                SourceInfo(
                    url="enterprise://knowledge_assistant/platform_runbook",
                    title="Platform Engineering Runbook",
                    snippet="AI-generated answer from knowledge base",
                    source_type="knowledge_assistant",
                ),
            ],
        )


# ---------------------------------------------------------------------------
# Enterprise tool fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def genie_tool() -> MockGenieTool:
    """MockGenieTool with financial data responses."""
    return MockGenieTool()


@pytest.fixture
def vector_search_tool_mock() -> MockVectorSearchTool:
    """MockVectorSearchTool with tech document chunks."""
    return MockVectorSearchTool()


@pytest.fixture
def knowledge_assistant_tool() -> MockKnowledgeAssistantTool:
    """MockKnowledgeAssistantTool with QA responses."""
    return MockKnowledgeAssistantTool()


@pytest.fixture
def enterprise_tools(
    genie_tool: MockGenieTool,
    vector_search_tool_mock: MockVectorSearchTool,
    knowledge_assistant_tool: MockKnowledgeAssistantTool,
) -> list[Any]:
    """All three mock enterprise tools."""
    return [genie_tool, vector_search_tool_mock, knowledge_assistant_tool]


@pytest.fixture
def enterprise_tool_registry(
    genie_tool: MockGenieTool,
    vector_search_tool_mock: MockVectorSearchTool,
    knowledge_assistant_tool: MockKnowledgeAssistantTool,
) -> ToolRegistry:
    """ToolRegistry with all enterprise tools registered."""
    registry = ToolRegistry()
    registry.register_external("genie", genie_tool)
    registry.register_external("vector_search", vector_search_tool_mock)
    registry.register_external("knowledge_assistant", knowledge_assistant_tool)
    return registry


def build_mock_llm_client() -> MagicMock:
    """Build a mock FrameworkLLMClient with deterministic defaults."""
    client = MagicMock(spec=FrameworkLLMClient)
    client.complete = AsyncMock(
        return_value=LLMResponse(content="mock output", usage={})
    )
    client.resolve_model = MagicMock(return_value="test-model")
    return client


async def collect_events(
    executor: Any,
    state: Any,
) -> list[StreamEvent]:
    """Drain the executor async iterator into a concrete event list."""
    events: list[StreamEvent] = []
    async for event in executor.execute(state):
        events.append(event)
    return events


T = TypeVar("T", bound=StreamEvent)


def events_of_type(events: list[StreamEvent], cls: type[T]) -> list[T]:
    """Return only events matching the requested StreamEvent subtype."""
    return [event for event in events if isinstance(event, cls)]
