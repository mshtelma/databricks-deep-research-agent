"""Tool protocol — defines the contract all tools (builtin, UC, enterprise) implement.

All tools expose a ``ResearchTool`` protocol with three members:

* ``definition`` — identity + JSON Schema for LLM function-calling.
* ``validate_arguments`` — clean / transform raw LLM args before execution.
* ``execute`` — async execution returning a ``ToolResult``.

Tool *dependencies* (search clients, tokens, domain filters) are injected via
the tool constructor — **not** via ``ToolContext``.  ``ToolContext`` carries only
per-call values that change between invocations (the current query, the shared
URL registry).

``UrlRegistry`` is a lightweight index-to-URL map shared across all tool calls
within a single workflow run.  The LLM sees integer indices only — never raw
URLs — which prevents hallucinated URL injection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Source kind enum
# ---------------------------------------------------------------------------


class SourceKind(StrEnum):
    """How a tool should be queried and how its results should be interpreted.

    Drives: query generation strategy, admission policy, result formatting.
    - web: keyword/BM25 search (Brave, Google) — synthetic relevance scores
    - vector_index: semantic embedding queries — trust upstream relevance_score
    - sql_analytics: NL→SQL (Genie) — structured tabular results
    - qa_assistant: NL question→NL answer (KA, endpoints) — prose answers
    - file: keyword search over uploaded files
    - builtin: framework internals (pool tools, crawl) — not a data source
    """

    web = "web"
    vector_index = "vector_index"
    sql_analytics = "sql_analytics"
    qa_assistant = "qa_assistant"
    file = "file"
    builtin = "builtin"
    delta_table = "delta_table"


class ToolKind(StrEnum):
    """Well-known tool kinds for YAML ``tools:`` declarations.

    Maps 1:1 with concrete tool implementations.  Custom kinds (not in this
    enum) are supported — the ``kind`` field on ``ToolDeclaration`` is typed
    as ``str``, not constrained to this enum.
    """

    web_search = "web_search"
    web_crawl = "web_crawl"
    file_search = "file_search"
    vector_search = "vector_search"
    genie = "genie"
    knowledge_assistant = "knowledge_assistant"
    compute = "compute"
    compute_namespace = "compute_namespace"
    delta_read = "delta_read"
    delta_grep = "delta_grep"
    delta_table_read = "delta_table_read"
    custom = "custom"


_TOOL_KIND_TO_SOURCE_KIND: dict[str, str] = {
    ToolKind.web_search: SourceKind.web,
    ToolKind.web_crawl: SourceKind.builtin,
    ToolKind.file_search: SourceKind.file,
    ToolKind.vector_search: SourceKind.vector_index,
    ToolKind.genie: SourceKind.sql_analytics,
    ToolKind.knowledge_assistant: SourceKind.qa_assistant,
    ToolKind.compute: SourceKind.builtin,
    ToolKind.compute_namespace: SourceKind.builtin,
    ToolKind.delta_read: SourceKind.delta_table,
    ToolKind.delta_grep: SourceKind.delta_table,
    ToolKind.delta_table_read: SourceKind.delta_table,
}


def tool_kind_to_source_kind(kind: str) -> str:
    """Map a ToolKind to a SourceKind.  Returns ``'builtin'`` for unknown kinds."""
    return _TOOL_KIND_TO_SOURCE_KIND.get(kind, SourceKind.builtin)


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolDefinition:
    """Tool definition — combines identity + schema for LLM function calling."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    source_type: str = "builtin"  # builtin, uc_function, uc_tool, enterprise
    source_kind: str = "builtin"  # query modality (SourceKind value)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SourceInfo:
    """A source reference from a tool result."""

    url: str
    canonical_url: str | None = None
    title: str = ""
    snippet: str = ""
    content: str | None = None
    source_type: str = "web"  # web, enterprise, file, etc.
    source_kind: str | None = None  # SourceKind value; preferred over source_type for routing/admission
    relevance_score: float | None = None


@dataclass(frozen=True)
class ToolResult:
    """Result returned by a tool execution."""

    content: str
    success: bool = True
    sources: list[SourceInfo] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True)
class ToolRef:
    """Reference to a tool by type and name.  Used in YAML workflow configs.

    Examples::

        ToolRef(type="builtin", name="web_search")
        ToolRef(type="uc_function", name="catalog.schema.my_function")
    """

    type: str  # "builtin", "uc_function", "uc_tool", "enterprise"
    name: str  # Tool identifier


# ---------------------------------------------------------------------------
# URL registry
# ---------------------------------------------------------------------------


class UrlRegistry:
    """Maps integer indices to URLs.  LLM sees indices only (security).

    Created per workflow execution, shared across all tool calls within a
    single workflow run.  ``web_search`` registers discovered URLs and returns
    indices; ``web_crawl`` resolves indices back to URLs for fetching.

    Internally backed by a *list* for O(1) index lookup and a reverse *dict*
    for deduplication.

    Ported / simplified from the existing app's ``tools/url_registry.py``.
    """

    __slots__ = ("_urls", "_url_to_index", "_url_failures", "_domain_failure_counts", "_domain_failure_classes")

    _DOMAIN_SUPPRESSION_THRESHOLD = 2

    def __init__(self) -> None:
        self._urls: list[str] = []
        self._url_to_index: dict[str, int] = {}
        self._url_failures: dict[str, str] = {}
        self._domain_failure_counts: dict[str, int] = {}
        self._domain_failure_classes: dict[str, str] = {}

    # -- mutators ------------------------------------------------------------

    def register(self, url: str) -> int:
        """Register a URL and return its integer index.

        If *url* was already registered, the existing index is returned
        (deduplication).
        """
        existing = self._url_to_index.get(url)
        if existing is not None:
            return existing

        index = len(self._urls)
        self._urls.append(url)
        self._url_to_index[url] = index
        return index

    # -- queries -------------------------------------------------------------

    def resolve(self, index: int) -> str | None:
        """Resolve an index back to its URL.  Returns ``None`` if not found."""
        if 0 <= index < len(self._urls):
            return self._urls[index]
        return None

    def get_all(self) -> list[tuple[int, str]]:
        """Return all ``(index, url)`` pairs in registration order."""
        return list(enumerate(self._urls))

    def get_failure(self, url: str) -> dict[str, str] | None:
        """Return cached non-retryable failure metadata for *url*, if any."""
        if url in self._url_failures:
            return {
                "scope": "url",
                "failure_class": self._url_failures[url],
            }

        domain = self._domain_for_url(url)
        if domain:
            count = self._domain_failure_counts.get(domain, 0)
            if count >= self._DOMAIN_SUPPRESSION_THRESHOLD:
                return {
                    "scope": "domain",
                    "failure_class": self._domain_failure_classes.get(domain, "domain_failure_threshold"),
                }
        return None

    def record_non_retryable_failure(self, url: str, failure_class: str) -> None:
        """Record a non-retryable failure for *url* and its domain."""
        self._url_failures[url] = failure_class

        domain = self._domain_for_url(url)
        if not domain:
            return
        self._domain_failure_counts[domain] = self._domain_failure_counts.get(domain, 0) + 1
        self._domain_failure_classes.setdefault(domain, failure_class)

    def clear_failure(self, url: str) -> None:
        """Clear an exact-URL cached failure after a successful retry."""
        self._url_failures.pop(url, None)

    # -- dunder helpers ------------------------------------------------------

    def __len__(self) -> int:
        return len(self._urls)

    def __contains__(self, url: str) -> bool:
        return url in self._url_to_index

    def __repr__(self) -> str:
        return f"UrlRegistry(count={len(self._urls)})"

    @staticmethod
    def _domain_for_url(url: str) -> str:
        return (urlparse(url).netloc or "").lower()


# ---------------------------------------------------------------------------
# Execution context
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolContext:
    """Per-call context passed to tools at execution time.

    Tool dependencies (search clients, domain filters, user tokens) are
    constructor-injected at tool creation time, **not** passed per-call.
    Only per-call values that change between invocations belong here.
    """

    query: str = ""
    url_registry: UrlRegistry | None = None
    current_step: Any | None = None
    background_summary: str = ""
    recent_observations: list[str] = field(default_factory=list)
    discovered_sources: list[Any] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ResearchTool(Protocol):
    """Protocol that all tools must implement.

    Builtin tools use constructor DI for dependencies::

        class WebSearchTool:
            def __init__(self, search_client: SearchClient) -> None:
                self._client = search_client

    Preferred YAML syntax (top-level ``tools:`` section)::

        tools:
          - name: earnings_index
            kind: vector_search
            config:
              index_name: prod_catalog.finance.earnings_idx
          - name: web_search
            kind: web_search

    Agent nodes reference tools by name::

        config:
          tools: [earnings_index, web_search]

    Legacy YAML syntax (still supported for backward compatibility)::

        tools:
          - type: builtin
            name: web_search
          - type: enterprise
            name: my_vector_search
    """

    @property
    def definition(self) -> ToolDefinition:
        """Tool definition combining name, description, and parameter schema."""
        ...

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Validate and potentially transform arguments before execution.

        The returned dict is the canonical input to ``execute()`` — combining
        validation and transformation prevents bugs where uncleaned args are
        passed to ``execute()``.

        Args:
            arguments: Raw arguments from LLM tool call.

        Returns:
            Validated / transformed arguments dict.

        Raises:
            ValueError: If arguments are invalid.
        """
        ...

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Execute the tool with validated arguments.

        Args:
            arguments: Validated arguments matching ``self.definition.parameters``.
            context: Execution context (query, URL registry).

        Returns:
            ``ToolResult`` with content, success status, optional sources / data.
        """
        ...
