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

import copy
import threading
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
    text_table = "text_table"


class ToolKind(StrEnum):
    """Well-known tool kinds for YAML ``tools:`` declarations.

    Maps 1:1 with concrete tool implementations.  Custom kinds (not in this
    enum) are supported — the ``kind`` field on ``ToolDeclaration`` is typed
    as ``str``, not constrained to this enum.
    """

    web_search = "web_search"
    web_crawl = "web_crawl"
    web_research = "web_research"
    academic_search = "academic_search"
    file_search = "file_search"
    vector_search = "vector_search"
    genie = "genie"
    knowledge_assistant = "knowledge_assistant"
    compute = "compute"
    compute_namespace = "compute_namespace"
    table_discovery = "table_discovery"
    table_search = "table_search"
    table_read = "table_read"
    table_neighbors = "table_neighbors"
    table_load = "table_load"
    table_aggregate = "table_aggregate"
    mcp = "mcp"
    custom = "custom"


# Tool kinds that reach UC-gated Databricks resources and therefore need a
# user identity (OBO) to behave correctly in a deployed app — running them as
# the service principal silently yields permission errors / empty results.
# Hosts gate "fail closed when OBO is missing" on this set
# (see ``workflow_requires_databricks``). Web/file/custom kinds are excluded.
DATABRICKS_BOUND_TOOL_KINDS: frozenset[str] = frozenset(
    {
        ToolKind.vector_search,
        ToolKind.genie,
        ToolKind.knowledge_assistant,
        ToolKind.table_discovery,
        ToolKind.table_search,
        ToolKind.table_read,
        ToolKind.table_neighbors,
        ToolKind.table_load,
        ToolKind.table_aggregate,
        ToolKind.compute,
        ToolKind.compute_namespace,
    }
)


_TOOL_KIND_TO_SOURCE_KIND: dict[str, str] = {
    ToolKind.web_search: SourceKind.web,
    ToolKind.web_crawl: SourceKind.builtin,
    ToolKind.web_research: SourceKind.web,
    # Academic retrievers return scholarly documents over the public web; treat
    # them as web sources so they admit to the pool via the keyword/BM25 path.
    ToolKind.academic_search: SourceKind.web,
    ToolKind.file_search: SourceKind.file,
    ToolKind.vector_search: SourceKind.vector_index,
    ToolKind.genie: SourceKind.sql_analytics,
    ToolKind.knowledge_assistant: SourceKind.qa_assistant,
    ToolKind.compute: SourceKind.builtin,
    ToolKind.compute_namespace: SourceKind.builtin,
    ToolKind.table_discovery: SourceKind.text_table,
    ToolKind.table_search: SourceKind.text_table,
    ToolKind.table_read: SourceKind.text_table,
    ToolKind.table_neighbors: SourceKind.text_table,
    ToolKind.table_load: SourceKind.text_table,
    ToolKind.table_aggregate: SourceKind.text_table,
    # MCP research tools are NL question→answer, and CITEABLE (non-builtin) so
    # their results flow through admission to the pool (spec §4.3 #11).
    ToolKind.mcp: SourceKind.qa_assistant,
}


def tool_kind_to_source_kind(kind: str) -> str:
    """Map a ToolKind to a SourceKind.  Returns ``'builtin'`` for unknown kinds."""
    return _TOOL_KIND_TO_SOURCE_KIND.get(kind, SourceKind.builtin)


# Each entry lists ``ToolFactoryContext`` field names a factory needs to
# successfully construct that tool kind. Used by the app's deploy-time and
# boot-time validators to fail loud when a workflow declares a kind whose
# runtime dependencies are unset (e.g. ``schema_cache`` is None because
# ``STORAGE_WAREHOUSE_ID`` was not propagated to the deployed app env). The
# inline ``if ctx.X is None: raise`` checks inside concrete factories remain
# the runtime backstop — this table is additive metadata.
_TOOL_KIND_REQUIRED_CTX: dict[str, frozenset[str]] = {
    ToolKind.web_search: frozenset({"search_client"}),
    ToolKind.web_research: frozenset({"search_client"}),
    ToolKind.vector_search: frozenset({"workspace_client"}),
    ToolKind.genie: frozenset({"workspace_client"}),
    ToolKind.knowledge_assistant: frozenset({"workspace_client"}),
    ToolKind.table_discovery: frozenset(
        {"table_registry", "table_discovery_provider"}
    ),
    ToolKind.table_search: frozenset(
        {"table_registry", "schema_cache", "sql_executor"}
    ),
    ToolKind.table_read: frozenset(
        {"table_registry", "schema_cache", "sql_executor"}
    ),
    ToolKind.table_neighbors: frozenset(
        {"table_registry", "schema_cache", "sql_executor"}
    ),
    ToolKind.table_load: frozenset(
        {"table_registry", "schema_cache", "sql_executor"}
    ),
    ToolKind.table_aggregate: frozenset(
        {"table_registry", "schema_cache", "sql_executor"}
    ),
    # web_crawl, file_search, compute, compute_namespace, custom: no required ctx
}


def required_ctx_fields_for_kind(kind: str) -> frozenset[str]:
    """Return ``ToolFactoryContext`` field names required to construct *kind*.

    Returns an empty frozenset for kinds with no statically-declared
    requirements (custom tools, or tools that are constructible without
    factory context fields). Callers should treat the empty set as
    "no precheck possible" rather than "no dependencies".
    """
    return _TOOL_KIND_REQUIRED_CTX.get(kind, frozenset())


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

    # Failures per domain before crawls of that domain are suppressed for the run.
    # Raised 2->4 so a couple of transient failures (one 403, one boilerplate page)
    # don't disable a whole domain whose other URLs may extract fine.
    _DOMAIN_SUPPRESSION_THRESHOLD = 4

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
# Table registry — in-memory structured table storage
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegisteredTable:
    """A table registered from any source (web, file, vector search, etc.)."""

    table_json: dict[str, Any]
    source_kind: str  # SourceKind value: "web", "file", "vector_index", etc.
    source_label: str  # URL, filename, index name
    markdown: str = ""  # Pre-rendered markdown for quick display


class TableRegistry:
    """Maps integer indices to structured tables within a workflow run.

    Source-agnostic — any tool can register tables.  Modeled after
    :class:`UrlRegistry`: created once per workflow run, shared across all
    tool calls via :class:`ToolContext`.

    Registration validates that ``table_json`` contains the required
    ``"headers"`` and ``"rows"`` keys.  A capacity limit prevents unbounded
    growth when crawling many pages.
    """

    __slots__ = ("_lock", "_tables", "_max_tables")

    _DEFAULT_MAX_TABLES = 200

    def __init__(self, *, max_tables: int = _DEFAULT_MAX_TABLES) -> None:
        self._lock = threading.Lock()
        self._tables: list[RegisteredTable] = []
        self._max_tables = max_tables

    def register(
        self,
        table_json: dict[str, Any],
        *,
        source_kind: str = "",
        source_label: str = "",
        markdown: str = "",
    ) -> int:
        """Validate *table_json*, store, and return its integer index.

        The dict is **deep-copied** on registration so callers cannot
        mutate the stored data after the fact.

        Raises:
            TypeError: If *table_json* is not a dict.
            ValueError: If *table_json* lacks ``"headers"`` or ``"rows"`` keys,
                or if the registry has reached its capacity limit.
        """
        if not isinstance(table_json, dict):
            raise TypeError(
                f"table_json must be a dict, got {type(table_json).__name__}"
            )
        if "headers" not in table_json or "rows" not in table_json:
            raise ValueError(
                "table_json must contain 'headers' and 'rows' keys; "
                f"got keys: {sorted(table_json.keys())}"
            )

        with self._lock:
            if len(self._tables) >= self._max_tables:
                raise ValueError(
                    f"TableRegistry capacity limit reached ({self._max_tables}). "
                    "Oldest tables are not evicted — raise the limit or reduce "
                    "the number of tables extracted per workflow."
                )

            index = len(self._tables)
            self._tables.append(
                RegisteredTable(
                    table_json=copy.deepcopy(table_json),
                    source_kind=source_kind,
                    source_label=source_label,
                    markdown=markdown,
                )
            )
            return index

    def resolve(self, index: int) -> RegisteredTable | None:
        """Resolve an index back to its :class:`RegisteredTable`.

        Returns ``None`` if *index* is out of range.
        """
        with self._lock:
            if 0 <= index < len(self._tables):
                return self._tables[index]
            return None

    def __len__(self) -> int:
        with self._lock:
            return len(self._tables)

    def __repr__(self) -> str:
        return f"TableRegistry(count={len(self._tables)})"


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
    table_registry: TableRegistry | None = None
    current_step: Any | None = None
    background_summary: str = ""
    recent_observations: list[str] = field(default_factory=list)
    discovered_sources: list[Any] = field(default_factory=list)
    read_only: bool = False
    # Runtime-capability attachment point. ``frozen=True`` prevents rebinding the
    # ``extras`` reference, but the contained dict is mutable — standard Python
    # idiom shared with :class:`ToolFactoryContext.extras`. Keys prefixed with
    # ``_framework_`` are reserved for framework use (approval broker, VFS,
    # todos store, etc.); user-chosen keys MUST NOT use this prefix.
    extras: dict[str, Any] = field(default_factory=dict)


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
