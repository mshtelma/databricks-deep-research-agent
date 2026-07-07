"""Builtin tool factory — creates web_search, web_crawl, file_search tools."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, cast

from databricks_deep_research.tools.catalog_types import CatalogCard, SafeProbe
from databricks_deep_research.tools.factory import ToolFactoryContext
from databricks_deep_research.tools.protocol import ResearchTool
from databricks_deep_research.workflow.definition import ToolDeclaration

logger = logging.getLogger(__name__)

_SUPPORTED_KINDS = frozenset({
    "web_search", "web_crawl", "web_research",
    "academic_search",
    "file_search", "compute", "compute_namespace",
    "table_discovery", "table_search", "table_read",
    "table_neighbors", "table_load", "table_aggregate",
    "read_skill", "python_function", "uc_function",
})
_FUNCTIONAL_TABLE_KINDS = frozenset({
    "table_search",
    "table_read",
    "table_neighbors",
    "table_load",
    "table_aggregate",
})

_SEARCH_PROVIDERS = frozenset({"brave", "jina", "databricks"})
_CRAWL_PROVIDERS = frozenset({"jina"})

# python_function construction-time code validation, cached by (code, modules)
# hash: validate_all() constructs every declared tool per request, so repeat
# validations of unchanged code must be free.
_PYFN_VALIDATION_CACHE: dict[tuple[str, frozenset[str]], str | None] = {}


def _validate_python_function_code(
    tool_name: str, code: str, extra_modules: frozenset[str]
) -> None:
    """Fail tool construction (pre-LLM-spend) on policy-violating code."""
    import hashlib

    from databricks_deep_research.tools.builtins._skill_script_runner import (
        ALLOWED_MODULES,
        SkillScriptPolicyError,
        validate_script_source,
    )

    key = (hashlib.sha256(code.encode("utf-8")).hexdigest(), extra_modules)
    if key not in _PYFN_VALIDATION_CACHE:
        try:
            validate_script_source(
                code, allowed_modules=ALLOWED_MODULES | extra_modules
            )
            _PYFN_VALIDATION_CACHE[key] = None
        except SkillScriptPolicyError as exc:
            _PYFN_VALIDATION_CACHE[key] = str(exc)
    error = _PYFN_VALIDATION_CACHE[key]
    if error is not None:
        raise ValueError(f"python_function '{tool_name}': {error}")


def _inprocess_python_function_allowed(ctx: ToolFactoryContext) -> bool:
    """Operator trust switch for the non-boundary in-process paths."""
    if ctx.extras.get("_allow_inprocess_python_function") is True:
        return True
    return os.environ.get(
        "DDR_ALLOW_INPROCESS_PYTHON_FUNCTION", ""
    ).strip().lower() in ("1", "true", "yes")


def _clean_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _first_config_str(config: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = _clean_str(config.get(key))
        if value:
            return value
    return None


def _roles_from_table_config(config: Mapping[str, Any]) -> dict[str, Any] | None:
    """Extract a RoleMap-compatible dict from YAML/app declaration config."""
    raw_roles = config.get("roles")
    if isinstance(raw_roles, Mapping):
        return dict(raw_roles)

    field_roles = config.get("field_roles")
    role_source = field_roles if isinstance(field_roles, Mapping) else config
    columns = _string_list(config.get("columns"))

    roles: dict[str, Any] = {}
    id_column = _first_config_str(
        role_source,
        "id",
        "id_column",
        "primary_key",
        "pk_column",
        "primary_key_column",
    )
    if id_column is None:
        id_column = "chunk_id" if "chunk_id" in columns else None
    content_column = _first_config_str(
        role_source,
        "content",
        "content_column",
        "text",
        "body",
        "structured_json",
    )
    order_column = _first_config_str(
        role_source,
        "order",
        "order_column",
        "order_by",
        "position_column",
        "sequence_column",
    )
    if order_column is None and "chunk_index" in columns:
        order_column = "chunk_index"
    partition_column = _first_config_str(
        role_source,
        "partition",
        "partition_column",
        "document_column",
        "source_column",
        "file_column",
    )
    if partition_column is None and "file_name" in columns:
        partition_column = "file_name"
    label_column = _first_config_str(role_source, "label", "label_column")
    type_column = _first_config_str(role_source, "type", "type_column")
    if type_column is None and "chunk_type" in columns:
        type_column = "chunk_type"
    date_column = _first_config_str(role_source, "date", "date_column")
    if date_column is None and "bulletin_date" in columns:
        date_column = "bulletin_date"

    for key, value in (
        ("id", id_column),
        ("content", content_column),
        ("order", order_column),
        ("partition", partition_column),
        ("label", label_column),
        ("type", type_column),
        ("date", date_column),
    ):
        if value:
            roles[key] = value

    if "id" not in roles or "content" not in roles:
        return None
    return roles


def _structured_passages_from_config(config: Mapping[str, Any]) -> dict[str, str]:
    raw = config.get("structured_passages")
    if isinstance(raw, Mapping):
        return {
            str(key): str(value)
            for key, value in raw.items()
            if str(key).strip() and str(value).strip()
        }
    parser = _clean_str(config.get("parser"))
    type_value = _clean_str(config.get("structured_type")) or _clean_str(
        config.get("type_value")
    )
    if parser and type_value:
        return {type_value: parser}
    return {}


def _numeric_columns_from_config(config: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(_string_list(config.get("numeric_columns")))


def _projection_columns_from_config(config: Mapping[str, Any]) -> list[str] | None:
    columns = _string_list(config.get("columns"))
    if not columns or columns == ["*"]:
        return None
    return columns


def _order_by_from_config(config: Mapping[str, Any]) -> list[str] | None:
    raw = config.get("order_by")
    if isinstance(raw, str) and raw.strip():
        return [raw.strip()]
    columns = _string_list(raw)
    return columns or None


def _table_fqn_from_decl(decl: ToolDeclaration) -> str | None:
    return _first_config_str(decl.config, "fqn", "table_name", "full_name")


def _table_binding_name_from_decl(decl: ToolDeclaration) -> str:
    return (
        _first_config_str(decl.config, "binding", "as_var", "binding_name")
        or decl.name
    )


def _ensure_declared_table_binding(
    decl: ToolDeclaration,
    ctx: ToolFactoryContext,
) -> str | None:
    """Register a declaration-local table binding when config names a table.

    Generated app and benchmark workflows historically declare table tools as
    concrete named tools with ``config.table_name``. The canonical table_*
    implementation addresses tables through registry bindings, so factory
    construction bridges that YAML shape into a BOUND binding once per
    resolver.
    """
    fqn = _table_fqn_from_decl(decl)
    if fqn is None:
        return None
    if ctx.table_registry is None:
        return None

    from databricks_deep_research.tools.builtins.text_table.binding import (
        BindingInfo,
        BindingSource,
    )
    from databricks_deep_research.tools.builtins.text_table.tools._common import (
        parse_roles,
    )

    binding_name = _table_binding_name_from_decl(decl)
    raw_roles = _roles_from_table_config(decl.config)
    roles = parse_roles(raw_roles) if raw_roles is not None else None

    source = BindingSource.BOUND if roles is not None else BindingSource.DISCOVERED
    existing = None
    if binding_name in ctx.table_registry:
        existing = ctx.table_registry.get(binding_name)
    if existing is not None:
        if existing.fqn == fqn:
            if existing.roles is None and roles is not None:
                ctx.table_registry.update_roles(binding_name, roles)
            return binding_name
        raise ValueError(
            f"table binding {binding_name!r} already registered for "
            f"{existing.fqn!r}; cannot also bind {fqn!r}"
        )

    info = BindingInfo(
        name=binding_name,
        fqn=fqn,
        source=source,
        description=decl.description or None,
        roles=roles,
        numeric_columns=_numeric_columns_from_config(decl.config),
        structured_passages=_structured_passages_from_config(decl.config),
    )
    if source is BindingSource.BOUND:
        ctx.table_registry.register_bound(info)
        return binding_name
    canonical_name, _warning = ctx.table_registry.register_discovered(info)
    return cast(str, canonical_name)


def _resolve_search_provider(
    provider: str,
    ctx: ToolFactoryContext,
    config: Mapping[str, Any] | None = None,
) -> Any:
    """Create a SearchClient for the named provider.

    ``config`` is the tool declaration's ``config`` block — used by providers that
    need per-tool settings (e.g. ``databricks`` reads ``model``/``model_family``).
    """
    config = config or {}
    if provider == "brave":
        api_key = ctx.api_keys.get("brave") or os.environ.get("BRAVE_API_KEY")
        if not api_key:
            raise ValueError(
                "Brave search requires BRAVE_API_KEY env var or "
                "api_keys['brave'] in ToolFactoryContext"
            )
        from databricks_deep_research.tools.builtins.brave_search import (
            BraveSearchAdapter,
        )

        return BraveSearchAdapter(api_key=api_key)

    if provider == "jina":
        api_key = ctx.api_keys.get("jina") or os.environ.get("JINA_API_KEY")
        from databricks_deep_research.tools.builtins.jina_search import (
            JinaSearchAdapter,
        )

        return JinaSearchAdapter(api_key=api_key)

    if provider == "databricks":
        return _build_databricks_search_provider(ctx, config)

    raise ValueError(
        f"Unknown search provider: {provider!r}. "
        f"Supported: {sorted(_SEARCH_PROVIDERS)}"
    )


def _build_databricks_search_provider(
    ctx: ToolFactoryContext, config: Mapping[str, Any]
) -> Any:
    """Build a :class:`DatabricksWebSearchAdapter` from factory context + config.

    Built-in web search is a model-serving call over the PUBLIC web, so it runs
    as the app / service-principal identity (the same one the LLM calls use) —
    NOT the OBO user (``ctx.user_token``), which is reserved for user-scoped data
    tools and need not carry the ``model-serving`` scope the foundation-model
    passthrough requires. Prefer ``ctx.serving_client_provider`` (the app's SP
    serving client); fall back to a ``ctx.workspace_client``-derived client for
    framework-only callers that don't set one. Endpoint comes from
    ``config['model']`` or the ``DATABRICKS_WEB_SEARCH_ENDPOINT`` env var.
    """
    model = config.get("model") or os.environ.get("DATABRICKS_WEB_SEARCH_ENDPOINT")
    if not model:
        raise ValueError(
            "Databricks built-in web search requires a serving endpoint: set "
            "config['model'] on the web_search tool or DATABRICKS_WEB_SEARCH_ENDPOINT"
        )

    serving_provider = ctx.serving_client_provider
    if serving_provider is not None:
        # App-supplied SP serving client (model serving runs as the app, not OBO).
        client_provider = serving_provider
    else:
        ws = ctx.workspace_client
        if ws is None:
            raise ValueError(
                "Databricks built-in web search requires a serving client: set "
                "ToolFactoryContext.serving_client_provider or workspace_client"
            )
        user_token = ctx.user_token
        cached: dict[str, Any] = {}

        def _client_provider() -> Any:
            # Reuse one AsyncOpenAI for the request lifetime (token is fixed per
            # request — OBO token, or an SDK-minted SP token valid well past a run).
            if "client" not in cached:
                from openai import AsyncOpenAI

                host = (getattr(ws.config, "host", "") or "").rstrip("/")
                if user_token:
                    token = user_token
                else:
                    headers = ws.config.authenticate() or {}
                    token = (headers.get("Authorization", "") or "").removeprefix("Bearer ").strip()
                cached["client"] = AsyncOpenAI(
                    api_key=token, base_url=f"{host}/serving-endpoints"
                )
            return cached["client"]

        client_provider = _client_provider

    from databricks_deep_research.tools.builtins.databricks_web_search import (
        build_databricks_web_search_adapter,
    )

    domain_filter = config.get("domain_filter")
    url_allowed = None
    restrict_to_domains: list[str] | None = None
    if domain_filter:
        from databricks_deep_research.tools.builtins.web_search import _domain_matches

        _df = list(domain_filter)
        url_allowed = lambda u: _domain_matches(u, _df)  # noqa: E731
        # The flat ``domain_filter`` list is an allowlist (``_domain_matches`` semantics),
        # so forward it for the allowlist push-down too. The adapter reduces it to the
        # pushable bare-domain subset (all-or-nothing) and keeps url_allowed authoritative.
        restrict_to_domains = _df

    return build_databricks_web_search_adapter(
        client_provider=client_provider,
        model=model,
        model_family=config.get("model_family"),
        max_results=config.get("max_results", 10),
        timeout_seconds=config.get("timeout_seconds", 30.0),
        resolve_redirects=config.get("resolve_redirects", True),
        url_allowed=url_allowed,
        restrict_to_domains=restrict_to_domains,
        push_allowed_domains=config.get("push_allowed_domains", True),
    )


def _resolve_crawl_provider(provider: str, ctx: ToolFactoryContext) -> Any:
    """Create a ContentCrawler for the named provider."""
    if provider == "jina":
        api_key = ctx.api_keys.get("jina") or os.environ.get("JINA_API_KEY")
        from databricks_deep_research.tools.builtins.jina_crawl import JinaCrawlAdapter

        return JinaCrawlAdapter(api_key=api_key)

    raise ValueError(
        f"Unknown crawl provider: {provider!r}. "
        f"Supported: {sorted(_CRAWL_PROVIDERS)}"
    )


class BuiltinToolFactory:
    """Creates web_search, web_crawl, and file_search tools from declarations."""

    catalog_cards: ClassVar[Mapping[str, CatalogCard]] = {
        "web_search": CatalogCard(
            summary="Search the public web and return ranked result snippets with source URLs.",
            input_prose=(
                "Provide a search query string. The tool returns the top results "
                "from the configured provider. Use focused phrasing — keywords or a "
                "short question — rather than long paragraphs."
            ),
            output_prose=(
                "Returns a list of search results. Each result includes a title, a "
                "URL, and a content snippet. URLs are registered with the framework "
                "so the LLM never sees raw URLs; cite results by the index assigned "
                "in the result block."
            ),
        ),
        "web_crawl": CatalogCard(
            summary="Fetch the body of a single web page and return cleaned content.",
            input_prose=(
                "Provide one URL to crawl. The tool dereferences the URL through the "
                "configured crawler (httpx + trafilatura by default), strips chrome, "
                "and returns the main page body."
            ),
            output_prose=(
                "Returns the page's cleaned text content, optionally with extracted "
                "tables. Long pages are truncated to the configured max_content_length."
            ),
        ),
        "web_research": CatalogCard(
            summary="Search the web and auto-fetch the top K result bodies in one call.",
            input_prose=(
                "Provide a search query string. The tool runs a search, then "
                "automatically crawls the top K results so the body content arrives "
                "in the first response — no follow-up crawl required."
            ),
            output_prose=(
                "Returns search results enriched with full page bodies for the top K "
                "hits. Each entry includes title, URL, snippet, and (where fetched) "
                "the cleaned page body."
            ),
        ),
        "academic_search": CatalogCard(
            summary="Search key-less scholarly APIs and return papers with abstracts and links.",
            input_prose=(
                "Provide a focused scholarly query string — author names, "
                "concepts, methods, or a paper title work best. The backing "
                "corpus is selected by the tool's configured provider (arxiv, "
                "openalex, pubmed_central, or semantic_scholar); no API key is "
                "required."
            ),
            output_prose=(
                "Returns a list of papers. Each entry includes a title, a "
                "landing-page or DOI URL, and an abstract (reconstructed where the "
                "provider ships it inverted). URLs are registered with the "
                "framework so cite results by the assigned index."
            ),
        ),
        "file_search": CatalogCard(
            summary="Search a local file index by keyword and return matching file paths.",
            input_prose=(
                "Provide a search query. The tool matches against the configured file "
                "index and returns paths plus snippet matches."
            ),
            output_prose=(
                "Returns a list of matching files with relevance-ranked snippets and "
                "absolute paths in the configured index root."
            ),
        ),
        "compute": CatalogCard(
            summary="Execute Python code in a sandboxed namespace and capture stdout/return value.",
            input_prose=(
                "Provide a Python code snippet as a string. The tool runs the code in "
                "a restricted namespace (configurable allowed_modules), captures "
                "stdout, and returns the trailing expression value when present."
            ),
            output_prose=(
                "Returns captured stdout and the value of the last expression. "
                "Variables defined here persist across compute calls in the same session "
                "via the compute namespace."
            ),
        ),
        "compute_namespace": CatalogCard(
            summary="List variables currently bound in the compute tool's namespace.",
            input_prose=(
                "Takes no arguments. Reports the current state of the sibling compute "
                "tool's namespace so subsequent compute calls can reference prior "
                "results by name."
            ),
            output_prose=(
                "Returns a mapping of variable names to short summaries (type and "
                "shape/length where applicable) for everything currently bound in the "
                "compute namespace."
            ),
        ),
        "table_discovery": CatalogCard(
            summary="List tables exposed to this agent and register discovered tables.",
            input_prose=(
                "Optionally provide a name_pattern substring filter and a detail level "
                "(basic, schema, full). Tables returned are added to the binding "
                "registry as DISCOVERED bindings so downstream table_* tools can "
                "reference them by name."
            ),
            output_prose=(
                "Returns a list of tables with name, fqn, and (when detail>=schema) "
                "column types. Use the binding name in subsequent table_search / "
                "table_read / table_aggregate calls."
            ),
        ),
        "table_search": CatalogCard(
            summary="Substring-search the content column of a registered table binding.",
            input_prose=(
                "Provide a binding name and a query substring; optionally narrow with a "
                "TableFilter `where` and project additional columns. The tool runs a "
                "parameterised SELECT with LIKE matching on the content column."
            ),
            output_prose=(
                "Returns matching rows projected to the binding's id and content "
                "columns plus any extras requested. Source identifiers are registered "
                "so citations resolve back to the originating row."
            ),
        ),
        "table_read": CatalogCard(
            summary="Read rows from a registered table binding with filter / projection / pagination.",
            input_prose=(
                "Provide a binding name and optional `where` filter, `columns` "
                "projection, `order_by` list (prefix '-' for DESC), `limit`, and "
                "`offset`. Use this when you already know which rows you need."
            ),
            output_prose=(
                "Returns a list of rows from the table, each row a mapping of column "
                "name to value, in the requested order."
            ),
        ),
        "table_neighbors": CatalogCard(
            summary="Fetch sibling rows around an anchor by partition and ordering columns.",
            input_prose=(
                "Provide a binding name, the anchor row id, and `before` / `after` "
                "window sizes. The tool returns rows in the same partition_column "
                "whose order_column falls within [order - before, order + after]."
            ),
            output_prose=(
                "Returns a contiguous window of rows from the binding ordered by the "
                "binding's order_column."
            ),
        ),
        "table_load": CatalogCard(
            summary="Materialise specific row(s) from a binding into the compute namespace as Table objects.",
            input_prose=(
                "Provide a binding name and one or more id values. Optionally pass "
                "`as_var` to bind the result under a specific name; otherwise the "
                "loaded row is exposed as `last_table` and appended to `tables`."
            ),
            output_prose=(
                "Returns the loaded row(s) as JSON. When a compute namespace is wired "
                "and `as_var` is provided, the row is also injected into the compute "
                "namespace as a `Table` object for use in compute() calls."
            ),
        ),
        "table_aggregate": CatalogCard(
            summary="Aggregate rows from a binding via count / sum / avg / min / max with optional GROUP BY.",
            input_prose=(
                "Provide a binding name and an op (count, sum, avg, min, max). For "
                "non-count ops a target column is required and must appear in the "
                "binding's numeric_columns. Optionally pass `where`, `group_by`, and "
                "`limit`."
            ),
            output_prose=(
                "Returns the aggregate result(s). When group_by is set, the response "
                "carries one row per group with the requested aggregate value."
            ),
        ),
        "python_function": CatalogCard(
            summary=(
                "Run a FIXED, design-time Python function (SMA, reshaping, "
                "forecast glue) in the run's sandboxed session."
            ),
            input_prose=(
                "Provide the declared parameters (they become globals). The "
                "function's code is fixed at design time; the script assigns "
                "'result'. Variables it defines persist in the session for "
                "later calls."
            ),
            output_prose=(
                "Returns the script's 'result' value plus captured stdout. "
                "With citeable enabled, the result is admitted as evidence."
            ),
        ),
        "uc_function": CatalogCard(
            summary=(
                "Invoke an existing Unity Catalog scalar function "
                "(catalog.schema.fn) via SQL under the caller's identity (OBO)."
            ),
            input_prose=(
                "Provide the function's declared arguments by name. Argument "
                "values are bound as SQL parameters; the function runs on a SQL "
                "warehouse and returns a single scalar."
            ),
            output_prose=(
                "Returns the function's scalar result. With citeable enabled "
                "(default), the result is admitted as evidence with a "
                "'uc-function://' source."
            ),
        ),
        "read_skill": CatalogCard(
            summary="Load the full body of an attached skill by name (progressive disclosure).",
            input_prose=(
                "Provide the exact name of a skill listed (name + description only) "
                "in the system prompt. The tool fetches that skill's full instructions "
                "from the configured skill store."
            ),
            output_prose=(
                "Returns the skill body as markdown. On an unknown name, returns a "
                "graceful miss listing the available skill names."
            ),
        ),
    }

    safe_probes: ClassVar[Mapping[str, SafeProbe | None]] = {
        "web_search": None,
        "web_crawl": None,
        "web_research": None,
        "academic_search": None,
        "file_search": None,
        "compute": None,
        "compute_namespace": None,
        "table_discovery": None,
        "table_search": None,
        "table_read": None,
        "table_neighbors": None,
        "table_load": None,
        "table_aggregate": None,
        "read_skill": None,
        "python_function": None,
        "uc_function": None,
    }

    def supports(self, kind: str) -> bool:
        return kind in _SUPPORTED_KINDS

    async def create(
        self, decl: ToolDeclaration, ctx: ToolFactoryContext
    ) -> ResearchTool:
        if decl.kind == "web_search":
            provider = decl.config.get("provider")
            if provider is None:
                # Legacy path: use pre-built ctx.search_client.
                if ctx.search_client is None:
                    raise ValueError(
                        f"search_client required in ToolFactoryContext for "
                        f"web_search tool '{decl.name}'"
                    )
                search_client = ctx.search_client
            else:
                search_client = _resolve_search_provider(provider, ctx, decl.config)

            from databricks_deep_research.tools.builtins.web_search import WebSearchTool

            return WebSearchTool(
                search_client=search_client,
                domain_filter=decl.config.get("domain_filter"),
                max_results=decl.config.get("max_results", 5),
                max_content_per_result=decl.config.get(
                    "max_content_per_result", 5000
                ),
                extract_tables=decl.config.get("extract_tables", True),
            )

        if decl.kind == "web_crawl":
            provider = decl.config.get("provider")
            if provider is not None:
                crawler = _resolve_crawl_provider(provider, ctx)
            else:
                crawler = ctx.crawler
            from databricks_deep_research.tools.builtins.web_crawl import WebCrawlTool

            return WebCrawlTool(
                crawler=crawler,
                timeout=decl.config.get("timeout", 30.0),
                max_content_length=decl.config.get("max_content_length", 50_000),
                extract_tables=decl.config.get("extract_tables", True),
            )

        if decl.kind == "web_research":
            # Merged tool: search + auto-crawl top K in one call. Lets the
            # researcher get real source bodies on the FIRST tool invocation
            # without relying on the LLM to orchestrate search→crawl correctly.
            provider = decl.config.get("provider")
            if provider is None:
                if ctx.search_client is None:
                    raise ValueError(
                        f"search_client required in ToolFactoryContext for "
                        f"web_research tool '{decl.name}' (set "
                        f"brave_api_key in ToolFactoryContext.from_defaults "
                        f"or provide search_client directly)"
                    )
                search_client = ctx.search_client
            else:
                search_client = _resolve_search_provider(provider, ctx, decl.config)

            crawl_provider = decl.config.get("crawl_provider")
            if crawl_provider is not None:
                crawler = _resolve_crawl_provider(crawl_provider, ctx)
            else:
                crawler = ctx.crawler

            from databricks_deep_research.tools.builtins.web_research import (
                WebResearchTool,
            )

            return WebResearchTool(
                search_client=search_client,
                crawler=crawler,
                auto_fetch_top_k=decl.config.get("auto_fetch_top_k", 5),
                total_results=decl.config.get("total_results", 10),
                max_body_chars=decl.config.get("max_body_chars", 8000),
            )

        if decl.kind == "academic_search":
            # Key-less scholarly retriever. config.provider selects the backing
            # corpus (arxiv [default], openalex, pubmed_central,
            # semantic_scholar). Network is constructed inside the tool from a
            # mockable http_fetch seam; no API key required (an optional key is
            # forwarded for providers that offer one — Semantic Scholar / NCBI).
            from databricks_deep_research.tools.builtins.academic_search import (
                ACADEMIC_PROVIDERS,
                DEFAULT_ACADEMIC_PROVIDER,
            )

            provider = (
                _clean_str(decl.config.get("provider")) or DEFAULT_ACADEMIC_PROVIDER
            ).lower()
            tool_cls = ACADEMIC_PROVIDERS.get(provider)
            if tool_cls is None:
                raise ValueError(
                    f"Unknown academic_search provider: {provider!r}. "
                    f"Supported: {sorted(ACADEMIC_PROVIDERS)}"
                )

            return tool_cls(
                http_fetch=ctx.extras.get("_academic_http_fetch"),
                name=decl.name,
                description=decl.description or "",
                max_results=decl.config.get("max_results", 5),
                max_content_chars=decl.config.get("max_content_chars", 8000),
                timeout_seconds=decl.config.get("timeout_seconds", 30.0),
                api_key=_clean_str(decl.config.get("api_key"))
                or ctx.api_keys.get(provider),
            )

        if decl.kind == "file_search":
            if ctx.file_index is None:
                raise ValueError(
                    f"file_index required in ToolFactoryContext for "
                    f"file_search tool '{decl.name}'"
                )
            from databricks_deep_research.tools.builtins.file_search import FileSearchTool

            return FileSearchTool(file_index=ctx.file_index)

        if decl.kind == "compute":
            from databricks_deep_research.tools.builtins.compute import PythonComputeTool

            return PythonComputeTool(
                name=decl.name,
                allowed_modules=decl.config.get("allowed_modules"),
                extra_modules=decl.config.get("extra_modules"),
                enable_dataframes=decl.config.get("enable_dataframes", False),
                max_execution_seconds=decl.config.get("max_execution_seconds", 10.0),
                max_output_chars=decl.config.get("max_output_chars", 10_000),
                max_code_length=decl.config.get("max_code_length", 20_000),
                description=decl.description,
            )

        if decl.kind == "compute_namespace":
            from databricks_deep_research.tools.builtins.compute import PythonComputeTool
            from databricks_deep_research.tools.builtins.compute_namespace import (
                ComputeNamespaceListTool,
            )

            def _resolve_compute() -> PythonComputeTool | None:
                """Lazy resolution: look up sibling 'compute' tool from resolver cache."""
                compute_name = decl.config.get("compute_tool_name", "compute")
                cached = ctx.extras.get("_resolver_cache", {}).get(compute_name)
                if isinstance(cached, PythonComputeTool):
                    return cached
                return None

            return ComputeNamespaceListTool(
                compute_resolver=_resolve_compute,
                name=decl.name,
                description=decl.description,
            )

        if decl.kind == "python_function":
            from databricks_deep_research.tools.builtins._skill_script_runner import (
                DATA_LIBS,
            )
            from databricks_deep_research.tools.builtins.compute import (
                PythonComputeTool,
            )
            from databricks_deep_research.tools.builtins.python_function import (
                PythonFunctionTool,
            )
            from databricks_deep_research.tools.code_executor import (
                RestrictedCodeExecutor,
                SandboxSession,
                SandboxSessionHolder,
            )

            code = decl.config.get("code")
            if not isinstance(code, str) or not code.strip():
                raise ValueError(
                    f"python_function '{decl.name}' requires non-empty config.code"
                )
            extra_modules = [
                m
                for m in (decl.config.get("extra_allowed_modules") or [])
                if isinstance(m, str) and m
            ]
            unknown_modules = [m for m in extra_modules if m not in DATA_LIBS]
            if unknown_modules:
                raise ValueError(
                    f"python_function '{decl.name}': extra_allowed_modules "
                    f"{unknown_modules} are not vetted data libraries; "
                    f"allowed: {sorted(DATA_LIBS)}"
                )
            data_lib_mode = decl.config.get("data_lib_mode", "facade")
            if data_lib_mode not in ("facade", "live"):
                raise ValueError(
                    f"python_function '{decl.name}': data_lib_mode must be "
                    f"'facade' or 'live', got {data_lib_mode!r}"
                )
            backend = decl.config.get("backend", "subprocess")
            _validate_python_function_code(decl.name, code, frozenset(extra_modules))
            params = [
                p for p in (decl.config.get("params") or []) if isinstance(p, dict)
            ]
            timeout_seconds = float(decl.config.get("timeout_seconds", 10.0))
            reads_namespace = [
                n
                for n in (decl.config.get("reads_namespace") or [])
                if isinstance(n, str) and n
            ]
            bind_result = decl.config.get("bind_result")
            bind_result = bind_result if isinstance(bind_result, str) and bind_result else None
            citeable = bool(decl.config.get("citeable", False))

            def _resolve_compute() -> PythonComputeTool | None:
                """Sibling lookup: the run's shared compute scratchpad, if any."""
                cached = ctx.extras.get("_resolver_cache", {})
                candidate = cached.get("compute")
                if isinstance(candidate, PythonComputeTool):
                    return candidate
                for value in cached.values():
                    if isinstance(value, PythonComputeTool):
                        return value
                return None

            if backend == "restricted":
                if not _inprocess_python_function_allowed(ctx):
                    raise ValueError(
                        f"python_function '{decl.name}': backend 'restricted' is "
                        "disabled on this host — in-process execution is not a "
                        "hard security boundary. Use the default 'subprocess' "
                        "backend, or have the operator enable "
                        "execution.allow_inprocess_python_function."
                    )
                return PythonFunctionTool(
                    name=decl.name,
                    code=code,
                    params=params,
                    description=decl.description,
                    backend="restricted",
                    restricted_executor=RestrictedCodeExecutor(
                        enable_dataframes=bool(extra_modules),
                        max_execution_seconds=timeout_seconds,
                    ),
                    compute_resolver=_resolve_compute,
                    reads_namespace=reads_namespace,
                    bind_result=bind_result,
                    citeable=citeable,
                    timeout_seconds=timeout_seconds,
                )
            if backend != "subprocess":
                raise ValueError(
                    f"python_function '{decl.name}': unknown backend {backend!r} "
                    "(expected 'subprocess' or 'restricted')"
                )
            if data_lib_mode == "live" and not _inprocess_python_function_allowed(ctx):
                raise ValueError(
                    f"python_function '{decl.name}': data_lib_mode 'live' exposes "
                    "the full pandas/numpy modules and requires the operator "
                    "trust switch; use the default 'facade' mode."
                )
            holder = ctx.extras.get("_sandbox_session")
            if not isinstance(holder, SandboxSessionHolder):
                holder = SandboxSessionHolder()
                ctx.extras["_sandbox_session"] = holder

            session_holder = holder

            def _get_session() -> SandboxSession:
                return session_holder.get_or_create(
                    wall_timeout_seconds=timeout_seconds,
                    extra_allowed_modules=extra_modules,
                    data_lib_mode=data_lib_mode,
                )

            return PythonFunctionTool(
                name=decl.name,
                code=code,
                params=params,
                description=decl.description,
                backend="subprocess",
                session_provider=_get_session,
                compute_resolver=_resolve_compute,
                extra_allowed_modules=extra_modules,
                data_lib_mode=data_lib_mode,
                reads_namespace=reads_namespace,
                bind_result=bind_result,
                citeable=citeable,
                timeout_seconds=timeout_seconds,
            )

        if decl.kind == "uc_function":
            if ctx.sql_executor is None:
                raise ValueError(
                    f"sql_executor required in ToolFactoryContext for "
                    f"uc_function tool '{decl.name}' — set STORAGE_WAREHOUSE_ID "
                    f"or TABLE_TOOLS_WAREHOUSE_ID so the OBO SQL executor is "
                    f"wired"
                )
            from databricks_deep_research.tools.builtins.uc_function import (
                UCFunctionTool,
            )

            function_name = decl.config.get("function")
            if not isinstance(function_name, str) or not function_name.strip():
                raise ValueError(
                    f"uc_function '{decl.name}' requires config.function "
                    f"(the 'catalog.schema.fn' FQN)"
                )
            uc_params = [
                p
                for p in (decl.config.get("params") or [])
                if isinstance(p, dict)
            ]
            return UCFunctionTool(
                name=decl.name,
                function_name=function_name.strip(),
                sql_executor=ctx.sql_executor,
                params=uc_params,
                description=decl.description,
                citeable=bool(decl.config.get("citeable", True)),
                returns_table=bool(decl.config.get("returns_table", False)),
            )

        if decl.kind == "read_skill":
            # The SkillStore is injected by the host via ctx.extras using the
            # framework-reserved ``_skill_store`` key (same DI idiom as the
            # compute resolver). The framework stays free of any persistence
            # dependency — the host supplies a FilesystemSkillStore or its own.
            skill_store = ctx.extras.get("_skill_store")
            if skill_store is None:
                raise ValueError(
                    f"read_skill tool '{decl.name}' requires a SkillStore in "
                    f"ToolFactoryContext.extras['_skill_store']"
                )
            from databricks_deep_research.tools.builtins.read_skill import (
                ReadSkillTool,
            )

            return ReadSkillTool(
                skill_store=skill_store,
                name=decl.name,
                description=decl.description,
            )

        if decl.kind == "table_discovery":
            if ctx.table_registry is None:
                raise ValueError(
                    f"table_registry required in ToolFactoryContext for "
                    f"table_discovery tool '{decl.name}'"
                )
            from databricks_deep_research.tools.builtins.text_table.tools.discovery import (
                TableDiscoveryTool,
            )

            return TableDiscoveryTool(
                provider=ctx.table_discovery_provider,
                registry=ctx.table_registry,
                schema_cache=ctx.schema_cache,
                sql_executor=ctx.sql_executor,
                name=decl.name,
                description=decl.description or None,
            )

        if decl.kind == "table_search":
            default_binding = _ensure_declared_table_binding(decl, ctx)
            if ctx.table_registry is None:
                raise ValueError(
                    f"table_registry required in ToolFactoryContext for "
                    f"table_search tool '{decl.name}'"
                )
            if ctx.schema_cache is None:
                raise ValueError(
                    f"schema_cache required in ToolFactoryContext for "
                    f"table_search tool '{decl.name}'"
                )
            if ctx.sql_executor is None:
                raise ValueError(
                    f"sql_executor required in ToolFactoryContext for "
                    f"table_search tool '{decl.name}'"
                )
            from databricks_deep_research.tools.builtins.text_table.tools.search import (
                TableSearchTool,
            )

            return TableSearchTool(
                registry=ctx.table_registry,
                schema_cache=ctx.schema_cache,
                sql_executor=ctx.sql_executor,
                name=decl.name,
                description=decl.description or None,
                default_binding=default_binding,
                default_columns=_projection_columns_from_config(decl.config),
            )

        if decl.kind == "table_read":
            default_binding = _ensure_declared_table_binding(decl, ctx)
            if ctx.table_registry is None:
                raise ValueError(
                    f"table_registry required in ToolFactoryContext for "
                    f"table_read tool '{decl.name}'"
                )
            if ctx.schema_cache is None:
                raise ValueError(
                    f"schema_cache required in ToolFactoryContext for "
                    f"table_read tool '{decl.name}'"
                )
            if ctx.sql_executor is None:
                raise ValueError(
                    f"sql_executor required in ToolFactoryContext for "
                    f"table_read tool '{decl.name}'"
                )
            from databricks_deep_research.tools.builtins.text_table.tools.read import (
                TableReadTool,
            )

            return TableReadTool(
                registry=ctx.table_registry,
                schema_cache=ctx.schema_cache,
                sql_executor=ctx.sql_executor,
                name=decl.name,
                description=decl.description or None,
                default_binding=default_binding,
                default_columns=_projection_columns_from_config(decl.config),
                default_order_by=_order_by_from_config(decl.config),
            )

        if decl.kind == "table_neighbors":
            default_binding = _ensure_declared_table_binding(decl, ctx)
            if ctx.table_registry is None:
                raise ValueError(
                    f"table_registry required in ToolFactoryContext for "
                    f"table_neighbors tool '{decl.name}'"
                )
            if ctx.schema_cache is None:
                raise ValueError(
                    f"schema_cache required in ToolFactoryContext for "
                    f"table_neighbors tool '{decl.name}'"
                )
            if ctx.sql_executor is None:
                raise ValueError(
                    f"sql_executor required in ToolFactoryContext for "
                    f"table_neighbors tool '{decl.name}'"
                )
            from databricks_deep_research.tools.builtins.text_table.tools.neighbors import (
                TableNeighborsTool,
            )

            return TableNeighborsTool(
                registry=ctx.table_registry,
                schema_cache=ctx.schema_cache,
                sql_executor=ctx.sql_executor,
                name=decl.name,
                description=decl.description or None,
                default_binding=default_binding,
            )

        if decl.kind == "table_load":
            default_binding = _ensure_declared_table_binding(decl, ctx)
            if ctx.table_registry is None:
                raise ValueError(
                    f"table_registry required in ToolFactoryContext for "
                    f"table_load tool '{decl.name}'"
                )
            if ctx.schema_cache is None:
                raise ValueError(
                    f"schema_cache required in ToolFactoryContext for "
                    f"table_load tool '{decl.name}'"
                )
            if ctx.sql_executor is None:
                raise ValueError(
                    f"sql_executor required in ToolFactoryContext for "
                    f"table_load tool '{decl.name}'"
                )
            from databricks_deep_research.tools.builtins.text_table.tools.load import (
                TableLoadTool,
            )

            # Optional: route compute namespace mutations through the sibling
            # compute tool's `inject_variable` API. We resolve eagerly via the
            # _resolver_cache so namespace_setter is bound before the first
            # tool call rather than per-call.
            namespace_setter = None
            compute_tool_name = decl.config.get("compute_tool_name", "compute")
            cached = ctx.extras.get("_resolver_cache", {}).get(compute_tool_name)
            if cached is not None and hasattr(cached, "inject_variable"):
                namespace_setter = cached.inject_variable

            return TableLoadTool(
                registry=ctx.table_registry,
                schema_cache=ctx.schema_cache,
                sql_executor=ctx.sql_executor,
                compute_namespace_setter=namespace_setter,
                name=decl.name,
                description=decl.description or None,
                default_binding=default_binding,
                default_columns=_projection_columns_from_config(decl.config),
                default_as_var=_clean_str(decl.config.get("store_in_compute"))
                or _clean_str(decl.config.get("as_var")),
            )

        if decl.kind == "table_aggregate":
            default_binding = _ensure_declared_table_binding(decl, ctx)
            if ctx.table_registry is None:
                raise ValueError(
                    f"table_registry required in ToolFactoryContext for "
                    f"table_aggregate tool '{decl.name}'"
                )
            if ctx.schema_cache is None:
                raise ValueError(
                    f"schema_cache required in ToolFactoryContext for "
                    f"table_aggregate tool '{decl.name}'"
                )
            if ctx.sql_executor is None:
                raise ValueError(
                    f"sql_executor required in ToolFactoryContext for "
                    f"table_aggregate tool '{decl.name}'"
                )
            from databricks_deep_research.tools.builtins.text_table.tools.aggregate import (
                TableAggregateTool,
            )

            return TableAggregateTool(
                registry=ctx.table_registry,
                schema_cache=ctx.schema_cache,
                sql_executor=ctx.sql_executor,
                name=decl.name,
                description=decl.description or None,
                default_binding=default_binding,
            )

        raise ValueError(f"Unsupported kind: {decl.kind}")
