"""Knowledge Assistant Tool for Databricks serving endpoints.

Provides question-answering via Databricks-hosted Knowledge Assistants.
Each configured endpoint creates a separate tool instance with a unique name.

Supports two modes:
1. System-configured (from app.yaml) - uses service principal authentication
2. User-configured (from UserDataSource) - uses OBO authentication

Includes MLflow tracing for observability.

Part of 007-enterprise-data-sources feature (T046).

Example configuration (config/app.yaml):
    knowledge_assistants:
      enabled: true
      endpoints:
        product_assistant:
          endpoint_name: product-knowledge-assistant
          description: Ask questions about our products
"""

import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from deep_research.agent.tools.base import (
    ResearchContext,
    ToolDefinition,
    ToolResult,
)
from deep_research.core.logging_utils import get_logger
from deep_research.services.metrics import record_source_query

if TYPE_CHECKING:
    from deep_research.models.data_source import UserDataSource
    from deep_research.services.obo_client import OBODatabricksClient

logger = get_logger(__name__)


@dataclass
class KACitation:
    """A citation from Knowledge Assistant response."""

    source: str
    title: str
    url: str | None
    snippet: str | None


class KnowledgeAssistantTool:
    """
    Knowledge Assistant tool implementing the ResearchTool protocol.

    Queries a Databricks-hosted Knowledge Assistant for answers with citations.
    Tool name is generated as 'ask_{endpoint_name}' to allow multiple KAs.

    Supports two authentication modes:
    1. Service principal (default) - for system-configured endpoints
    2. OBO (on-behalf-of) - for user-configured endpoints via UserDataSource

    Features:
    - Research context passing (include_context parameter)
    - Confidence level extraction from response
    - Internal reference tracking for citations
    """

    def __init__(
        self,
        *,
        endpoint_name: str,
        tool_name: str | None = None,
        description: str | None = None,
        # OBO authentication support
        obo_client: "OBODatabricksClient | None" = None,
        data_source: "UserDataSource | None" = None,
        # Context passing support
        pass_context: bool = False,
        max_context_chars: int = 4000,
    ) -> None:
        """Initialize the Knowledge Assistant tool.

        Args:
            endpoint_name: Databricks serving endpoint name for the KA.
            tool_name: Custom tool name. Defaults to 'ask_{endpoint_name}'.
            description: Custom description for LLM. Defaults to generic.
            obo_client: OBO client for user authentication. If provided with
                       data_source, uses OBO auth instead of service principal.
            data_source: UserDataSource configuration from database.
            pass_context: Whether to include research context with questions.
            max_context_chars: Maximum context characters to include.
        """
        self._endpoint_name = endpoint_name
        self._obo_client = obo_client
        self._data_source = data_source
        self._pass_context = pass_context
        self._max_context_chars = max_context_chars

        # Determine if using OBO authentication
        self._use_obo = obo_client is not None and data_source is not None

        # Generate tool name
        if tool_name:
            self._tool_name = tool_name
        elif self._use_obo and data_source:
            # For user sources, use source name for uniqueness
            safe_name = data_source.name.replace(" ", "_").replace("-", "_").lower()
            self._tool_name = f"ask_{safe_name}"
        else:
            self._tool_name = f"ask_{endpoint_name.replace('-', '_')}"

        # Generate description
        if description:
            self._description = description
        elif self._use_obo and data_source and data_source.description:
            self._description = (
                f"{data_source.description} "
                "Returns an answer with source citations when available."
            )
        else:
            self._description = (
                f"Ask the '{endpoint_name}' Knowledge Assistant a question. "
                "Returns an answer with source citations when available."
            )

        # Lazy-loaded client (for non-OBO mode)
        self._client: Any = None

        # Build parameters schema
        params: dict[str, Any] = {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": (
                        "Question to ask the Knowledge Assistant. "
                        "Be specific and clear for best results."
                    ),
                },
            },
            "required": ["question"],
        }

        # Add include_context parameter if context passing is enabled
        if self._pass_context:
            params["properties"]["include_context"] = {
                "type": "boolean",
                "description": (
                    "Include research context (findings from other sources) "
                    "with the question to help the assistant provide more "
                    "relevant answers. Default: true."
                ),
                "default": True,
            }

        self._definition = ToolDefinition(
            name=self._tool_name,
            description=self._description,
            parameters=params,
            source_type="knowledge_assistant",
        )

    def _get_client(self) -> Any:
        """Get or create WorkspaceClient (for non-OBO mode)."""
        if self._client is None:
            try:
                from databricks.sdk import WorkspaceClient

                self._client = WorkspaceClient()
                logger.info(
                    "WorkspaceClient initialized for Knowledge Assistant",
                    endpoint=self._endpoint_name,
                )
            except ImportError as exc:
                raise ImportError(
                    "databricks-sdk package not installed. "
                    "Install with: pip install databricks-sdk"
                ) from exc
        return self._client

    async def _get_obo_client(self, user_token: str) -> Any:
        """Get OBO-authenticated WorkspaceClient.

        Args:
            user_token: User's OAuth token for OBO auth.

        Returns:
            WorkspaceClient with user's permissions.
        """
        if not self._obo_client:
            raise ValueError("OBO client not configured")

        return await self._obo_client.get_client(user_token)

    def _add_research_context(
        self,
        question: str,
        context: ResearchContext,
    ) -> str:
        """Add research context to the question for better answers.

        Includes relevant findings from other sources to help the
        Knowledge Assistant provide more contextual answers.

        Args:
            question: Original user question.
            context: Research context with findings and sources.

        Returns:
            Enhanced question with context prefix.
        """
        context_parts: list[str] = []

        # Add query context from plugin_data if available
        original_query = context.plugin_data.get("original_query")
        if original_query:
            context_parts.append(f"Research Topic: {original_query}")

        # Add relevant evidence found so far
        if context.evidence_registry:
            evidence_items = list(context.evidence_registry.values())
            if evidence_items:
                source_summaries: list[str] = []
                total_chars = 0

                for evidence in evidence_items[:10]:  # Limit to top 10 items
                    if isinstance(evidence, dict):
                        title = evidence.get("title", "Untitled")
                        snippet = evidence.get("snippet", evidence.get("content", ""))
                    else:
                        title = getattr(evidence, "title", "Untitled")
                        snippet = getattr(evidence, "snippet", getattr(evidence, "content", ""))

                    summary = f"- {title}"
                    if snippet:
                        snippet_preview = str(snippet)[:200]
                        summary += f": {snippet_preview}..."

                    if total_chars + len(summary) > self._max_context_chars:
                        break

                    source_summaries.append(summary)
                    total_chars += len(summary)

                if source_summaries:
                    context_parts.append(
                        "Relevant findings from other sources:\n" +
                        "\n".join(source_summaries)
                    )

        # Combine context with question
        if context_parts:
            context_text = "\n\n".join(context_parts)
            return (
                f"[Research Context]\n{context_text}\n\n"
                f"[Question]\n{question}"
            )

        return question

    @property
    def definition(self) -> ToolDefinition:
        """Return tool definition for LLM function calling."""
        return self._definition

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ResearchContext,
    ) -> ToolResult:
        """Execute Knowledge Assistant query and return answer.

        Supports OBO authentication when configured with obo_client and data_source.
        Supports context passing when pass_context is enabled.
        Includes MLflow tracing for observability.

        Args:
            arguments: Tool arguments containing 'question', optionally 'include_context'
            context: Research context with identity, registries, and user_token

        Returns:
            ToolResult with answer content, sources, and confidence level
        """
        question = arguments.get("question", "")
        include_context = arguments.get("include_context", True)

        logger.info(
            "KA_TOOL_EXECUTE",
            tool_name=self._tool_name,
            endpoint_name=self._endpoint_name,
            question=question[:100],
            obo_mode=self._use_obo,
            context_passing=self._pass_context,
            include_context=include_context,
        )

        # Add research context if enabled and requested
        enhanced_question = question
        if self._pass_context and include_context:
            enhanced_question = self._add_research_context(question, context)

        start_time = time.perf_counter()

        # Determine source name for attribution (used in tracing and response)
        source_name = (
            self._data_source.name if self._data_source
            else self._endpoint_name
        )

        try:
            # Get appropriate client based on auth mode
            if self._use_obo:
                if not context.user_token:
                    logger.info(
                        "KA_SPAN_ATTRS",
                        success=False,
                        error_type="MissingToken",
                    )
                    return ToolResult(
                        content="OBO authentication required but no user token available.",
                        success=False,
                        error="Missing user_token in context",
                    )
                client = await self._get_obo_client(context.user_token)
            else:
                client = self._get_client()

            # Query the KA serving endpoint (sync API, run in executor).
            # KA endpoints use the Responses API format: they require the
            # ``input`` field and return ``output`` with ``output_text``
            # items.  The SDK's high-level ``serving_endpoints.query()``
            # does not deserialize the ``output`` field, so we call the
            # REST API directly through ``api_client.do()`` which returns
            # the raw dict.
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.api_client.do(
                    "POST",
                    f"/serving-endpoints/{self._endpoint_name}/invocations",
                    body={
                        "input": [
                            {"role": "user", "content": enhanced_question}
                        ]
                    },
                ),
            )

            duration_ms = (time.perf_counter() - start_time) * 1000

            # Parse response with confidence level
            answer, citations, confidence = self._parse_response_with_confidence(response)

            if not answer:
                logger.info(
                    "KA_SPAN_ATTRS",
                    duration_ms=duration_ms,
                    result_count=0,
                    success=True,
                    has_answer=False,
                )
                # Record metrics for monitoring (T108)
                record_source_query(
                    source_type="knowledge_assistant",
                    source_name=source_name,
                    latency_ms=duration_ms,
                    success=True,
                )
                return ToolResult(
                    content="The Knowledge Assistant could not provide an answer.",
                    success=True,
                    sources=[],
                    data={
                        "question": question,
                        "has_answer": False,
                        "context_included": self._pass_context and include_context,
                    },
                )

            # Build sources list for citation tracking
            sources: list[dict[str, Any]] = []
            citation_text: list[str] = []

            # Generate unique URL per response to avoid dedup collisions
            import hashlib
            response_hash = hashlib.sha256(answer[:200].encode()).hexdigest()[:12] if answer else "empty"

            for idx, citation in enumerate(citations):
                # Build navigable workspace URL
                if citation.url:
                    source_url = citation.url
                else:
                    from deep_research.core.auth import get_workspace_host
                    workspace_host = get_workspace_host()
                    if workspace_host:
                        source_url = f"{workspace_host}/ml/endpoints/{self._endpoint_name}#{response_hash}-{idx}"
                    else:
                        source_url = f"ka://{self._endpoint_name}/{response_hash}/{idx}"

                # Title fallback: citation title → source display name → endpoint name
                source_title = citation.title or source_name or self._endpoint_name

                source_entry: dict[str, Any] = {
                    "type": "knowledge_assistant",
                    "source_name": source_name,
                    "endpoint_name": self._endpoint_name,
                    "source": citation.source,
                    "title": source_title,
                    "url": source_url,
                    "snippet": citation.snippet,
                    "citation_index": idx,
                    "confidence_level": confidence,
                }

                # Add internal reference tracking
                if citation.source:
                    source_entry["internal_reference"] = {
                        "source_type": "knowledge_assistant",
                        "reference_id": citation.source,
                        "document_title": citation.title,
                    }

                sources.append(source_entry)

                if citation.title:
                    citation_text.append(f"[{idx + 1}] {citation.title}")

            # Format content with citations and confidence
            content = answer
            if citation_text:
                content += "\n\nSources:\n" + "\n".join(citation_text)

            if confidence and confidence != "unknown":
                content += f"\n\n[Confidence: {confidence}]"

            # Log final metrics
            logger.info(
                "KA_SPAN_ATTRS",
                duration_ms=duration_ms,
                result_count=len(citations),
                success=True,
                has_answer=True,
                confidence_level=confidence,
            )

            # Record metrics for monitoring (T108)
            record_source_query(
                source_type="knowledge_assistant",
                source_name=source_name,
                latency_ms=duration_ms,
                success=True,
            )

            return ToolResult(
                content=content,
                success=True,
                sources=sources,
                data={
                    "question": question,
                    "has_answer": True,
                    "citation_count": len(citations),
                    "endpoint_name": self._endpoint_name,
                    "source_name": source_name,
                    "confidence_level": confidence,
                    "context_included": self._pass_context and include_context,
                    "obo_authenticated": self._use_obo,
                },
            )

        except ImportError as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.info(
                "KA_SPAN_ATTRS",
                duration_ms=duration_ms,
                success=False,
                error_type="ImportError",
            )
            # Record error metrics for monitoring (T108)
            record_source_query(
                source_type="knowledge_assistant",
                source_name=source_name,
                latency_ms=duration_ms,
                success=False,
                error="ImportError: SDK not installed",
            )
            logger.error("Databricks SDK not available", error=str(e))
            return ToolResult(
                content="Knowledge Assistant is not available. SDK not installed.",
                success=False,
                error=str(e),
            )
        except Exception as e:
            error_msg = str(e)
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Provide helpful error messages for common OBO failures
            if self._use_obo:
                if "PERMISSION_DENIED" in error_msg or "403" in error_msg:
                    error_msg = (
                        f"Permission denied: You don't have access to endpoint "
                        f"'{self._endpoint_name}'. Please verify your permissions."
                    )
                elif "NOT_FOUND" in error_msg or "404" in error_msg:
                    error_msg = f"Endpoint not found: '{self._endpoint_name}' does not exist."

            # Log error info
            logger.info(
                "KA_SPAN_ATTRS",
                duration_ms=duration_ms,
                result_count=0,
                success=False,
                error_type=type(e).__name__,
            )

            # Record error metrics for monitoring (T108)
            record_source_query(
                source_type="knowledge_assistant",
                source_name=source_name,
                latency_ms=duration_ms,
                success=False,
                error=error_msg[:200],
            )

            logger.error(
                "KNOWLEDGE_ASSISTANT_ERROR",
                error=error_msg,
                error_type=type(e).__name__,
                endpoint=self._endpoint_name,
                obo_mode=self._use_obo,
                duration_ms=duration_ms,
                exc_info=True,
            )
            return ToolResult(
                content=f"Query failed: {error_msg[:500]}",
                success=False,
                error=error_msg,
            )

    def _parse_response(self, response: Any) -> tuple[str, list[KACitation]]:
        """Parse Knowledge Assistant response (legacy method).

        Args:
            response: Raw response from serving_endpoints.query()

        Returns:
            Tuple of (answer_text, list_of_citations)
        """
        answer, citations, _ = self._parse_response_with_confidence(response)
        return answer, citations

    def _parse_response_with_confidence(
        self,
        response: Any,
    ) -> tuple[str, list[KACitation], str]:
        """Parse Knowledge Assistant response with confidence level.

        Handles two response formats:
        1. **Responses API** (current) — raw dict returned by
           ``api_client.do()`` with an ``output`` array containing
           ``output_text`` items and ``url_citation`` annotations.
        2. **Legacy predictions** — ``QueryEndpointResponse`` with
           ``predictions[0].choices[0].message``.

        Args:
            response: Raw dict (Responses API) or QueryEndpointResponse object.

        Returns:
            Tuple of (answer_text, list_of_citations, confidence_level)
            confidence_level is one of: "high", "medium", "low", "unknown"
        """
        answer = ""
        citations: list[KACitation] = []
        confidence = "unknown"

        try:
            # ----- Responses API format (raw dict with "output") -----
            if isinstance(response, dict) and "output" in response:
                return self._parse_responses_api(response)

            # ----- Legacy format: predictions[0].choices[0].message -----
            predictions = getattr(response, "predictions", None)
            if not predictions or len(predictions) == 0:
                return answer, citations, confidence

            first_pred = predictions[0]

            if isinstance(first_pred, dict):
                choices = first_pred.get("choices", [])
            else:
                choices = getattr(first_pred, "choices", [])

            if not choices or len(choices) == 0:
                return answer, citations, confidence

            first_choice = choices[0]

            if isinstance(first_choice, dict):
                message = first_choice.get("message", {})
                confidence = (
                    first_choice.get("confidence")
                    or message.get("confidence", "unknown")
                )
            else:
                message = getattr(first_choice, "message", {})
                confidence = getattr(
                    first_choice, "confidence",
                    getattr(message, "confidence", "unknown")
                )

            if isinstance(message, dict):
                answer = message.get("content", "")
                raw_citations = message.get("citations", [])
            else:
                answer = getattr(message, "content", "")
                raw_citations = getattr(message, "citations", [])

            if confidence == "unknown" and raw_citations:
                if len(raw_citations) >= 3:
                    confidence = "high"
                elif len(raw_citations) >= 1:
                    confidence = "medium"
                else:
                    confidence = "low"

            for cit in raw_citations:
                if isinstance(cit, dict):
                    citations.append(KACitation(
                        source=cit.get("source", ""),
                        title=cit.get("title", ""),
                        url=cit.get("url"),
                        snippet=cit.get("snippet"),
                    ))
                else:
                    citations.append(KACitation(
                        source=getattr(cit, "source", ""),
                        title=getattr(cit, "title", ""),
                        url=getattr(cit, "url", None),
                        snippet=getattr(cit, "snippet", None),
                    ))

        except Exception as e:
            response_type = type(response).__name__
            predictions_info = "unavailable"
            try:
                preds = getattr(response, "predictions", None)
                if preds is not None:
                    if preds:
                        predictions_info = (
                            f"len={len(preds)}, "
                            f"first_type={type(preds[0]).__name__}"
                        )
                    else:
                        predictions_info = "empty_list"
                else:
                    predictions_info = "None"
            except Exception:
                predictions_info = "unparseable"

            logger.warning(
                "KA_RESPONSE_PARSE_FAILED",
                error=str(e)[:200],
                error_type=type(e).__name__,
                response_type=response_type,
                predictions_info=predictions_info,
                endpoint=self._endpoint_name,
            )

        return answer, citations, confidence

    def _parse_responses_api(
        self,
        response: dict[str, Any],
    ) -> tuple[str, list[KACitation], str]:
        """Parse the Responses API format returned by KA endpoints.

        The response has ``output`` → list of message objects → ``content``
        → list of ``output_text`` items, each with ``text`` and optional
        ``annotations`` (``url_citation``).

        Returns:
            Tuple of (answer_text, list_of_citations, confidence_level)
        """
        text_parts: list[str] = []
        citations: list[KACitation] = []

        for output_item in response.get("output", []):
            if output_item.get("type") != "message":
                continue
            for content_item in output_item.get("content", []):
                if content_item.get("type") != "output_text":
                    continue
                text = content_item.get("text", "")
                if text:
                    text_parts.append(text)
                for ann in content_item.get("annotations", []):
                    if ann.get("type") == "url_citation":
                        citations.append(KACitation(
                            source=ann.get("url", ""),
                            title=ann.get("title", ""),
                            url=ann.get("url"),
                            snippet=text[:200] if text else None,
                        ))

        answer = "".join(text_parts)

        # Infer confidence from citation count
        if len(citations) >= 3:
            confidence = "high"
        elif len(citations) >= 1:
            confidence = "medium"
        else:
            confidence = "unknown"

        return answer, citations, confidence

    def validate_arguments(self, arguments: dict[str, Any]) -> list[str]:
        """Validate question arguments.

        Args:
            arguments: Raw arguments from LLM

        Returns:
            List of error messages (empty if valid)
        """
        errors: list[str] = []

        # Required: question
        question = arguments.get("question")
        if not question:
            errors.append("'question' is required")
        elif not isinstance(question, str):
            errors.append("'question' must be a string")
        elif len(question) > 2000:
            errors.append("'question' must be 2000 characters or less")

        return errors


def create_knowledge_assistant_tools_from_config(
    config: Any,
) -> list[KnowledgeAssistantTool]:
    """Create KnowledgeAssistantTool instances from app configuration.

    Args:
        config: KnowledgeAssistantsConfig from app_config

    Returns:
        List of KnowledgeAssistantTool instances, one per enabled endpoint
    """
    tools: list[KnowledgeAssistantTool] = []

    if not config or not getattr(config, "enabled", False):
        return tools

    endpoints = getattr(config, "endpoints", {})

    for name, endpoint_config in endpoints.items():
        if not getattr(endpoint_config, "enabled", True):
            logger.debug("Skipping disabled Knowledge Assistant endpoint", endpoint=name)
            continue

        try:
            tool = KnowledgeAssistantTool(
                endpoint_name=endpoint_config.endpoint_name,
                tool_name=getattr(endpoint_config, "tool_name", None),
                description=getattr(endpoint_config, "description", None),
            )
            tools.append(tool)
            logger.info(
                "Created Knowledge Assistant tool",
                tool_name=tool.definition.name,
            )
        except Exception as e:
            logger.warning(
                "Failed to create Knowledge Assistant tool",
                endpoint=name,
                error=str(e),
            )

    return tools


def create_knowledge_assistant_from_user_source(
    data_source: "UserDataSource",
    obo_client: "OBODatabricksClient",
) -> KnowledgeAssistantTool:
    """Create a KnowledgeAssistantTool from a user-configured data source.

    This creates an OBO-authenticated tool that operates with the user's
    permissions when querying the Knowledge Assistant endpoint.

    Args:
        data_source: UserDataSource from database with type KNOWLEDGE_ASSISTANT.
        obo_client: OBO client for user authentication.

    Returns:
        KnowledgeAssistantTool configured for OBO authentication.

    Raises:
        ValueError: If data_source is not a Knowledge Assistant type.
    """
    from deep_research.models.data_source import DataSourceType

    if data_source.type != DataSourceType.KNOWLEDGE_ASSISTANT.value:
        raise ValueError(
            f"Expected KNOWLEDGE_ASSISTANT data source, got {data_source.type}"
        )

    config = data_source.config or {}

    # Extract endpoint name from config or endpoint_identifier
    endpoint_name = config.get("endpoint_name") or data_source.endpoint_identifier
    if not endpoint_name:
        raise ValueError("Knowledge Assistant source requires endpoint_name")

    # Extract optional settings
    pass_context = config.get("pass_context", False)
    max_context_chars = config.get("max_context_chars", 4000)

    tool = KnowledgeAssistantTool(
        endpoint_name=endpoint_name,
        description=data_source.description,
        obo_client=obo_client,
        data_source=data_source,
        pass_context=pass_context,
        max_context_chars=max_context_chars,
    )

    logger.info(
        "Created OBO Knowledge Assistant tool from user source",
        tool_name=tool.definition.name,
        source_name=data_source.name,
        endpoint=endpoint_name,
        pass_context=pass_context,
    )

    return tool
