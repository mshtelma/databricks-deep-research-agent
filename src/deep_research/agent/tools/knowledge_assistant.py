"""Knowledge Assistant Tool for Databricks serving endpoints.

Provides question-answering via Databricks-hosted Knowledge Assistants.
Each configured endpoint creates a separate tool instance with a unique name.

Example configuration (config/app.yaml):
    knowledge_assistants:
      enabled: true
      endpoints:
        product_assistant:
          endpoint_name: product-knowledge-assistant
          description: Ask questions about our products
"""

from dataclasses import dataclass
from typing import Any

from deep_research.agent.tools.base import (
    ResearchContext,
    ResearchTool,
    ToolDefinition,
    ToolResult,
)
from deep_research.core.logging_utils import get_logger

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

    Requires Databricks SDK with proper authentication (WorkspaceClient).
    """

    def __init__(
        self,
        *,
        endpoint_name: str,
        tool_name: str | None = None,
        description: str | None = None,
    ) -> None:
        """Initialize the Knowledge Assistant tool.

        Args:
            endpoint_name: Databricks serving endpoint name for the KA.
            tool_name: Custom tool name. Defaults to 'ask_{endpoint_name}'.
            description: Custom description for LLM. Defaults to generic.
        """
        self._endpoint_name = endpoint_name

        # Generate tool name
        self._tool_name = tool_name or f"ask_{endpoint_name.replace('-', '_')}"

        # Generate description
        self._description = description or (
            f"Ask the '{endpoint_name}' Knowledge Assistant a question. "
            "Returns an answer with source citations when available."
        )

        # Lazy-loaded client
        self._client: Any = None

        self._definition = ToolDefinition(
            name=self._tool_name,
            description=self._description,
            parameters={
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
            },
        )

    def _get_client(self) -> Any:
        """Get or create WorkspaceClient."""
        if self._client is None:
            try:
                from databricks.sdk import WorkspaceClient

                self._client = WorkspaceClient()
                logger.info(
                    "WorkspaceClient initialized for Knowledge Assistant",
                    endpoint=self._endpoint_name,
                )
            except ImportError:
                raise ImportError(
                    "databricks-sdk package not installed. "
                    "Install with: pip install databricks-sdk"
                )
        return self._client

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

        Args:
            arguments: Tool arguments containing 'question'
            context: Research context with identity and registries

        Returns:
            ToolResult with answer content and source tracking
        """
        question = arguments.get("question", "")

        try:
            client = self._get_client()

            # Query the KA serving endpoint
            response = client.serving_endpoints.query(
                name=self._endpoint_name,
                dataframe_records=[
                    {
                        "messages": [
                            {"role": "user", "content": question}
                        ]
                    }
                ],
            )

            # Parse response
            answer, citations = self._parse_response(response)

            if not answer:
                return ToolResult(
                    content="The Knowledge Assistant could not provide an answer.",
                    success=True,
                    sources=[],
                    data={"question": question, "has_answer": False},
                )

            # Build sources list for citation tracking
            sources: list[dict[str, Any]] = []
            citation_text: list[str] = []

            for idx, citation in enumerate(citations):
                sources.append({
                    "type": "knowledge_assistant",
                    "endpoint_name": self._endpoint_name,
                    "source": citation.source,
                    "title": citation.title,
                    "url": citation.url,
                    "snippet": citation.snippet,
                    "citation_index": idx,
                })
                if citation.title:
                    citation_text.append(f"[{idx + 1}] {citation.title}")

            # Format content with citations
            content = answer
            if citation_text:
                content += "\n\nSources:\n" + "\n".join(citation_text)

            return ToolResult(
                content=content,
                success=True,
                sources=sources,
                data={
                    "question": question,
                    "has_answer": True,
                    "citation_count": len(citations),
                    "endpoint_name": self._endpoint_name,
                },
            )

        except ImportError as e:
            logger.error("Databricks SDK not available", error=str(e))
            return ToolResult(
                content="Knowledge Assistant is not available. SDK not installed.",
                success=False,
                error=str(e),
            )
        except Exception as e:
            logger.error(
                "Knowledge Assistant error",
                error=str(e),
                endpoint=self._endpoint_name,
            )
            return ToolResult(
                content=f"Query failed: {e}",
                success=False,
                error=str(e),
            )

    def _parse_response(self, response: Any) -> tuple[str, list[KACitation]]:
        """Parse Knowledge Assistant response.

        Args:
            response: Raw response from serving_endpoints.query()

        Returns:
            Tuple of (answer_text, list_of_citations)
        """
        answer = ""
        citations: list[KACitation] = []

        try:
            # Navigate response structure
            # response.predictions[0].choices[0].message
            predictions = getattr(response, "predictions", None)
            if not predictions or len(predictions) == 0:
                return answer, citations

            first_pred = predictions[0]

            # Handle dict or object
            if isinstance(first_pred, dict):
                choices = first_pred.get("choices", [])
            else:
                choices = getattr(first_pred, "choices", [])

            if not choices or len(choices) == 0:
                return answer, citations

            first_choice = choices[0]

            # Get message
            if isinstance(first_choice, dict):
                message = first_choice.get("message", {})
            else:
                message = getattr(first_choice, "message", {})

            # Extract content
            if isinstance(message, dict):
                answer = message.get("content", "")
                raw_citations = message.get("citations", [])
            else:
                answer = getattr(message, "content", "")
                raw_citations = getattr(message, "citations", [])

            # Parse citations
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
            logger.warning("Failed to parse KA response", error=str(e))

        return answer, citations

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
