"""Databricks Knowledge Assistant tool — Q&A over serving endpoints.

Wraps a Databricks serving endpoint that implements a Q&A interface into
the ``ResearchTool`` protocol.
"""

from __future__ import annotations

import logging
from typing import Any

from databricks_deep_research.tools.protocol import (
    ResearchTool,
    SourceInfo,
    SourceKind,
    ToolContext,
    ToolDefinition,
    ToolResult,
)

logger = logging.getLogger(__name__)


class DatabricksKnowledgeAssistantTool:
    """Queries a Databricks serving endpoint for Q&A.

    Implements the :class:`ResearchTool` protocol.
    """

    def __init__(
        self,
        workspace_client: Any,
        name: str,
        endpoint_name: str,
        description: str = "",
    ) -> None:
        self._ws = workspace_client
        self._name = name
        self._endpoint_name = endpoint_name
        self._description = description or f"Knowledge assistant via {endpoint_name}"

        self._definition = ToolDefinition(
            name=name,
            description=self._description,
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
            source_kind=SourceKind.qa_assistant,
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        question = arguments.get("question")
        if not question or not isinstance(question, str):
            raise ValueError("'question' is required and must be a non-empty string")
        if len(question) > 2000:
            raise ValueError("'question' must be at most 2000 characters")
        return {"question": question.strip()}

    async def execute(
        self, arguments: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        question = arguments["question"]
        source_url = f"enterprise://knowledge_assistant/{self._endpoint_name}"

        try:
            # Query the serving endpoint
            response = self._ws.serving_endpoints.query(
                name=self._endpoint_name,
                inputs=[{"query": question}],
            )

            # Extract answer from response
            content = _extract_answer(response)

            if context.url_registry:
                context.url_registry.register(source_url)

            sources = [SourceInfo(
                url=source_url,
                title=f"Knowledge Assistant: {question[:80]}",
                snippet=content[:300],
                source_type="enterprise",
                source_kind=SourceKind.qa_assistant,
            )]

            logger.info(
                "KNOWLEDGE_ASSISTANT_RESULTS tool=%s endpoint=%s question=%s",
                self._name, self._endpoint_name, question[:100],
            )

            return ToolResult(
                content=content,
                success=True,
                sources=sources,
                data={
                    "endpoint_name": self._endpoint_name,
                    "source_kind": SourceKind.qa_assistant,
                    "empty_result": not content.strip(),
                },
            )

        except Exception as exc:
            logger.exception(
                "KNOWLEDGE_ASSISTANT_ERROR tool=%s endpoint=%s question=%s",
                self._name, self._endpoint_name, question[:100],
            )
            return ToolResult(
                content=f"Knowledge assistant query failed: {exc}",
                success=False,
                error=str(exc),
            )


def _extract_answer(response: Any) -> str:
    """Extract the answer text from a serving endpoint response."""
    # Handle DataframeSplitInput / predictions format
    predictions = getattr(response, "predictions", None)
    if predictions:
        if isinstance(predictions, list):
            return "\n".join(str(p) for p in predictions)
        return str(predictions)

    # Handle dict-like responses
    if hasattr(response, "outputs"):
        outputs = response.outputs
        if isinstance(outputs, list):
            return "\n".join(str(o) for o in outputs)
        return str(outputs)

    # Fallback: try to get text from the response
    if hasattr(response, "choices"):
        choices = response.choices
        if choices:
            first = choices[0]
            if hasattr(first, "text"):
                return first.text
            if hasattr(first, "message") and hasattr(first.message, "content"):
                return first.message.content

    return str(response)
