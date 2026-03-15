"""Databricks Genie tool — natural language SQL analytics.

Wraps the Databricks Genie API (start conversation → create message → poll
until complete → format results) into the ``ResearchTool`` protocol.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any

from databricks_deep_research.tools.protocol import (
    SourceInfo,
    SourceKind,
    ToolContext,
    ToolDefinition,
    ToolResult,
)

logger = logging.getLogger(__name__)

_GENIE_MAX_WAIT = 180.0  # seconds


class DatabricksGenieTool:
    """Queries a Databricks Genie space using natural language.

    Implements the :class:`ResearchTool` protocol.
    """

    def __init__(
        self,
        workspace_client: Any,
        name: str,
        space_id: str,
        description: str = "",
    ) -> None:
        self._ws = workspace_client
        self._name = name
        self._space_id = space_id
        self._description = description or f"Natural language SQL via Genie space {space_id}"

        self._definition = ToolDefinition(
            name=name,
            description=self._description,
            parameters={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Natural language question about the data.",
                    },
                },
                "required": ["question"],
            },
            source_type="enterprise",
            source_kind=SourceKind.sql_analytics,
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
        source_url = f"enterprise://genie/{self._space_id}"

        try:
            # Wait for a completed response using the SDK's built-in waiter.
            message = self._ws.genie.start_conversation_and_wait(
                space_id=self._space_id,
                content=question,
                timeout=timedelta(seconds=_GENIE_MAX_WAIT),
            )

            conversation_id = message.conversation_id
            message_id = message.message_id

            # Format results
            content = _format_genie_result(
                self._ws,
                self._space_id,
                conversation_id,
                message_id,
                message,
            )

            if context.url_registry:
                context.url_registry.register(source_url)

            sources = [SourceInfo(
                url=source_url,
                title=f"Genie: {question[:80]}",
                snippet=content[:300],
                content=content,
                source_type="enterprise",
                source_kind=SourceKind.sql_analytics,
            )]

            logger.info(
                "GENIE_RESULTS tool=%s space_id=%s question=%s",
                self._name, self._space_id, question[:100],
            )

            return ToolResult(
                content=content,
                success=True,
                sources=sources,
                data={
                    "space_id": self._space_id,
                    "conversation_id": conversation_id,
                    "source_kind": SourceKind.sql_analytics,
                    "empty_result": content.strip() == "No results returned from Genie.",
                },
            )

        except Exception as exc:
            error_text = str(exc).lower()
            if "timed out" in error_text or "timeout" in error_text:
                return ToolResult(
                    content=f"Genie query timed out after {_GENIE_MAX_WAIT:.1f}s",
                    success=False,
                    error="timeout",
                )
            logger.exception(
                "GENIE_ERROR tool=%s space_id=%s question=%s",
                self._name, self._space_id, question[:100],
            )
            return ToolResult(
                content=f"Genie query failed: {exc}",
                success=False,
                error=str(exc),
            )


def _format_genie_result(
    workspace_client: Any,
    space_id: str,
    conversation_id: str,
    message_id: str,
    message: Any,
) -> str:
    """Format a Genie message result into readable text."""
    parts: list[str] = []
    seen_statement_ids: set[str] = set()

    # Extract attachments (SQL, tables, text)
    attachments = getattr(message, "attachments", None) or []
    for attachment in attachments:
        # Text content
        text_content = getattr(attachment, "text", None)
        if text_content and hasattr(text_content, "content"):
            parts.append(text_content.content)

        # Query result (tabular data)
        query_result = getattr(attachment, "query", None)
        if query_result:
            sql = getattr(query_result, "query", None)
            if sql:
                parts.append(f"SQL: {sql}")

            description = getattr(query_result, "description", None)
            if description:
                parts.append(f"Description: {description}")

            _append_inline_query_table(parts, query_result)

        attachment_id = getattr(attachment, "attachment_id", None)
        if attachment_id:
            try:
                response = workspace_client.genie.get_message_query_result_by_attachment(
                    space_id=space_id,
                    conversation_id=conversation_id,
                    message_id=message_id,
                    attachment_id=attachment_id,
                )
            except Exception:
                logger.debug(
                    "GENIE_QUERY_RESULT_ATTACHMENT_FAILED space_id=%s conversation_id=%s message_id=%s attachment_id=%s",
                    space_id,
                    conversation_id,
                    message_id,
                    attachment_id,
                    exc_info=True,
                )
            else:
                _append_statement_response(parts, response, seen_statement_ids)

    if not seen_statement_ids:
        try:
            response = workspace_client.genie.get_message_query_result(
                space_id=space_id,
                conversation_id=conversation_id,
                message_id=message_id,
            )
        except Exception:
            logger.debug(
                "GENIE_QUERY_RESULT_FALLBACK_FAILED space_id=%s conversation_id=%s message_id=%s",
                space_id,
                conversation_id,
                message_id,
                exc_info=True,
            )
        else:
            _append_statement_response(parts, response, seen_statement_ids)

    if not parts:
        return "No results returned from Genie."

    return "\n".join(parts)


def _append_inline_query_table(parts: list[str], query_result: Any) -> None:
    """Append any inline query table embedded directly in an attachment."""
    columns = getattr(query_result, "columns", None) or []
    rows = getattr(query_result, "data", None) or []
    if not columns or not rows:
        return
    col_names = [getattr(c, "name", str(c)) for c in columns]
    _append_table(parts, col_names, rows)


def _append_statement_response(
    parts: list[str],
    response: Any,
    seen_statement_ids: set[str],
) -> None:
    """Append a statement response table, deduplicating repeated attachment results."""
    statement_response = getattr(response, "statement_response", None)
    if statement_response is None:
        return

    statement_id = getattr(statement_response, "statement_id", None)
    if statement_id and statement_id in seen_statement_ids:
        return
    if statement_id:
        seen_statement_ids.add(statement_id)

    manifest = getattr(statement_response, "manifest", None)
    result = getattr(statement_response, "result", None)
    schema = getattr(manifest, "schema", None) if manifest is not None else None
    columns = getattr(schema, "columns", None) or []
    rows = getattr(result, "data_array", None) or []
    if not columns or not rows:
        return

    col_names = [getattr(c, "name", str(c)) for c in columns]
    _append_table(parts, col_names, rows)


def _append_table(parts: list[str], columns: list[str], rows: list[Any]) -> None:
    """Append a simple pipe-delimited table preview."""
    header = " | ".join(columns)
    parts.append(header)
    parts.append("-" * len(header))
    for row in rows[:50]:
        row_vals = [str(v) for v in (row if isinstance(row, list) else [row])]
        parts.append(" | ".join(row_vals))
    if len(rows) > 50:
        parts.append(f"... ({len(rows) - 50} more rows)")
