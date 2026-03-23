"""Genie Tool for natural language SQL queries.

Implements GenieTool for querying enterprise databases via Genie AI/BI
with OBO authentication, conversation context, tabular result formatting,
and MLflow tracing for observability.

Part of 007-enterprise-data-sources feature (T022, T023).
"""

import asyncio
import time
from typing import Any

from deep_research.agent.tools.base import (
    ResearchContext,
    ToolDefinition,
    ToolResult,
)
from deep_research.core.logging_utils import get_logger
from deep_research.services.metrics import record_source_query
from deep_research.services.obo_client import OBODatabricksClient

logger = get_logger(__name__)

# Default row limit for tabular results
DEFAULT_MAX_ROWS = 100


class GenieTool:
    """Genie tool for natural language SQL queries with OBO authentication.

    Key features:
    - Uses user's OBO token for queries (respects their permissions)
    - Supports conversation context for follow-up queries
    - Returns tabular results with SQL transparency
    - Generates narrative summaries when helpful

    The tool maintains conversation state to support follow-up questions
    that reference previous queries/results.
    """

    def __init__(
        self,
        obo_client: OBODatabricksClient,
        space_id: str,
        name: str,
        description: str | None = None,
        example_questions: list[str] | None = None,
        max_rows: int = DEFAULT_MAX_ROWS,
    ) -> None:
        """Initialize the Genie tool.

        Args:
            obo_client: OBO client for user authentication.
            space_id: Genie space ID.
            name: Display name for the tool.
            description: Tool description for LLM.
            example_questions: Example questions to show in description.
            max_rows: Maximum rows to return (truncates larger results).
        """
        self._obo_client = obo_client
        self._space_id = space_id
        self._name = name
        self._max_rows = max_rows
        self._example_questions = example_questions or []

        # Generate tool name from space ID
        safe_id = space_id.replace("-", "_").replace(".", "_")
        self._tool_name = f"query_genie_{safe_id}"

        # Build description
        if description:
            self._description = description
        else:
            self._description = (
                f"Query '{name}' using natural language. "
                "Translates your question to SQL and returns structured data. "
                "Use for analytics, aggregations, and data exploration."
            )

        if self._example_questions:
            examples = ", ".join(f'"{q}"' for q in self._example_questions[:3])
            self._description += f" Example questions: {examples}"

        # Conversation state for follow-ups
        self._conversation_id: str | None = None

    @property
    def definition(self) -> ToolDefinition:
        """Return tool definition for LLM function calling.

        Includes is_follow_up parameter for conversation context.
        """
        return ToolDefinition(
            name=self._tool_name,
            description=self._description,
            parameters={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": (
                            "Your question in natural language. Be specific about "
                            "the data you want: metrics, time periods, filters, etc."
                        ),
                    },
                    "is_follow_up": {
                        "type": "boolean",
                        "description": (
                            "Set to true if this question refers to previous results "
                            "(e.g., 'break that down by region'). Default: false."
                        ),
                        "default": False,
                    },
                },
                "required": ["question"],
            },
            source_type="genie",
        )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ResearchContext,
    ) -> ToolResult:
        """Execute Genie query with OBO authentication.

        Args:
            arguments: Tool arguments containing 'question', optional 'is_follow_up'.
            context: Research context with user_token for OBO.

        Returns:
            ToolResult with formatted query results, SQL, and source tracking.
        """
        question = arguments.get("question", "")
        is_follow_up = arguments.get("is_follow_up", False)

        logger.info(
            "GENIE_TOOL_EXECUTE",
            tool_name=self._tool_name,
            space_id=self._space_id,
            question=question[:100],
            is_follow_up=is_follow_up,
            has_conversation_context=self._conversation_id is not None,
        )

        if not question:
            return ToolResult(
                content="Error: 'question' is required.",
                success=False,
                error="Missing required argument: question",
            )

        start_time = time.perf_counter()

        try:
            # Get OBO-authenticated client
            client = await self._obo_client.get_client(context.user_token)

            # Execute query via executor (SDK is synchronous)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._execute_query(client, question, is_follow_up),
            )

            duration_ms = (time.perf_counter() - start_time) * 1000

            if result.get("error"):
                # Genie returned a structured error in the response
                logger.info(
                    "GENIE_RESULT_ERROR",
                    error=result["error"][:200],
                    space_id=self._space_id,
                )
            elif not result.get("columns") and not result.get("narrative") and not result.get("sql"):
                # Truly empty response — no data, no narrative, no SQL
                logger.warning(
                    "GENIE_EMPTY_RESPONSE",
                    space_id=self._space_id,
                    question=question[:100],
                    result_keys=[k for k, v in result.items() if v],
                )

            # Format the response
            content = self._format_result(result)
            row_count = result.get("row_count", 0)

            # Build source for citation tracking (unique URL per query via message_id)
            msg_id = result.get("message_id", "")

            # Build navigable workspace URL (fragment preserves dedup uniqueness)
            from deep_research.core.auth import get_workspace_host
            workspace_host = get_workspace_host()
            if workspace_host:
                source_url = f"{workspace_host}/sql/genie/spaces/{self._space_id}#{msg_id}"
            else:
                source_url = f"genie://{self._space_id}/{msg_id}"

            sources = [{
                "type": "genie",
                "source_name": self._name,
                "space_id": self._space_id,
                "url": source_url,
                "title": self._name,
                "content": content[:3000],
                "generated_sql": result.get("sql"),
                "row_count": row_count,
            }]

            # Log final metrics
            logger.info(
                "GENIE_SPAN_ATTRS",
                duration_ms=duration_ms,
                result_count=row_count,
                success=True,
                has_response=True,
                truncated=result.get("truncated", False),
                has_sql=result.get("sql") is not None,
            )

            # Record metrics for monitoring (T108)
            record_source_query(
                source_type="genie",
                source_name=self._name,
                latency_ms=duration_ms,
                success=True,
            )

            return ToolResult(
                content=content,
                success=True,
                sources=sources,
                data={
                    "question": question,
                    "space_id": self._space_id,
                    "sql": result.get("sql"),
                    "row_count": row_count,
                    "truncated": result.get("truncated", False),
                },
            )

        except Exception as e:
            error_msg = str(e)
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Provide helpful error messages
            if "PERMISSION_DENIED" in error_msg or "403" in error_msg:
                error_msg = (
                    f"Permission denied: You don't have access to Genie space '{self._space_id}'. "
                    "Please verify your permissions."
                )
            elif "NOT_FOUND" in error_msg or "404" in error_msg:
                error_msg = f"Genie space not found: '{self._space_id}' does not exist."
            elif "ambiguous" in error_msg.lower():
                error_msg = (
                    "Your question is ambiguous. Please be more specific about "
                    "which data, metrics, or time period you're asking about."
                )

            # Log error info
            logger.info(
                "GENIE_SPAN_ATTRS",
                duration_ms=duration_ms,
                result_count=0,
                success=False,
                error_type=type(e).__name__,
            )

            # Record error metrics for monitoring (T108)
            record_source_query(
                source_type="genie",
                source_name=self._name,
                latency_ms=duration_ms,
                success=False,
                error=error_msg[:200],
            )

            logger.error(
                "GENIE_QUERY_ERROR",
                error=error_msg,
                error_type=type(e).__name__,
                space_id=self._space_id,
                duration_ms=duration_ms,
                exc_info=True,
            )

            return ToolResult(
                content=f"Query failed: {error_msg[:500]}",
                success=False,
                error=error_msg,
            )

    def _execute_query(
        self,
        client: Any,
        question: str,
        is_follow_up: bool,
    ) -> dict[str, Any]:
        """Execute the Genie query (synchronous) with two-step data retrieval.

        Step 1: Send question and get GenieMessage (contains SQL, narrative, metadata).
        Step 2: Fetch actual tabular data via get_message_attachment_query_result().

        Handles conversation context for follow-up questions.
        Uses SDK's *_and_wait() methods that handle polling internally.
        """
        # Step 1: Verify Genie API availability (narrow AttributeError catch)
        try:
            genie = client.genie
        except AttributeError:
            logger.warning(
                "GENIE_API_NOT_AVAILABLE: Genie API not available in current SDK version",
            )
            return {
                "error": "Genie API not available. Please upgrade Databricks SDK.",
                "sql": None,
                "columns": [],
                "rows": [],
                "row_count": 0,
            }

        # Step 2: Start or continue conversation
        if is_follow_up and self._conversation_id:
            message = genie.create_message_and_wait(
                space_id=self._space_id,
                conversation_id=self._conversation_id,
                content=question,
            )
        else:
            message = genie.start_conversation_and_wait(
                space_id=self._space_id,
                content=question,
            )
            self._conversation_id = message.conversation_id

        # Step 3: Log the message structure for debugging
        logger.info(
            "GENIE_MESSAGE_RECEIVED",
            space_id=self._space_id,
            message_id=getattr(message, "message_id", None),
            status=str(getattr(message, "status", "unknown")),
            has_query_result=message.query_result is not None,
            statement_id=getattr(message.query_result, "statement_id", None) if message.query_result else None,
            attachment_count=len(message.attachments) if message.attachments else 0,
            has_content=bool(getattr(message, "content", None)),
            has_error=bool(getattr(message, "error", None)),
        )

        # Step 4: Extract metadata from message (SQL from attachments, narrative, status)
        result = self._extract_result(message)

        # Propagate message_id for unique source URLs
        result["message_id"] = getattr(message, "message_id", None) or getattr(message, "id", None) or ""

        # Step 5: If message has query attachments, fetch actual tabular data
        attachment_id = None
        if message.attachments:
            for att in message.attachments:
                if att.query:  # GenieQueryAttachment present = SQL query was generated
                    attachment_id = att.attachment_id
                    break

        if attachment_id and message.conversation_id:
            msg_id = message.message_id or message.id
            try:
                query_result_response = genie.get_message_attachment_query_result(
                    space_id=self._space_id,
                    conversation_id=message.conversation_id,
                    message_id=msg_id,
                    attachment_id=attachment_id,
                )
                self._parse_statement_response(query_result_response, result)
            except Exception as e:
                logger.warning(
                    "GENIE_QUERY_RESULT_FETCH_ERROR",
                    error=str(e)[:300],
                    error_type=type(e).__name__,
                    space_id=self._space_id,
                    message_id=msg_id,
                    attachment_id=attachment_id,
                )
                # Still return what we have (SQL, narrative) — just without tabular data
        elif message.query_result and message.query_result.statement_id:
            # Fallback: no attachment_id but statement_id exists — try deprecated API
            msg_id = message.message_id or message.id
            try:
                query_result_response = genie.get_message_query_result(
                    space_id=self._space_id,
                    conversation_id=message.conversation_id,
                    message_id=msg_id,
                )
                self._parse_statement_response(query_result_response, result)
            except Exception as e:
                logger.warning(
                    "GENIE_QUERY_RESULT_FALLBACK_ERROR",
                    error=str(e)[:300],
                    error_type=type(e).__name__,
                    space_id=self._space_id,
                    statement_id=message.query_result.statement_id,
                )
        else:
            logger.info(
                "GENIE_NO_QUERY_ATTACHMENT",
                space_id=self._space_id,
                reason="Message has no query attachments — likely a text-only response",
            )

        return result

    def _parse_statement_response(
        self,
        response: Any,
        result: dict[str, Any],
    ) -> None:
        """Parse StatementResponse and populate result dict with columns/rows.

        Called after the second API call (get_message_attachment_query_result or
        get_message_query_result) to extract tabular data from the response.
        """
        stmt = getattr(response, "statement_response", None)
        if not stmt:
            logger.warning("GENIE_EMPTY_STATEMENT_RESPONSE", space_id=self._space_id)
            return

        # Extract columns from manifest.schema.columns
        manifest = getattr(stmt, "manifest", None)
        if manifest:
            schema = getattr(manifest, "schema", None)
            if schema and schema.columns:
                result["columns"] = [
                    col.name for col in schema.columns if col.name
                ]
                logger.debug(
                    "GENIE_COLUMNS_EXTRACTED",
                    column_count=len(result["columns"]),
                    columns=result["columns"][:10],  # Log first 10
                )

        # Extract SQL statement_id (if not already set from attachments)
        if not result.get("sql"):
            statement_id = getattr(stmt, "statement_id", None)
            if statement_id:
                result["sql_statement_id"] = statement_id

        # Extract rows from result.data_array
        stmt_result = getattr(stmt, "result", None)
        if stmt_result and stmt_result.data_array:
            all_rows = stmt_result.data_array
            result["row_count"] = len(all_rows)
            if len(all_rows) > self._max_rows:
                result["rows"] = all_rows[:self._max_rows]
                result["truncated"] = True
            else:
                result["rows"] = all_rows

            logger.info(
                "GENIE_DATA_EXTRACTED",
                row_count=result["row_count"],
                column_count=len(result.get("columns", [])),
                truncated=result.get("truncated", False),
                space_id=self._space_id,
            )
        else:
            logger.info(
                "GENIE_NO_DATA_ROWS",
                space_id=self._space_id,
                has_manifest=manifest is not None,
                has_result=stmt_result is not None,
                has_data_array=bool(getattr(stmt_result, "data_array", None)) if stmt_result else False,
            )

        # Check statement status for additional context
        status = getattr(stmt, "status", None)
        if status:
            status_state = getattr(status, "state", None)
            if status_state:
                state_value = getattr(status_state, "value", str(status_state))
                if state_value in ("FAILED", "CLOSED", "CANCELED"):
                    error_msg = getattr(status, "error", None)
                    result["error"] = f"Statement {state_value}: {error_msg or 'unknown'}"
                    logger.warning(
                        "GENIE_STATEMENT_FAILED",
                        state=state_value,
                        error=str(error_msg)[:200] if error_msg else None,
                    )

    def _extract_result(self, message: Any) -> dict[str, Any]:
        """Extract metadata from GenieMessage (step 1 of two-step retrieval).

        Only extracts metadata available directly on GenieMessage:
        - Error/status checks
        - SQL from attachments (message.attachments[*].query.query)
        - row_count hint from query_result metadata
        - Narrative from message content

        Actual tabular data (columns, rows) is fetched in step 2 via
        get_message_attachment_query_result() and parsed by _parse_statement_response().
        """
        result: dict[str, Any] = {
            "sql": None,
            "columns": [],
            "rows": [],
            "row_count": 0,
            "truncated": False,
            "narrative": None,
        }

        # Check for errors
        error = getattr(message, "error", None)
        if error:
            error_msg = getattr(error, "message", str(error))
            result["error"] = error_msg
            logger.warning("GENIE_MESSAGE_ERROR", error=error_msg[:200])
            return result

        # Check message status
        status = getattr(message, "status", None)
        if status:
            status_value = getattr(status, "value", str(status))
            if status_value in ("FAILED", "CANCELLED"):
                result["error"] = f"Query {status_value.lower()}"
                logger.warning("GENIE_MESSAGE_STATUS_FAILED", status=status_value)
                return result

        # Extract SQL from attachments (correct SDK path)
        attachments = getattr(message, "attachments", None) or []
        for att in attachments:
            query_att = getattr(att, "query", None)
            if query_att:
                sql = getattr(query_att, "query", None)
                if sql:
                    result["sql"] = sql
                break  # Use first query attachment

        # Extract row_count hint from query_result metadata
        qr = getattr(message, "query_result", None)
        if qr:
            row_count = getattr(qr, "row_count", None)
            if row_count is not None:
                result["row_count"] = row_count

        # Extract narrative/content
        content = getattr(message, "content", None)
        if content:
            result["narrative"] = content

        return result

    def _format_result(self, result: dict[str, Any]) -> str:
        """Format Genie result for LLM consumption."""
        parts: list[str] = []

        # Show generated SQL (helps with transparency)
        if result.get("sql"):
            parts.append("**Generated SQL:**")
            parts.append(f"```sql\n{result['sql']}\n```")
            parts.append("")

        # Show narrative if available
        if result.get("narrative"):
            parts.append("**Summary:**")
            parts.append(result["narrative"])
            parts.append("")

        # Show tabular data
        columns = result.get("columns", [])
        rows = result.get("rows", [])

        if columns and rows:
            parts.append(f"**Results ({result.get('row_count', len(rows))} rows):**")

            # Format as markdown table
            header = "| " + " | ".join(str(c) for c in columns) + " |"
            separator = "| " + " | ".join("---" for _ in columns) + " |"
            parts.append(header)
            parts.append(separator)

            for row in rows[:50]:  # Limit displayed rows
                row_str = "| " + " | ".join(str(v) if v is not None else "" for v in row) + " |"
                parts.append(row_str)

            if result.get("truncated"):
                parts.append(f"\n*Results truncated. Showing {self._max_rows} of {result['row_count']} rows.*")

        elif result.get("error"):
            parts.append(f"**Error:** {result['error']}")

        elif not result.get("narrative") and not result.get("sql"):
            parts.append("No data returned for this query.")
        # If we have narrative/sql but no tabular data, that's fine — already appended above

        return "\n".join(parts)

    def validate_arguments(self, arguments: dict[str, Any]) -> list[str]:
        """Validate query arguments.

        Args:
            arguments: Raw arguments from LLM.

        Returns:
            List of error messages (empty if valid).
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

        # Optional: is_follow_up
        is_follow_up = arguments.get("is_follow_up")
        if is_follow_up is not None and not isinstance(is_follow_up, bool):
            errors.append("'is_follow_up' must be a boolean")

        return errors

    def clear_conversation(self) -> None:
        """Clear conversation state.

        Call this when starting a new research session to ensure
        fresh conversation context.
        """
        self._conversation_id = None
