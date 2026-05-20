"""Observability helpers for the Agent Designer feature.

All five signals delegate to the same ``MetricsSink`` used by the storage
layer (``storage.observability``).  The default sink emits one structured
log line per emission so that the host log pipeline (Splunk / Grafana Loki)
can derive dashboards without a separate Prometheus exporter.

Signals implemented here (server-side only):

* ``agent_designer.registry_fetch_ms``   — histogram, GET /registry latency
* ``agent_designer.validation_error``    — counter, POST /validate failures
* ``agent_designer.save_etag_conflict``  — counter, 409 from PATCH /agents-v2
* ``agent_designer.designer_save_latency`` — histogram, POST/PATCH success latency
* ``agent_designer.chat_mutation``       — structured log, per tool-call in chat

Deferred (V1.5, client-side):
* ``agent_designer.dnd_drop_failed``     — must be emitted from the React
  drag-and-drop handler, not the server.  No server-side data is available at
  the point a D&D operation fails.

Pattern: re-uses ``deep_research.storage.observability.get_sink()`` so that
tests can swap sinks via ``use_sink()`` without any additional test-doubles.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Literal

from deep_research.storage.observability import get_sink

logger = logging.getLogger("agent_designer.metrics")

# Maximum characters allowed in the args_summary field of a chat mutation log.
_ARGS_SUMMARY_MAX_CHARS = 500


def record_registry_fetch(duration_ms: float) -> None:
    """Record latency for a successful GET /registry response.

    Args:
        duration_ms: Wall-clock time in milliseconds for the registry build.
    """
    get_sink().histogram("agent_designer.registry_fetch_ms", duration_ms)


def record_validation_error(error_kind: str) -> None:
    """Increment the validation-error counter for a POST /validate failure.

    Args:
        error_kind: Short classifier for the error (e.g. ``"validation"``,
            ``"schema"``, ``"syntax"``).  Becomes the ``error_kind`` label on
            the metric so dashboards can group by failure category.
    """
    get_sink().counter("agent_designer.validation_error", 1, error_kind=error_kind)


def record_etag_conflict() -> None:
    """Increment the counter for a 409 ETag conflict on PATCH /agents-v2/{id}."""
    get_sink().counter("agent_designer.save_etag_conflict", 1)


def record_save_latency(operation: str, duration_ms: float) -> None:
    """Record latency for a successful POST or PATCH /agents-v2 operation.

    Args:
        operation: Either ``"create"`` (POST) or ``"update"`` (PATCH).
        duration_ms: Wall-clock time in milliseconds for the successful save.
    """
    get_sink().histogram(
        "agent_designer.designer_save_latency",
        duration_ms,
        operation=operation,
    )


def log_chat_mutation(
    tool_name: str,
    args_summary: dict[str, object],
    validation_errors_count: int,
    outcome: str,
) -> None:
    """Emit a structured log line for every tool-call dispatched in a chat turn.

    The ``args_summary`` dict is serialised to JSON and **truncated to
    ``_ARGS_SUMMARY_MAX_CHARS`` characters** before logging so that large AST
    payloads are never written to the log stream.

    Args:
        tool_name: Name of the designer tool called (e.g. ``"add_block"``).
        args_summary: A shallow summary of the tool arguments.  Do **not** pass
            raw, full AST dicts — callers should extract only the fields useful
            for debugging (e.g. ``{"kind": "agent", "parent_path": "root"}``).
        validation_errors_count: Number of validation errors in the resulting
            AST (0 means a clean mutation).
        outcome: Short result descriptor — one of ``"success"``, ``"error"``,
            or ``"validation_failed"``.
    """
    raw = json.dumps(args_summary, default=str)
    truncated = raw[:_ARGS_SUMMARY_MAX_CHARS]

    logger.info(
        "agent_designer.chat_mutation tool=%s outcome=%s validation_errors=%d args=%s",
        tool_name,
        outcome,
        validation_errors_count,
        truncated,
        extra={
            "metric_name": "agent_designer.chat_mutation",
            "tool_name": tool_name,
            "outcome": outcome,
            "validation_errors_count": validation_errors_count,
            "args_summary": truncated,
        },
    )


def record_yaml_import_outcome(
    outcome: Literal["success", "schema_error", "too_large", "unsafe", "registry_version_mismatch"],
) -> None:
    """Increment the YAML import outcome counter for POST /import-yaml.

    Args:
        outcome: Result of the import attempt — one of ``"success"``,
            ``"schema_error"``, ``"too_large"``, ``"unsafe"``, or
            ``"registry_version_mismatch"``.  Becomes the ``outcome`` label
            on the metric so dashboards can group by failure category.
    """
    get_sink().counter("agent_designer.yaml_import_outcome", 1, outcome=outcome)


def record_yaml_export_ms(duration_ms: float) -> None:
    """Histogram of YAML export latency for GET /{id}/yaml.

    Args:
        duration_ms: Wall-clock time in milliseconds for the serialisation.
    """
    get_sink().histogram("agent_designer.yaml_export_ms", duration_ms)


def record_mermaid_export_ms(duration_ms: float) -> None:
    """Histogram of Mermaid export latency for GET /{id}/mermaid.

    Args:
        duration_ms: Wall-clock time in milliseconds for the serialisation.
    """
    get_sink().histogram("agent_designer.mermaid_export_ms", duration_ms)


def record_revision_write_failed() -> None:
    """Increment the counter for a best-effort revision write failure.

    Called when writing an AgentRevision snapshot fails after a successful
    primary create/update.  The primary operation is NOT affected.
    """
    get_sink().counter("agent_designer.revision_write_failed", 1)


def record_token_refresh_attempt(
    outcome: Literal["success", "failure", "noop"],
) -> None:
    """Increment the OBO token refresh attempt counter.

    Args:
        outcome: Result of the refresh attempt — one of ``"success"``,
            ``"failure"``, or ``"noop"`` (feature disabled / not needed).
            Becomes the ``outcome`` label on the metric.
    """
    get_sink().counter("agent_designer.token_refresh_attempt", 1, outcome=outcome)


def record_token_refresh_failure(
    error_kind: Literal["expired_refresh", "network", "permission"],
) -> None:
    """Increment the OBO token refresh failure counter.

    Args:
        error_kind: Category of the failure — one of
            ``"expired_refresh"`` (token already past expiry),
            ``"network"`` (connectivity / timeout), or
            ``"permission"`` (403 / authorization error).
            Becomes the ``error_kind`` label on the metric.
    """
    get_sink().counter("agent_designer.token_refresh_failure", 1, error_kind=error_kind)


def log_run_principal(principal: dict[str, Any]) -> None:
    """Audit log for run principal (user vs service-principal).

    Emits a single structured log line that must contain BOTH
    ``requested_by_user_id`` AND ``executed_as_sp_id`` so that the security
    audit pipeline can correlate every execution with the human who triggered
    it and the identity under which it ran.

    Args:
        principal: Dict with keys:

            * ``requested_by_user_id`` (``str``) — ID of the human user who
              initiated the run.
            * ``executed_as_sp_id`` (``str | None``) — UUID of the service
              principal that will execute the workflow, or ``None`` when the
              workflow runs as the caller.
            * ``run_kind`` (``"caller" | "sp"``) — short discriminator for
              dashboards and alerting rules.

    Example (SP run)::

        log_run_principal({
            "requested_by_user_id": "user-abc",
            "executed_as_sp_id": "11111111-2222-3333-4444-555555555555",
            "run_kind": "sp",
        })

    Example (caller run)::

        log_run_principal({
            "requested_by_user_id": "user-abc",
            "executed_as_sp_id": None,
            "run_kind": "caller",
        })

    Security: raw tokens must NEVER be included in ``principal``.
    """
    requested_by_user_id: Any = principal.get("requested_by_user_id")
    executed_as_sp_id: Any = principal.get("executed_as_sp_id")
    run_kind: Any = principal.get("run_kind", "caller")

    logger.info(
        "agent_designer.run_principal requested_by_user_id=%s executed_as_sp_id=%s run_kind=%s",
        requested_by_user_id,
        executed_as_sp_id,
        run_kind,
        extra={
            "metric_name": "agent_designer.run_principal",
            "requested_by_user_id": requested_by_user_id,
            "executed_as_sp_id": executed_as_sp_id,
            "run_kind": run_kind,
        },
    )
