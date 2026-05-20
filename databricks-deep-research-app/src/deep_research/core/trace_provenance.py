"""Single source of truth for the ``dr.*`` MLflow provenance tag schema.

Centralized so every surface (main DRE app, designer chat, deployed shell
apps) calls the same helper and we cannot drift on field names. The schema
is documented in the tracing-unification plan:

| Tag                | Surface(s)                          |
| ------------------ | ----------------------------------- |
| dr.surface         | all (designer-chat/main-chat/shell-app) |
| dr.workspace_id    | all                                  |
| dr.user_id         | all                                  |
| dr.session_id      | designer-chat, main-chat            |
| dr.agent_v2_id     | designer-chat (post-propose), main-chat, shell-app |
| dr.agent_name      | same as above                        |
| dr.app_name        | shell-app                            |
| dr.revision_id     | shell-app (correlation to a specific deploy) |
| dr.workflow_id     | main-chat, shell-app                |
| dr.workflow_name   | main-chat, shell-app                |
| dr.topology        | main-chat, shell-app                |
| dr.grounding_mode  | main-chat, shell-app                |
| dr.query_preview   | all (bounded to 200 chars)          |

Callers pass keys WITHOUT the ``dr.`` prefix — the helper adds it. Missing /
empty values are dropped so a partially-known surface (e.g. designer-chat
before ``agent_v2_id`` is minted) does not pollute the tag space with
empty strings.
"""
from __future__ import annotations

import logging
from typing import Any

try:
    import mlflow
except ImportError:  # pragma: no cover - mlflow always installed in app envs
    mlflow = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Length cap per tag value. MLflow tag values are stored unbounded but the
# Databricks UI truncates long values; bounding here keeps the trace-list
# views readable and prevents accidentally tagging a 10KB query.
_MAX_TAG_VALUE_CHARS = 512


def set_trace_provenance(**tags: Any) -> None:
    """Tag the current MLflow trace with ``dr.*`` provenance fields.

    Safe to call when mlflow is not installed, when no trace is currently
    active, or when called multiple times within a single trace (later calls
    overwrite earlier values for the same key — used to fill ``agent_v2_id``
    after ``propose_workflow`` mints it mid-conversation).

    Args:
        **tags: Provenance fields (e.g. ``surface="designer-chat"``,
            ``agent_v2_id=...``). Keys are added with the ``dr.`` prefix.
            ``None`` and empty-string values are silently dropped.
    """
    if mlflow is None:
        return
    cleaned: dict[str, str] = {}
    for key, value in tags.items():
        if value is None or value == "":
            continue
        cleaned[f"dr.{key}"] = str(value)[:_MAX_TAG_VALUE_CHARS]
    if not cleaned:
        return
    try:
        for k, v in cleaned.items():
            mlflow.set_trace_tag(k, v)
    except Exception as exc:  # pragma: no cover - defensive
        # Never surface tagging failures to the caller. Trace-list views
        # losing provenance is a UX gripe, not a runtime error.
        logger.debug("set_trace_provenance failed: %s", exc)
