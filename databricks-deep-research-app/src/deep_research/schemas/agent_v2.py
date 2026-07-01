"""Pydantic schemas for Agent Designer V1 CRUD.

These schemas wrap the framework's `WorkflowDefinition` AST and add
app-level metadata (id, owner, visibility, etag, timestamps).
"""
from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from databricks_deep_research.workflow.loader import load_workflow_from_dict
from pydantic import BaseModel, ConfigDict, Field, field_validator

from deep_research.agent_designer.semantic_validation import (
    semantic_validation_errors,
)
from deep_research.agent_designer.workflow_validation import WorkflowValidationResult
from deep_research.models.visibility import AgentVisibility


def _enforce_semantic_validation(definition: dict[str, Any]) -> None:
    """W10: run semantic checks at write time so direct API callers cannot
    persist an AST that the runtime would reject.

    The structural loader above catches malformed JSON; this catches
    undeclared-tool references and required-config gaps. Pre-W10 these
    only ran when the frontend explicitly called
    ``POST /agent-designer/validate`` before save — any caller that
    bypassed that route (chat-assistant patches, CLI scripts, future
    integrations) could land a broken AST.
    """
    errors = semantic_validation_errors(definition)
    if errors:
        # Pydantic surfaces this as a 422 with the joined message. We
        # preserve the per-error path so the UI can highlight offenders.
        detail = "; ".join(
            f"{e.path or '<root>'}: {e.message}" for e in errors
        )
        raise ValueError(f"Workflow semantic validation failed: {detail}")


# Visibility settable by users (no SYSTEM)
UserSettableVisibility = Literal["private", "workspace"]


class CreateAgentV2Request(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=255)
    description: str | None = None
    avatar_url: str | None = Field(default=None, max_length=512)
    visibility: UserSettableVisibility = "private"
    definition: dict[str, Any] = Field(description="Workflow AST")

    @field_validator("definition")
    @classmethod
    def _validate_definition(cls, v: dict[str, Any]) -> dict[str, Any]:
        # Round-trip through the framework loader to enforce structural validity.
        # Raises a Pydantic ValidationError-compatible error if the AST is invalid.
        try:
            load_workflow_from_dict(v)
        except Exception as exc:
            raise ValueError(f"Invalid workflow definition: {exc}") from exc
        # W10: ALSO enforce semantic checks (undeclared-tool refs, required
        # tool config). Previously these only ran via the optional
        # /agent-designer/validate endpoint, leaving direct API callers free
        # to persist runtime-broken ASTs.
        _enforce_semantic_validation(v)
        return v


class UpdateAgentV2Request(BaseModel):
    """Partial update payload. V1 limitation: a field set to None is treated as
    "not provided" by the service (existing value preserved); clearing a field
    back to null is not expressible until V1.5 introduces a sentinel."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = None
    avatar_url: str | None = Field(default=None, max_length=512)
    visibility: UserSettableVisibility | None = None
    definition: dict[str, Any] | None = None

    @field_validator("definition")
    @classmethod
    def _validate_definition(cls, v: dict[str, Any] | None) -> dict[str, Any] | None:
        if v is None:
            return None
        try:
            load_workflow_from_dict(v)
        except Exception as exc:
            raise ValueError(f"Invalid workflow definition: {exc}") from exc
        _enforce_semantic_validation(v)
        return v


class AgentV2Response(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    owner_id: str
    name: str
    description: str | None
    avatar_url: str | None
    visibility: AgentVisibility
    definition: dict[str, Any]
    schema_version: int
    etag: str
    created_at: datetime
    updated_at: datetime
    validation: WorkflowValidationResult | None = Field(
        default=None,
        description=(
            "Save-time workflow validation result (advisory). Carries the "
            "verdict (pass/needs_revision/fail/skipped) + summary + directives "
            "so the UI can surface it. None when the gate was skipped OR when a "
            "background validation is still running (see validation_pending)."
        ),
    )
    validation_pending: bool = Field(
        default=False,
        description=(
            "True when an advisory validation for the CURRENT definition is "
            "still running in the background (cache miss on save). The verdict "
            "is delivered asynchronously: poll GET /agents-v2/{id} until this is "
            "False, then read `validation`. Distinguishes 'validation pending' "
            "from 'validation not applicable' (both have validation=None)."
        ),
    )


class AgentV2Summary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    description: str | None
    visibility: AgentVisibility
    owner_id: str
    updated_at: datetime
    node_count: int = 0
    in_app_active: bool = Field(
        default=False,
        description=(
            "True if this agent has at least one active in_app deployment "
            "(i.e., the chat composer should list it). Computed via JOIN against "
            "agent_deployments at list time."
        ),
    )
    default_verify_sources: bool = Field(
        default=True,
        description=(
            "The agent's authored default for expensive NLI 'verify sources': "
            "True if its synthesizer uses grounding_mode='reclaim' (verify), False "
            "for classical_lite/none (cite-only). The chat composer seeds the "
            "'Verify sources' toggle from this so it reflects the agent's intent; "
            "the toggle then fully overrides it per run."
        ),
    )


class AgentV2ListResponse(BaseModel):
    items: list[AgentV2Summary]
    total: int
