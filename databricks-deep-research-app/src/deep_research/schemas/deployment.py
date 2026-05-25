"""Pydantic schemas for the Agent Designer Deployment Feature (Phase 1).

The four mode-specific config models form a discriminated union on the
``mode`` literal field. The union is validated at the API layer before any
DB write -- this matches the existing ``agents_v2.definition`` pattern (see
``schemas/agent_v2.py``) and avoids DB-level JSON schema constraints.

Mirrors plan Section B.1 (DeploymentMode/DeploymentStatus enums) and
Section B.7 (request/response shapes).
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from deep_research.models.agent_deployment import DeploymentMode, DeploymentStatus

# ---------------------------------------------------------------------------
# Deploy-here error kinds — single source of truth (Section S)
# Mirrored verbatim in frontend/src/types/deployment.ts as DeployHereErrorKind.
# ---------------------------------------------------------------------------

DeployHereErrorKind = Literal[
    "mode_does_not_support_inline_deploy",
    "deploy_already_in_progress",
    "missing_workspace_permission",
    "missing_obo_token",
    "artifact_too_large",
    "redeploy_requires_confirmation",
    "app_name_collision",
    "framework_tag_unreachable",
    "reachability_timeout",
    "reachability_failed",
]

DeployHereProbeStatus = Literal["ok", "denied", "unknown"]


class CanDeployHereResponse(BaseModel):
    """Response for GET /api/v1/deployments/can-deploy-here.

    ``actor`` distinguishes an OBO-scoped probe (Databricks Apps runtime where
    the user's token is forwarded) from a service-principal fallback (local
    dev / non-Apps environments). The UI shows a small hint when actor is
    "sp_fallback" to clarify that ownership semantics may differ.
    """

    model_config = ConfigDict(extra="forbid")

    can_deploy: bool
    reason: DeployHereErrorKind | None = None
    probe_status: DeployHereProbeStatus
    actor: Literal["obo", "sp_fallback"]

# W16: SQL-identifier validation regexes for Mode 4 (Lakeflow batch).
# Pre-W16 the schema only enforced non-empty strings; the template
# interpolated raw user input directly into SQL, allowing malformed/
# injection-capable artifacts (codex finding). Even though Phase 2 does
# not execute the SQL server-side, the artifact is exported and pasted
# into a Lakeflow pipeline by the user — so the rendered SQL must be
# safe by construction.
_UC_IDENT = r"[A-Za-z_][A-Za-z0-9_-]*"
_UC_TABLE_RE = re.compile(rf"^{_UC_IDENT}\.{_UC_IDENT}\.{_UC_IDENT}$")
_SQL_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# Endpoint names follow the `dr-agent-*` / `dr-shell-*` etc. convention,
# but Mode 4 accepts ANY pre-existing serving endpoint, so the rule is
# the union of accepted endpoint-name characters in the Databricks model
# serving API: lowercase letters, digits, hyphens, underscores; cannot
# start with a hyphen.
_ENDPOINT_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")

# ---------------------------------------------------------------------------
# Per-mode config models. Each carries a Literal[mode] discriminator so
# Pydantic can dispatch deserialization to the right subtype.
# ---------------------------------------------------------------------------


class _ConfigBase(BaseModel):
    """Forbid extra keys on every deployment config to catch typos early."""

    model_config = ConfigDict(extra="forbid")


class InAppDeploymentConfig(_ConfigBase):
    """Config for Mode 1 -- in-app picker. No external resources.

    Visibility on AgentV2 governs availability; nothing else to configure.
    """

    mode: Literal[DeploymentMode.IN_APP] = DeploymentMode.IN_APP


class ShellAppDeploymentConfig(_ConfigBase):
    """Config for Mode 2 -- standalone Databricks App. Phase 2 deliverable.

    The pinned framework ref is captured at deploy time so that a re-deploy
    of the same revision still pins the same Git ref.
    """

    mode: Literal[DeploymentMode.SHELL_APP] = DeploymentMode.SHELL_APP
    app_name: str = Field(
        ...,
        min_length=2,
        max_length=30,
        pattern=r"^dr-shell-[a-z0-9-]+$",
        description="Databricks App name, 2-30 chars, must start with 'dr-shell-'.",
    )
    framework_git_tag: str = Field(
        ...,
        min_length=1,
        description="Git ref of databricks-deep-research-agent.",
    )
    target: str = Field(default="dev", description="Databricks bundle target.")
    brave_secret_scope: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Optional Databricks secret scope containing BRAVE_API_KEY. "
            "Defaults to server deploy-here settings."
        ),
    )
    brave_secret_key: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Optional key within brave_secret_scope. Defaults to server "
            "deploy-here settings."
        ),
    )


class MlflowAgentDeploymentConfig(_ConfigBase):
    """Config for Mode 3 -- MLflow Responses agent + Databricks App.

    Phase 3 deliverable. ``endpoint_name`` is auto-generated when omitted.
    """

    mode: Literal[DeploymentMode.MLFLOW_AGENT] = DeploymentMode.MLFLOW_AGENT
    uc_catalog: str = Field(..., min_length=1, max_length=255)
    uc_schema: str = Field(..., min_length=1, max_length=255)
    uc_model_name: str = Field(..., min_length=1, max_length=255)
    endpoint_name: str | None = Field(
        default=None,
        max_length=255,
        pattern=r"^dr-agent-[a-z0-9-]+$",
        description="Optional endpoint override (must start with 'dr-agent-').",
    )
    env_overrides: dict[str, str] = Field(default_factory=dict)


class BatchDeploymentConfig(_ConfigBase):
    """Config for Mode 4 -- Lakeflow batch (or Workflow + SQL fallback).

    Phase 2 deliverable. Decoupled from Mode 3: ``target_endpoint`` may be
    a Mode 3 deployment name OR any pre-existing serving endpoint name.
    """

    mode: Literal[DeploymentMode.BATCH] = DeploymentMode.BATCH
    target_endpoint: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Serving endpoint to call via ai_query (any endpoint).",
    )
    input_table: str = Field(
        ...,
        min_length=1,
        description="3-level Unity Catalog name: catalog.schema.table.",
    )
    output_table: str = Field(..., min_length=1)
    prompt_column: str = Field(..., min_length=1)
    response_format: dict[str, Any] | None = Field(
        default=None,
        description="Optional ai_query responseFormat STRUCT spec.",
    )

    # W16 — identifier validation. Pre-W16 the template interpolated raw
    # strings directly into SQL. Now each field must match a strict regex
    # before render; the BatchTranslator additionally backtick-quotes
    # identifiers in the SQL to defend in depth.
    @field_validator("target_endpoint")
    @classmethod
    def _validate_endpoint(cls, v: str) -> str:
        if not _ENDPOINT_NAME_RE.fullmatch(v):
            raise ValueError(
                "target_endpoint must match ^[A-Za-z_][A-Za-z0-9_-]*$ "
                "(serving endpoint name)"
            )
        return v

    @field_validator("input_table", "output_table")
    @classmethod
    def _validate_uc_table(cls, v: str) -> str:
        if not _UC_TABLE_RE.fullmatch(v):
            raise ValueError(
                "must be a 3-level Unity Catalog name "
                "(catalog.schema.table) using identifier-safe characters"
            )
        return v

    @field_validator("prompt_column")
    @classmethod
    def _validate_prompt_column(cls, v: str) -> str:
        if not _SQL_IDENT_RE.fullmatch(v):
            raise ValueError(
                "prompt_column must match ^[A-Za-z_][A-Za-z0-9_]*$ "
                "(SQL identifier)"
            )
        return v


# Discriminated union of all four mode configs. Pydantic validates the
# ``mode`` literal and dispatches to the matching subclass.
DeploymentConfig = Annotated[
    InAppDeploymentConfig
    | ShellAppDeploymentConfig
    | MlflowAgentDeploymentConfig
    | BatchDeploymentConfig,
    Field(discriminator="mode"),
]


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class CreateDeploymentRequest(BaseModel):
    """POST /api/v1/deployments body."""

    model_config = ConfigDict(extra="forbid")

    agent_id: UUID
    revision_id: UUID
    config: DeploymentConfig


class DeploymentResponse(BaseModel):
    """Single AgentDeployment row, serialized for API responses."""

    model_config = ConfigDict(extra="forbid", from_attributes=True)

    id: UUID
    agent_id: UUID
    revision_id: UUID
    mode: DeploymentMode
    status: DeploymentStatus
    config: dict[str, Any]
    endpoint_name: str | None
    model_name: str | None
    external_resource_ids: dict[str, Any] | None
    error_message: str | None
    cleanup_attempts: int
    cancel_requested: bool = False
    deployed_by: str
    created_at: datetime
    updated_at: datetime
    deactivated_at: datetime | None


class DeploymentListResponse(BaseModel):
    """GET /api/v1/deployments paginated list response."""

    model_config = ConfigDict(extra="forbid")

    items: list[DeploymentResponse]
    next_cursor: str | None = None


class DeploymentStatusResponse(BaseModel):
    """GET /api/v1/deployments/{id}/status lightweight poll response."""

    model_config = ConfigDict(extra="forbid")

    status: DeploymentStatus
    updated_at: datetime
    error_message: str | None = None
    external_resource_ids: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Capability probes (split fast/slow per plan Section B.7)
# ---------------------------------------------------------------------------


class CanRunFastResponse(BaseModel):
    """Visibility-only probe response, target latency <100 ms.

    The frontend renders the agent picker from this response, then refines
    on hover/select via the slow probe.
    """

    model_config = ConfigDict(extra="forbid")

    can_run: bool
    reasons: list[str] = Field(default_factory=list)


class CanRunSlowResponse(BaseModel):
    """UC-permission probe response, 5 s timeout, 5 min TTL cache.

    Phase 1: returns the same shape as the fast probe (UC probe is Phase 3).
    """

    model_config = ConfigDict(extra="forbid")

    can_run: bool
    reasons: list[str] = Field(default_factory=list)
    cached: bool = False


class ActiveDeploymentSummary(BaseModel):
    """One row in the 409 response body when deletion is blocked."""

    model_config = ConfigDict(extra="forbid")

    id: UUID
    mode: DeploymentMode
    status: DeploymentStatus
    endpoint_name: str | None = None


class ActiveDeploymentsErrorResponse(BaseModel):
    """HTTP 409 body when ``DELETE /agents-v2/{id}`` is blocked.

    Plan Section N.1.
    """

    model_config = ConfigDict(extra="forbid")

    error_kind: Literal["active_deployments_exist"] = "active_deployments_exist"
    active_count: int
    deployments: list[ActiveDeploymentSummary]
    message: str = (
        "Deactivate all deployments before deleting, or use ?force=true"
    )


__all__ = [
    "ActiveDeploymentSummary",
    "ActiveDeploymentsErrorResponse",
    "BatchDeploymentConfig",
    "CanDeployHereResponse",
    "CanRunFastResponse",
    "CanRunSlowResponse",
    "CreateDeploymentRequest",
    "DeployHereErrorKind",
    "DeploymentConfig",
    "DeploymentListResponse",
    "DeploymentResponse",
    "DeploymentStatusResponse",
    "InAppDeploymentConfig",
    "MlflowAgentDeploymentConfig",
    "ShellAppDeploymentConfig",
]
