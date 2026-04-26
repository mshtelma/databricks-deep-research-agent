"""Custom Agent API endpoints.

Provides CRUD operations for user-created custom agents
with preset step management.

Part of US6 - Custom Agent Configurations (T078).
"""

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession

from deep_research.core.app_config import get_app_config
from deep_research.core.deps import get_custom_agent_service
from deep_research.core.exceptions import ConflictError, NotFoundError, ValidationError
from deep_research.middleware.auth import AuthenticatedUser, CurrentUser
from deep_research.models.custom_agent import (
    AgentVisibility as ModelAgentVisibility,
)
from deep_research.schemas.custom_agent import (
    AgentOutputFormat,
    AgentResearchDepth,
    AgentSourceScope,
    AgentVisibility,
    AgentWorkflowMode,
    CreateCustomAgentRequest,
    CreatePresetStepRequest,
    CustomAgentListResponse,
    CustomAgentResponse,
    CustomAgentSummary,
    PresetStepResponse,
    UpdateCustomAgentRequest,
    UpdatePresetStepRequest,
)
from deep_research.services._protocols import ICustomAgentService

router = APIRouter(prefix="/custom-agents", tags=["Custom Agents"])


def _agent_to_response(agent: Any) -> CustomAgentResponse:
    """Convert agent view / ORM object to response schema."""
    preset_steps = [
        PresetStepResponse(
            id=step.id,
            agent_id=step.agent_id,
            title=step.title,
            description=step.description,
            order=step.order,
            is_required=step.is_required,
            source_hints=step.source_hints,
            source_scope=AgentSourceScope(step.source_scope) if step.source_scope else None,
            created_at=step.created_at,
            updated_at=step.updated_at,
        )
        for step in sorted(agent.preset_steps, key=lambda s: s.order)
    ]

    # Check for stale model override warnings
    model_override_warnings: list[dict[str, str]] = []
    if agent.model_overrides:
        app_config = get_app_config()
        for tier_name, endpoint_id in agent.model_overrides.items():
            if endpoint_id not in app_config.endpoints:
                model_override_warnings.append({
                    "tier": tier_name,
                    "endpoint": endpoint_id,
                    "message": f"Endpoint '{endpoint_id}' not found in current config",
                })

    return CustomAgentResponse(
        id=agent.id,
        owner_id=agent.owner_id,
        name=agent.name,
        description=agent.description,
        avatar_url=agent.avatar_url,
        system_prompt_template_id=agent.system_prompt_template_id,
        synthesis_template_id=agent.synthesis_template_id,
        source_scope=AgentSourceScope(agent.source_scope),
        enabled_sources=agent.enabled_sources,
        disabled_sources=agent.disabled_sources or [],
        use_planner=agent.use_planner,
        default_depth=AgentResearchDepth(agent.default_depth),
        default_mode=AgentWorkflowMode(agent.default_mode),
        enable_clarification=agent.enable_clarification,
        output_format=AgentOutputFormat(agent.output_format),
        output_schema=agent.output_schema,
        visibility=AgentVisibility(agent.visibility),
        model_overrides=agent.model_overrides,
        domain_filter_mode=agent.domain_filter_mode,
        include_domains=agent.include_domains,
        exclude_domains=agent.exclude_domains,
        model_override_warnings=model_override_warnings,
        preset_steps=preset_steps,
        created_at=agent.created_at,
        updated_at=agent.updated_at,
    )


def _agent_to_summary(agent: Any) -> CustomAgentSummary:
    """Convert agent view / ORM object to summary schema."""
    has_source_config = (
        (agent.source_scope and agent.source_scope != "all")
        or bool(agent.enabled_sources)
    )

    # Compute capability tags for display
    capabilities: list[str] = []
    if agent.source_scope in ("all", "web_only"):
        capabilities.append("web_search")
    if agent.source_scope in ("all", "enterprise_only"):
        capabilities.append("enterprise_sources")
    if agent.default_mode != "planner":
        capabilities.append("manual_workflow")
    if agent.output_format == "json":
        capabilities.append("structured_output")
    if agent.system_prompt_template_id or agent.synthesis_template_id:
        capabilities.append("custom_prompts")

    return CustomAgentSummary(
        id=agent.id,
        owner_id=agent.owner_id,
        name=agent.name,
        description=agent.description,
        avatar_url=agent.avatar_url,
        visibility=AgentVisibility(agent.visibility),
        source_scope=AgentSourceScope(agent.source_scope),
        default_mode=AgentWorkflowMode(agent.default_mode),
        default_depth=AgentResearchDepth(agent.default_depth),
        preset_step_count=len(agent.preset_steps),
        has_model_overrides=bool(agent.model_overrides),
        has_domain_filter=agent.domain_filter_mode is not None,
        has_source_config=bool(has_source_config),
        capabilities=capabilities,
        created_at=agent.created_at,
    )


# =============================================================================
# List and Get Endpoints
# =============================================================================


@router.get("", response_model=CustomAgentListResponse)
async def list_custom_agents(
    request: Request,
    user: CurrentUser,
    service: ICustomAgentService = Depends(get_custom_agent_service),
    visibility: AgentVisibility | None = Query(None, description="Filter by visibility"),
    source_scope: AgentSourceScope | None = Query(None, description="Filter by source scope"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> CustomAgentListResponse:
    """List custom agents accessible to the current user.

    Returns user-owned agents, workspace-visible agents, and system agents.
    """
    # Convert schema enums to model enum values if provided
    vis_value = visibility.value if visibility else None
    scope_value = source_scope.value if source_scope else None

    agents, total = await service.get_accessible_agents(
        user_id=user.user_id,
        visibility=vis_value,
        source_scope=scope_value,
        limit=limit,
        offset=offset,
    )

    # Count agents by category
    user_agents = sum(1 for a in agents if a.owner_id == user.user_id)
    workspace_agents = sum(
        1 for a in agents
        if a.visibility == ModelAgentVisibility.WORKSPACE.value and a.owner_id != user.user_id
    )
    system_agents = sum(1 for a in agents if a.visibility == ModelAgentVisibility.SYSTEM.value)

    return CustomAgentListResponse(
        agents=[_agent_to_summary(a) for a in agents],
        total=total,
        user_agents=user_agents,
        workspace_agents=workspace_agents,
        system_agents=system_agents,
    )


@router.get("/{agent_id}", response_model=CustomAgentResponse)
async def get_custom_agent(
    agent_id: UUID,
    request: Request,
    user: CurrentUser,
    service: ICustomAgentService = Depends(get_custom_agent_service),
) -> CustomAgentResponse:
    """Get details of a specific custom agent.

    Returns the agent if owned by user, workspace-visible, or system.
    """
    agent = await service.get_accessible(agent_id, user.user_id)

    if not agent:
        raise NotFoundError("CustomAgent", str(agent_id))

    return _agent_to_response(agent)


# =============================================================================
# Create Endpoint
# =============================================================================


@router.post("", response_model=CustomAgentResponse, status_code=201)
async def create_custom_agent(
    request: Request,
    request_body: CreateCustomAgentRequest,
    user: AuthenticatedUser,
    service: ICustomAgentService = Depends(get_custom_agent_service),
) -> CustomAgentResponse:
    """Create a new custom agent.

    Optionally create preset steps inline with the agent.
    """
    # Check for duplicate name
    existing = await service.get_by_name(user.user_id, request_body.name)
    if existing:
        raise ConflictError(f"Agent with name '{request_body.name}' already exists")

    # Convert preset steps to dicts
    preset_steps: list[dict[str, Any]] | None = None
    if request_body.preset_steps:
        preset_steps = []
        for step in request_body.preset_steps:
            step_dict: dict[str, Any] = {
                "title": step.title,
                "description": step.description,
                "order": step.order,
                "is_required": step.is_required,
                "source_scope": step.source_scope.value if step.source_scope else None,
            }
            if step.source_hints:
                step_dict["source_hints"] = step.source_hints.model_dump()
            preset_steps.append(step_dict)

    # Validate model_overrides tier names at creation time (M2)
    _VALID_TIERS = {"simple", "analytical", "complex", "synthesis"}
    if request_body.model_overrides:
        for tier_name in request_body.model_overrides:
            if tier_name not in _VALID_TIERS:
                raise ValidationError(
                    f"Unknown model tier: {tier_name!r}. "
                    f"Valid tiers: {', '.join(sorted(_VALID_TIERS))}"
                )

    agent = await service.create_agent(
        owner_id=user.user_id,
        name=request_body.name,
        description=request_body.description,
        avatar_url=request_body.avatar_url,
        system_prompt_template_id=request_body.system_prompt_template_id,
        synthesis_template_id=request_body.synthesis_template_id,
        source_scope=request_body.source_scope.value,
        enabled_sources=request_body.enabled_sources,
        disabled_sources=request_body.disabled_sources,
        use_planner=request_body.use_planner,
        default_depth=request_body.default_depth.value,
        default_mode=request_body.default_mode.value,
        enable_clarification=request_body.enable_clarification,
        output_format=request_body.output_format.value,
        output_schema=request_body.output_schema,
        visibility=request_body.visibility.value,
        preset_steps=preset_steps,
        model_overrides=request_body.model_overrides,
        domain_filter_mode=request_body.domain_filter_mode,
        include_domains=request_body.include_domains,
        exclude_domains=request_body.exclude_domains,
    )

    return _agent_to_response(agent)


# =============================================================================
# Update Endpoint
# =============================================================================


@router.patch("/{agent_id}", response_model=CustomAgentResponse)
async def update_custom_agent(
    agent_id: UUID,
    request: Request,
    request_body: UpdateCustomAgentRequest,
    user: AuthenticatedUser,
    service: ICustomAgentService = Depends(get_custom_agent_service),
) -> CustomAgentResponse:
    """Update a custom agent.

    Only the agent owner can update. System agents cannot be modified.
    """
    agent = await service.get_for_user(agent_id, user.user_id)

    if not agent:
        raise NotFoundError("CustomAgent", str(agent_id))

    # Check if it's a system agent
    if agent.visibility == ModelAgentVisibility.SYSTEM.value:
        raise ValidationError("System agents cannot be modified")

    # Check for duplicate name if changing
    if request_body.name is not None and request_body.name != agent.name:
        existing = await service.get_by_name(user.user_id, request_body.name)
        if existing:
            raise ConflictError(f"Agent with name '{request_body.name}' already exists")

    # Update fields
    if request_body.name is not None:
        agent.name = request_body.name
    if request_body.description is not None:
        agent.description = request_body.description
    if request_body.avatar_url is not None:
        agent.avatar_url = request_body.avatar_url
    if request_body.system_prompt_template_id is not None:
        agent.system_prompt_template_id = request_body.system_prompt_template_id
    if request_body.synthesis_template_id is not None:
        agent.synthesis_template_id = request_body.synthesis_template_id
    if request_body.source_scope is not None:
        agent.source_scope = request_body.source_scope.value
    if request_body.enabled_sources is not None:
        agent.enabled_sources = request_body.enabled_sources
    if request_body.disabled_sources is not None:
        agent.disabled_sources = request_body.disabled_sources
    if request_body.use_planner is not None:
        agent.use_planner = request_body.use_planner
    if request_body.default_depth is not None:
        agent.default_depth = request_body.default_depth.value
    if request_body.default_mode is not None:
        agent.default_mode = request_body.default_mode.value
    if request_body.enable_clarification is not None:
        agent.enable_clarification = request_body.enable_clarification
    if request_body.output_format is not None:
        agent.output_format = request_body.output_format.value
    if request_body.output_schema is not None:
        agent.output_schema = request_body.output_schema
    if request_body.visibility is not None:
        # Cannot set visibility to system
        if request_body.visibility == AgentVisibility.SYSTEM:
            raise ValidationError("Cannot set visibility to 'system'")
        agent.visibility = request_body.visibility.value
    if request_body.model_overrides is not None:
        # Validate model_overrides tier names at update time (M2)
        _VALID_TIERS = {"simple", "analytical", "complex", "synthesis"}
        for tier_name in request_body.model_overrides:
            if tier_name not in _VALID_TIERS:
                raise ValidationError(
                    f"Unknown model tier: {tier_name!r}. "
                    f"Valid tiers: {', '.join(sorted(_VALID_TIERS))}"
                )
        agent.model_overrides = request_body.model_overrides
    if request_body.domain_filter_mode is not None:
        agent.domain_filter_mode = request_body.domain_filter_mode
    if request_body.include_domains is not None:
        agent.include_domains = request_body.include_domains
    if request_body.exclude_domains is not None:
        agent.exclude_domains = request_body.exclude_domains

    agent.updated_at = datetime.now(UTC)
    updated = await service.update(agent)

    return _agent_to_response(updated)


# =============================================================================
# Delete Endpoint
# =============================================================================


@router.delete("/{agent_id}", status_code=204)
async def delete_custom_agent(
    agent_id: UUID,
    request: Request,
    user: AuthenticatedUser,
    service: ICustomAgentService = Depends(get_custom_agent_service),
) -> None:
    """Delete a custom agent.

    Only the agent owner can delete. System agents cannot be deleted.
    Preset steps are automatically deleted via cascade.
    """
    agent = await service.get_for_user(agent_id, user.user_id)

    if not agent:
        raise NotFoundError("CustomAgent", str(agent_id))

    # Check if it's a system agent
    if agent.visibility == ModelAgentVisibility.SYSTEM.value:
        raise ValidationError("System agents cannot be deleted")

    await service.delete(agent)


# =============================================================================
# Preset Step Endpoints
# =============================================================================


@router.get("/{agent_id}/steps", response_model=list[PresetStepResponse])
async def list_preset_steps(
    agent_id: UUID,
    request: Request,
    user: CurrentUser,
    service: ICustomAgentService = Depends(get_custom_agent_service),
) -> list[PresetStepResponse]:
    """List preset steps for a custom agent.

    Returns steps ordered by execution order.
    """
    # Verify agent exists and is accessible
    agent = await service.get_accessible(agent_id, user.user_id)
    if not agent:
        raise NotFoundError("CustomAgent", str(agent_id))

    steps = await service.get_agent_preset_steps(agent_id)

    return [
        PresetStepResponse(
            id=step.id,
            agent_id=step.agent_id,
            title=step.title,
            description=step.description,
            order=step.order,
            is_required=step.is_required,
            source_hints=step.source_hints,
            source_scope=AgentSourceScope(step.source_scope) if step.source_scope else None,
            created_at=step.created_at,
            updated_at=step.updated_at,
        )
        for step in steps
    ]


@router.post("/{agent_id}/steps", response_model=PresetStepResponse, status_code=201)
async def create_preset_step(
    agent_id: UUID,
    request: Request,
    request_body: CreatePresetStepRequest,
    user: AuthenticatedUser,
    service: ICustomAgentService = Depends(get_custom_agent_service),
) -> PresetStepResponse:
    """Create a preset step for a custom agent.

    Only the agent owner can add steps. System agents cannot be modified.
    """
    # Verify agent exists and is owned by user
    agent = await service.get_for_user(agent_id, user.user_id)
    if not agent:
        raise NotFoundError("CustomAgent", str(agent_id))

    # Check if it's a system agent
    if agent.visibility == ModelAgentVisibility.SYSTEM.value:
        raise ValidationError("Cannot add steps to system agents")

    # Convert source hints
    source_hints = None
    if request_body.source_hints:
        source_hints = request_body.source_hints.model_dump()

    step = await service.create_preset_step(
        agent_id=agent_id,
        title=request_body.title,
        description=request_body.description,
        order=request_body.order,
        is_required=request_body.is_required,
        source_hints=source_hints,
        source_scope=request_body.source_scope.value if request_body.source_scope else None,
    )

    return PresetStepResponse(
        id=step.id,
        agent_id=step.agent_id,
        title=step.title,
        description=step.description,
        order=step.order,
        is_required=step.is_required,
        source_hints=step.source_hints,
        source_scope=AgentSourceScope(step.source_scope) if step.source_scope else None,
        created_at=step.created_at,
        updated_at=step.updated_at,
    )


@router.patch("/{agent_id}/steps/{step_id}", response_model=PresetStepResponse)
async def update_preset_step(
    agent_id: UUID,
    step_id: UUID,
    request: Request,
    request_body: UpdatePresetStepRequest,
    user: AuthenticatedUser,
    service: ICustomAgentService = Depends(get_custom_agent_service),
) -> PresetStepResponse:
    """Update a preset step.

    Only the agent owner can update steps. System agents cannot be modified.
    """
    # Verify agent exists and is owned by user
    agent = await service.get_for_user(agent_id, user.user_id)
    if not agent:
        raise NotFoundError("CustomAgent", str(agent_id))

    # Check if it's a system agent
    if agent.visibility == ModelAgentVisibility.SYSTEM.value:
        raise ValidationError("Cannot modify steps of system agents")

    # Get the step
    step = await service.get_preset_step(step_id, agent_id)
    if not step:
        raise NotFoundError("AgentPresetStep", str(step_id))

    # Update fields
    if request_body.title is not None:
        step.title = request_body.title
    if request_body.description is not None:
        step.description = request_body.description
    if request_body.order is not None:
        step.order = request_body.order
    if request_body.is_required is not None:
        step.is_required = request_body.is_required
    if request_body.source_hints is not None:
        step.source_hints = request_body.source_hints.model_dump()
    if request_body.source_scope is not None:
        step.source_scope = request_body.source_scope.value

    updated_step = await service.update_preset_step(step)

    return PresetStepResponse(
        id=updated_step.id,
        agent_id=updated_step.agent_id,
        title=updated_step.title,
        description=updated_step.description,
        order=updated_step.order,
        is_required=updated_step.is_required,
        source_hints=updated_step.source_hints,
        source_scope=AgentSourceScope(updated_step.source_scope) if updated_step.source_scope else None,
        created_at=updated_step.created_at,
        updated_at=updated_step.updated_at,
    )


@router.delete("/{agent_id}/steps/{step_id}", status_code=204)
async def delete_preset_step(
    agent_id: UUID,
    step_id: UUID,
    request: Request,
    user: AuthenticatedUser,
    service: ICustomAgentService = Depends(get_custom_agent_service),
) -> None:
    """Delete a preset step.

    Only the agent owner can delete steps. System agents cannot be modified.
    """
    # Verify agent exists and is owned by user
    agent = await service.get_for_user(agent_id, user.user_id)
    if not agent:
        raise NotFoundError("CustomAgent", str(agent_id))

    # Check if it's a system agent
    if agent.visibility == ModelAgentVisibility.SYSTEM.value:
        raise ValidationError("Cannot delete steps from system agents")

    # Get and delete the step
    step = await service.get_preset_step(step_id, agent_id)
    if not step:
        raise NotFoundError("AgentPresetStep", str(step_id))

    await service.delete_preset_step(step)


@router.post("/{agent_id}/steps/reorder", response_model=list[PresetStepResponse])
async def reorder_preset_steps(
    agent_id: UUID,
    request: Request,
    step_order: list[UUID],
    user: AuthenticatedUser,
    service: ICustomAgentService = Depends(get_custom_agent_service),
) -> list[PresetStepResponse]:
    """Reorder preset steps for a custom agent.

    Pass a list of step IDs in the desired order.
    Only the agent owner can reorder steps.
    """
    # Verify agent exists and is owned by user
    agent = await service.get_for_user(agent_id, user.user_id)
    if not agent:
        raise NotFoundError("CustomAgent", str(agent_id))

    # Check if it's a system agent
    if agent.visibility == ModelAgentVisibility.SYSTEM.value:
        raise ValidationError("Cannot reorder steps of system agents")

    steps = await service.reorder_preset_steps(agent_id, step_order)

    return [
        PresetStepResponse(
            id=step.id,
            agent_id=step.agent_id,
            title=step.title,
            description=step.description,
            order=step.order,
            is_required=step.is_required,
            source_hints=step.source_hints,
            source_scope=AgentSourceScope(step.source_scope) if step.source_scope else None,
            created_at=step.created_at,
            updated_at=step.updated_at,
        )
        for step in steps
    ]
