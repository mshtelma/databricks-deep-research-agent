"""CustomAgentService - CRUD operations for custom agents and preset steps.

Manages user-created custom agents with support for:
- CRUD with cascade delete for preset steps
- Agent resolution by ID or name
- Accessible agents listing (own + workspace + system)

Part of US6 - Custom Agent Configurations (T077).
"""

import logging
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import selectinload

from deep_research.models.custom_agent import (
    AgentPresetStep,
    AgentSourceScope,
    AgentVisibility,
    AgentWorkflowMode,
    CustomAgent,
)
from deep_research.services.base import BaseRepository

logger = logging.getLogger(__name__)


class CustomAgentService(BaseRepository[CustomAgent]):
    """Service for managing custom agents and preset steps.

    Extends BaseRepository[CustomAgent] for standard CRUD operations.
    Provides specialized methods for:
    - Creating agents with inline preset steps
    - Managing preset steps within agents
    - Resolving agents by ID or name
    - Listing accessible agents (own + workspace + system)
    """

    model = CustomAgent

    # =========================================================================
    # Agent Creation
    # =========================================================================

    async def create_agent(
        self,
        owner_id: str,
        name: str,
        description: str | None = None,
        avatar_url: str | None = None,
        system_prompt_template_id: UUID | None = None,
        synthesis_template_id: UUID | None = None,
        source_scope: str = AgentSourceScope.ALL.value,
        enabled_sources: list[str] | None = None,
        disabled_sources: list[str] | None = None,
        use_planner: bool = True,
        default_depth: str = "medium",
        default_mode: str = AgentWorkflowMode.PLANNER.value,
        enable_clarification: bool = True,
        output_format: str = "markdown",
        output_schema: dict[str, Any] | None = None,
        visibility: str = AgentVisibility.PRIVATE.value,
        preset_steps: list[dict[str, Any]] | None = None,
        model_overrides: dict[str, str] | None = None,
        domain_filter_mode: str | None = None,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
    ) -> CustomAgent:
        """Create a new custom agent with optional preset steps.

        Args:
            owner_id: Databricks workspace user ID.
            name: Display name for the agent.
            description: Human-readable description.
            avatar_url: URL for agent avatar.
            system_prompt_template_id: FK to system prompt template.
            synthesis_template_id: FK to synthesis template.
            source_scope: Source scope (all, enterprise_only, web_only).
            enabled_sources: Explicit source whitelist.
            disabled_sources: Source blacklist.
            use_planner: Whether to use AI planner.
            default_depth: Default research depth.
            default_mode: Default workflow mode.
            enable_clarification: Whether to enable clarification.
            output_format: Output format (markdown, json).
            output_schema: JSON schema for structured output.
            visibility: Visibility level.
            preset_steps: Optional list of preset step definitions.

        Returns:
            Created agent with preset steps.
        """
        agent = CustomAgent(
            owner_id=owner_id,
            name=name,
            description=description,
            avatar_url=avatar_url,
            system_prompt_template_id=system_prompt_template_id,
            synthesis_template_id=synthesis_template_id,
            source_scope=source_scope,
            enabled_sources=enabled_sources,
            disabled_sources=disabled_sources or [],
            use_planner=use_planner,
            default_depth=default_depth,
            default_mode=default_mode,
            enable_clarification=enable_clarification,
            output_format=output_format,
            output_schema=output_schema,
            visibility=visibility,
            model_overrides=model_overrides,
            domain_filter_mode=domain_filter_mode,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
        )

        agent = await self.add(agent)

        # Create preset steps if provided
        if preset_steps:
            for step_def in preset_steps:
                await self.create_preset_step(
                    agent_id=agent.id,
                    title=step_def.get("title", "Untitled Step"),
                    description=step_def.get("description"),
                    order=step_def.get("order", 1),
                    is_required=step_def.get("is_required", True),
                    source_hints=step_def.get("source_hints"),
                    source_scope=step_def.get("source_scope"),
                )

        # Refresh to load relationships
        await self._session.refresh(agent, ["preset_steps"])

        logger.info(
            "Created custom agent",
            extra={
                "AGENT_ID": str(agent.id),
                "OWNER_ID": owner_id,
                "AGENT_NAME": name,
                "PRESET_STEPS_COUNT": len(preset_steps) if preset_steps else 0,
            },
        )

        return agent

    # =========================================================================
    # Preset Step Management
    # =========================================================================

    async def create_preset_step(
        self,
        agent_id: UUID,
        title: str,
        description: str | None = None,
        order: int = 1,
        is_required: bool = True,
        source_hints: dict[str, Any] | None = None,
        source_scope: str | None = None,
    ) -> AgentPresetStep:
        """Create a preset step for an agent.

        Args:
            agent_id: Parent agent ID.
            title: Step title.
            description: Step description.
            order: Execution order.
            is_required: Whether step is required.
            source_hints: Source selection hints.
            source_scope: Optional source scope override.

        Returns:
            Created preset step.
        """
        step = AgentPresetStep(
            agent_id=agent_id,
            title=title,
            description=description,
            order=order,
            is_required=is_required,
            source_hints=source_hints,
            source_scope=source_scope,
        )

        self._session.add(step)
        await self._session.flush()
        await self._session.refresh(step)

        logger.info(
            "Created preset step",
            extra={
                "step_id": str(step.id),
                "agent_id": str(agent_id),
                "title": title,
                "order": order,
            },
        )

        return step

    async def get_preset_step(
        self,
        step_id: UUID,
        agent_id: UUID,
    ) -> AgentPresetStep | None:
        """Get a preset step by ID within an agent.

        Args:
            step_id: Step ID.
            agent_id: Parent agent ID.

        Returns:
            Preset step if found, None otherwise.
        """
        result = await self._session.execute(
            select(AgentPresetStep).where(
                and_(
                    AgentPresetStep.id == step_id,
                    AgentPresetStep.agent_id == agent_id,
                )
            )
        )
        return result.scalar_one_or_none()

    async def update_preset_step(
        self,
        step: AgentPresetStep,
    ) -> AgentPresetStep:
        """Persist changes to a preset step.

        Args:
            step: Step with modifications.

        Returns:
            Updated step.
        """
        step.updated_at = datetime.now(UTC)
        await self._session.flush()
        await self._session.refresh(step)
        return step

    async def delete_preset_step(
        self,
        step: AgentPresetStep,
    ) -> None:
        """Delete a preset step.

        Args:
            step: Step to delete.
        """
        await self._session.delete(step)
        await self._session.flush()

    async def get_agent_preset_steps(
        self,
        agent_id: UUID,
    ) -> list[AgentPresetStep]:
        """Get all preset steps for an agent, ordered by execution order.

        Args:
            agent_id: Agent ID.

        Returns:
            List of preset steps ordered by execution order.
        """
        result = await self._session.execute(
            select(AgentPresetStep)
            .where(AgentPresetStep.agent_id == agent_id)
            .order_by(AgentPresetStep.order)
        )
        return list(result.scalars().all())

    async def reorder_preset_steps(
        self,
        agent_id: UUID,
        step_order: list[UUID],
    ) -> list[AgentPresetStep]:
        """Reorder preset steps for an agent.

        Args:
            agent_id: Agent ID.
            step_order: List of step IDs in desired order.

        Returns:
            Updated preset steps.
        """
        steps = await self.get_agent_preset_steps(agent_id)
        step_map = {step.id: step for step in steps}

        for order, step_id in enumerate(step_order, start=1):
            if step_id in step_map:
                step_map[step_id].order = order

        await self._session.flush()
        return await self.get_agent_preset_steps(agent_id)

    # =========================================================================
    # Agent Resolution
    # =========================================================================

    async def resolve_agent_for_request(
        self,
        user_id: str,
        agent_id: UUID | None = None,
        agent_name: str | None = None,
    ) -> CustomAgent | None:
        """Resolve an agent by ID or name for a user.

        Resolves in the following order:
        1. By ID if provided (must be accessible to user)
        2. By name if provided (searches user's agents first, then workspace/system)

        Args:
            user_id: User ID making the request.
            agent_id: Optional agent ID to resolve.
            agent_name: Optional agent name to resolve.

        Returns:
            Resolved agent if found and accessible, None otherwise.
        """
        if agent_id:
            return await self.get_accessible(agent_id, user_id)

        if agent_name:
            # First try user's own agents
            agent = await self.get_by_name(user_id, agent_name)
            if agent:
                return agent

            # Then try workspace/system agents by name
            result = await self._session.execute(
                select(CustomAgent)
                .options(selectinload(CustomAgent.preset_steps))
                .where(
                    and_(
                        CustomAgent.name == agent_name,
                        CustomAgent.visibility.in_(
                            [AgentVisibility.WORKSPACE.value, AgentVisibility.SYSTEM.value]
                        ),
                    )
                )
            )
            return result.scalar_one_or_none()

        return None

    # =========================================================================
    # Query Methods
    # =========================================================================

    async def get_accessible_agents(
        self,
        user_id: str,
        visibility: str | None = None,
        source_scope: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[CustomAgent], int]:
        """Get custom agents accessible to a user.

        Returns agents that are:
        - Owned by the user (any visibility)
        - OR workspace-visible
        - OR system agents

        Args:
            user_id: Databricks workspace user ID.
            visibility: Optional filter by visibility.
            source_scope: Optional filter by source scope.
            limit: Maximum number of agents.
            offset: Number of agents to skip.

        Returns:
            Tuple of (agents, total_count).
        """
        # Build conditions: own agents OR workspace visible OR system
        access_conditions = or_(
            CustomAgent.owner_id == user_id,
            CustomAgent.visibility == AgentVisibility.WORKSPACE.value,
            CustomAgent.visibility == AgentVisibility.SYSTEM.value,
        )

        conditions = [access_conditions]

        if visibility:
            conditions.append(CustomAgent.visibility == visibility)

        if source_scope:
            conditions.append(CustomAgent.source_scope == source_scope)

        # Get total count
        count_query = select(func.count(CustomAgent.id)).where(and_(*conditions))
        count_result = await self._session.execute(count_query)
        total = count_result.scalar() or 0

        # Get agents with preset steps
        query = (
            select(CustomAgent)
            .options(selectinload(CustomAgent.preset_steps))
            .where(and_(*conditions))
            .order_by(CustomAgent.name)
            .limit(limit)
            .offset(offset)
        )
        result = await self._session.execute(query)
        agents = list(result.scalars().all())

        return agents, total

    async def get_for_user(
        self,
        agent_id: UUID,
        user_id: str,
    ) -> CustomAgent | None:
        """Get an agent by ID with user ownership check.

        Args:
            agent_id: Agent ID.
            user_id: User ID (for ownership check).

        Returns:
            Agent if found and owned by user, None otherwise.
        """
        result = await self._session.execute(
            select(CustomAgent)
            .options(selectinload(CustomAgent.preset_steps))
            .where(
                and_(
                    CustomAgent.id == agent_id,
                    CustomAgent.owner_id == user_id,
                )
            )
        )
        return result.scalar_one_or_none()

    async def get_accessible(
        self,
        agent_id: UUID,
        user_id: str,
    ) -> CustomAgent | None:
        """Get an agent by ID if accessible to user.

        Agent is accessible if:
        - Owned by the user
        - OR workspace-visible
        - OR system agent

        Args:
            agent_id: Agent ID.
            user_id: User ID.

        Returns:
            Agent if accessible, None otherwise.
        """
        result = await self._session.execute(
            select(CustomAgent)
            .options(selectinload(CustomAgent.preset_steps))
            .where(
                and_(
                    CustomAgent.id == agent_id,
                    or_(
                        CustomAgent.owner_id == user_id,
                        CustomAgent.visibility == AgentVisibility.WORKSPACE.value,
                        CustomAgent.visibility == AgentVisibility.SYSTEM.value,
                    ),
                )
            )
        )
        return result.scalar_one_or_none()

    async def get_by_name(
        self,
        owner_id: str,
        name: str,
    ) -> CustomAgent | None:
        """Get an agent by name for a specific owner.

        Args:
            owner_id: Owner user ID.
            name: Agent name.

        Returns:
            Agent if found, None otherwise.
        """
        result = await self._session.execute(
            select(CustomAgent)
            .options(selectinload(CustomAgent.preset_steps))
            .where(
                and_(
                    CustomAgent.owner_id == owner_id,
                    CustomAgent.name == name,
                )
            )
        )
        return result.scalar_one_or_none()

    async def get_system_agents(self) -> list[CustomAgent]:
        """Get all system-provided agents.

        Returns:
            List of system agents.
        """
        result = await self._session.execute(
            select(CustomAgent)
            .options(selectinload(CustomAgent.preset_steps))
            .where(CustomAgent.visibility == AgentVisibility.SYSTEM.value)
            .order_by(CustomAgent.name)
        )
        return list(result.scalars().all())

    async def get_workspace_agents(self) -> list[CustomAgent]:
        """Get all workspace-visible agents.

        Returns:
            List of workspace agents.
        """
        result = await self._session.execute(
            select(CustomAgent)
            .options(selectinload(CustomAgent.preset_steps))
            .where(CustomAgent.visibility == AgentVisibility.WORKSPACE.value)
            .order_by(CustomAgent.name)
        )
        return list(result.scalars().all())
