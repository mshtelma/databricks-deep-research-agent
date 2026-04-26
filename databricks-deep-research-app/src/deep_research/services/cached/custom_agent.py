"""Cache-backed `ICustomAgentService` — routes custom-agent CRUD through `StorageStack`.

Agent records live in the ``custom_agents`` list table (cold-path list/upsert/delete).
Preset steps are denormalized into the ``steps`` JSONB column on each agent row —
a JSON array of dicts with keys:
    {id, order, title, description, is_required, source_hints, source_scope,
     created_at, updated_at}

Return shape: every method returns a `CustomAgentView` dataclass that exposes the
same attribute names that the 14 call sites rely on. `preset_steps` is always a
list of `PresetStepView` objects (sorted by ``order``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from deep_research.services._cached_base import _CachedServiceBase
from deep_research.services._protocols import ICustomAgentService

if TYPE_CHECKING:
    from deep_research.storage.factory import StorageStack

logger = logging.getLogger(__name__)

_TABLE = "custom_agents"
_PK = "id"


# ---------------------------------------------------------------------------
# View objects (legacy-compatible DTOs)
# ---------------------------------------------------------------------------


@dataclass
class PresetStepView:
    """Lightweight DTO mirroring `AgentPresetStep` ORM attribute surface."""

    id: UUID
    agent_id: UUID
    title: str
    description: str | None
    order: int
    is_required: bool
    source_hints: dict[str, Any] | None
    source_scope: str | None
    created_at: datetime
    updated_at: datetime

    # Compat props used by api/v1/custom_agents.py
    @property
    def step_scope(self) -> Any:
        if self.source_scope:
            try:
                from deep_research.models.custom_agent import AgentSourceScope
                return AgentSourceScope(self.source_scope)
            except ValueError:
                pass
        return None

    def get_preferred_sources(self) -> list[str]:
        if self.source_hints and "preferred_sources" in self.source_hints:
            result = self.source_hints.get("preferred_sources", [])
            return list(result) if result else []
        return []

    def get_search_queries(self) -> list[str]:
        if self.source_hints and "search_queries" in self.source_hints:
            result = self.source_hints.get("search_queries", [])
            return list(result) if result else []
        return []


@dataclass
class CustomAgentView:
    """Read-only DTO mirroring the legacy `CustomAgent` ORM attribute surface."""

    id: UUID
    owner_id: str
    name: str
    description: str | None
    avatar_url: str | None
    system_prompt_template_id: UUID | None
    synthesis_template_id: UUID | None
    source_scope: str
    enabled_sources: list[str] | None
    disabled_sources: list[str]
    use_planner: bool
    default_depth: str
    default_mode: str
    enable_clarification: bool
    output_format: str
    output_schema: dict[str, Any] | None
    visibility: str
    model_overrides: dict[str, str] | None
    domain_filter_mode: str | None
    include_domains: list[str] | None
    exclude_domains: list[str] | None
    created_at: datetime
    updated_at: datetime
    preset_steps: list[PresetStepView] = field(default_factory=list)

    # Compat props
    @property
    def visibility_level(self) -> Any:
        from deep_research.models.custom_agent import AgentVisibility
        return AgentVisibility(self.visibility)

    @property
    def scope(self) -> Any:
        from deep_research.models.custom_agent import AgentSourceScope
        return AgentSourceScope(self.source_scope)

    @property
    def workflow_mode(self) -> Any:
        from deep_research.models.custom_agent import AgentWorkflowMode
        return AgentWorkflowMode(self.default_mode)

    @property
    def research_depth(self) -> Any:
        from deep_research.models.custom_agent import AgentResearchDepth
        return AgentResearchDepth(self.default_depth)

    @property
    def format(self) -> Any:
        from deep_research.models.custom_agent import AgentOutputFormat
        return AgentOutputFormat(self.output_format)

    @property
    def is_workspace_visible(self) -> bool:
        from deep_research.models.custom_agent import AgentVisibility
        return self.visibility in (
            AgentVisibility.WORKSPACE.value,
            AgentVisibility.SYSTEM.value,
        )

    @property
    def is_system_agent(self) -> bool:
        from deep_research.models.custom_agent import AgentVisibility
        return self.visibility == AgentVisibility.SYSTEM.value

    def has_preset_steps(self) -> bool:
        return len(self.preset_steps) > 0

    def get_ordered_preset_steps(self) -> list[PresetStepView]:
        return sorted(self.preset_steps, key=lambda s: s.order)

    # Null relationships expected by template-related code
    system_prompt_template: Any = field(default=None, repr=False)
    synthesis_template: Any = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Row serialisation helpers
# ---------------------------------------------------------------------------


def _uuid(v: Any) -> UUID:
    if isinstance(v, UUID):
        return v
    return UUID(str(v))


def _opt_uuid(v: Any) -> UUID | None:
    if v is None:
        return None
    return _uuid(v)


def _dt(v: Any) -> datetime:
    if isinstance(v, datetime):
        return v
    return datetime.fromisoformat(str(v))


def _step_from_dict(d: dict[str, Any]) -> PresetStepView:
    return PresetStepView(
        id=_uuid(d["id"]),
        agent_id=_uuid(d["agent_id"]),
        title=str(d.get("title", "")),
        description=d.get("description"),
        order=int(d.get("order", 1)),
        is_required=bool(d.get("is_required", True)),
        source_hints=d.get("source_hints"),
        source_scope=d.get("source_scope"),
        created_at=_dt(d.get("created_at", datetime.now(UTC))),
        updated_at=_dt(d.get("updated_at", datetime.now(UTC))),
    )


def _step_to_dict(step: PresetStepView) -> dict[str, Any]:
    return {
        "id": str(step.id),
        "agent_id": str(step.agent_id),
        "title": step.title,
        "description": step.description,
        "order": step.order,
        "is_required": step.is_required,
        "source_hints": step.source_hints,
        "source_scope": step.source_scope,
        "created_at": step.created_at.isoformat(),
        "updated_at": step.updated_at.isoformat(),
    }


def _row_to_view(row: dict[str, Any]) -> CustomAgentView:
    steps_raw: list[dict[str, Any]] = row.get("steps") or []
    steps = sorted([_step_from_dict(s) for s in steps_raw], key=lambda s: s.order)
    return CustomAgentView(
        id=_uuid(row["id"]),
        owner_id=str(row["owner_id"]),
        name=str(row["name"]),
        description=row.get("description"),
        avatar_url=row.get("avatar_url"),
        system_prompt_template_id=_opt_uuid(row.get("system_prompt_template_id")),
        synthesis_template_id=_opt_uuid(row.get("synthesis_template_id")),
        source_scope=str(row.get("source_scope", "all")),
        enabled_sources=row.get("enabled_sources"),
        disabled_sources=row.get("disabled_sources") or [],
        use_planner=bool(row.get("use_planner", True)),
        default_depth=str(row.get("default_depth", "medium")),
        default_mode=str(row.get("default_mode", "planner")),
        enable_clarification=bool(row.get("enable_clarification", True)),
        output_format=str(row.get("output_format", "markdown")),
        output_schema=row.get("output_schema"),
        visibility=str(row.get("visibility", "private")),
        model_overrides=row.get("model_overrides"),
        domain_filter_mode=row.get("domain_filter_mode"),
        include_domains=row.get("include_domains"),
        exclude_domains=row.get("exclude_domains"),
        created_at=_dt(row.get("created_at", datetime.now(UTC))),
        updated_at=_dt(row.get("updated_at", datetime.now(UTC))),
        preset_steps=steps,
    )


def _view_to_row(agent: CustomAgentView) -> dict[str, Any]:
    return {
        "id": str(agent.id),
        "owner_id": agent.owner_id,
        "name": agent.name,
        "description": agent.description,
        "avatar_url": agent.avatar_url,
        "system_prompt_template_id": (
            str(agent.system_prompt_template_id)
            if agent.system_prompt_template_id
            else None
        ),
        "synthesis_template_id": (
            str(agent.synthesis_template_id)
            if agent.synthesis_template_id
            else None
        ),
        "source_scope": agent.source_scope,
        "enabled_sources": agent.enabled_sources,
        "disabled_sources": agent.disabled_sources,
        "use_planner": agent.use_planner,
        "default_depth": agent.default_depth,
        "default_mode": agent.default_mode,
        "enable_clarification": agent.enable_clarification,
        "output_format": agent.output_format,
        "output_schema": agent.output_schema,
        "visibility": agent.visibility,
        "model_overrides": agent.model_overrides,
        "domain_filter_mode": agent.domain_filter_mode,
        "include_domains": agent.include_domains,
        "exclude_domains": agent.exclude_domains,
        "created_at": agent.created_at.isoformat(),
        "updated_at": agent.updated_at.isoformat(),
        "steps": [_step_to_dict(s) for s in agent.preset_steps],
    }


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class CachedCustomAgentService(_CachedServiceBase, ICustomAgentService):
    """``ICustomAgentService`` backed by ``StorageStack`` cold-path list tables."""

    _service_name = "custom_agent"

    def __init__(self, stack: "StorageStack") -> None:
        super().__init__(stack)

    # -- Reads ---------------------------------------------------------------

    async def _get_all_rows(self) -> list[dict[str, Any]]:
        return await self._cold_list_rows(_TABLE)

    async def get_for_user(
        self, agent_id: UUID, user_id: str
    ) -> CustomAgentView | None:
        rows = await self._cold_list_rows(_TABLE, {"id": str(agent_id), "owner_id": user_id})
        if not rows:
            return None
        return _row_to_view(rows[0])

    async def get_accessible(
        self, agent_id: UUID, user_id: str
    ) -> CustomAgentView | None:
        from deep_research.models.custom_agent import AgentVisibility

        rows = await self._cold_list_rows(_TABLE, {"id": str(agent_id)})
        if not rows:
            return None
        row = rows[0]
        vis = row.get("visibility", "private")
        owner = row.get("owner_id")
        if (
            owner == user_id
            or vis == AgentVisibility.WORKSPACE.value
            or vis == AgentVisibility.SYSTEM.value
        ):
            return _row_to_view(row)
        return None

    async def get_by_name(
        self, owner_id: str, name: str
    ) -> CustomAgentView | None:
        rows = await self._cold_list_rows(_TABLE, {"owner_id": owner_id, "name": name})
        if not rows:
            return None
        return _row_to_view(rows[0])

    async def get_system_agents(self) -> list[CustomAgentView]:
        from deep_research.models.custom_agent import AgentVisibility

        rows = await self._cold_list_rows(
            _TABLE, {"visibility": AgentVisibility.SYSTEM.value}, order_by="name"
        )
        return [_row_to_view(r) for r in rows]

    async def get_workspace_agents(self) -> list[CustomAgentView]:
        from deep_research.models.custom_agent import AgentVisibility

        rows = await self._cold_list_rows(
            _TABLE, {"visibility": AgentVisibility.WORKSPACE.value}, order_by="name"
        )
        return [_row_to_view(r) for r in rows]

    async def resolve_agent_for_request(
        self,
        user_id: str,
        agent_id: UUID | None = None,
        agent_name: str | None = None,
    ) -> CustomAgentView | None:
        if agent_id:
            return await self.get_accessible(agent_id, user_id)
        if agent_name:
            agent = await self.get_by_name(user_id, agent_name)
            if agent:
                return agent
            # Workspace/system lookup by name
            all_rows = await self._get_all_rows()
            for row in all_rows:
                from deep_research.models.custom_agent import AgentVisibility
                if row.get("name") == agent_name and row.get("visibility") in (
                    AgentVisibility.WORKSPACE.value,
                    AgentVisibility.SYSTEM.value,
                ):
                    return _row_to_view(row)
        return None

    async def get_accessible_agents(
        self,
        user_id: str,
        visibility: str | None = None,
        source_scope: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[CustomAgentView], int]:
        from deep_research.models.custom_agent import AgentVisibility

        all_rows = await self._get_all_rows()

        accessible: list[dict[str, Any]] = []
        for row in all_rows:
            vis = row.get("visibility", "private")
            owner = row.get("owner_id")
            if not (
                owner == user_id
                or vis == AgentVisibility.WORKSPACE.value
                or vis == AgentVisibility.SYSTEM.value
            ):
                continue
            if visibility is not None and vis != visibility:
                continue
            if source_scope is not None and row.get("source_scope") != source_scope:
                continue
            accessible.append(row)

        accessible.sort(key=lambda r: r.get("name", ""))
        total = len(accessible)
        page = accessible[offset: offset + limit]
        return [_row_to_view(r) for r in page], total

    # -- Writes --------------------------------------------------------------

    async def create_agent(
        self,
        owner_id: str,
        name: str,
        description: str | None = None,
        avatar_url: str | None = None,
        system_prompt_template_id: UUID | None = None,
        synthesis_template_id: UUID | None = None,
        source_scope: str = "all",
        enabled_sources: list[str] | None = None,
        disabled_sources: list[str] | None = None,
        use_planner: bool = True,
        default_depth: str = "medium",
        default_mode: str = "planner",
        enable_clarification: bool = True,
        output_format: str = "markdown",
        output_schema: dict[str, Any] | None = None,
        visibility: str = "private",
        preset_steps: list[dict[str, Any]] | None = None,
        model_overrides: dict[str, str] | None = None,
        domain_filter_mode: str | None = None,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
    ) -> CustomAgentView:
        now = datetime.now(UTC)
        agent_id = uuid4()

        step_views: list[PresetStepView] = []
        if preset_steps:
            for i, step_def in enumerate(preset_steps, start=1):
                step_views.append(
                    PresetStepView(
                        id=uuid4(),
                        agent_id=agent_id,
                        title=step_def.get("title", "Untitled Step"),
                        description=step_def.get("description"),
                        order=step_def.get("order", i),
                        is_required=step_def.get("is_required", True),
                        source_hints=step_def.get("source_hints"),
                        source_scope=step_def.get("source_scope"),
                        created_at=now,
                        updated_at=now,
                    )
                )

        view = CustomAgentView(
            id=agent_id,
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
            created_at=now,
            updated_at=now,
            preset_steps=step_views,
        )
        await self._cold_upsert_row(_TABLE, _view_to_row(view), pk=_PK)
        logger.info("Created custom agent %s for owner %s", agent_id, owner_id)
        return view

    async def update(self, agent: Any) -> Any:
        """Persist changes from a CustomAgent-like object (ORM compat shim)."""
        if isinstance(agent, CustomAgentView):
            agent.updated_at = datetime.now(UTC)
            await self._cold_upsert_row(_TABLE, _view_to_row(agent), pk=_PK)
            return agent

        # Legacy ORM object — convert fields we know about
        agent_id = _uuid(agent.id)
        rows = await self._cold_list_rows(_TABLE, {"id": str(agent_id)})
        if not rows:
            return agent
        view = _row_to_view(rows[0])

        # Copy all scalar fields from the ORM object onto the view
        for attr in (
            "name", "description", "avatar_url", "source_scope",
            "enabled_sources", "disabled_sources", "use_planner",
            "default_depth", "default_mode", "enable_clarification",
            "output_format", "output_schema", "visibility",
            "model_overrides", "domain_filter_mode",
            "include_domains", "exclude_domains",
        ):
            val = getattr(agent, attr, None)
            if val is not None or attr in (
                "description", "avatar_url", "output_schema",
                "model_overrides", "domain_filter_mode", "include_domains",
                "exclude_domains", "enabled_sources",
            ):
                setattr(view, attr, val)

        view.system_prompt_template_id = _opt_uuid(
            getattr(agent, "system_prompt_template_id", None)
        )
        view.synthesis_template_id = _opt_uuid(
            getattr(agent, "synthesis_template_id", None)
        )
        view.updated_at = datetime.now(UTC)
        await self._cold_upsert_row(_TABLE, _view_to_row(view), pk=_PK)
        return view

    async def delete(self, agent: Any) -> None:
        agent_id = str(_uuid(agent.id))
        await self._cold_delete_row(_TABLE, agent_id, pk=_PK)
        logger.info("Deleted custom agent %s", agent_id)

    # -- Preset step management ----------------------------------------------

    async def get_agent_preset_steps(
        self, agent_id: UUID
    ) -> list[PresetStepView]:
        rows = await self._cold_list_rows(_TABLE, {"id": str(agent_id)})
        if not rows:
            return []
        view = _row_to_view(rows[0])
        return view.preset_steps  # already sorted by order

    async def get_preset_step(
        self, step_id: UUID, agent_id: UUID
    ) -> PresetStepView | None:
        steps = await self.get_agent_preset_steps(agent_id)
        for step in steps:
            if step.id == step_id:
                return step
        return None

    async def create_preset_step(
        self,
        agent_id: UUID,
        title: str,
        description: str | None = None,
        order: int = 1,
        is_required: bool = True,
        source_hints: dict[str, Any] | None = None,
        source_scope: str | None = None,
    ) -> PresetStepView:
        rows = await self._cold_list_rows(_TABLE, {"id": str(agent_id)})
        if not rows:
            raise ValueError(f"CustomAgent {agent_id} not found")
        view = _row_to_view(rows[0])

        now = datetime.now(UTC)
        step = PresetStepView(
            id=uuid4(),
            agent_id=agent_id,
            title=title,
            description=description,
            order=order,
            is_required=is_required,
            source_hints=source_hints,
            source_scope=source_scope,
            created_at=now,
            updated_at=now,
        )
        view.preset_steps.append(step)
        view.updated_at = now
        await self._cold_upsert_row(_TABLE, _view_to_row(view), pk=_PK)
        return step

    async def update_preset_step(self, step: Any) -> Any:
        """Persist changes to a step object (mutated in-place by the caller)."""
        if not isinstance(step, PresetStepView):
            raise TypeError(f"Expected PresetStepView, got {type(step)}")
        rows = await self._cold_list_rows(_TABLE, {"id": str(step.agent_id)})
        if not rows:
            return step
        view = _row_to_view(rows[0])

        # Replace the matching step in the list
        step.updated_at = datetime.now(UTC)
        view.preset_steps = [
            step if s.id == step.id else s for s in view.preset_steps
        ]
        view.updated_at = datetime.now(UTC)
        await self._cold_upsert_row(_TABLE, _view_to_row(view), pk=_PK)
        return step

    async def delete_preset_step(self, step: Any) -> None:
        if not isinstance(step, PresetStepView):
            raise TypeError(f"Expected PresetStepView, got {type(step)}")
        rows = await self._cold_list_rows(_TABLE, {"id": str(step.agent_id)})
        if not rows:
            return
        view = _row_to_view(rows[0])
        view.preset_steps = [s for s in view.preset_steps if s.id != step.id]
        view.updated_at = datetime.now(UTC)
        await self._cold_upsert_row(_TABLE, _view_to_row(view), pk=_PK)

    async def reorder_preset_steps(
        self,
        agent_id: UUID,
        step_order: list[UUID],
    ) -> list[PresetStepView]:
        rows = await self._cold_list_rows(_TABLE, {"id": str(agent_id)})
        if not rows:
            return []
        view = _row_to_view(rows[0])
        step_map = {s.id: s for s in view.preset_steps}

        for new_order, step_id in enumerate(step_order, start=1):
            if step_id in step_map:
                step_map[step_id].order = new_order
                step_map[step_id].updated_at = datetime.now(UTC)

        view.preset_steps = sorted(view.preset_steps, key=lambda s: s.order)
        view.updated_at = datetime.now(UTC)
        await self._cold_upsert_row(_TABLE, _view_to_row(view), pk=_PK)
        return view.preset_steps
