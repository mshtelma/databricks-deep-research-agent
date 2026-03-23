"""Planner agent - creates structured research plans.

Supports two planning modes:
1. Basic planner (run_planner): Web search only
2. Source-aware planner (run_source_aware_planner): Enterprise + web sources

Part of 007-enterprise-data-sources feature (T037).
"""

from typing import Any
from uuid import uuid4

from mlflow.entities import SpanType
from pydantic import BaseModel, Field

from deep_research.agent.config import get_endpoint_override, get_step_limits
from deep_research.agent.prompts.planner import PLANNER_SYSTEM_PROMPT, PLANNER_USER_PROMPT
from deep_research.agent.prompts.source_aware_planner import (
    SOURCE_AWARE_PLANNER_NO_LANDSCAPE_PROMPT,
    SOURCE_AWARE_PLANNER_SYSTEM_PROMPT,
    SOURCE_AWARE_PLANNER_USER_PROMPT,
)
from deep_research.agent.state import Plan, PlanStep, ResearchState, StepStatus, StepType
from deep_research.core.logging_utils import get_logger, truncate
from deep_research.core.tracing import safe_tool_span
from deep_research.core.tracing_constants import (
    ATTR_PLAN_ITERATION,
    ATTR_PLAN_STEPS_COUNT,
    ATTR_PLAN_THOUGHT,
    PHASE_PLAN,
    research_span_name,
    truncate_for_attr,
)
from deep_research.services.llm.client import LLMClient
from deep_research.services.llm.types import ModelTier

logger = get_logger(__name__)


class PlanStepOutput(BaseModel):
    """Output schema for a plan step."""

    id: str
    title: str
    description: str
    step_type: str
    needs_search: bool
    status: str = "pending"  # For preserving completed step status


class PlanOutput(BaseModel):
    """Output schema for Planner agent."""

    id: str
    title: str
    thought: str
    has_enough_context: bool = False
    steps: list[PlanStepOutput]


async def run_planner(state: ResearchState, llm: LLMClient) -> ResearchState:
    """Run the Planner agent to create a research plan.

    Args:
        state: Current research state.
        llm: LLM client for completions.

    Returns:
        Updated state with research plan.
    """
    # Use plan_iterations + 1 for span naming (before increment)
    iteration = state.plan_iterations + 1
    span_name = research_span_name(PHASE_PLAN, "planner", iteration=iteration)

    async with safe_tool_span(span_name, SpanType.AGENT, {
        ATTR_PLAN_ITERATION: iteration,
        "query": truncate_for_attr(state.query, 100),
        "previous_observations_count": len(state.all_observations),
    }) as span:

        logger.info(
            "PLANNER_CREATING_PLAN",
            query=truncate(state.query, 80),
            iteration=iteration,
            prev_observations=len(state.all_observations),
        )

        # Increment iteration for replanning
        state.plan_iterations += 1

        if state.plan_iterations > state.max_plan_iterations:
            logger.warning(
                "MAX_PLAN_ITERATIONS_REACHED",
                max_iterations=state.max_plan_iterations,
            )
            if span:
                span.set_attributes({"max_iterations_reached": True})
            return state

        # Get completed steps from previous plan (for preservation during ADJUST)
        completed_steps: list[PlanStep] = []
        if state.current_plan:
            completed_steps = state.get_completed_steps()

        if span:
            span.set_attributes({"preserved_steps_count": len(completed_steps)})

        # Format completed steps for prompt
        completed_steps_str = ""
        if completed_steps:
            completed_steps_str = "\n".join(
                f"- [{s.id}] {s.title} (COMPLETED)"
                for s in completed_steps
            )
            completed_steps_str += f"\n\nNOTE: {len(completed_steps)} step(s) already completed. Only output NEW steps."
        else:
            completed_steps_str = "(No completed steps - this is the initial plan)"

        # Format previous observations
        observations_str = ""
        if state.all_observations:
            observations_str = "\n\n---\n\n".join(
                f"**Step {i + 1}:**\n{obs}" for i, obs in enumerate(state.all_observations)
            )
        else:
            observations_str = "(No previous observations)"

        # Format reflector feedback
        reflector_feedback = ""
        if state.last_reflection:
            reflector_feedback = (
                f"Decision: {state.last_reflection.decision.value}\n"
                f"Reasoning: {state.last_reflection.reasoning}"
            )
            if state.last_reflection.suggested_changes:
                reflector_feedback += (
                    "\nSuggested changes:\n"
                    + "\n".join(f"- {c}" for c in state.last_reflection.suggested_changes)
                )

        # Get step limits and guidance from per-depth config
        depth = state.resolve_depth()
        step_limits = get_step_limits(depth)
        step_guidance = step_limits.prompt_guidance or ""

        # Build messages
        messages = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": PLANNER_USER_PROMPT.format(
                    query=state.query,
                    min_steps=step_limits.min,
                    max_steps=step_limits.max,
                    step_prompt_guidance=step_guidance,
                    background_results=state.background_investigation_results
                    or "(No background investigation)",
                    file_context=state.get_file_context_for_prompt(max_chars=10_000) or "(No uploaded files)",
                    completed_steps=completed_steps_str,
                    all_observations=observations_str,
                    reflector_feedback=reflector_feedback or "(First planning iteration)",
                    iteration=state.plan_iterations,
                ),
            },
        ]

        try:
            response = await llm.complete(
                messages=messages,
                tier=ModelTier.ANALYTICAL,
                endpoint_override=get_endpoint_override(state, ModelTier.ANALYTICAL),
                structured_output=PlanOutput,
            )

            output = response.structured or PlanOutput.model_validate_json(response.content)

            # Convert LLM output to new steps, skipping any that match completed step IDs
            completed_ids = {s.id for s in completed_steps}
            new_steps = [
                PlanStep(
                    id=step.id,
                    title=step.title,
                    description=step.description,
                    step_type=StepType(step.step_type),
                    needs_search=step.needs_search,
                    status=StepStatus(step.status) if step.status != "pending" else StepStatus.PENDING,
                )
                for step in output.steps
                if step.id not in completed_ids  # Don't duplicate completed steps
            ]

            # Enforce step limits - truncate if LLM exceeds max
            max_new_steps = step_limits.max - len(completed_steps)
            if len(new_steps) > max_new_steps:
                logger.warning(
                    "PLANNER_STEPS_EXCEEDED_LIMIT",
                    returned_steps=len(new_steps),
                    completed_steps=len(completed_steps),
                    max_allowed=step_limits.max,
                    truncating_to=max_new_steps,
                )
                new_steps = new_steps[:max_new_steps]

            # Merge: completed steps (preserved) + new steps from LLM
            final_steps = completed_steps + new_steps

            state.current_plan = Plan(
                id=output.id or str(uuid4()),
                title=output.title,
                thought=output.thought,
                steps=final_steps,
                has_enough_context=output.has_enough_context,
                iteration=state.plan_iterations,
            )

            # Set step index to first non-completed step (resume from where we left off)
            state.current_step_index = len(completed_steps)

            # Set output attributes for trace
            if span:
                span.set_attributes({
                    ATTR_PLAN_STEPS_COUNT: len(final_steps),
                    ATTR_PLAN_THOUGHT: truncate_for_attr(output.thought, 200),
                    "plan.title": truncate_for_attr(output.title, 100),
                    "plan.new_steps_count": len(new_steps),
                    "plan.has_enough_context": output.has_enough_context,
                })

            # Log step details
            step_summaries = [f"{s.step_type.value}:{truncate(s.title, 30)}" for s in final_steps[:5]]
            logger.info(
                "PLAN_CREATED",
                title=truncate(output.title, 60),
                thought=truncate(output.thought, 100),
                total_steps=len(final_steps),
                preserved_steps=len(completed_steps),
                new_steps=len(new_steps),
                step_summaries=step_summaries,
                has_enough_context=output.has_enough_context,
            )

        except Exception as e:
            logger.error(
                "PLANNER_ERROR",
                error_type=type(e).__name__,
                error=str(e)[:200],
            )
            if span:
                span.set_attributes({
                    "error": str(e)[:200],
                    "error_type": type(e).__name__,
                })
            # Create minimal fallback plan
            state.current_plan = Plan(
                id=str(uuid4()),
                title="Research Plan",
                thought=f"Fallback plan due to error: {e}",
                steps=[
                    PlanStep(
                        id="step-1",
                        title="Research the topic",
                        description=f"Search for information about: {state.query}",
                        step_type=StepType.RESEARCH,
                        needs_search=True,
                    )
                ],
                iteration=state.plan_iterations,
            )
            state.current_step_index = 0

        return state


# =============================================================================
# Source-Aware Planner (007-enterprise-data-sources, T037)
# =============================================================================


class SourceHintOutput(BaseModel):
    """Output schema for a source hint."""

    source_name: str
    source_type: str
    priority: int = 2
    query_hint: str | None = None
    reasoning: str | None = None


class SourceAwarePlanStepOutput(BaseModel):
    """Output schema for a plan step with source hints."""

    id: str
    title: str
    description: str
    step_type: str
    needs_search: bool
    source_hints: list[SourceHintOutput] = Field(default_factory=list)
    exclude_sources: list[str] = Field(default_factory=list)
    status: str = "pending"


class SourceAwarePlanOutput(BaseModel):
    """Output schema for source-aware planner."""

    id: str
    title: str
    thought: str
    has_enough_context: bool = False
    steps: list[SourceAwarePlanStepOutput]


async def run_source_aware_planner(state: ResearchState, llm: LLMClient) -> ResearchState:
    """Run the source-aware planner that leverages data landscape.

    Creates research plans with source hints based on:
    - Data landscape from background discovery
    - User's enabled enterprise data sources
    - Query characteristics

    Args:
        state: Current research state with optional data_landscape.
        llm: LLM client for completions.

    Returns:
        Updated state with source-aware research plan.
    """
    # Check if source-aware planning is enabled
    from deep_research.core.app_config import get_app_config

    config = get_app_config()
    source_routing: dict[str, Any] = getattr(config, "source_routing", {})

    # Fall back to basic planner if source routing disabled or no landscape
    if not source_routing.get("enabled", True):
        logger.debug("Source routing disabled, using basic planner")
        return await run_planner(state, llm)

    # Use plan_iterations + 1 for span naming
    iteration = state.plan_iterations + 1
    span_name = research_span_name(PHASE_PLAN, "source_aware_planner", iteration=iteration)

    async with safe_tool_span(span_name, SpanType.AGENT, {
        ATTR_PLAN_ITERATION: iteration,
        "query": truncate_for_attr(state.query, 100),
        "has_data_landscape": state.data_landscape is not None,
        "previous_observations_count": len(state.all_observations),
    }) as span:

        logger.info(
            "SOURCE_AWARE_PLANNER_CREATING_PLAN",
            query=truncate(state.query, 80),
            iteration=iteration,
            has_landscape=state.data_landscape is not None,
            prev_observations=len(state.all_observations),
        )

        # Increment iteration for replanning
        state.plan_iterations += 1

        if state.plan_iterations > state.max_plan_iterations:
            logger.warning(
                "MAX_PLAN_ITERATIONS_REACHED",
                max_iterations=state.max_plan_iterations,
            )
            if span:
                span.set_attributes({"max_iterations_reached": True})
            return state

        # Get completed steps from previous plan
        completed_steps: list[PlanStep] = []
        if state.current_plan:
            completed_steps = state.get_completed_steps()

        if span:
            span.set_attributes({"preserved_steps_count": len(completed_steps)})

        # Format completed steps for prompt
        completed_steps_str = ""
        if completed_steps:
            completed_steps_str = "\n".join(
                f"- [{s.id}] {s.title} (COMPLETED)"
                for s in completed_steps
            )
            completed_steps_str += f"\n\nNOTE: {len(completed_steps)} step(s) already completed. Only output NEW steps."
        else:
            completed_steps_str = "(No completed steps - this is the initial plan)"

        # Format previous observations
        observations_str = ""
        if state.all_observations:
            observations_str = "\n\n---\n\n".join(
                f"**Step {i + 1}:**\n{obs}" for i, obs in enumerate(state.all_observations)
            )
        else:
            observations_str = "(No previous observations)"

        # Format reflector feedback
        reflector_feedback = ""
        if state.last_reflection:
            reflector_feedback = (
                f"Decision: {state.last_reflection.decision.value}\n"
                f"Reasoning: {state.last_reflection.reasoning}"
            )
            if state.last_reflection.suggested_changes:
                reflector_feedback += (
                    "\nSuggested changes:\n"
                    + "\n".join(f"- {c}" for c in state.last_reflection.suggested_changes)
                )

        # Get step limits and guidance
        depth = state.resolve_depth()
        step_limits = get_step_limits(depth)
        step_guidance = step_limits.prompt_guidance or ""

        # Get data landscape summary for prompt
        data_landscape_str = "(No enterprise data sources available)"
        if state.data_landscape and hasattr(state.data_landscape, "to_planner_summary"):
            data_landscape_str = state.data_landscape.to_planner_summary()

        # Select prompt based on landscape availability
        if state.data_landscape and state.data_landscape.discovery_results:
            user_prompt = SOURCE_AWARE_PLANNER_USER_PROMPT
            system_prompt = SOURCE_AWARE_PLANNER_SYSTEM_PROMPT
        else:
            user_prompt = SOURCE_AWARE_PLANNER_NO_LANDSCAPE_PROMPT
            system_prompt = SOURCE_AWARE_PLANNER_SYSTEM_PROMPT

        # Build messages
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": user_prompt.format(
                    query=state.query,
                    min_steps=step_limits.min,
                    max_steps=step_limits.max,
                    step_prompt_guidance=step_guidance,
                    data_landscape=data_landscape_str,
                    background_results=state.background_investigation_results
                    or "(No background investigation)",
                    file_context=state.get_file_context_for_prompt(max_chars=10_000) or "(No uploaded files)",
                    completed_steps=completed_steps_str,
                    all_observations=observations_str,
                    reflector_feedback=reflector_feedback or "(First planning iteration)",
                    iteration=state.plan_iterations,
                ),
            },
        ]

        try:
            response = await llm.complete(
                messages=messages,
                tier=ModelTier.ANALYTICAL,
                endpoint_override=get_endpoint_override(state, ModelTier.ANALYTICAL),
                structured_output=SourceAwarePlanOutput,
            )

            if response.structured:
                output = response.structured
            else:
                output = SourceAwarePlanOutput.model_validate_json(response.content)

            # Convert LLM output to PlanStep objects (compatible with existing state)
            completed_ids = {s.id for s in completed_steps}
            new_steps = [
                PlanStep(
                    id=step.id,
                    title=step.title,
                    description=step.description,
                    step_type=StepType(step.step_type),
                    needs_search=step.needs_search,
                    status=StepStatus(step.status) if step.status != "pending" else StepStatus.PENDING,
                )
                for step in output.steps
                if step.id not in completed_ids
            ]

            # Store source hints in state for researcher to use
            # Store as a dict mapping step_id -> source_hints
            source_hints_map: dict[str, list[dict[str, Any]]] = {}
            for step in output.steps:
                if step.source_hints:
                    source_hints_map[step.id] = [
                        {
                            "source_name": h.source_name,
                            "source_type": h.source_type,
                            "priority": h.priority,
                            "query_hint": h.query_hint,
                            "reasoning": h.reasoning,
                        }
                        for h in step.source_hints
                    ]

            # Store in phase results for researcher access
            state.add_phase_result("source_hints", source_hints_map)

            # Enforce step limits
            max_new_steps = step_limits.max - len(completed_steps)
            if len(new_steps) > max_new_steps:
                logger.warning(
                    "SOURCE_AWARE_PLANNER_STEPS_EXCEEDED",
                    returned_steps=len(new_steps),
                    max_allowed=max_new_steps,
                )
                new_steps = new_steps[:max_new_steps]

            # Merge completed + new steps
            final_steps = completed_steps + new_steps

            state.current_plan = Plan(
                id=output.id or str(uuid4()),
                title=output.title,
                thought=output.thought,
                steps=final_steps,
                has_enough_context=output.has_enough_context,
                iteration=state.plan_iterations,
            )

            state.current_step_index = len(completed_steps)

            if span:
                span.set_attributes({
                    ATTR_PLAN_STEPS_COUNT: len(final_steps),
                    ATTR_PLAN_THOUGHT: truncate_for_attr(output.thought, 200),
                    "plan.title": truncate_for_attr(output.title, 100),
                    "plan.new_steps_count": len(new_steps),
                    "plan.has_enough_context": output.has_enough_context,
                    "plan.steps_with_hints": len(source_hints_map),
                })

            logger.info(
                "SOURCE_AWARE_PLAN_CREATED",
                title=truncate(output.title, 60),
                thought=truncate(output.thought, 100),
                total_steps=len(final_steps),
                steps_with_hints=len(source_hints_map),
                has_enough_context=output.has_enough_context,
            )

        except Exception as e:
            logger.error(
                "SOURCE_AWARE_PLANNER_ERROR",
                error_type=type(e).__name__,
                error=str(e)[:200],
            )
            if span:
                span.set_attributes({
                    "error": str(e)[:200],
                    "error_type": type(e).__name__,
                })
            # Fall back to basic planner on error
            logger.info("Falling back to basic planner due to error")
            return await run_planner(state, llm)

        return state
