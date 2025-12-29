"""Planner agent - creates structured research plans."""

from uuid import uuid4

import mlflow
from pydantic import BaseModel

from src.agent.prompts.planner import PLANNER_SYSTEM_PROMPT, PLANNER_USER_PROMPT
from src.agent.state import Plan, PlanStep, ResearchState, StepStatus, StepType
from src.core.logging_utils import get_logger, truncate
from src.services.llm.client import LLMClient
from src.services.llm.types import ModelTier

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


@mlflow.trace(name="planner", span_type="AGENT")
async def run_planner(state: ResearchState, llm: LLMClient) -> ResearchState:
    """Run the Planner agent to create a research plan.

    Args:
        state: Current research state.
        llm: LLM client for completions.

    Returns:
        Updated state with research plan.
    """
    logger.info(
        "PLANNER_CREATING_PLAN",
        query=truncate(state.query, 80),
        iteration=state.plan_iterations + 1,
        prev_observations=len(state.all_observations),
    )

    # Increment iteration for replanning
    state.plan_iterations += 1

    if state.plan_iterations > state.max_plan_iterations:
        logger.warning(
            "MAX_PLAN_ITERATIONS_REACHED",
            max_iterations=state.max_plan_iterations,
        )
        return state

    # Get completed steps from previous plan (for preservation during ADJUST)
    completed_steps: list[PlanStep] = []
    if state.current_plan:
        completed_steps = state.get_completed_steps()

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

    # Build messages
    messages = [
        {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": PLANNER_USER_PROMPT.format(
                query=state.query,
                background_results=state.background_investigation_results
                or "(No background investigation)",
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
            structured_output=PlanOutput,
        )

        if response.structured:
            output = response.structured
        else:
            output = PlanOutput.model_validate_json(response.content)

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
