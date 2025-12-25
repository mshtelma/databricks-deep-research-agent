"""Reflector agent - evaluates research progress and decides next action."""

from typing import Literal

import mlflow
from pydantic import BaseModel

from src.agent.prompts.reflector import (
    REFLECTOR_SYSTEM_PROMPT,
    REFLECTOR_USER_PROMPT,
)
from src.agent.state import ReflectionDecision, ReflectionResult, ResearchState
from src.core.logging_utils import get_logger, log_agent_decision
from src.services.llm.client import LLMClient
from src.services.llm.types import ModelTier

logger = get_logger(__name__)


class ReflectorOutput(BaseModel):
    """Output schema for Reflector agent."""

    # Use Literal to constrain allowed values in JSON schema
    decision: Literal["continue", "adjust", "complete"]
    reasoning: str
    suggested_changes: list[str] | None = None


@mlflow.trace(name="reflector", span_type="AGENT")
async def run_reflector(state: ResearchState, llm: LLMClient) -> ResearchState:
    """Run the Reflector agent to evaluate research progress.

    Args:
        state: Current research state.
        llm: LLM client for completions.

    Returns:
        Updated state with reflection decision.
    """
    if not state.current_plan:
        logger.warning("REFLECTOR_NO_PLAN")
        return state

    logger.info(
        "REFLECTOR_EVALUATING",
        step=f"{state.current_step_index + 1}/{len(state.current_plan.steps)}",
        observations=len(state.all_observations),
        sources=len(state.sources),
    )

    # Format plan summary
    plan_summary = f"Title: {state.current_plan.title}\n"
    plan_summary += f"Thought: {state.current_plan.thought}\n"
    plan_summary += "Steps:\n"
    for i, step in enumerate(state.current_plan.steps):
        status = step.status.value
        plan_summary += f"  {i + 1}. [{status}] {step.title}\n"

    # Format all observations
    observations_str = ""
    if state.all_observations:
        observations_str = "\n\n---\n\n".join(
            f"**Step {i + 1}:**\n{obs}" for i, obs in enumerate(state.all_observations)
        )

    # Current step info
    current_step_num = state.current_step_index + 1
    total_steps = len(state.current_plan.steps)
    step = state.current_plan.steps[state.current_step_index] if state.current_step_index < total_steps else None

    messages = [
        {"role": "system", "content": REFLECTOR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": REFLECTOR_USER_PROMPT.format(
                query=state.query,
                iteration=state.plan_iterations,
                plan_summary=plan_summary,
                current_step=current_step_num,
                total_steps=total_steps,
                step_title=step.title if step else "N/A",
                observation=state.last_observation or "(No observation)",
                all_observations=observations_str or "(No observations yet)",
                sources_count=len(state.sources),
            ),
        },
    ]

    try:
        response = await llm.complete(
            messages=messages,
            tier=ModelTier.SIMPLE,
            structured_output=ReflectorOutput,
        )

        if response.structured:
            output = response.structured
        else:
            output = ReflectorOutput.model_validate_json(response.content)

        # Create reflection result (normalize case as defense-in-depth)
        decision = ReflectionDecision(output.decision.lower())
        result = ReflectionResult(
            decision=decision,
            reasoning=output.reasoning,
            suggested_changes=output.suggested_changes,
        )

        state.last_reflection = result
        state.reflection_history.append(result)

        log_agent_decision(
            logger,
            decision=decision.value,
            reasoning=output.reasoning,
            suggested_changes=output.suggested_changes,
        )

    except Exception as e:
        logger.error(
            "REFLECTOR_ERROR",
            error_type=type(e).__name__,
            error=str(e)[:200],
        )
        # Default to continue on error
        state.last_reflection = ReflectionResult(
            decision=ReflectionDecision.CONTINUE,
            reasoning=f"Default continue due to error: {e}",
        )

    return state
