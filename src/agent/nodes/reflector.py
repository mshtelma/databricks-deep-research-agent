"""Reflector agent - evaluates research progress and decides next action."""

from typing import Literal

import mlflow
from mlflow.entities import SpanType
from pydantic import BaseModel

from src.agent.prompts.reflector import (
    REFLECTOR_SYSTEM_PROMPT,
    REFLECTOR_USER_PROMPT,
)
from src.agent.state import (
    PlanStep,
    ReflectionDecision,
    ReflectionResult,
    ResearchState,
    StepStatus,
)
from src.core.logging_utils import get_logger, log_agent_decision
from src.core.tracing_constants import (
    ATTR_DECISION,
    ATTR_DECISION_REASONING,
    ATTR_STEP_INDEX,
    PHASE_REFLECT,
    research_span_name,
    truncate_for_attr,
)
from src.services.llm.client import LLMClient
from src.services.llm.types import ModelTier

logger = get_logger(__name__)


class ReflectorOutput(BaseModel):
    """Output schema for Reflector agent with coverage analysis."""

    # Coverage analysis fields (NEW - all have defaults for backward compatibility)
    remaining_topics: list[str] = []
    covered_topics: list[str] = []
    coverage_gaps: list[str] = []

    # Use Literal to constrain allowed values in JSON schema
    decision: Literal["continue", "adjust", "complete"]
    reasoning: str
    suggested_changes: list[str] | None = None


def _format_remaining_steps(state: ResearchState) -> str:
    """Format pending steps for coverage analysis.

    Args:
        state: Current research state with plan.

    Returns:
        Formatted string of remaining (pending) steps.
    """
    if not state.current_plan:
        return "(No plan)"

    remaining = []
    for i, step in enumerate(state.current_plan.steps):
        if step.status == StepStatus.PENDING:
            remaining.append(f"- Step {i + 1}: {step.title}\n  {step.description}")

    return "\n".join(remaining) if remaining else "(All steps completed)"


def _format_source_topics(state: ResearchState) -> str:
    """Format source topics for coverage check.

    Args:
        state: Current research state with sources.

    Returns:
        Formatted string of source titles and snippets.
    """
    if not state.sources:
        return "(No sources yet)"

    topics = []
    for source in state.sources[:15]:  # Limit for prompt size
        if source.title:
            topics.append(f"- {source.title}")
            if source.snippet:
                topics.append(f"  {source.snippet[:100]}...")

    return "\n".join(topics)


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

    # Use 1-based indexing for user-facing span names
    step_number = state.current_step_index + 1
    span_name = research_span_name(PHASE_REFLECT, "reflector", step=step_number)

    with mlflow.start_span(name=span_name, span_type=SpanType.AGENT) as span:
        total_steps = len(state.current_plan.steps)
        span.set_attributes({
            ATTR_STEP_INDEX: step_number,
            "step.total": total_steps,
            "observations_count": len(state.all_observations),
            "sources_count": len(state.sources),
        })

        logger.info(
            "REFLECTOR_EVALUATING",
            step=f"{step_number}/{total_steps}",
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
        current_step: PlanStep | None = state.current_plan.steps[state.current_step_index] if state.current_step_index < total_steps else None

        # Coverage analysis data
        remaining_steps = _format_remaining_steps(state)
        source_topics = _format_source_topics(state)
        min_steps = state.get_min_steps()
        steps_completed = len(state.get_completed_steps())

        messages = [
            {"role": "system", "content": REFLECTOR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": REFLECTOR_USER_PROMPT.format(
                    query=state.query,
                    iteration=state.plan_iterations,
                    plan_summary=plan_summary,
                    remaining_steps=remaining_steps,
                    current_step=current_step_num,
                    total_steps=total_steps,
                    step_title=current_step.title if current_step else "N/A",
                    observation=state.last_observation or "(No observation)",
                    all_observations=observations_str or "(No observations yet)",
                    sources_count=len(state.sources),
                    source_topics=source_topics,
                    min_steps=min_steps,
                    steps_completed=steps_completed,
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

            # Set decision attributes for trace
            span.set_attributes({
                ATTR_DECISION: decision.value,
                ATTR_DECISION_REASONING: truncate_for_attr(output.reasoning, 300),
                "decision.has_suggested_changes": output.suggested_changes is not None and len(output.suggested_changes) > 0,
                "coverage.gaps_count": len(output.coverage_gaps),
                "coverage.covered_topics_count": len(output.covered_topics),
                "coverage.remaining_topics_count": len(output.remaining_topics),
            })

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
            span.set_attributes({
                "error": str(e)[:200],
                "error_type": type(e).__name__,
            })
            # Default to continue on error
            state.last_reflection = ReflectionResult(
                decision=ReflectionDecision.CONTINUE,
                reasoning=f"Default continue due to error: {e}",
            )

        return state
