"""
Centralized routing policy for multi-agent workflow.

This module contains pure functions that make routing decisions based on state,
ensuring consistent and testable deadlock prevention logic.
"""

import time
from typing import Dict, Any, Tuple, Optional
from enum import Enum

from .grounding import VerificationLevel, FactualityReport
from .logging import get_logger

logger = get_logger(__name__)


class TerminationReason(Enum):
    """Reasons for terminating the workflow."""
    FACT_CHECK_LOOP_LIMIT = "fact_check_loop_limit_reached"
    TOTAL_STEP_LIMIT = "total_step_limit_reached" 
    WALL_CLOCK_LIMIT = "wall_clock_time_limit_reached"
    NO_PROGRESS_LIMIT = "no_progress_limit_reached"
    EMERGENCY_BRAKE = "emergency_circuit_breaker_triggered"
    QUALITY_ACHIEVED = "factuality_quality_achieved"
    PLAN_COMPLETE = "plan_complete_no_further_work"


class ProgressMetrics:
    """Tracks progress between fact checking iterations."""
    
    def __init__(self, state: Dict[str, Any]):
        self.current_sources = len(state.get("search_results", []))
        self.current_citations = len(state.get("citations", []))
        self.current_factuality = state.get("factuality_score", 0.0)
        self.current_ungrounded = state.get("ungrounded_claims", 0)
        
    def has_progress_since(self, previous_metrics: Optional['ProgressMetrics'], min_delta: float = 0.05) -> Tuple[bool, str]:
        """Check if meaningful progress has been made since previous iteration."""
        if previous_metrics is None:
            return True, "first_iteration"
            
        # Check for new sources or citations
        if (self.current_sources > previous_metrics.current_sources or 
            self.current_citations > previous_metrics.current_citations):
            return True, "new_evidence_added"
            
        # Check for factuality improvement
        factuality_delta = self.current_factuality - previous_metrics.current_factuality
        if factuality_delta >= min_delta:
            return True, f"factuality_improved_by_{factuality_delta:.3f}"
            
        # Check for reduction in ungrounded claims
        if self.current_ungrounded < previous_metrics.current_ungrounded:
            return True, "ungrounded_claims_reduced"
            
        return False, "no_meaningful_progress"


def should_terminate_workflow(state: Dict[str, Any]) -> Tuple[bool, TerminationReason, str]:
    """
    Determine if workflow should terminate immediately.
    
    This is the FIRST check - if this returns True, workflow MUST terminate
    regardless of factuality scores or other considerations.
    
    Returns: (should_terminate, reason, explanation)
    """
    
    # Check fact check loop limit (HARD LIMIT - cannot be bypassed)
    fact_check_loops = state.get("fact_check_loops", 0)
    max_fact_check_loops = state.get("max_fact_check_loops", 2)
    
    if fact_check_loops >= max_fact_check_loops:
        explanation = f"Fact check loops exhausted: {fact_check_loops}/{max_fact_check_loops}"
        logger.warning(f"TERMINATION: {explanation}")
        return True, TerminationReason.FACT_CHECK_LOOP_LIMIT, explanation
    
    # Check total workflow steps (emergency circuit breaker)
    total_steps = state.get("total_workflow_steps", 0)
    max_total_steps = state.get("max_total_steps", 50)
    
    if total_steps >= max_total_steps:
        explanation = f"Total workflow steps exhausted: {total_steps}/{max_total_steps}"
        logger.error(f"EMERGENCY TERMINATION: {explanation}")
        return True, TerminationReason.TOTAL_STEP_LIMIT, explanation
    
    # Check wall clock time limit
    start_time = state.get("workflow_start_time")
    max_wall_clock_seconds = state.get("max_wall_clock_seconds", 120)
    
    if start_time:
        elapsed = time.time() - start_time
        if elapsed >= max_wall_clock_seconds:
            explanation = f"Wall clock time limit exceeded: {elapsed:.1f}s/{max_wall_clock_seconds}s"
            logger.warning(f"TERMINATION: {explanation}")
            return True, TerminationReason.WALL_CLOCK_LIMIT, explanation
    
    # Check for repeated no-progress cycles
    no_progress_count = state.get("consecutive_no_progress_cycles", 0)
    max_no_progress = state.get("max_no_progress_cycles", 1)  # CRITICAL FIX: More aggressive
    
    if no_progress_count >= max_no_progress:
        explanation = f"No progress limit reached: {no_progress_count}/{max_no_progress} cycles"
        logger.warning(f"TERMINATION: {explanation}")
        return True, TerminationReason.NO_PROGRESS_LIMIT, explanation
    
    # CRITICAL FIX: Emergency brake for structural errors
    error_count = state.get("structural_error_count", 0)
    if error_count >= 3:  # If we've seen 3+ structural errors, force stop
        explanation = f"Repeated structural errors detected: {error_count} errors"
        logger.error(f"EMERGENCY TERMINATION: {explanation}")
        return True, TerminationReason.EMERGENCY_BRAKE, explanation
    
    # CRITICAL FIX: Check for repeated same steps (infinite loop detection)
    last_steps = state.get("last_executed_steps", [])
    if len(last_steps) >= 5:  # Look at last 5 steps
        if len(set(last_steps[-5:])) <= 2:  # If only 1-2 unique steps in last 5
            explanation = f"Infinite loop detected - repeating steps: {last_steps[-5:]}"
            logger.error(f"EMERGENCY TERMINATION: {explanation}")
            return True, TerminationReason.EMERGENCY_BRAKE, explanation
    
    return False, None, ""


def should_request_more_research(
    state: Dict[str, Any], 
    report: FactualityReport,
    progress_metrics: ProgressMetrics
) -> Tuple[bool, str]:
    """
    Determine if more research should be requested.
    
    Only called AFTER should_terminate_workflow returns False.
    
    Returns: (should_request, reason)
    """
    
    # Check if we have remaining loops
    fact_check_loops = state.get("fact_check_loops", 0)
    max_fact_check_loops = state.get("max_fact_check_loops", 2)
    remaining_loops = max_fact_check_loops - fact_check_loops
    
    if remaining_loops <= 0:
        return False, "no_remaining_fact_check_loops"
    
    # CRITICAL FIX: Check if additional research was explicitly disabled (placeholder content)
    if state.get("skip_additional_research", False):
        return False, "additional_research_disabled_placeholder_content"
    
    # CRITICAL FIX: Memory protection - check memory before allowing more research
    try:
        import psutil
        current_mb = psutil.Process().memory_info().rss / 1024 / 1024
        if current_mb > 1500:  # Conservative threshold
            logger.warning(f"Memory threshold exceeded ({current_mb:.0f}MB) - stopping research cycles")
            return False, f"memory_protection_prevents_research_{current_mb:.0f}MB"
    except ImportError:
        pass  # Continue without memory check
    
    # CRITICAL FIX: Check if we made any real progress (new observations)
    # Only check this on subsequent iterations, not the first one
    current_obs_count = len(state.get("observations", []))
    previous_obs_count = state.get("previous_observations_count")
    
    if previous_obs_count is not None and current_obs_count <= previous_obs_count:
        logger.warning(f"No new observations added ({current_obs_count} vs {previous_obs_count}) - stopping research cycles")
        return False, "no_new_observations_stopping_cycles"
    
    # CRITICAL FIX: Check ungrounded claims FIRST before factuality threshold
    # High ungrounded claims should trigger research even if overall factuality is acceptable
    if report.total_claims > 0:
        ungrounded_ratio = report.ungrounded_claims / report.total_claims
        if ungrounded_ratio >= 0.4:  # 40% or more ungrounded
            return True, f"high_ungrounded_ratio_{ungrounded_ratio:.2f}_remaining_loops_{remaining_loops}"
    
    # Check factuality score threshold
    verification_level = state.get("verification_level", VerificationLevel.MODERATE)
    min_factuality = _get_min_factuality_threshold(verification_level)
    
    if report.overall_factuality_score < min_factuality:
        return True, f"low_factuality_{report.overall_factuality_score:.2f}_vs_{min_factuality}_remaining_loops_{remaining_loops}"
    
    # Check if progress is possible (have we tried this before?)
    previous_metrics = state.get("previous_progress_metrics")
    if previous_metrics:
        has_progress, progress_reason = progress_metrics.has_progress_since(previous_metrics)
        if not has_progress:
            # No progress = stop requesting research regardless of factuality
            logger.warning(f"No progress detected: {progress_reason} - stopping research cycle")
            return False, f"no_progress_detected_{progress_reason}"
    
    # If factuality is acceptable and ungrounded ratio is low, no need for more research
    if report.overall_factuality_score >= min_factuality:
        return False, f"factuality_threshold_met_{report.overall_factuality_score:.2f}>={min_factuality}"
    
    return False, "factuality_acceptable"


def should_replan(state: Dict[str, Any], report: FactualityReport) -> Tuple[bool, str]:
    """
    Determine if replanning is needed. 
    
    CRITICAL FIX: Never regenerate completed plans to prevent memory explosion.
    
    Returns: (should_replan, reason)
    """
    
    plan = state.get("current_plan")
    if not plan:
        return False, "no_current_plan"
    
    # CRITICAL FIX: Never regenerate completed plans - this caused the memory explosion
    plan_complete = bool(hasattr(plan, "is_complete") and plan.is_complete())
    if plan_complete:
        # Accept low factuality rather than regenerating and causing memory explosion
        return False, f"plan_complete_accepting_factuality_{report.overall_factuality_score:.2f}"
    
    # Only allow replanning during early iterations (reduced from 3 to 2)
    plan_iterations = state.get("plan_iterations", 0)
    max_plan_iterations = min(state.get("max_plan_iterations", 3), 2)  # Hard cap at 2
    
    if plan_iterations >= max_plan_iterations:
        return False, f"plan_iteration_limit_{plan_iterations}/{max_plan_iterations}"
    
    # Only replan if plan quality is VERY poor AND we haven't executed much
    completed_steps = len(state.get("completed_steps", []))
    if completed_steps > 2:
        return False, f"too_much_work_completed_{completed_steps}_steps"
    
    # Memory protection: check if we should avoid replanning due to memory pressure
    try:
        import psutil
        current_mb = psutil.Process().memory_info().rss / 1024 / 1024
        if current_mb > 1200:  # Conservative threshold
            return False, f"memory_protection_prevents_replanning_{current_mb:.0f}MB"
    except ImportError:
        pass  # Continue without memory check
    
    # DISABLED: The original condition that caused memory explosion
    # This line caused 14→17 point plan regeneration and memory death
    # if plan_complete and report.overall_factuality_score < 0.6:
    #     return True, f"plan_complete_but_low_factuality_{report.overall_factuality_score:.2f}"
    
    return False, "replanning_disabled_for_memory_safety"


def determine_next_node(state: Dict[str, Any], report: FactualityReport) -> Tuple[str, str]:
    """
    Main routing function that determines the next node.
    
    This function implements the complete routing logic with proper precedence:
    1. Check for mandatory termination conditions
    2. Check agent failure rates and avoid problematic agents
    3. Check for quality degradation acceptance
    4. Check for progress-based research requests  
    5. Check for replanning needs
    6. Default to reporter
    
    Returns: (next_node, reasoning)
    """
    
    # Step 1: Check for mandatory termination (CANNOT be bypassed)
    should_terminate, termination_reason, explanation = should_terminate_workflow(state)
    if should_terminate:
        return "reporter", f"TERMINATE_{termination_reason.value}_{explanation}"
    
    # Step 2: Track quality metrics for this iteration
    state = track_quality_degradation(state, "factuality", report.overall_factuality_score, 0.6)
    
    # Step 3: Check if we should accept degraded quality to prevent loops
    should_accept, accept_reason = should_accept_degraded_quality(
        state, "factuality", report.overall_factuality_score, 0.6
    )
    if should_accept:
        return "reporter", f"ACCEPT_DEGRADED_QUALITY_{accept_reason}"
    
    # Step 4: Calculate current progress metrics
    current_metrics = ProgressMetrics(state)
    
    # Step 5: Check agent failure rates before routing
    avoid_researcher, avoid_reason = should_avoid_agent(state, "researcher")
    avoid_planner, planner_reason = should_avoid_agent(state, "planner")
    
    # Step 6: Check if more research is warranted (unless researcher is failing)
    if not avoid_researcher:
        should_research, research_reason = should_request_more_research(state, report, current_metrics)
        if should_research:
            # Store current metrics for next iteration comparison
            state["previous_progress_metrics"] = current_metrics
            # CRITICAL FIX: Store observations count to track progress
            state["previous_observations_count"] = len(state.get("observations", []))
            return "researcher", f"REQUEST_RESEARCH_{research_reason}"
    else:
        logger.warning(f"Skipping researcher due to high failure rate: {avoid_reason}")
    
    # Step 7: Check if replanning is needed (unless planner is failing)
    if not avoid_planner:
        should_replan_flag, replan_reason = should_replan(state, report)
        if should_replan_flag:
            state = increment_quality_attempts(state)
            return "planner", f"REPLAN_{replan_reason}"
    else:
        logger.warning(f"Skipping planner due to high failure rate: {planner_reason}")
    
    # Step 8: Default to completion
    return "reporter", f"COMPLETE_factuality_{report.overall_factuality_score:.2f}_acceptable"


def _get_min_factuality_threshold(verification_level: VerificationLevel) -> float:
    """Get minimum factuality threshold for verification level."""
    if verification_level == VerificationLevel.STRICT:
        return 0.9
    elif verification_level == VerificationLevel.MODERATE:
        return 0.6  # CRITICAL FIX: Reduced from 0.7 to prevent unnecessary cycles
    else:  # LENIENT
        return 0.5


def update_progress_tracking(state: Dict[str, Any], had_progress: bool) -> Dict[str, Any]:
    """Update progress tracking counters in state."""
    if had_progress:
        state["consecutive_no_progress_cycles"] = 0
    else:
        current_count = state.get("consecutive_no_progress_cycles", 0)
        state["consecutive_no_progress_cycles"] = current_count + 1
    
    return state


def initialize_workflow_limits(state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Initialize workflow limits and counters for a new request."""
    
    # Initialize counters
    state.setdefault("fact_check_loops", 0)
    state.setdefault("total_workflow_steps", 0)
    state.setdefault("consecutive_no_progress_cycles", 0)
    
    # CRITICAL FIX: Set VERY aggressive limits to prevent recursion limit hits
    state.setdefault("max_fact_check_loops", config.get("research", {}).get("max_fact_check_loops", 1))  # Reduced from 2
    state.setdefault("max_total_steps", config.get("workflow", {}).get("max_total_steps", 8))  # VERY aggressive limit
    
    # Handle both workflow and timeouts config for wall clock limits
    timeout_config = config.get("timeouts", {})
    workflow_config = config.get("workflow", {})
    default_timeout = timeout_config.get("total_workflow_timeout", workflow_config.get("max_wall_clock_seconds", 60))
    state.setdefault("max_wall_clock_seconds", default_timeout)
    
    state.setdefault("max_no_progress_cycles", config.get("workflow", {}).get("max_no_progress_cycles", 1))  # Reduced from 2
    
    # Initialize structural error tracking
    state.setdefault("structural_error_count", 0)
    state.setdefault("last_executed_steps", [])
    
    # Set start time
    state.setdefault("workflow_start_time", time.time())
    
    return state


def track_structural_error(state: Dict[str, Any], error_message: str) -> Dict[str, Any]:
    """Track a structural error in the workflow state."""
    current_count = state.get("structural_error_count", 0)
    state["structural_error_count"] = current_count + 1
    
    logger.error(f"Structural error #{current_count + 1}: {error_message}")
    
    # Keep track of recent structural errors for debugging
    errors = state.get("recent_structural_errors", [])
    errors.append({
        "error": error_message,
        "timestamp": time.time(),
        "count": current_count + 1
    })
    # Keep only last 5 errors
    state["recent_structural_errors"] = errors[-5:]
    
    return state


def track_step_execution(state: Dict[str, Any], step_id: str) -> Dict[str, Any]:
    """
    Track step executions to detect infinite loops.
    
    Args:
        state: Current workflow state
        step_id: ID of the step being executed
        
    Returns:
        Updated state with step execution tracking
    """
    # Initialize tracking if needed
    if "step_execution_counts" not in state:
        state["step_execution_counts"] = {}
    
    # Increment execution count for this step
    current_count = state["step_execution_counts"].get(step_id, 0)
    state["step_execution_counts"][step_id] = current_count + 1
    
    # Warn if step is executing repeatedly
    if current_count >= 2:
        logger.warning(f"Step {step_id} has executed {current_count + 1} times - possible infinite loop")
        
        # Trigger structural error if too many repetitions
        if current_count >= 3:
            state = track_structural_error(
                state, 
                f"Step {step_id} executed {current_count + 1} times - infinite loop detected"
            )
    
    return state


def should_skip_failed_step(state: Dict[str, Any], step_id: str, max_retries: int = 3) -> bool:
    """
    Determine if a failed step should be skipped after multiple attempts.
    
    Args:
        state: Current workflow state
        step_id: ID of the step to check
        max_retries: Maximum number of retries before skipping
    
    Returns:
        True if step should be skipped, False otherwise
    """
    # Track failed step attempts
    failed_attempts = state.get("failed_step_attempts", {})
    current_attempts = failed_attempts.get(step_id, 0)
    
    if current_attempts >= max_retries:
        logger.warning(f"Step {step_id} has failed {current_attempts} times - will skip")
        return True
    
    return False


def increment_failed_attempts(state: Dict[str, Any], step_id: str) -> Dict[str, Any]:
    """
    Track failed attempts for circuit breaker.
    
    Args:
        state: Current workflow state
        step_id: ID of the step that failed
    
    Returns:
        Updated state with incremented failure count
    """
    failed_attempts = state.get("failed_step_attempts", {})
    failed_attempts[step_id] = failed_attempts.get(step_id, 0) + 1
    state["failed_step_attempts"] = failed_attempts
    
    logger.info(f"Step {step_id} has now failed {failed_attempts[step_id]} time(s)")
    
    return state


def track_executed_step(state: Dict[str, Any], step_name: str) -> Dict[str, Any]:
    """Track an executed step to detect infinite loops."""
    steps = state.get("last_executed_steps", [])
    steps.append(step_name)
    
    # Keep only last 10 steps for loop detection
    state["last_executed_steps"] = steps[-10:]
    
    logger.debug(f"Executed step: {step_name}, recent steps: {steps[-5:]}")
    
    return state


def track_agent_failure(state: Dict[str, Any], agent_name: str, error_type: str, error_message: str) -> Dict[str, Any]:
    """Track agent failures to build failure memory."""
    failure_history = state.get("agent_failure_history", {})
    agent_failures = failure_history.get(agent_name, [])
    
    failure_record = {
        "error_type": error_type,
        "error_message": error_message[:200],  # Truncate long messages
        "timestamp": time.time(),
        "failure_count": len(agent_failures) + 1
    }
    
    agent_failures.append(failure_record)
    # Keep only last 5 failures per agent
    failure_history[agent_name] = agent_failures[-5:]
    state["agent_failure_history"] = failure_history
    
    logger.warning(f"Agent failure tracked: {agent_name} - {error_type}: {error_message[:100]}")
    
    return state


def get_agent_failure_rate(state: Dict[str, Any], agent_name: str, time_window: float = 300) -> float:
    """Calculate agent failure rate within time window (seconds)."""
    failure_history = state.get("agent_failure_history", {})
    agent_failures = failure_history.get(agent_name, [])
    
    if not agent_failures:
        return 0.0
    
    current_time = time.time()
    recent_failures = [
        f for f in agent_failures 
        if current_time - f["timestamp"] <= time_window
    ]
    
    # Calculate failure rate as failures per minute
    if time_window > 0:
        return len(recent_failures) / (time_window / 60.0)
    else:
        return 0.0


def should_avoid_agent(state: Dict[str, Any], agent_name: str, max_failure_rate: float = 2.0) -> Tuple[bool, str]:
    """Determine if an agent should be avoided due to high failure rate."""
    failure_rate = get_agent_failure_rate(state, agent_name)
    
    if failure_rate >= max_failure_rate:
        return True, f"agent_{agent_name}_failure_rate_{failure_rate:.1f}_exceeds_{max_failure_rate}"
    
    return False, ""


def track_quality_degradation(state: Dict[str, Any], metric_name: str, current_value: float, target_value: float) -> Dict[str, Any]:
    """Track quality metrics to detect degradation patterns."""
    quality_history = state.get("quality_degradation_history", {})
    metric_history = quality_history.get(metric_name, [])
    
    degradation_record = {
        "current_value": current_value,
        "target_value": target_value,
        "degradation": max(0, target_value - current_value),
        "timestamp": time.time()
    }
    
    metric_history.append(degradation_record)
    # Keep only last 10 measurements
    quality_history[metric_name] = metric_history[-10:]
    state["quality_degradation_history"] = quality_history
    
    return state


def should_accept_degraded_quality(state: Dict[str, Any], metric_name: str, current_value: float, target_value: float, max_attempts: int = 3) -> Tuple[bool, str]:
    """Determine if degraded quality should be accepted to prevent infinite loops."""
    attempts = state.get("quality_improvement_attempts", 0)
    
    if attempts >= max_attempts:
        degradation = target_value - current_value
        return True, f"quality_degradation_accepted_after_{attempts}_attempts_degradation_{degradation:.3f}"
    
    # Check if we're in a degradation pattern
    quality_history = state.get("quality_degradation_history", {})
    metric_history = quality_history.get(metric_name, [])
    
    if len(metric_history) >= 3:
        # Check if quality has been consistently poor
        recent_values = [m["current_value"] for m in metric_history[-3:]]
        if all(v < target_value * 0.8 for v in recent_values):  # All below 80% of target
            return True, f"persistent_quality_degradation_pattern_detected"
    
    return False, ""


def increment_quality_attempts(state: Dict[str, Any]) -> Dict[str, Any]:
    """Increment quality improvement attempt counter."""
    state["quality_improvement_attempts"] = state.get("quality_improvement_attempts", 0) + 1
    return state
