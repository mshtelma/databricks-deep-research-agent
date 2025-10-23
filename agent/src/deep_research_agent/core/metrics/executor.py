"""Execution engine for calculation plans."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Dict, List

from langchain_core.messages import AIMessage

from .. import get_logger
from ..report_generation.models import CalculationContext, Calculation
from .models import CalculationEvent, CalculationPlan, CalculationTask, CalculationProvenance, CalculationValidation
from .sandbox import SafePythonExecutor, ExecutionResult
from .data_context import MetricDataContext
from .katex_formatter import convert_to_katex


logger = get_logger(__name__)


class CalculationExecutor:
    """Execute calculation plans with deterministic fallbacks."""

    def __init__(
        self,
        llm,
        python_runner=None,
        config: Dict[str, Any] = None
    ) -> None:
        self._llm = llm
        self._python_runner = python_runner
        self._config = config or {}
        
        # Initialize sandboxed executor
        # Config can be nested under 'metrics' key or at root level
        metrics_config = self._config.get('metrics', self._config)
        execution_config = metrics_config.get('execution', {})
        self._sandbox = SafePythonExecutor(
            timeout_seconds=execution_config.get('timeout_seconds', 5.0),
            enable_pandas=True
        )
        self._use_sandbox = execution_config.get('enable_sandbox', True)

    async def execute(
        self,
        plan: CalculationPlan,
        context: CalculationContext,
    ) -> tuple[CalculationContext, List[CalculationEvent], List[AIMessage]]:
        """Execute tasks in order and update the calculation context.

        Uses sandboxed Python execution for deterministic calculations with full provenance.
        """
        events: List[CalculationEvent] = []

        # Build mutable mapping for quick updates
        existing_calcs: Dict[str, Calculation] = {
            calc.description: calc for calc in context.calculations
        }
        
        # Create data context for calculations
        data_ctx = MetricDataContext(context.extracted_data)
        
        logger.info(f"[EXECUTOR] Starting execution of {len(plan.tasks)} tasks")
        logger.info(f"[EXECUTOR] Data context: {data_ctx}")

        for task in plan.ordered_tasks():
            logger.debug(f"[EXECUTOR] Executing task: {task.task_id} - {task.description}")
            
            event = CalculationEvent(
                task_id=task.task_id,
                status="pending",
                attempts=0
            )

            # Check if calculation already exists
            if task.output_key in existing_calcs:
                event.status = "existing"
                event.result = existing_calcs[task.output_key].result
                event.attempts = 0
                logger.debug(f"[EXECUTOR] Task {task.task_id} uses existing calculation")
            else:
                # Execute calculation
                if self._use_sandbox and task.code:
                    # Use sandboxed execution for safety
                    result = await self._execute_sandboxed(task, data_ctx)
                    event.status = result.get('status', 'computed')
                    event.result = result.get('result')
                    event.attempts = result.get('attempts', 1)
                    event.message = result.get('error_message')
                    
                    # Check if we need research
                    if result.get('needs_research'):
                        event.needs_research = True
                        event.research_query = result.get('research_query')
                else:
                    # Fallback to placeholder for tasks without code
                    result = self._execute_placeholder(task, context)
                    event.status = "computed"
                    event.result = result

            events.append(event)
            
            # Create Calculation object with full provenance for successful computations
            if event.status == "computed" and event.result is not None:
                # Validate result type before creating Calculation
                if not self._is_valid_result_type(event.result):
                    logger.error(
                        f"[EXECUTOR] Task {task.task_id} produced invalid result type: "
                        f"{type(event.result).__name__}. Expected float, str, or None. "
                        f"Skipping calculation creation."
                    )
                    event.status = "failed"
                    event.message = f"Invalid result type: {type(event.result).__name__}"
                else:
                    calculation = self._build_calculation_with_provenance(
                        task, event, plan, context
                    )
                    if calculation:
                        context.calculations.append(calculation)
                        logger.debug(
                            f"[EXECUTOR] Added calculation: {calculation.description} "
                            f"= {calculation.result}"
                        )
                        # 🔍 DEBUG: Track where calculation results go
                        logger.info(
                            f"[DEBUG_NA_BUG] ✅ Calculation created: "
                            f"task_id={task.task_id}, "
                            f"description={calculation.description}, "
                            f"result={calculation.result}, "
                            f"formula={getattr(calculation, 'formula', 'N/A')}"
                        )
                        logger.info(
                            f"[DEBUG_NA_BUG] 📍 Stored in: context.calculations[{len(context.calculations)-1}]"
                        )
                        logger.info(
                            f"[DEBUG_NA_BUG] ❓ Check: Are observations being updated with this value?"
                        )
        
        # Count statistics
        success_count = sum(1 for e in events if e.status in ["computed", "existing"])
        failure_count = sum(1 for e in events if e.status == "failed")
        research_needed = sum(1 for e in events if getattr(e, 'needs_research', False))

        messages: List[AIMessage] = [
            AIMessage(
                content=(
                    f"CalculationExecutor completed execution: "
                    f"{success_count} successful, {failure_count} failed, "
                    f"{research_needed} need research. "
                    f"Created {len([e for e in events if e.status == 'computed'])} new calculations."
                ),
                name="calculation_executor",
            )
        ]
        return context, events, messages
    
    async def _execute_sandboxed(
        self,
        task: CalculationTask,
        data_ctx: MetricDataContext
    ) -> Dict[str, Any]:
        """Execute a task using sandboxed Python execution.
        
        Args:
            task: Calculation task to execute
            data_ctx: Data context for calculations
        
        Returns:
            Dict with status, result, attempts, error_message
        """
        if not task.code:
            return {
                'status': 'failed',
                'result': None,
                'attempts': 0,
                'error_message': 'No code provided for task'
            }
        
        # Validate code first
        is_valid, validation_error = self._sandbox.validate_code(task.code)
        if not is_valid:
            logger.warning(f"[EXECUTOR] Code validation failed: {validation_error}")
            return {
                'status': 'failed',
                'result': None,
                'attempts': 1,
                'error_message': f"Code validation failed: {validation_error}"
            }
        
        # Execute in sandbox
        logger.debug(f"[EXECUTOR] Executing sandboxed code:\n{task.code}")
        exec_result = self._sandbox.execute(
            code=task.code,
            # FIX: Provide both 'ctx' and 'calc_context' for backward compatibility
            # Some generated code may reference calc_context instead of ctx
            context={'ctx': data_ctx, 'calc_context': data_ctx}
        )
        
        if exec_result.success:
            logger.debug(
                f"[EXECUTOR] Execution successful: {exec_result.result} "
                f"({exec_result.execution_time:.3f}s)"
            )
            return {
                'status': 'computed',
                'result': exec_result.result,
                'attempts': 1,
                'execution_time': exec_result.execution_time
            }

        logger.warning(
            f"[EXECUTOR] Execution failed: {exec_result.error_type} - "
            f"{exec_result.error_message}"
        )

        needs_research = self._is_data_missing_error(exec_result)
        graceful_result = None if exec_result.error_type != "SecurityError" else None
        status = 'failed'

        return {
            'status': status,
            'result': graceful_result,
            'attempts': 1,
            'error_message': f"{exec_result.error_type}: {exec_result.error_message}",
            'needs_research': needs_research,
            'research_query': task.fallback_research_query if needs_research else None
        }
    
    def _is_valid_result_type(self, result: Any) -> bool:
        """Check if result has a valid type for Calculation model.
        
        Args:
            result: The result value to validate
        
        Returns:
            True if result is float, int, str, bool, or None
        """
        # Calculation model expects Union[float, str]
        # We accept int, bool as they're compatible with float
        # We also accept None as it indicates missing data (graceful failure)
        return isinstance(result, (float, int, str, bool, type(None)))
    
    def _is_data_missing_error(self, exec_result: ExecutionResult) -> bool:
        """Check if execution failure was due to missing data.
        
        Args:
            exec_result: Execution result from sandbox
        
        Returns:
            True if error indicates missing data
        """
        if not exec_result.error_message:
            return False
        
        # Check for common missing data indicators
        missing_indicators = [
            'NoneType',
            'KeyError',
            'not found',
            'missing',
            'None'
        ]
        
        error_msg = exec_result.error_message.lower()
        return any(indicator.lower() in error_msg for indicator in missing_indicators)

    def _execute_placeholder(
        self,
        task: CalculationTask,
        context: CalculationContext,
    ) -> Any:
        """Very small deterministic executor supporting trivial formulas.

        This is intentionally basic—when the planner begins emitting richer instructions we will
        replace this with a sandboxed Python runner. The placeholder currently supports ratios and
        deltas referencing numeric inputs provided inline.
        """
        try:
            if task.operation == "ratio" and {"numerator", "denominator"} <= task.inputs.keys():
                numerator = float(task.inputs["numerator"])
                denominator = float(task.inputs["denominator"])
                return numerator / denominator if denominator else math.nan
            if task.operation == "delta" and {"minuend", "subtrahend"} <= task.inputs.keys():
                return float(task.inputs["minuend"]) - float(task.inputs["subtrahend"])
        except Exception:  # pragma: no cover - defensive fallback
            return None
        return task.inputs.get("value")
    
    def _build_calculation_with_provenance(
        self,
        task: CalculationTask,
        event: CalculationEvent,
        plan: CalculationPlan,
        context: CalculationContext
    ) -> Calculation:
        """Build a Calculation object with full provenance tracking.
        
        Args:
            task: The calculation task that was executed
            event: The execution event with results
            plan: The full calculation plan (for dependencies)
            context: The calculation context (for observation tracking)
        
        Returns:
            Calculation object with provenance metadata
        """
        # Get dependencies for this task
        dependencies = list(plan.dependencies.get(task.task_id, []))
        
        # Track observation sources used in this calculation
        observation_ids = self._trace_observation_sources(task, context)
        
        # Generate KaTeX formula if possible
        formula_katex = None
        if task.code:
            formula_katex = convert_to_katex(task.code)
        
        # Create provenance metadata
        provenance = CalculationProvenance(
            observation_ids=observation_ids,
            source_tasks=dependencies,
            code_cell_id=task.task_id,
            notes=f"Executed at {datetime.utcnow().isoformat()}"
        )
        
        # Create validation metadata
        validation = CalculationValidation(
            status="success",
            message="Calculation completed successfully",
            attempts=event.attempts
        )
        
        # Determine unit from task or default
        unit = task.expected_unit if task.expected_unit else "unitless"
        
        # Build the calculation
        calculation = Calculation(
            description=task.description,
            formula=task.code if task.code else str(task.inputs),
            inputs=task.inputs,
            result=event.result,
            unit=unit,
            calculation_id=task.task_id,
            source_tasks=dependencies,
            provenance=provenance,
            validation=validation,
            formula_katex=formula_katex,
            confidence=0.9  # High confidence for sandboxed executions
        )
        
        return calculation
    
    def _trace_observation_sources(
        self,
        task: CalculationTask,
        context: CalculationContext
    ) -> List[str]:
        """Trace which observations contributed to this calculation.
        
        This is a heuristic approach that looks for entities/metrics mentioned
        in the task inputs and matches them to data points.
        
        Args:
            task: The calculation task
            context: The calculation context with extracted data
        
        Returns:
            List of observation IDs (or entity-metric keys)
        """
        observation_sources = []
        
        # Extract entities and metrics from task inputs
        for input_key, input_value in task.inputs.items():
            # Try to match input values to data points
            for dp in context.extracted_data:
                # Simple heuristic: if the value matches and entity/metric appear in task description
                if (dp.value == input_value or 
                    str(input_value).lower() in task.description.lower()):
                    # Create a unique identifier for the observation source
                    obs_id = f"{dp.entity}:{dp.metric}"
                    if obs_id not in observation_sources:
                        observation_sources.append(obs_id)
        
        # If no sources found, try to extract from task description
        if not observation_sources and context.extracted_data:
            # Look for entity names in task description
            for dp in context.extracted_data[:10]:  # Limit to first 10 to avoid noise
                if dp.entity.lower() in task.description.lower():
                    obs_id = f"{dp.entity}:{dp.metric}"
                    if obs_id not in observation_sources:
                        observation_sources.append(obs_id)
        
        return observation_sources[:5]  # Limit to top 5 sources for readability
