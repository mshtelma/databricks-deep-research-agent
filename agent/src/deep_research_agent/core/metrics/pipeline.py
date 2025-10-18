"""High-level orchestration for the metric pipeline."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage

from .models import CalculationPlan, MetricSpecBundle
from .state import MetricPipelineState
from .spec_analyzer import MetricSpecAnalyzer
from .planner import CalculationPlanner
from .executor import CalculationExecutor
from .instrumentation import MetricPipelineInstrumentation


class MetricPipeline:
    """Co-ordinates metric specification, planning, and execution.
    
    Features:
    - Caching of plans and results for performance
    - Instrumentation for observability
    - Graceful error handling with fallbacks
    """

    def __init__(
        self,
        reporter_like,
        llm,
        *,
        python_runner=None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._reporter_like = reporter_like
        self._llm = llm
        self._python_runner = python_runner
        self._config = config or {}
        
        # Extract metrics config
        metrics_config = self._config.get('metrics', {})
        
        self._spec_analyzer = MetricSpecAnalyzer(reporter_like)
        codegen_config = metrics_config.get('codegen', {})
        code_llm = getattr(reporter_like, 'codegen_llm', None)
        self._planner = CalculationPlanner(llm, code_llm=code_llm)
        self._executor = CalculationExecutor(
            llm,
            python_runner=python_runner,
            config=metrics_config
        )
        
        # Initialize caching and instrumentation
        performance_config = metrics_config.get('performance', {})
        self._enable_caching = performance_config.get('enable_caching', True)
        self._plan_cache: Dict[str, CalculationPlan] = {}
        self._result_cache: Dict[str, Any] = {}
        self._max_cache_size = performance_config.get('max_cache_size', 100)
        
        # Instrumentation
        enable_detailed_logging = performance_config.get('enable_detailed_logging', False)
        self._instrumentation = MetricPipelineInstrumentation(enable_detailed_logging)
    
    def _compute_hash(self, data: Any) -> str:
        """Compute stable hash for caching."""
        try:
            json_str = json.dumps(data, sort_keys=True, default=str)
            return hashlib.sha256(json_str.encode()).hexdigest()[:16]
        except (TypeError, ValueError):
            # Fallback for non-serializable data
            return hashlib.sha256(str(data).encode()).hexdigest()[:16]
    
    def _get_cached_plan(self, cache_key: str) -> Optional[CalculationPlan]:
        """Get cached plan if available."""
        if not self._enable_caching:
            return None
        
        plan = self._plan_cache.get(cache_key)
        if plan:
            self._instrumentation.record_cache_hit("plan")
            return plan
        
        self._instrumentation.record_cache_miss("plan")
        return None
    
    def _cache_plan(self, cache_key: str, plan: CalculationPlan) -> None:
        """Cache a plan."""
        if not self._enable_caching:
            return
        
        # Simple LRU-like eviction if cache is full
        if len(self._plan_cache) >= self._max_cache_size:
            # Remove oldest entry (first item)
            self._plan_cache.pop(next(iter(self._plan_cache)), None)
        
        self._plan_cache[cache_key] = plan

    async def run(
        self,
        findings: Dict[str, Any],
        state: Optional[MetricPipelineState] = None,
    ) -> tuple[MetricPipelineState, List[AIMessage]]:
        """Execute the full pipeline and return updated state plus instrumentation.
        
        Features:
        - Plan caching based on table specifications
        - Performance instrumentation
        - Graceful error handling with fallbacks
        """
        from .. import get_logger
        logger = get_logger(__name__)
        
        state = state or MetricPipelineState()
        
        # Start instrumentation tracking
        run_id = f"run_{state.iteration_count + 1}"
        self._instrumentation.start_run(run_id)
        
        # Get config values (only set max_iterations if state is new)
        if state.iteration_count == 0:
            # Config can be nested under 'metrics' key or at root level
            metrics_config = self._config.get('metrics', self._config)
            feedback_config = metrics_config.get('feedback', {})
            max_iterations = feedback_config.get('max_iterations', 3)
            state.max_iterations = max_iterations

        # Guard against infinite loops
        if not state.can_iterate():
            logger.warning(
                f"[METRIC PIPELINE] Reached max iterations ({state.max_iterations}), "
                "stopping pipeline execution"
            )
            
            # Generate final metrics
            metrics = self._instrumentation.get_pipeline_metrics(run_id, state.iteration_count)
            logger.info(
                f"[METRIC PIPELINE] Final metrics: {metrics.success_rate:.1%} success rate, "
                f"{metrics.cache_hit_rate:.1%} cache hit rate"
            )
            
            messages = [
                AIMessage(
                    content=f"Metric pipeline stopped: reached max iterations ({state.max_iterations})",
                    name="metric_pipeline"
                )
            ]
            return state, messages
        
        # Increment iteration counter
        state.increment_iteration()
        logger.info(
            f"[METRIC PIPELINE] Starting iteration {state.iteration_count}/{state.max_iterations}"
        )

        messages: List[AIMessage] = []

        try:
            # Phase 1: Analyze specifications
            spec_bundle, calc_context, analyzer_messages = await self._spec_analyzer.analyze(findings)
            messages.extend(analyzer_messages)

            # Phase 2: Create calculation plan (with caching)
            cache_key = self._compute_hash({
                "table_specs": [spec.model_dump() for spec in calc_context.table_specifications],
                "extracted_data": len(calc_context.extracted_data)
            })
            
            plan = self._get_cached_plan(cache_key)
            if plan:
                logger.info(f"[METRIC PIPELINE] Using cached plan (key: {cache_key})")
                planner_messages = [
                    AIMessage(
                        content="Using cached calculation plan",
                        name="calculation_planner"
                    )
                ]
            else:
                plan, planner_messages = await self._planner.create_plan(calc_context)
                self._cache_plan(cache_key, plan)
                logger.info(f"[METRIC PIPELINE] Created new plan with {len(plan.tasks)} tasks")
            
            messages.extend(planner_messages)

            # Phase 3: Execute calculations
            calc_context, events, executor_messages = await self._executor.execute(plan, calc_context)
            messages.extend(executor_messages)

            # Update state
            state.spec_bundle = spec_bundle
            state.calculation_plan = plan
            state.calculation_context = calc_context
            state.execution_summary = events
            state.pending_research_queries = [
                event.research_query
                for event in events
                if getattr(event, "needs_research", False) and event.research_query
            ]
            state.touch()
            
            # Track research feedback
            if state.pending_research_queries:
                self._instrumentation.record_research_feedback(len(state.pending_research_queries))
            
            # Log results with metrics
            metrics = self._instrumentation.get_pipeline_metrics(run_id, state.iteration_count)
            
            if state.pending_research_queries:
                logger.info(
                    f"[METRIC PIPELINE] Iteration {state.iteration_count} complete: "
                    f"{len(state.pending_research_queries)} research queries needed "
                    f"({metrics.success_rate:.1%} success rate)"
                )
            else:
                logger.info(
                    f"[METRIC PIPELINE] Iteration {state.iteration_count} complete: "
                    f"all calculations successful ({metrics.success_rate:.1%} success rate, "
                    f"{metrics.cache_hit_rate:.1%} cache hit rate)"
                )

        except Exception as e:
            logger.error(f"[METRIC PIPELINE] Pipeline execution failed: {e}", exc_info=True)
            # Graceful degradation - return partial state
            messages.append(
                AIMessage(
                    content=f"Metric pipeline encountered an error: {str(e)}. Returning partial results.",
                    name="metric_pipeline"
                )
            )
            state.touch()

        return state, messages
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get caching statistics for monitoring."""
        return self._instrumentation.get_cache_stats()
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get execution history for debugging."""
        return self._instrumentation.get_execution_history()
    
    def clear_caches(self) -> None:
        """Clear all caches."""
        self._plan_cache.clear()
        self._result_cache.clear()
        logger = __import__('deep_research_agent.core', fromlist=['get_logger']).get_logger(__name__)
        logger.info("[METRIC PIPELINE] Caches cleared")
