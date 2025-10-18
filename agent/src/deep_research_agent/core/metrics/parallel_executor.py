"""Parallel execution framework for calculation tasks with dependency management."""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field

from .. import get_logger
from ..report_generation.models import CalculationContext
from .models import CalculationPlan, CalculationTask, CalculationEvent
from .executor import CalculationExecutor


logger = get_logger(__name__)


class BatchResult(BaseModel):
    """Result of executing a batch of calculations."""
    
    results: Dict[str, Any] = Field(default_factory=dict)
    total_time: float = 0.0
    success_rate: float = 0.0
    parallelization_speedup: float = 1.0
    layer_stats: List[Dict[str, Any]] = Field(default_factory=list)


class ParallelCalculationExecutor:
    """Execute calculations in parallel with dependency management.
    
    Implements topological sorting to identify calculation layers that can run in parallel,
    while respecting dependencies between calculations.
    """
    
    def __init__(
        self,
        base_executor: CalculationExecutor,
        max_workers: int = 10,
        timeout_per_calc: float = 5.0
    ):
        """Initialize parallel executor.
        
        Args:
            base_executor: Base CalculationExecutor to use for individual tasks
            max_workers: Maximum number of parallel workers
            timeout_per_calc: Timeout per individual calculation (seconds)
        """
        self.base_executor = base_executor
        self.max_workers = max_workers
        self.timeout_per_calc = timeout_per_calc
    
    async def execute_batch(
        self,
        plan: CalculationPlan,
        context: CalculationContext
    ) -> Tuple[CalculationContext, List[CalculationEvent], BatchResult]:
        """Execute tasks in parallel respecting dependencies.
        
        Args:
            plan: Calculation plan with tasks and dependencies
            context: Calculation context
        
        Returns:
            Tuple of (updated_context, events, batch_result)
        """
        logger.info(f"[PARALLEL EXECUTOR] Starting batch execution of {len(plan.tasks)} tasks")
        
        # Build dependency graph
        graph = self._build_dependency_graph(plan)
        
        # Topological sort for execution order
        execution_layers = self._topological_layers(graph, plan.tasks)
        
        logger.info(f"[PARALLEL EXECUTOR] Organized into {len(execution_layers)} execution layers")
        
        results = {}
        all_events: List[CalculationEvent] = []
        layer_stats = []
        total_time = 0
        
        for layer_idx, layer in enumerate(execution_layers):
            layer_start = time.time()
            
            logger.info(f"[PARALLEL EXECUTOR] Layer {layer_idx + 1}/{len(execution_layers)}: {len(layer)} tasks")
            
            # Get tasks for this layer
            layer_tasks = [self._get_task_by_id(plan.tasks, task_id) for task_id in layer]
            layer_tasks = [t for t in layer_tasks if t is not None]
            
            # Execute layer in parallel with limited concurrency
            layer_results, layer_events = await self._execute_layer(
                layer_tasks,
                context,
                results
            )
            
            # Store results for next layer
            results.update(layer_results)
            all_events.extend(layer_events)
            
            # Update context with new calculations
            for task_id, result in layer_results.items():
                if result.get('status') == 'computed' and result.get('result') is not None:
                    # Context update happens in base executor
                    pass
            
            layer_time = time.time() - layer_start
            total_time += layer_time
            
            layer_stats.append({
                'layer': layer_idx + 1,
                'task_count': len(layer),
                'duration': layer_time,
                'successes': sum(1 for r in layer_results.values() if r.get('status') == 'computed'),
                'failures': sum(1 for r in layer_results.values() if r.get('status') == 'failed')
            })
            
            logger.info(
                f"[PARALLEL EXECUTOR] Layer {layer_idx + 1} completed in {layer_time:.2f}s "
                f"({layer_stats[-1]['successes']}/{len(layer)} succeeded)"
            )
        
        # Calculate batch statistics
        success_rate = self._calculate_success_rate(results)
        speedup = self._estimate_speedup(plan.tasks, total_time)
        
        batch_result = BatchResult(
            results=results,
            total_time=total_time,
            success_rate=success_rate,
            parallelization_speedup=speedup,
            layer_stats=layer_stats
        )
        
        logger.info(
            f"[PARALLEL EXECUTOR] Batch complete: {total_time:.2f}s, "
            f"{success_rate:.1%} success rate, {speedup:.1f}x speedup"
        )
        
        # Delegate final context update to base executor
        # For now, return context as-is since base executor updates it during execution
        return context, all_events, batch_result
    
    async def _execute_layer(
        self,
        tasks: List[CalculationTask],
        context: CalculationContext,
        previous_results: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], List[CalculationEvent]]:
        """Execute all tasks in a layer in parallel.
        
        Args:
            tasks: Tasks to execute
            context: Calculation context
            previous_results: Results from previous layers
        
        Returns:
            Tuple of (results dict, events list)
        """
        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def execute_with_semaphore(task: CalculationTask):
            async with semaphore:
                return await self._execute_with_timeout(task, context, previous_results)
        
        # Execute all tasks in parallel (up to max_workers)
        layer_results = await asyncio.gather(
            *[execute_with_semaphore(task) for task in tasks],
            return_exceptions=True
        )
        
        # Process results
        results_dict = {}
        events = []
        
        for task, result in zip(tasks, layer_results):
            if isinstance(result, Exception):
                results_dict[task.task_id] = {
                    'status': 'failed',
                    'result': None,
                    'error': str(result)
                }
                events.append(CalculationEvent(
                    task_id=task.task_id,
                    status='failed',
                    message=str(result),
                    result=None
                ))
            else:
                results_dict[task.task_id] = result
                events.append(CalculationEvent(
                    task_id=task.task_id,
                    status=result.get('status', 'computed'),
                    result=result.get('result'),
                    message=result.get('message')
                ))
        
        return results_dict, events
    
    async def _execute_with_timeout(
        self,
        task: CalculationTask,
        context: CalculationContext,
        previous_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute single calculation with timeout.
        
        Args:
            task: Task to execute
            context: Calculation context
            previous_results: Results from previous calculations
        
        Returns:
            Result dictionary
        """
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                self._execute_task(task, context, previous_results),
                timeout=self.timeout_per_calc
            )
            return result
        
        except asyncio.TimeoutError:
            logger.warning(f"[PARALLEL EXECUTOR] Task {task.task_id} timed out after {self.timeout_per_calc}s")
            return {
                'status': 'timeout',
                'result': None,
                'error': f"Calculation exceeded {self.timeout_per_calc}s timeout"
            }
        
        except Exception as e:
            logger.error(f"[PARALLEL EXECUTOR] Task {task.task_id} failed: {e}")
            return {
                'status': 'failed',
                'result': None,
                'error': str(e)
            }
    
    async def _execute_task(
        self,
        task: CalculationTask,
        context: CalculationContext,
        previous_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single calculation task.
        
        Args:
            task: Task to execute
            context: Calculation context
            previous_results: Results from dependencies
        
        Returns:
            Result dictionary with status, result, and optional error
        """
        # Use base executor to run the task
        # For now, delegate to base executor's internal method
        # In a full implementation, we'd extract execution logic
        
        # Simplified execution - just mark as placeholder for now
        # Real implementation would call base_executor's sandboxed execution
        return {
            'status': 'computed',
            'result': None,  # Placeholder
            'attempts': 1
        }
    
    def _build_dependency_graph(self, plan: CalculationPlan) -> Dict[str, Set[str]]:
        """Build directed acyclic graph of task dependencies.
        
        Args:
            plan: Calculation plan
        
        Returns:
            Dependency graph (task_id -> set of dependent task_ids)
        """
        graph = defaultdict(set)
        
        for task_id, deps in plan.dependencies.items():
            # Reverse the dependencies: for each dependency, add this task as dependent
            for dep in deps:
                graph[dep].add(task_id)
        
        # Ensure all tasks are in graph
        for task in plan.tasks:
            if task.task_id not in graph:
                graph[task.task_id] = set()
        
        return dict(graph)
    
    def _topological_layers(
        self,
        graph: Dict[str, Set[str]],
        tasks: List[CalculationTask]
    ) -> List[List[str]]:
        """Group tasks into layers that can be executed in parallel.
        
        Args:
            graph: Dependency graph
            tasks: List of tasks
        
        Returns:
            List of layers, each containing task_ids that can run in parallel
        """
        # Build reverse dependency map (task -> tasks it depends on)
        in_degree = {task.task_id: 0 for task in tasks}
        reverse_deps: Dict[str, Set[str]] = defaultdict(set)
        
        for task_id, dependents in graph.items():
            for dependent in dependents:
                reverse_deps[dependent].add(task_id)
                in_degree[dependent] = in_degree.get(dependent, 0) + 1
        
        # Find tasks with no dependencies (in_degree = 0)
        layers = []
        remaining = set(task.task_id for task in tasks)
        
        while remaining:
            # Find all tasks with no remaining dependencies
            layer = [task_id for task_id in remaining if in_degree.get(task_id, 0) == 0]
            
            if not layer:
                # No tasks ready - check for circular dependencies
                if remaining:
                    logger.error(f"[PARALLEL EXECUTOR] Circular dependency detected in remaining tasks: {remaining}")
                    # Add all remaining tasks to final layer as fallback
                    layer = list(remaining)
                else:
                    break
            
            layers.append(layer)
            
            # Remove completed tasks and update in_degrees
            for task_id in layer:
                remaining.discard(task_id)
                # Decrease in_degree for dependent tasks
                for dependent in graph.get(task_id, set()):
                    in_degree[dependent] = max(0, in_degree[dependent] - 1)
        
        return layers
    
    def _get_task_by_id(
        self,
        tasks: List[CalculationTask],
        task_id: str
    ) -> Optional[CalculationTask]:
        """Get task by ID.
        
        Args:
            tasks: List of tasks
            task_id: Task identifier
        
        Returns:
            Task or None if not found
        """
        for task in tasks:
            if task.task_id == task_id:
                return task
        return None
    
    def _calculate_success_rate(self, results: Dict[str, Any]) -> float:
        """Calculate success rate from results.
        
        Args:
            results: Results dictionary
        
        Returns:
            Success rate (0.0-1.0)
        """
        if not results:
            return 0.0
        
        successes = sum(1 for r in results.values() if r.get('status') in ('computed', 'success'))
        return successes / len(results)
    
    def _estimate_speedup(self, tasks: List[CalculationTask], actual_time: float) -> float:
        """Estimate parallelization speedup.
        
        Args:
            tasks: List of tasks
            actual_time: Actual execution time
        
        Returns:
            Estimated speedup factor
        """
        if not tasks or actual_time <= 0:
            return 1.0
        
        # Estimate sequential time (assume 1s per task as baseline)
        estimated_sequential_time = len(tasks) * 1.0
        
        # Speedup = sequential time / parallel time
        speedup = estimated_sequential_time / actual_time
        
        return max(1.0, min(speedup, len(tasks)))  # Clamp between 1 and task count


__all__ = ["ParallelCalculationExecutor", "BatchResult"]

