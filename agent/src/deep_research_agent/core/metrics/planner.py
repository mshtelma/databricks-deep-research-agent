"""Calculation planner for metric derivation."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Set

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from .. import get_logger
from ..report_generation.models import Calculation, CalculationContext, DataPoint
from .models import CalculationPlan, CalculationTask, CalculationTaskType, MissingDataRequest


logger = get_logger(__name__)


class CodeGenerationRequest(BaseModel):
    """Structured request for code generation."""
    
    metric_name: str
    description: str
    available_entities: List[str]
    available_metrics: Dict[str, List[str]]  # entity -> list of metrics
    operation_type: str  # ratio, delta, aggregation, etc.


class GeneratedCalculation(BaseModel):
    """LLM-generated calculation specification."""
    
    metric_name: str
    description: str
    operation: str
    code: str
    depends_on_metrics: List[str] = Field(default_factory=list)
    depends_on_entities: List[str] = Field(default_factory=list)
    inputs_missing: bool = False
    missing_data_explanation: Optional[str] = None
    research_query: Optional[str] = None


class CalculationPlanner:
    """Create an ordered plan describing how to derive metrics.
    
    This planner uses LLM to generate Python code for calculations,
    analyzes dependencies, and flags missing data for research feedback.
    """

    def __init__(self, llm) -> None:
        self._llm = llm

    async def create_plan(
        self,
        calc_context: CalculationContext,
    ) -> tuple[CalculationPlan, List[AIMessage]]:
        """Derive a calculation plan from context with intelligent code generation.
        
        Process:
        1. Analyze what calculations are needed (gap analysis)
        2. Generate Python code using MetricDataContext API
        3. Build dependency graph
        4. Flag missing data for research
        """
        logger.info("[PLANNER] Starting intelligent calculation planning")
        
        # If calculations already exist in context, use them
        if calc_context.calculations:
            logger.info(f"[PLANNER] Using {len(calc_context.calculations)} existing calculations")
            return await self._plan_from_existing(calc_context)
        
        # Otherwise, perform gap analysis and generate code
        logger.info("[PLANNER] Performing gap analysis and code generation")
        tasks, dependencies = await self._analyze_and_generate(calc_context)
        
        plan = CalculationPlan(tasks=tasks, dependencies=dependencies)
        
        # Count statistics
        research_needed = sum(1 for t in tasks if t.requires_research)
        has_code = sum(1 for t in tasks if t.code)
        
        messages: List[AIMessage] = [
            AIMessage(
                content=(
                    f"CalculationPlanner generated plan with {len(tasks)} tasks: "
                    f"{has_code} with executable code, {research_needed} need research."
                ),
                name="calculation_planner",
            )
        ]
        
        logger.info(
            f"[PLANNER] Plan complete: {len(tasks)} tasks, "
            f"{research_needed} need research"
        )
        
        return plan, messages
    
    async def _plan_from_existing(
        self,
        calc_context: CalculationContext
    ) -> tuple[CalculationPlan, List[AIMessage]]:
        """Create plan from existing calculations in context."""
        tasks: List[CalculationTask] = []
        dependencies: Dict[str, Set[str]] = {}
        
        for calc_index, calculation in enumerate(calc_context.calculations):
            task_id = f"calc_{calc_index+1}"
            
            # Try to extract code from formula if it looks like Python
            code = None
            if calculation.formula and self._looks_like_python(calculation.formula):
                code = f"result = {calculation.formula}"
            
            task = CalculationTask(
                task_id=task_id,
                operation=CalculationTaskType.FORMULA,
                description=calculation.description,
                inputs=calculation.inputs,
                output_key=calculation.description,
                code=code,
            )
            tasks.append(task)
            dependencies[task_id] = set()
        
        plan = CalculationPlan(tasks=tasks, dependencies=dependencies)
        
        messages: List[AIMessage] = [
            AIMessage(
                content=(
                    f"CalculationPlanner created plan from {len(tasks)} existing calculations."
                ),
                name="calculation_planner",
            )
        ]
        
        return plan, messages
    
    async def _analyze_and_generate(
        self,
        calc_context: CalculationContext
    ) -> tuple[List[CalculationTask], Dict[str, Set[str]]]:
        """Analyze requirements and generate calculation code.
        
        Returns:
            (tasks, dependencies) tuple
        """
        # Step 1: Analyze available data
        available_data = self._analyze_available_data(calc_context)
        
        # Step 2: Analyze required metrics from table specs
        required_metrics = self._analyze_required_metrics(calc_context)
        
        # Step 3: Perform gap analysis
        gaps = self._find_gaps(available_data, required_metrics)
        
        if not gaps:
            logger.info("[PLANNER] No gaps found, all required metrics available")
            return [], {}
        
        logger.info(f"[PLANNER] Found {len(gaps)} metric gaps to fill")
        
        # Step 4: Generate code for missing metrics
        tasks = []
        dependencies: Dict[str, Set[str]] = {}
        
        for gap_index, gap in enumerate(gaps):
            task = await self._generate_calculation_task(
                gap,
                available_data,
                task_id=f"calc_{gap_index+1}"
            )
            tasks.append(task)
            dependencies[task.task_id] = set()  # TODO: Parse dependencies from code
        
        return tasks, dependencies
    
    def _analyze_available_data(self, calc_context: CalculationContext) -> Dict[str, Any]:
        """Analyze what data is available."""
        entities: Set[str] = set()
        entity_metrics: Dict[str, Set[str]] = {}
        
        for dp in calc_context.extracted_data:
            entities.add(dp.entity)
            if dp.entity not in entity_metrics:
                entity_metrics[dp.entity] = set()
            entity_metrics[dp.entity].add(dp.metric)
        
        return {
            'entities': list(entities),
            'entity_metrics': {k: list(v) for k, v in entity_metrics.items()},
            'data_point_count': len(calc_context.extracted_data)
        }
    
    def _analyze_required_metrics(self, calc_context: CalculationContext) -> List[str]:
        """Analyze what metrics are required from table specs."""
        required = set()
        
        for table_spec in calc_context.table_specifications:
            # Metrics are typically in columns
            if hasattr(table_spec, 'column_metrics'):
                required.update(table_spec.column_metrics)
        
        return list(required)
    
    def _find_gaps(
        self,
        available_data: Dict[str, Any],
        required_metrics: List[str]
    ) -> List[Dict[str, Any]]:
        """Find gaps between required and available metrics."""
        gaps = []
        
        # Get all available metrics across all entities
        all_available_metrics = set()
        for metrics in available_data['entity_metrics'].values():
            all_available_metrics.update(metrics)
        
        # Find missing metrics
        for metric in required_metrics:
            # Check if metric exists directly
            if metric in all_available_metrics:
                continue
            
            # Check if it's a derived metric (contains keywords like rate, ratio, difference)
            if self._is_derived_metric(metric):
                gaps.append({
                    'metric_name': metric,
                    'type': 'derived',
                    'available_data': available_data
                })
        
        return gaps
    
    def _is_derived_metric(self, metric: str) -> bool:
        """Check if metric name suggests it's a derived calculation."""
        derived_keywords = [
            'rate', 'ratio', 'percent', 'percentage', 'difference', 
            'delta', 'change', 'growth', 'average', 'mean', 'total',
            'sum', 'effective', 'marginal', 'net'
        ]
        metric_lower = metric.lower()
        return any(keyword in metric_lower for keyword in derived_keywords)
    
    async def _generate_calculation_task(
        self,
        gap: Dict[str, Any],
        available_data: Dict[str, Any],
        task_id: str
    ) -> CalculationTask:
        """Generate a calculation task for a missing metric.
        
        Uses LLM to generate Python code using MetricDataContext API.
        """
        metric_name = gap['metric_name']
        
        # Build prompt for code generation
        prompt = self._build_code_generation_prompt(metric_name, available_data)
        
        try:
            # Use LLM to generate code
            generated = await self._call_llm_for_code(prompt)
            
            return CalculationTask(
                task_id=task_id,
                operation=CalculationTaskType.FORMULA,
                description=generated.get('description', f"Calculate {metric_name}"),
                inputs={},
                output_key=metric_name,
                code=generated.get('code'),
                requires_research=generated.get('inputs_missing', False),
                fallback_research_query=generated.get('research_query'),
            )
        
        except Exception as e:
            logger.warning(f"[PLANNER] Code generation failed for {metric_name}: {e}")
            
            # Fallback: create task flagged for research
            return CalculationTask(
                task_id=task_id,
                operation=CalculationTaskType.FORMULA,
                description=f"Calculate {metric_name}",
                inputs={},
                output_key=metric_name,
                code=None,
                requires_research=True,
                fallback_research_query=f"Find data to calculate {metric_name}",
            )
    
    def _build_code_generation_prompt(
        self,
        metric_name: str,
        available_data: Dict[str, Any]
    ) -> str:
        """Build prompt for LLM code generation."""
        entities = available_data['entities']
        entity_metrics = available_data['entity_metrics']
        
        # Format available data nicely
        data_summary = []
        for entity in entities[:5]:  # Limit to first 5 entities
            metrics = entity_metrics.get(entity, [])
            data_summary.append(f"- {entity}: {', '.join(metrics[:10])}")
        
        prompt = f"""Generate Python code to calculate the metric: {metric_name}

Available Data Context:
{chr(10).join(data_summary)}

Available API:
- ctx.get_scalar(entity, metric) -> float  # Get single value
- ctx.get_table(entity) -> pd.DataFrame    # Get all metrics for entity
- ctx.get_all_entities() -> List[str]      # List all entities
- math, statistics modules available

Task: Write Python code to calculate {metric_name} using the available data.

Requirements:
1. Assign result to variable named 'result'
2. Use the ctx API to access data
3. Handle missing data gracefully (check for None)
4. Include brief comment explaining calculation

Example for "effective_tax_rate":
```python
# Calculate effective tax rate as (total_tax / gross_income) * 100
gross = ctx.get_scalar('Spain', 'gross_income')
tax = ctx.get_scalar('Spain', 'total_tax_paid')
if gross and tax and gross > 0:
    result = (tax / gross) * 100
else:
    result = None
```

Generate code for {metric_name}:"""
        
        return prompt
    
    async def _call_llm_for_code(self, prompt: str) -> Dict[str, Any]:
        """Call LLM to generate calculation code."""
        messages = [
            SystemMessage(content="You are an expert at writing Python calculation code."),
            HumanMessage(content=prompt)
        ]
        
        try:
            response = await self._llm.ainvoke(messages)
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # Extract code from response
            code = self._extract_code_from_response(response_text)
            
            return {
                'code': code,
                'description': f"Generated calculation",
                'inputs_missing': False
            }
        
        except Exception as e:
            logger.error(f"[PLANNER] LLM call failed: {e}")
            return {
                'code': None,
                'description': "Failed to generate code",
                'inputs_missing': True,
                'research_query': "Find more data for calculation"
            }
    
    def _extract_code_from_response(self, response: str) -> Optional[str]:
        """Extract Python code from LLM response."""
        # Try to find code block
        code_block_pattern = r'```python\s+(.*?)\s+```'
        match = re.search(code_block_pattern, response, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        
        # Try to find any code block
        code_block_pattern = r'```\s+(.*?)\s+```'
        match = re.search(code_block_pattern, response, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        
        # Fallback: return whole response if it looks like code
        if 'result =' in response:
            return response.strip()
        
        return None
    
    def _looks_like_python(self, text: str) -> bool:
        """Check if text looks like Python code."""
        python_indicators = ['=', 'ctx.get_', 'def ', 'if ', 'for ', 'while ', 'import ']
        return any(indicator in text for indicator in python_indicators)
