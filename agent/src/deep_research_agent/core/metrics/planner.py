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
from .formula_extractor import FormulaExtractor, FormulaSpec
from .validator import FormulaValidator
from .planner_prompts import (
    CODEGEN_STRICT_SYSTEM_PROMPT,
    CODEGEN_STRICT_RULES,
    CODEGEN_USER_TEMPLATE,
    CODE_REPAIR_USER_TEMPLATE,
)


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
    
    This planner uses 3-tier formula discovery (pattern matching, LLM extraction, synthesis)
    to find formulas from observations, then generates executable Python code.
    Analyzes dependencies and flags missing data for research feedback.
    """

    def __init__(self, llm, *, code_llm=None) -> None:
        self._llm = llm
        self._code_llm = code_llm or llm
        self._formula_extractor = FormulaExtractor(self._llm)
        self._formula_validator = FormulaValidator()
        self._system_prompt = CODEGEN_STRICT_SYSTEM_PROMPT
        self._rules_block = CODEGEN_STRICT_RULES

    async def create_plan(
        self,
        calc_context: CalculationContext,
    ) -> tuple[CalculationPlan, List[AIMessage]]:
        """Derive a calculation plan from context.

        Supports two modes:
        1. Unified Planning (NEW): Direct conversion from unified plan calculations
        2. Legacy Pipeline: Gap analysis + code generation for missing metrics

        Mode is detected from calc_context.metadata['unified_plan'] flag.
        """
        # Check if this is from unified planning
        is_unified = calc_context.metadata and calc_context.metadata.get('unified_plan', False)

        if is_unified:
            logger.info("[PLANNER] Using unified planning mode - skipping gap analysis")

            # Unified planning: calculations are already complete, just convert to tasks
            existing_tasks = []
            existing_dependencies: Dict[str, Set[str]] = {}

            if calc_context.calculations:
                logger.info(f"[PLANNER] Converting {len(calc_context.calculations)} unified plan calculations to tasks")
                existing_tasks = self._convert_calculations_to_tasks(calc_context.calculations)
                # Initialize dependencies for existing tasks
                for task in existing_tasks:
                    existing_dependencies[task.task_id] = set()

            plan = CalculationPlan(tasks=existing_tasks, dependencies=existing_dependencies)

            messages: List[AIMessage] = [
                AIMessage(
                    content=(
                        f"CalculationPlanner processed unified plan with {len(existing_tasks)} tasks. "
                        f"No gap analysis needed - all data links are explicit."
                    ),
                    name="calculation_planner",
                )
            ]

            logger.info(
                f"[PLANNER] Unified plan processing complete: {len(existing_tasks)} tasks"
            )

            return plan, messages

        # LEGACY PATH: Gap analysis + code generation
        logger.info("[PLANNER] Starting intelligent calculation planning")

        # Step 1: Always preserve existing calculations from Stage 1B
        existing_tasks = []
        existing_dependencies: Dict[str, Set[str]] = {}
        if calc_context.calculations:
            logger.info(f"[PLANNER] Found {len(calc_context.calculations)} existing calculations from Stage 1B")
            existing_tasks = self._convert_calculations_to_tasks(calc_context.calculations)
            # Initialize dependencies for existing tasks
            for task in existing_tasks:
                existing_dependencies[task.task_id] = set()

        # Step 2: ALWAYS perform gap analysis to find missing metrics
        logger.info("[PLANNER] Performing gap analysis to identify missing metrics")
        gap_tasks, gap_dependencies = await self._analyze_and_generate(
            calc_context,
            existing_metrics=[calc.description for calc in calc_context.calculations] if calc_context.calculations else []
        )

        # Step 3: Combine existing + generated tasks
        all_tasks = existing_tasks + gap_tasks
        all_dependencies = {**existing_dependencies, **gap_dependencies}

        plan = CalculationPlan(tasks=all_tasks, dependencies=all_dependencies)

        # Count statistics
        research_needed = sum(1 for t in all_tasks if t.requires_research)
        has_code = sum(1 for t in all_tasks if t.code)

        messages: List[AIMessage] = [
            AIMessage(
                content=(
                    f"CalculationPlanner generated plan with {len(all_tasks)} tasks: "
                    f"{len(existing_tasks)} from Stage 1B, {len(gap_tasks)} generated for gaps, "
                    f"{has_code} with executable code, {research_needed} need research."
                ),
                name="calculation_planner",
            )
        ]

        logger.info(
            f"[PLANNER] Plan complete: {len(all_tasks)} tasks total "
            f"({len(existing_tasks)} existing + {len(gap_tasks)} generated), "
            f"{research_needed} need research"
        )

        return plan, messages
    
    def _convert_calculations_to_tasks(
        self,
        calculations: List[Calculation]
    ) -> List[CalculationTask]:
        """Convert Stage 1B Calculation objects to CalculationTask objects.
        
        This helper extracts the logic from _plan_from_existing without creating
        a full plan, allowing existing calculations to be combined with gap-filling tasks.
        
        Args:
            calculations: List of Calculation objects from Stage 1B
        
        Returns:
            List of CalculationTask objects
        """
        tasks: List[CalculationTask] = []
        
        for calc_index, calculation in enumerate(calculations):
            task_id = f"existing_{calc_index+1}"
            
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
        
        return tasks
    
    async def _plan_from_existing(
        self,
        calc_context: CalculationContext
    ) -> tuple[CalculationPlan, List[AIMessage]]:
        """Create plan from existing calculations in context.
        
        DEPRECATED: This method is kept for backward compatibility but is no longer
        the primary path. The new create_plan() method now always performs gap analysis.
        """
        tasks = self._convert_calculations_to_tasks(calc_context.calculations)
        dependencies: Dict[str, Set[str]] = {}
        
        for task in tasks:
            dependencies[task.task_id] = set()
        
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
        calc_context: CalculationContext,
        existing_metrics: Optional[List[str]] = None
    ) -> tuple[List[CalculationTask], Dict[str, Set[str]]]:
        """Analyze requirements and generate calculation code using formula discovery.
        
        Args:
            calc_context: Calculation context with data and specs
            existing_metrics: List of metric names already provided by Stage 1B calculations
        
        Returns:
            (tasks, dependencies) tuple
        """
        # Step 1: Analyze available data
        available_data = self._analyze_available_data(calc_context)
        
        # Step 2: Analyze required metrics from table specs
        required_metrics = self._analyze_required_metrics(calc_context)
        
        # Step 3: Perform gap analysis (skip metrics already covered by existing calculations)
        gaps = self._find_gaps(available_data, required_metrics, existing_metrics or [])
        
        if not gaps:
            logger.info("[PLANNER] No gaps found, all required metrics available")
            return [], {}
        
        logger.info(f"[PLANNER] Found {len(gaps)} metric gaps to fill")
        
        # Step 4: FORMULA DISCOVERY - Extract formulas from observations
        gap_metrics = [gap['metric_name'] for gap in gaps]
        observations = self._prepare_observations_for_extraction(calc_context)
        
        logger.info(f"[PLANNER] Running formula discovery for {len(gap_metrics)} metrics")
        extracted_formulas = await self._formula_extractor.extract_formulas_from_observations(
            observations,
            gap_metrics
        )
        logger.info(f"[PLANNER] Found {len(extracted_formulas)} formulas")
        
        # Step 5: Generate code for missing metrics using discovered formulas
        tasks = []
        dependencies: Dict[str, Set[str]] = {}
        
        for gap_index, gap in enumerate(gaps):
            metric_name = gap['metric_name']
            formula_spec = extracted_formulas.get(metric_name)
            
            task = await self._generate_calculation_task_with_formula(
                gap,
                formula_spec,
                available_data,
                calc_context,
                task_id=f"gap_{gap_index+1}"  # Use 'gap_' prefix to avoid conflicts with 'existing_' tasks
            )
            tasks.append(task)
            
            # Extract dependencies from formula or code
            deps = self._extract_dependencies(task, available_data)
            dependencies[task.task_id] = deps
        
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
        required_metrics: List[str],
        existing_metrics: List[str]
    ) -> List[Dict[str, Any]]:
        """Find gaps between required and available metrics.
        
        Args:
            available_data: Available data from Stage 1B extractions
            required_metrics: Metrics required by table specifications
            existing_metrics: Metrics already provided by Stage 1B calculations
        
        Returns:
            List of gap descriptions for metrics that need to be calculated
        """
        gaps = []
        
        # Get all available metrics across all entities
        all_available_metrics = set()
        for metrics in available_data['entity_metrics'].values():
            all_available_metrics.update(metrics)
        
        # Add existing calculation metrics to available set
        all_available_metrics.update(existing_metrics)
        
        # Find missing metrics
        for metric in required_metrics:
            # Check if metric exists directly (from data extraction or existing calculations)
            if metric in all_available_metrics:
                logger.debug(f"[PLANNER] Metric '{metric}' already available, skipping")
                continue
            
            # Metric is missing - add to gaps regardless of whether it looks "derived"
            # The LLM will attempt to generate code; if data is unavailable, it will be flagged for research
            metric_type = 'derived' if self._is_derived_metric(metric) else 'direct'
            gaps.append({
                'metric_name': metric,
                'type': metric_type,
                'available_data': available_data
            })
            logger.info(f"[PLANNER] Gap identified: '{metric}' ({metric_type})")
        
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
    
    def _prepare_observations_for_extraction(
        self,
        calc_context: CalculationContext
    ) -> List[Dict[str, Any]]:
        """Prepare observations from context for formula extraction.
        
        Args:
            calc_context: Calculation context with observations
        
        Returns:
            List of observation dictionaries with 'id' and 'content' fields
        """
        observations = []
        
        # Strategy 1: Check metadata for source observations (primary source)
        obs_list = []
        if hasattr(calc_context, 'metadata') and isinstance(calc_context.metadata, dict):
            if 'source_observations' in calc_context.metadata:
                obs_list = calc_context.metadata['source_observations']
                logger.info(f"[PLANNER] Found {len(obs_list)} observations in metadata")
        
        # Strategy 2: Check direct attributes
        if not obs_list:
            if hasattr(calc_context, 'observations'):
                obs_list = calc_context.observations
            elif hasattr(calc_context, 'source_observations'):
                obs_list = calc_context.source_observations
        
        # Strategy 3: Extract from data points (fallback)
        if not obs_list:
            logger.warning("[PLANNER] No observations found in calc_context, trying data points")
            for dp in calc_context.extracted_data:
                if hasattr(dp, 'source_observation'):
                    obs_list.append(dp.source_observation)
        
        # Convert to standard format with validation
        for i, obs in enumerate(obs_list):
            try:
                if isinstance(obs, dict):
                    content = obs.get('content', obs.get('text', obs.get('full_content', '')))
                    observations.append({
                        'id': obs.get('id', obs.get('step_id', f'obs_{i}')),
                        'content': str(content) if content else ''
                    })
                elif isinstance(obs, str):
                    observations.append({
                        'id': f'obs_{i}',
                        'content': obs
                    })
                elif hasattr(obs, 'content') or hasattr(obs, 'full_content'):
                    content = getattr(obs, 'full_content', None) or getattr(obs, 'content', '')
                    observations.append({
                        'id': getattr(obs, 'id', getattr(obs, 'step_id', f'obs_{i}')),
                        'content': str(content) if content else ''
                    })
            except Exception as e:
                logger.warning(f"[PLANNER] Failed to parse observation {i}: {e}")
                continue
        
        logger.info(f"[PLANNER] Prepared {len(observations)} observations for formula extraction")
        return observations
    
    def _extract_dependencies(
        self,
        task: CalculationTask,
        available_data: Dict[str, Any]
    ) -> Set[str]:
        """Extract metric dependencies from task code.
        
        Args:
            task: Calculation task
            available_data: Available data context
        
        Returns:
            Set of metric names this task depends on
        """
        if not task.code:
            return set()
        
        dependencies = set()
        
        # Look for ctx.get_scalar() calls
        scalar_pattern = r"ctx\.get_scalar\(['\"](\w+)['\"],\s*['\"](\w+)['\"]"
        matches = re.findall(scalar_pattern, task.code)
        
        for entity, metric in matches:
            dependencies.add(metric)
        
        # Also check depends_on_metrics field
        if task.depends_on_metrics:
            dependencies.update(task.depends_on_metrics)
        
        return dependencies
    
    async def _generate_calculation_task_with_formula(
        self,
        gap: Dict[str, Any],
        formula_spec: Optional[FormulaSpec],
        available_data: Dict[str, Any],
        calc_context: CalculationContext,
        task_id: str
    ) -> CalculationTask:
        """Generate a calculation task using discovered formula.
        
        Args:
            gap: Gap description
            formula_spec: Extracted formula (if found)
            available_data: Available data context
            calc_context: Full calculation context
            task_id: Task identifier
        
        Returns:
            CalculationTask with code and metadata
        """
        metric_name = gap['metric_name']
        
        if formula_spec:
            # We have a formula! Generate code from it
            logger.info(f"[PLANNER] Generating code from formula for '{metric_name}': {formula_spec.raw_formula}")
            
            try:
                code = await self._generate_code_from_formula(
                    metric_name,
                    formula_spec,
                    available_data,
                    calc_context
                )
                
                # Strip any import statements (defense in depth)
                code = self._strip_imports(code)
                # Convert f-strings to .format() (defense in depth - already done in _generate_code_from_formula)
                code = self._convert_fstrings_to_format(code)
                # Wrap code to force scalar results (fixes dict/list returns)
                code = self._wrap_code_for_scalar_result(code)

                # Validate the generated code
                available_vars = list(available_data['entity_metrics'].keys()) if available_data['entity_metrics'] else []
                validation = self._formula_validator.validate_formula(
                    code,
                    available_vars
                )
                
                if not validation.valid:
                    logger.warning(f"[PLANNER] Code validation failed: {validation.reason}")
                    # Try to fix
                    if validation.suggested_fix:
                        code = await self._fix_code_with_llm(code, validation.suggested_fix)
                        # Convert f-strings again after LLM fix
                        code = self._convert_fstrings_to_format(code)
                
                repaired_code = await self._attempt_repairs_if_needed(
                    metric_name=metric_name,
                    original_code=code,
                    formula=formula_spec.raw_formula,
                    available_data=available_data,
                    calc_context=calc_context,
                )

                return CalculationTask(
                    task_id=task_id,
                    operation=CalculationTaskType.FORMULA,
                    description=f"Calculate {metric_name}",
                    inputs={},
                    output_key=metric_name,
                    code=repaired_code,
                    formula_display=formula_spec.raw_formula,
                    depends_on_metrics=formula_spec.variables,
                    requires_research=False,
                )
            
            except Exception as e:
                logger.error(f"[PLANNER] Failed to generate code from formula: {e}")
                # Fall through to legacy method
        
        # Fallback: use legacy code generation
        return await self._generate_calculation_task(gap, available_data, task_id)
    
    async def _generate_calculation_task(
        self,
        gap: Dict[str, Any],
        available_data: Dict[str, Any],
        task_id: str
    ) -> CalculationTask:
        """Generate a calculation task for a missing metric.
        
        Uses LLM to generate Python code using MetricDataContext API.
        If LLM fails, provides a reasonable fallback.
        """
        metric_name = gap['metric_name']
        
        # Try LLM code generation
        try:
            prompt = self._build_code_generation_prompt(metric_name, available_data)
            generated = await self._call_llm_for_code(prompt)

            code = generated.get('code')

            if code and 'result' in code:
                code = await self._attempt_repairs_if_needed(
                    metric_name=metric_name,
                    original_code=code,
                    formula=None,
                    available_data=available_data,
                    calc_context=calc_context,
                )
                return CalculationTask(
                    task_id=task_id,
                    operation=CalculationTaskType.FORMULA,
                    description=generated.get('description', f"Calculate {metric_name}"),
                    inputs={},
                    output_key=metric_name,
                    code=code,
                    requires_research=False,
                )

        except Exception as e:
            logger.warning(f"[PLANNER] LLM code generation failed for {metric_name}: {e}")
        
        # Fallback: Generate template code that returns None
        # This is better than crashing and allows the Executor to continue
        logger.info(f"[PLANNER] Using fallback code template for {metric_name}")
        
        fallback_code = f"""# Placeholder for {metric_name}
# Data not available in observations - marked for research
result = None  # Mark as missing data
"""
        
        return CalculationTask(
            task_id=task_id,
            operation=CalculationTaskType.FORMULA,
            description=f"Calculate {metric_name} (needs data)",
            inputs={},
            output_key=metric_name,
            code=fallback_code,
            requires_research=True,
            fallback_research_query=f"Find data to calculate {metric_name}",
        )
    
    def _build_code_generation_prompt(
        self,
        metric_name: str,
        available_data: Dict[str, Any],
        formula: Optional[str] = None,
    ) -> str:
        """Build prompt for LLM code generation."""
        entities = available_data['entities']
        entity_metrics = available_data['entity_metrics']

        data_summary = []
        for entity in entities[:5]:  # Limit to first 5 entities
            metrics = entity_metrics.get(entity, [])
            data_summary.append(f"- {entity}: {', '.join(metrics[:10])}")

        return CODEGEN_USER_TEMPLATE.format(
            metric_name=metric_name,
            formula=formula or "(formula discovery not available)",
            data_summary="\n".join(data_summary) or "(no metrics available)",
        )
    
    async def _call_llm_for_code(self, prompt: str) -> Dict[str, Any]:
        """Call LLM to generate calculation code."""
        messages = [
            SystemMessage(content="You are an expert at writing Python calculation code."),
            HumanMessage(content=prompt)
        ]
        
        llm = self._code_llm

        try:
            response = await llm.ainvoke(messages)

            response_text = self._extract_response_text(response)
            code = self._extract_code_from_response(response_text)

            sanitized_code = self._sanitize_generated_code(code) if code else None

            return {
                'code': sanitized_code,
                'description': "Generated calculation",
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
    
    def _strip_imports(self, code: str) -> str:
        """Remove ALL import statements using both line-based and AST-based stripping.

        This is an aggressive approach to ensure NO imports slip through:
        1. Line-based stripping (catches simple imports)
        2. AST-based stripping (catches complex/nested imports)
        3. Comment-out as fallback (if AST fails)

        Args:
            code: Python code string

        Returns:
            Code with all import statements removed
        """
        original_code = code

        # Step 1: Line-based stripping
        lines = code.split('\n')
        cleaned = []
        removed_count = 0
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('import ') or stripped.startswith('from '):
                logger.warning(f"[PLANNER] Line-based: Removed prohibited import: {stripped}")
                removed_count += 1
            else:
                cleaned.append(line)

        code = '\n'.join(cleaned)

        # Step 2: AST-based stripping (catches imports that line-based missed)
        try:
            import ast
            tree = ast.parse(code, mode='exec')

            # Remove Import and ImportFrom nodes from the AST
            new_body = []
            ast_removed = 0
            for node in tree.body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    ast_removed += 1
                    # Log what we're removing
                    if isinstance(node, ast.Import):
                        names = [alias.name for alias in node.names]
                        logger.warning(f"[PLANNER] AST-based: Removed import: import {', '.join(names)}")
                    else:  # ImportFrom
                        module = node.module or ''
                        names = [alias.name for alias in node.names]
                        logger.warning(f"[PLANNER] AST-based: Removed import: from {module} import {', '.join(names)}")
                else:
                    new_body.append(node)

            if ast_removed > 0:
                tree.body = new_body
                # Unparse the tree back to code (Python 3.9+)
                code = ast.unparse(tree)
                logger.info(f"[PLANNER] AST stripping removed {ast_removed} additional import nodes")

        except Exception as e:
            # If AST parsing fails, try comment-out approach
            logger.warning(f"[PLANNER] AST stripping failed ({e}), using comment-out fallback")
            code = self._comment_out_imports(code)

        if code != original_code:
            logger.info(f"[PLANNER] Import stripping complete: removed {removed_count} line-based imports")

        return code

    def _comment_out_imports(self, code: str) -> str:
        """Convert import statements to comments as last resort.

        This is a nuclear option when AST-based stripping fails.

        Args:
            code: Python code string

        Returns:
            Code with imports converted to comments
        """
        lines = code.split('\n')
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('import ') or stripped.startswith('from '):
                lines[i] = '# DISABLED: ' + line
                logger.warning(f"[PLANNER] Commented out import: {stripped}")
        return '\n'.join(lines)

    def _wrap_code_for_scalar_result(self, code: str) -> str:
        """Ensure code returns scalar value, not dict/list.

        Wraps generated code to force extraction of scalar value from dicts/lists
        if the LLM accidentally returns those instead of a single number/string/None.

        Args:
            code: Python code string

        Returns:
            Code with result type enforcement wrapper
        """
        # Don't wrap if already has the wrapper
        if 'FORCE SCALAR RESULT' in code:
            return code

        wrapper = f"""
{code}

# FORCE SCALAR RESULT
if isinstance(result, dict):
    # Extract from dict - try common keys
    result = result.get('value', result.get('result', result.get('annual_rent', None)))
    logger.warning("[EXECUTOR] Extracted scalar value from dict result")
elif isinstance(result, list):
    # Take first element from list
    result = result[0] if result else None
    logger.warning("[EXECUTOR] Extracted scalar value from list result")
"""
        logger.debug("[PLANNER] Added scalar result enforcement wrapper")
        return wrapper

    def _convert_fstrings_to_format(self, code: str) -> str:
        """Convert all f-strings to .format() calls to eliminate syntax errors.

        Converts: f"{var:.2f}" -> "{:.2f}".format(var)
        Handles spaces: f"{var :.2f}" -> "{:.2f}".format(var)

        This completely avoids the "Space not allowed in string format specifier" error
        by using str.format() which doesn't have this issue.

        Supports all f-string variants:
        - Regular: f"..." and f'...'
        - Capital F: F"..." and F'...'
        - Raw f-strings: fr"...", rf"...", Fr"...", etc.
        - Triple-quoted: f'''...''' and f\"\"\"...\"\"\" (basic support)

        Args:
            code: Python code string

        Returns:
            Code with all f-strings converted to .format() calls
        """
        # Match all f-string variants with proper quote escaping
        # Pattern matches: [r|R]?[f|F]"..." or [r|R]?[f|F]'...'
        def replace_fstring(match):
            # Determine which quote type was used
            if match.group(1) is not None:
                quote = '"'
                content = match.group(1)
            else:
                quote = "'"
                content = match.group(2)

            # Extract variables and build format string
            vars_found = []

            def replace_braces(m):
                inner = m.group(1)

                # Skip empty (escaped braces {{ or }})
                if not inner:
                    return m.group(0)

                # Split on colon to separate variable from format specifier
                if ':' in inner:
                    var_part, format_part = inner.split(':', 1)
                    var_name = var_part.strip()  # Remove any spaces around variable name
                    vars_found.append(var_name)
                    return '{:' + format_part + '}'  # Keep format spec, drop variable name
                else:
                    var_name = inner.strip()
                    vars_found.append(var_name)
                    return '{}'

            # Replace all {var} or {var:fmt} patterns
            new_content = re.sub(r'\{([^}]*)\}', replace_braces, content)

            # Build .format() call
            if vars_found:
                return f'{quote}{new_content}{quote}.format({", ".join(vars_found)})'
            else:
                # No variables, just remove f prefix
                return f'{quote}{content}{quote}'

        # Pattern to match ALL f-string variants:
        # - [r|R]?[f|F]"..." (double quotes)
        # - [r|R]?[f|F]'...' (single quotes)
        # Matches: f"...", F"...", rf"...", fr"...", RF"...", etc.
        pattern = r'(?:[rR]?[fF]|[fF][rR])"((?:[^"\\]|\\.)*)"|(?:[rR]?[fF]|[fF][rR])\'((?:[^\'\\]|\\.)*)\''
        converted = re.sub(pattern, replace_fstring, code)

        if converted != code:
            num_fstrings = code.count('f"') + code.count("f'")
            logger.info(f"[PLANNER] Converted {num_fstrings} f-strings to .format() calls")

        return converted

    def _sanitize_format_specifiers(self, code: str) -> str:
        """Remove spaces before colons in f-string format specifiers.

        DEPRECATED: Use _convert_fstrings_to_format() instead.
        This method is kept for backward compatibility but the converter is more reliable.

        Fixes invalid syntax like f"{value :.2f}" to f"{value:.2f}"

        Args:
            code: Python code string

        Returns:
            Sanitized code with fixed format specifiers
        """
        # Pattern: matches {variable_name <spaces> :format_spec}
        # Captures the variable name and format spec, removes spaces between them
        pattern = r'\{([^}]+?)\s+:([^}]+?)\}'
        sanitized = re.sub(pattern, r'{\1:\2}', code)

        if sanitized != code:
            logger.info("[PLANNER] Sanitized f-string format specifiers (removed spaces before colons)")

        return sanitized
    
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
    
    async def _generate_code_from_formula(
        self,
        metric_name: str,
        formula_spec: FormulaSpec,
        available_data: Dict[str, Any],
        calc_context: CalculationContext
    ) -> str:
        """Generate executable Python code from formula specification.
        
        Args:
            metric_name: Name of metric being calculated
            formula_spec: Formula specification with raw formula and variables
            available_data: Available data context
            calc_context: Full calculation context
        
        Returns:
            Python code string
        """
        # Format available data for prompt
        data_summary = []
        entities = available_data.get('entities', [])
        entity_metrics = available_data.get('entity_metrics', {})
        
        for entity in entities[:5]:  # Limit to first 5
            metrics = entity_metrics.get(entity, [])
            if metrics:
                data_summary.append(f"- {entity}: {', '.join(list(metrics)[:10])}")
        
        prompt = f"""Convert this formula to executable Python code using the MetricDataContext API.

⛔ ABSOLUTELY FORBIDDEN - YOU WILL BREAK THE CODE IF YOU DO THESE ⛔
═══════════════════════════════════════════════════════════════════════════

❌ NEVER EVER DO THIS - WILL CAUSE INSTANT FAILURE:
```python
import math  # ❌ FORBIDDEN - Causes security violation
from datetime import datetime  # ❌ FORBIDDEN - Not allowed
import re  # ❌ FORBIDDEN - Will fail validation
result = {{'value': 123, 'error': None}}  # ❌ FORBIDDEN - Must return scalar, not dict!
result = [100, 200]  # ❌ FORBIDDEN - Must return single value, not list!
display = tax  # ❌ FORBIDDEN - Don't format strings in calculation code!
```

✅ ONLY DO THIS:
```python
# No imports needed - everything is pre-available!
result = value ** 0.5  # ✓ Use operators, not math.sqrt()
result = 123  # ✓ Return single number, NOT dict or list
display = "{{0:.0f}}".format(tax)  # ✓ Correct string formatting!
```

═══════════════════════════════════════════════════════════════════════════

⚠️ CRITICAL STRING FORMATTING RULE ⚠️
USE .format() FOR ALL STRING FORMATTING!

✅ CORRECT - Use .format():
    display = "{{0:.0f}} / {{1:.0f}}".format(tax, income)
    info = "Rate: {{0:.2%}}".format(rate)
    text = "Value is {{0}}".format(value)

✅ MORE EXAMPLES:
    result_text = "{{:.2f}}".format(calculation_result)
    summary = "Tax: {{0}}, Income: {{1}}".format(total_tax, gross_income)

Formula: {formula_spec.raw_formula}
Metric: {metric_name}
Variables in formula: {', '.join(formula_spec.variables)}

Available Data:
{chr(10).join(data_summary)}

API to use:
- ctx.get_scalar(entity, metric, dimensions=None) -> float or None
  Example: net = ctx.get_scalar("Spain", "net_income")
  Example with dimensions: net = ctx.get_scalar("Spain", "net_income", {{"scenario": "Single"}})

Requirements:
1. Convert the formula to Python code
2. Use ctx.get_scalar() to retrieve each variable
3. Handle None values (missing data) gracefully
4. **CRITICAL: ALWAYS assign final result to 'result' variable** (not to a dict, list, or other structure)
5. Include error handling for division by zero
6. Preserve the formula for transparency
7. DO NOT use undefined functions - if the formula contains function calls like spanish_tax(), 
   either replace with actual calculation or use placeholder values
8. Use only standard Python operators and built-in functions (abs, min, max, sum, round, pow)

**ABSOLUTELY FORBIDDEN:**
- ✗ `import math` or any import statement
- ✗ `from X import Y` or any from-import
- ✗ `__import__()` or dynamic imports
- All math functions are already available - DO NOT import them

If you need math functions, use ONLY these pre-approved built-ins:
- abs(), min(), max(), sum(), round(), pow()
- Standard operators: +, -, *, /, //, %, **

BAD EXAMPLE (DO NOT DO THIS):
```python
import math  # ✗ FORBIDDEN - will fail validation
result = math.sqrt(value)
```

GOOD EXAMPLE:
```python
# Use power operator instead of math.sqrt
result = value ** 0.5  # ✓ CORRECT
```

**CRITICAL REQUIREMENTS:**
- The code MUST include a line assigning to 'result' variable: `result = <value>`
- The result MUST be a single number (float/int), string, or None - NEVER a dict or list
- NEVER assign a dictionary like `result = {'value': 123}` - assign the value directly: `result = 123`
- Generate valid Python code that will pass ast.parse() validation
- Use try-except blocks to handle errors and set `result = None` on failure

GOOD Example:
Formula: "effective_tax_rate = (total_tax / gross_income) * 100"

Generated code:
```python
# Calculate effective tax rate: (total_tax / gross_income) * 100
try:
    # Get required data
    total_tax = ctx.get_scalar("entity", "total_tax")
    gross_income = ctx.get_scalar("entity", "gross_income")

    # Calculate
    if total_tax is not None and gross_income is not None and gross_income > 0:
        result = (total_tax / gross_income) * 100  # ✓ Assigns single number to result
        # ✓ Use .format() for string formatting
        formula_display = "({0:.0f} / {1:.0f}) * 100".format(total_tax, gross_income)
    else:
        result = None  # ✓ Assigns None on missing data
        error_message = "Missing data for calculation"
except Exception as e:
    result = None  # ✓ Assigns None on error
    error_message = str(e)
```

BAD Example (DO NOT DO THIS):
```python
# ✗ WRONG: Assigns dict instead of single value
result = {{'annual_rent': None, 'error': 'missing data', 'monthly_rent': None}}

# ✗ WRONG: Assigns list instead of single value  
result = [100, 200, 300]

# ✗ WRONG: Does not assign to 'result' variable at all
monthly_rent = 1500
annual_rent = monthly_rent * 12
```

Now generate code for: {metric_name}
Formula: {formula_spec.raw_formula}
"""
        
        try:
            response = await self._llm.ainvoke([
                SystemMessage(content=(
                    "You are an expert at converting formulas to Python code. "
                    "CRITICAL: Use .format() for all string formatting. "
                    "Example: '{:.2f}'.format(value) or '{0} divided by {1}'.format(a, b). "
                    "Return only the Python code block."
                )),
                HumanMessage(content=prompt)
            ])
            
            response_text = self._extract_response_text(response)
            code = self._extract_code_from_response(response_text)

            if not code:
                raise ValueError("Failed to extract code from LLM response")

            return self._sanitize_generated_code(code)
        
        except Exception as e:
            logger.error(f"[PLANNER] Code generation from formula failed: {e}")
            # Fallback: simple code generation
            return self._generate_simple_code_from_formula(formula_spec, entities[0] if entities else "entity")
    
    def _generate_simple_code_from_formula(
        self,
        formula_spec: FormulaSpec,
        default_entity: str
    ) -> str:
        """Generate simple code from formula as fallback.
        
        Args:
            formula_spec: Formula specification
            default_entity: Default entity to use
        
        Returns:
            Simple Python code
        """
        # Generate variable retrieval code
        var_lines = []
        for var in formula_spec.variables:
            var_lines.append(f"    {var} = ctx.get_scalar('{default_entity}', '{var}')")
        
        # Simple formula evaluation
        formula_clean = formula_spec.raw_formula.replace('÷', '/').replace('×', '*')
        
        code = f"""# Calculate using formula: {formula_spec.raw_formula}
try:
{chr(10).join(var_lines)}
    
    # Check all variables available
    if all(v is not None for v in [{', '.join(formula_spec.variables)}]):
        result = {formula_clean}
    else:
        result = None
except Exception as e:
    result = None
    error_message = str(e)
"""
        return code
    
    async def _fix_code_with_llm(self, code: str, error: str) -> str:
        """Auto-fix code generation errors.
        
        Args:
            code: Original code with errors
            error: Error description
        
        Returns:
            Fixed code
        """
        fix_prompt = f"""The following Python code has an error. Please fix it.

Original Code:
```python
{code}
```

Error: {error}

Please return only the corrected Python code (in a code block).
"""
        
        try:
            response = await self._llm.ainvoke([
                SystemMessage(content="You are a Python debugging expert. Return only corrected code."),
                HumanMessage(content=fix_prompt)
            ])
            
            # Extract content, handling multiple content blocks
            if hasattr(response, 'content'):
                response_text = response.content
                # If content is a list of content blocks, extract text from them
                if isinstance(response_text, list):
                    text_parts = []
                    for block in response_text:
                        if isinstance(block, dict) and 'text' in block:
                            text_parts.append(block['text'])
                        elif isinstance(block, str):
                            text_parts.append(block)
                        elif hasattr(block, 'text'):
                            text_parts.append(block.text)
                    response_text = '\n'.join(text_parts) if text_parts else str(response_text)
            else:
                response_text = str(response)
            
            fixed_code = self._extract_code_from_response(response_text)
            
            return fixed_code if fixed_code else code  # Return original if extraction fails
        
        except Exception as e:
            logger.error(f"[PLANNER] Code fix failed: {e}")
            return code  # Return original on failure
