"""
Instruction analyzer for extracting structured requirements from natural language instructions.

Uses LLM-based extraction with fallback mechanisms to parse user instructions into
structured requirement objects.
"""

import logging
import json
import re
from typing import Optional, Dict, Any, List, Tuple
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

from .requirements import (
    RequirementSet, RequiredDataPoint, OutputFormat, TableSpecification,
    NarrativeSpecification, Constraint, AssumptionRequirement,
    OutputFormatType, TableStructureType, ConstraintType,
    RequirementExtractionResult
)
from .response_handlers import extract_text_from_response

logger = logging.getLogger(__name__)


class InstructionAnalyzer:
    """
    Analyzes natural language instructions to extract structured requirements.
    
    Uses a multi-stage approach:
    1. LLM-based extraction with structured prompting
    2. Rule-based fallback for common patterns
    3. Validation and confidence scoring
    """
    
    def __init__(self, llm: Optional[BaseChatModel] = None):
        """
        Initialize the instruction analyzer.
        
        Args:
            llm: Language model for extraction. If None, will use rule-based fallback only.
        """
        self.llm = llm
        self.extraction_cache: Dict[str, RequirementExtractionResult] = {}
    
    def extract_requirements(
        self, 
        instruction: str, 
        use_cache: bool = True
    ) -> RequirementExtractionResult:
        """
        Extract structured requirements from natural language instruction.
        
        Args:
            instruction: The natural language instruction from the user
            use_cache: Whether to use cached results if available
            
        Returns:
            RequirementExtractionResult containing extracted requirements
        """
        logger.info(f"INSTRUCTION_ANALYZER: Extracting requirements from: {instruction[:100]}...")
        
        # Check cache first
        cache_key = self._generate_cache_key(instruction)
        if use_cache and cache_key in self.extraction_cache:
            logger.info("INSTRUCTION_ANALYZER: Using cached requirements")
            return self.extraction_cache[cache_key]
        
        # Try LLM extraction first
        result = None
        if self.llm:
            try:
                result = self._llm_extraction(instruction)
                logger.info(f"INSTRUCTION_ANALYZER: LLM extraction confidence: {result.confidence_score:.2f}")
            except Exception as e:
                logger.warning(f"INSTRUCTION_ANALYZER: LLM extraction failed: {e}")
        
        # Fallback to rule-based extraction if needed
        if not result or result.confidence_score < 0.5:
            logger.info("INSTRUCTION_ANALYZER: Using rule-based fallback")
            fallback_result = self._rule_based_extraction(instruction)
            
            if not result:
                result = fallback_result
            else:
                # Combine LLM and rule-based results
                result = self._combine_extraction_results(result, fallback_result)
        
        # Validate and enhance the results
        result = self._validate_and_enhance(result, instruction)
        
        # Cache the result
        if use_cache:
            self.extraction_cache[cache_key] = result
        
        logger.info(f"INSTRUCTION_ANALYZER: Final extraction confidence: {result.confidence_score:.2f}")
        return result
    
    def _llm_extraction(self, instruction: str) -> RequirementExtractionResult:
        """Extract requirements using LLM with structured prompting."""
        
        system_prompt = """You are an expert at analyzing research instructions and extracting structured requirements.

Your task is to analyze the user's instruction and extract ALL requirements including:
1. Output formats (table, narrative, visualization, etc.)
2. Specific data points that must be included
3. Constraints (word limits, formatting requirements)
4. Table structures if tables are requested
5. Assumptions that should be stated explicitly

Be EXHAUSTIVE. Missing requirements lead to incorrect outputs.

For tables, pay special attention to:
- What the rows represent
- What the columns represent  
- Required data points in the table
- Any sorting or formatting requirements

For narratives, identify:
- Word count limits (look for ≤, <, "less than", "under", "max")
- Required sections
- Style requirements

For data points, list every specific metric mentioned:
- Financial figures (take-home pay, tax rates, etc.)
- Dates and time periods
- Assumptions to state
- Sources to cite

Return your analysis as JSON with this exact structure:
{
    "output_formats": [
        {
            "type": "table|narrative|visualization|mixed",
            "table_spec": {
                "structure_type": "comparative|summary|data_matrix",
                "rows_represent": "what rows represent",
                "columns_represent": "what columns represent", 
                "required_data_points": ["list", "of", "data", "points"],
                "format_specification": "markdown|csv|json"
            },
            "narrative_spec": {
                "word_limit": number or null,
                "required_sections": ["section1", "section2"],
                "style": "professional|academic|casual",
                "numbered_format": true|false
            }
        }
    ],
    "required_data_points": [
        {
            "name": "data_point_name",
            "description": "what this represents",
            "is_critical": true|false
        }
    ],
    "constraints": [
        {
            "type": "word_count|section_count|format_requirement|data_requirement|style_requirement|methodology|rigor|other",
            "value": "constraint value",
            "description": "human readable description"
        }
    ],
    "assumptions": [
        {
            "category": "tax|calculation|data",
            "description": "assumption to state"
        }
    ],
    "confidence": 0.0-1.0
}"""
        
        user_prompt = f"""Analyze this instruction and extract ALL requirements:

{instruction}

Return complete JSON with all identified requirements."""
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]

            response = self.llm.invoke(messages)
            # Handle both string and structured list responses
            response_text = extract_text_from_response(response)

            # Parse JSON response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found in LLM response")
            
            parsed_data = json.loads(json_match.group())
            
            # Convert to structured objects
            requirements = self._parse_llm_response(parsed_data, instruction)
            
            confidence = parsed_data.get('confidence', 0.8)
            
            return RequirementExtractionResult(
                requirements=requirements,
                extraction_method="llm",
                confidence_score=confidence,
                fallback_applied=False
            )
            
        except Exception as e:
            logger.error(f"INSTRUCTION_ANALYZER: LLM extraction error: {e}")
            raise
    
    def _parse_llm_response(self, data: Dict[str, Any], instruction: str) -> RequirementSet:
        """Parse LLM JSON response into RequirementSet."""
        
        # Parse output formats
        output_formats = []
        for fmt_data in data.get('output_formats', []):
            output_format = OutputFormat(
                type=OutputFormatType(fmt_data['type'])
            )
            
            # Add table spec if present
            if 'table_spec' in fmt_data and fmt_data['table_spec']:
                ts = fmt_data['table_spec']
                output_format.table_spec = TableSpecification(
                    structure_type=TableStructureType(ts.get('structure_type', 'comparative')),
                    rows_represent=ts.get('rows_represent', ''),
                    columns_represent=ts.get('columns_represent', ''),
                    required_data_points=ts.get('required_data_points', []),
                    format_specification=ts.get('format_specification', 'markdown')
                )
            
            # Add narrative spec if present
            if 'narrative_spec' in fmt_data and fmt_data['narrative_spec']:
                ns = fmt_data['narrative_spec']
                output_format.narrative_spec = NarrativeSpecification(
                    word_limit=ns.get('word_limit'),
                    required_sections=ns.get('required_sections', []),
                    style=ns.get('style', 'professional'),
                    numbered_format=ns.get('numbered_format', False)
                )
            
            output_formats.append(output_format)
        
        # Parse data points
        data_points = []
        for dp_data in data.get('required_data_points', []):
            data_points.append(RequiredDataPoint(
                name=dp_data['name'],
                description=dp_data['description'],
                is_critical=dp_data.get('is_critical', True)
            ))
        
        # Parse constraints
        constraints = []
        for c_data in data.get('constraints', []):
            # Pass string directly to Constraint - let Pydantic validator normalize
            constraints.append(Constraint(
                type=c_data['type'],  # Validator will normalize unknown types to OTHER
                value=c_data['value'],
                description=c_data['description']
            ))
        
        # Parse assumptions
        assumptions = []
        for a_data in data.get('assumptions', []):
            assumptions.append(AssumptionRequirement(
                category=a_data['category'],
                description=a_data['description']
            ))
        
        # CRITICAL FIX: Synchronize data points with table specifications
        for output_format in output_formats:
            if output_format.table_spec:
                # Add all critical data points to table spec
                for data_point in data_points:
                    if data_point.is_critical:
                        dp_name = data_point.name
                        if dp_name not in output_format.table_spec.required_data_points:
                            output_format.table_spec.required_data_points.append(dp_name)
                
                logger.debug(f"INSTRUCTION_ANALYZER: Table spec synchronized with {len(output_format.table_spec.required_data_points)} data points")
        
        return RequirementSet(
            output_formats=output_formats,
            required_data_points=data_points,
            constraints=constraints,
            assumptions=assumptions,
            original_instruction=instruction
        )
    
    def _rule_based_extraction(self, instruction: str) -> RequirementExtractionResult:
        """Fallback rule-based extraction for common patterns."""
        
        logger.info("INSTRUCTION_ANALYZER: Applying rule-based extraction patterns")
        
        instruction_lower = instruction.lower()
        output_formats = []
        constraints = []
        data_points = []
        warnings = []
        
        # Table detection
        if 'table' in instruction_lower:
            logger.info("INSTRUCTION_ANALYZER: 🔍 Table requirement detected")
            
            table_spec = TableSpecification(
                structure_type=TableStructureType.COMPARATIVE,
                rows_represent="items",
                columns_represent="attributes",
                required_data_points=[],
                format_specification="markdown"
            )
            
            # Look for row/column specifications
            row_col_pattern = r'rows?\s*[=:]\s*([^,\n]+).*columns?\s*[=:]\s*([^,\n]+)'
            match = re.search(row_col_pattern, instruction_lower)
            if match:
                table_spec.rows_represent = match.group(1).strip()
                table_spec.columns_represent = match.group(2).strip()
                logger.info(f"INSTRUCTION_ANALYZER: 📊 Table structure: {table_spec.rows_represent} × {table_spec.columns_represent}")
            
            output_formats.append(OutputFormat(
                type=OutputFormatType.TABLE,
                table_spec=table_spec
            ))
        
        # Word count detection
        word_patterns = [
            r'≤\s*(\d+)\s*words?',
            r'less than\s*(\d+)\s*words?',
            r'max\s*(\d+)\s*words?',
            r'under\s*(\d+)\s*words?'
        ]
        
        for pattern in word_patterns:
            match = re.search(pattern, instruction_lower)
            if match:
                word_limit = int(match.group(1))
                logger.info(f"INSTRUCTION_ANALYZER: ✂️ Word limit detected: {word_limit}")
                constraints.append(Constraint(
                    type=ConstraintType.WORD_COUNT,
                    value=word_limit,
                    description=f"Maximum {word_limit} words"
                ))
                
                # Add narrative format if word limit found
                narrative_spec = NarrativeSpecification(
                    word_limit=word_limit,
                    style="professional"
                )
                output_formats.append(OutputFormat(
                    type=OutputFormatType.NARRATIVE,
                    narrative_spec=narrative_spec
                ))
                break
        
        # Common data point patterns
        data_keywords = {
            'net take-home': 'Net take-home income after taxes',
            'effective tax rate': 'Effective tax rate calculation',
            'annual rent': 'Annual rental costs',
            'daycare': 'Daycare costs and subsidies',
            'family benefits': 'Family benefits and allowances',
            'disposable income': 'Disposable income calculation',
            'exchange rates': 'Currency exchange rates used',
            'assumptions': 'Key assumptions made in analysis'
        }
        
        found_data_points = []
        for keyword, description in data_keywords.items():
            if keyword in instruction_lower:
                found_data_points.append(keyword)
                data_points.append(RequiredDataPoint(
                    name=keyword.replace(' ', '_'),
                    description=description,
                    is_critical=True
                ))
        
        if found_data_points:
            logger.info(f"INSTRUCTION_ANALYZER: 📊 Data points detected: {found_data_points}")
        
        # Default to narrative if no format detected
        if not output_formats:
            output_formats.append(OutputFormat(
                type=OutputFormatType.NARRATIVE,
                narrative_spec=NarrativeSpecification()
            ))
            warnings.append("No specific output format detected, defaulting to narrative")
        
        requirements = RequirementSet(
            output_formats=output_formats,
            required_data_points=data_points,
            constraints=constraints,
            original_instruction=instruction
        )
        
        # Estimate confidence based on what we found
        confidence = 0.3  # Base confidence for rule-based
        if any('table' in instruction_lower for _ in [1]):
            confidence += 0.2
        if constraints:
            confidence += 0.2
        if data_points:
            confidence += 0.1 * min(len(data_points), 3) / 3
        
        return RequirementExtractionResult(
            requirements=requirements,
            extraction_method="rule_based",
            confidence_score=min(confidence, 0.8),  # Cap at 0.8 for rule-based
            warnings=warnings,
            fallback_applied=True
        )
    
    def _combine_extraction_results(
        self, 
        llm_result: RequirementExtractionResult,
        rule_result: RequirementExtractionResult
    ) -> RequirementExtractionResult:
        """Combine LLM and rule-based extraction results."""
        
        logger.info("INSTRUCTION_ANALYZER: Combining LLM and rule-based results")
        
        # Use LLM result as base
        combined_reqs = llm_result.requirements
        
        # Add any missing constraints from rule-based extraction (prevent actual duplicates)
        def constraint_signature(c):
            return f"{c.type}:{c.value}:{c.description}"
        
        existing_signatures = {constraint_signature(c) for c in combined_reqs.constraints}
        
        for constraint in rule_result.requirements.constraints:
            sig = constraint_signature(constraint)
            if sig not in existing_signatures:
                combined_reqs.constraints.append(constraint)
                existing_signatures.add(sig)
                logger.info(f"INSTRUCTION_ANALYZER: Added unique constraint: {constraint.description}")
            else:
                logger.debug(f"INSTRUCTION_ANALYZER: Skipped duplicate constraint: {constraint.description}")
        
        # Add any missing data points (prevent duplicates)
        existing_dp_names = {dp.name for dp in combined_reqs.required_data_points}
        
        for data_point in rule_result.requirements.required_data_points:
            if data_point.name not in existing_dp_names:
                combined_reqs.required_data_points.append(data_point)
                existing_dp_names.add(data_point.name)
                logger.info(f"INSTRUCTION_ANALYZER: Added unique data point: {data_point.name}")
            else:
                logger.debug(f"INSTRUCTION_ANALYZER: Skipped duplicate data point: {data_point.name}")
        
        # Combine warnings
        combined_warnings = llm_result.warnings + rule_result.warnings
        
        # Average confidence scores
        combined_confidence = (llm_result.confidence_score + rule_result.confidence_score) / 2
        
        return RequirementExtractionResult(
            requirements=combined_reqs,
            extraction_method="hybrid",
            confidence_score=combined_confidence,
            warnings=combined_warnings,
            fallback_applied=rule_result.fallback_applied
        )
    
    def _validate_and_enhance(
        self, 
        result: RequirementExtractionResult, 
        instruction: str
    ) -> RequirementExtractionResult:
        """Validate and enhance extraction results."""
        
        # Validate completeness
        is_complete, issues = result.requirements.validate_completeness()
        if not is_complete:
            result.validation_errors.extend(issues)
            logger.warning(f"INSTRUCTION_ANALYZER: Validation issues: {issues}")
        
        # Enhance with success criteria
        success_criteria = self._generate_success_criteria(result.requirements, instruction)
        result.requirements.success_criteria = success_criteria
        
        # Assess complexity
        complexity = self._assess_complexity(result.requirements)
        result.requirements.complexity_level = complexity
        
        # Estimate research steps needed
        steps = self._estimate_research_steps(result.requirements)
        result.requirements.estimated_research_steps = steps
        
        logger.info(f"INSTRUCTION_ANALYZER: Requirements complexity: {complexity}, estimated steps: {steps}")
        
        return result
    
    def _generate_success_criteria(self, requirements: RequirementSet, instruction: str) -> List[str]:
        """Generate success criteria based on requirements."""
        criteria = []
        
        if requirements.requires_table():
            criteria.append("Generate table with correct row/column structure")
            criteria.append("Include all required data points in table")
        
        if requirements.requires_narrative():
            word_limit = requirements.get_word_limit()
            if word_limit:
                criteria.append(f"Keep narrative under {word_limit} words")
        
        if requirements.required_data_points:
            criteria.append(f"Include all {len(requirements.required_data_points)} required data points")
        
        if requirements.assumptions:
            criteria.append("State all required assumptions explicitly")
        
        # Add format-specific criteria
        for constraint in requirements.constraints:
            if constraint.is_hard_requirement:
                criteria.append(f"Satisfy constraint: {constraint.description}")
        
        return criteria
    
    def _assess_complexity(self, requirements: RequirementSet) -> str:
        """Assess the complexity level of the requirements."""
        
        complexity_score = 0
        
        # Output format complexity
        if len(requirements.output_formats) > 1:
            complexity_score += 2
        
        if requirements.requires_table():
            complexity_score += 2
            
        # Data point complexity
        critical_data_points = len(requirements.get_critical_data_points())
        if critical_data_points > 5:
            complexity_score += 2
        elif critical_data_points > 2:
            complexity_score += 1
        
        # Constraint complexity
        if len(requirements.constraints) > 2:
            complexity_score += 1
        
        # Assumption complexity
        if len(requirements.assumptions) > 3:
            complexity_score += 1
        
        if complexity_score >= 6:
            return "complex"
        elif complexity_score >= 3:
            return "moderate"
        else:
            return "simple"
    
    def _estimate_research_steps(self, requirements: RequirementSet) -> int:
        """Estimate number of research steps needed."""
        
        base_steps = 3  # Background, planning, basic research
        
        # Add steps for data complexity
        critical_data_points = len(requirements.get_critical_data_points())
        additional_steps = min(critical_data_points // 3, 4)  # Max 4 additional steps
        
        # Add step for fact checking if many data points
        if critical_data_points > 5:
            additional_steps += 1
        
        # Add step for complex output formatting
        if len(requirements.output_formats) > 1:
            additional_steps += 1
        
        return base_steps + additional_steps
    
    def _generate_cache_key(self, instruction: str) -> str:
        """Generate cache key for instruction."""
        import hashlib
        return hashlib.md5(instruction.encode()).hexdigest()[:12]