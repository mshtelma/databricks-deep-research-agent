"""Formula extraction and discovery for metric calculations.

Implements a 3-tier formula discovery system:
1. Pattern matching: Regex-based extraction from observations
2. LLM extraction: Structured prompt for formula discovery
3. Synthesis: Infer formulas from examples
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field

from langchain_core.messages import SystemMessage, HumanMessage

from .. import get_logger


logger = get_logger(__name__)


class FormulaSpec(BaseModel):
    """Specification of an extracted or synthesized formula."""
    
    metric_name: str
    raw_formula: str = Field(description="Human-readable formula expression")
    variables: List[str] = Field(default_factory=list, description="Variables used in formula")
    confidence: float = Field(default=0.8, description="Confidence score 0.0-1.0")
    source_obs_ids: List[str] = Field(default_factory=list, description="Source observation IDs")
    source_text: Optional[str] = Field(default=None, description="Excerpt containing formula")
    extraction_method: str = Field(default="pattern", description="pattern|llm|synthesis")


class FormulaExtractor:
    """Extract formulas from observations using pattern matching + LLM."""
    
    FORMULA_PATTERNS = [
        # Direct formula patterns
        r"(\w+)\s*=\s*([^\.;]+)",  # metric = formula
        r"calculated\s+as[:\s]+([^\.]+)",  # "calculated as: formula"
        r"formula[:\s]+([^\.]+)",  # "formula: ..."
        r"computed\s+by[:\s]+([^\.]+)",  # "computed by: ..."
        r"equals?\s+([^\.]+)",  # "equals ..."
        # Math operation patterns
        r"(\w+)\s*÷\s*(\w+)",  # division with ÷
        r"(\w+)\s*/\s*(\w+)",  # division with /
        r"(\w+)\s*[-−]\s*(\w+)",  # subtraction
        r"(\w+)\s*\+\s*(\w+)",  # addition
        r"(\w+)\s*[×*]\s*(\w+)",  # multiplication
    ]
    
    def __init__(self, llm):
        """Initialize formula extractor.
        
        Args:
            llm: Language model for formula extraction
        """
        self._llm = llm
    
    async def extract_formulas_from_observations(
        self,
        observations: List[Dict[str, Any]],
        required_metrics: List[str]
    ) -> Dict[str, FormulaSpec]:
        """Extract formulas using three-tier approach.
        
        Args:
            observations: List of observation dictionaries with 'content' field
            required_metrics: List of metric names needing formulas
        
        Returns:
            Dictionary mapping metric name to FormulaSpec
        """
        logger.info(f"[FORMULA EXTRACTOR] Extracting formulas for {len(required_metrics)} metrics")
        formulas: Dict[str, FormulaSpec] = {}
        
        # Tier 1: Direct pattern matching
        logger.info("[FORMULA EXTRACTOR] Tier 1: Pattern matching")
        for obs in observations:
            content = obs.get("content", "")
            obs_id = obs.get("id", "unknown")
            
            for metric in required_metrics:
                if metric in formulas:
                    continue  # Already found
                
                formula_text = self._extract_with_patterns(content, metric)
                if formula_text:
                    variables = self._extract_variables_from_formula(formula_text)
                    formulas[metric] = FormulaSpec(
                        metric_name=metric,
                        raw_formula=formula_text,
                        variables=variables,
                        confidence=0.9,
                        source_obs_ids=[obs_id],
                        source_text=content[:200],  # First 200 chars for context
                        extraction_method="pattern"
                    )
                    logger.info(f"[FORMULA EXTRACTOR] Pattern match found for '{metric}': {formula_text}")
        
        # Tier 2: LLM extraction for missing formulas
        missing = [m for m in required_metrics if m not in formulas]
        if missing and len(missing) <= 20:  # Limit to avoid token overflow
            logger.info(f"[FORMULA EXTRACTOR] Tier 2: LLM extraction for {len(missing)} missing metrics")
            relevant_obs = self._filter_relevant_observations(observations, missing)
            if relevant_obs:
                llm_formulas = await self._extract_with_llm(relevant_obs, missing)
                formulas.update(llm_formulas)
        
        # Tier 3: Synthesis from examples (last resort)
        still_missing = [m for m in required_metrics if m not in formulas]
        if still_missing and len(still_missing) <= 10:  # More expensive, limit further
            logger.info(f"[FORMULA EXTRACTOR] Tier 3: Synthesis for {len(still_missing)} missing metrics")
            synthetic_formulas = await self._synthesize_formulas(observations, still_missing)
            formulas.update(synthetic_formulas)
        
        logger.info(f"[FORMULA EXTRACTOR] Found {len(formulas)} formulas out of {len(required_metrics)} required")
        return formulas
    
    def _extract_with_patterns(self, content: str, metric: str) -> Optional[str]:
        """Extract formula using regex patterns.
        
        Args:
            content: Text content to search
            metric: Metric name to look for
        
        Returns:
            Extracted formula string or None
        """
        # Normalize metric name for matching
        metric_normalized = metric.lower().replace('_', ' ')
        content_lower = content.lower()
        
        # Check if metric is mentioned in content
        if metric_normalized not in content_lower and metric.lower() not in content_lower:
            return None
        
        # Find the sentence containing the metric
        sentences = re.split(r'[.!?]\s+', content)
        for sentence in sentences:
            if metric_normalized in sentence.lower() or metric.lower() in sentence.lower():
                # Try each pattern
                for pattern in self.FORMULA_PATTERNS:
                    match = re.search(pattern, sentence, re.IGNORECASE)
                    if match:
                        # Extract formula part
                        if len(match.groups()) >= 1:
                            formula = match.group(1).strip()
                            # Clean up common noise
                            formula = formula.rstrip('.,;:')
                            if len(formula) > 5:  # Minimum length check
                                return formula
        
        return None
    
    def _extract_variables_from_formula(self, formula: str) -> List[str]:
        """Extract variable names from formula string.
        
        Args:
            formula: Formula string
        
        Returns:
            List of variable names found in formula
        """
        # Match word sequences (potential variables)
        words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', formula)
        
        # Filter out common operators and keywords
        operators = {'and', 'or', 'not', 'the', 'a', 'an', 'is', 'as', 'by', 'of', 'to', 'in'}
        variables = [w for w in words if w.lower() not in operators and len(w) > 1]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_vars = []
        for var in variables:
            if var not in seen:
                seen.add(var)
                unique_vars.append(var)
        
        return unique_vars
    
    def _filter_relevant_observations(
        self,
        observations: List[Dict[str, Any]],
        metrics: List[str]
    ) -> List[Dict[str, Any]]:
        """Filter observations that likely contain information about metrics.
        
        Args:
            observations: All observations
            metrics: Metric names to look for
        
        Returns:
            Filtered observations
        """
        relevant = []
        metric_patterns = [m.lower().replace('_', ' ') for m in metrics]
        
        for obs in observations:
            content = obs.get("content", "").lower()
            # Check if any metric is mentioned
            for pattern in metric_patterns:
                if pattern in content or any(word in content for word in pattern.split()):
                    relevant.append(obs)
                    break
        
        # Limit to avoid token overflow
        return relevant[:10]
    
    async def _extract_with_llm(
        self,
        observations: List[Dict[str, Any]],
        missing_metrics: List[str]
    ) -> Dict[str, FormulaSpec]:
        """Use LLM to extract formulas from text passages.
        
        Args:
            observations: Relevant observations
            missing_metrics: Metrics needing formulas
        
        Returns:
            Dictionary of FormulaSpecs
        """
        # Format observations for prompt
        obs_text = "\n\n".join([
            f"[Observation {i+1}]\n{obs.get('content', '')[:500]}"
            for i, obs in enumerate(observations)
        ])
        
        prompt = f"""You are a formula extraction expert. Find calculation formulas in the provided text.

Observations from research:
{obs_text}

Find formulas for these metrics: {', '.join(missing_metrics)}

Rules:
1. Look for explicit formulas, equations, or calculation descriptions
2. Identify component variables mentioned in the text
3. Express in standard mathematical notation (use +, -, *, /, parentheses)
4. If multiple formulas exist, choose the most standard/common one
5. Only return formulas you are confident about

Return JSON format (valid JSON only):
{{
    "metric_name": {{
        "formula": "python_expression",
        "variables": ["var1", "var2"],
        "confidence": 0.0-1.0,
        "source_text": "brief excerpt containing formula"
    }}
}}

Example:
{{
    "effective_tax_rate": {{
        "formula": "(total_tax / gross_income) * 100",
        "variables": ["total_tax", "gross_income"],
        "confidence": 0.9,
        "source_text": "calculated as total tax divided by gross income"
    }}
}}"""
        
        try:
            response = await self._llm.ainvoke([
                SystemMessage(content="You are a formula extraction expert. Return valid JSON only."),
                HumanMessage(content=prompt)
            ])
            
            # Extract content, handling multiple content blocks
            if hasattr(response, 'content'):
                content = response.content
                # If content is a list of content blocks, extract text from them
                if isinstance(content, list):
                    text_parts = []
                    for block in content:
                        if isinstance(block, dict) and 'text' in block:
                            text_parts.append(block['text'])
                        elif isinstance(block, str):
                            text_parts.append(block)
                        elif hasattr(block, 'text'):
                            text_parts.append(block.text)
                    content = '\n'.join(text_parts) if text_parts else str(content)
            else:
                content = str(response)
            
            return self._parse_llm_formulas(content, observations, "llm")
        
        except Exception as e:
            logger.error(f"[FORMULA EXTRACTOR] LLM extraction failed: {e}")
            return {}
    
    async def _synthesize_formulas(
        self,
        observations: List[Dict[str, Any]],
        missing_metrics: List[str]
    ) -> Dict[str, FormulaSpec]:
        """Synthesize formulas from examples when not explicitly stated.
        
        Args:
            observations: All observations
            missing_metrics: Metrics needing formulas
        
        Returns:
            Dictionary of synthesized FormulaSpecs
        """
        # Format observations
        obs_text = "\n\n".join([
            f"[Context {i+1}]\n{obs.get('content', '')[:300]}"
            for i, obs in enumerate(observations[:15])
        ])
        
        prompt = f"""You are a calculation expert. Infer formulas from examples and context.

Context and examples from research:
{obs_text}

Metrics needing formulas: {', '.join(missing_metrics)}

Tasks:
1. Look for numerical examples showing calculations
2. Identify patterns and relationships between values
3. Infer the most likely formula based on domain knowledge
4. Consider standard formulas in the domain (tax, finance, etc.)

Example inference:
- If text says "Spain: gross €50,000, tax €15,000, net €35,000"
- Infer: net_income = gross_income - tax

Return JSON format (valid JSON only) with inferred formulas.
Use confidence 0.5-0.7 since they're inferred, not explicit.

{{
    "metric_name": {{
        "formula": "inferred_formula",
        "variables": ["var1", "var2"],
        "confidence": 0.5-0.7,
        "source_text": "explanation of inference"
    }}
}}"""
        
        try:
            response = await self._llm.ainvoke([
                SystemMessage(content="You are a calculation expert. Return valid JSON only."),
                HumanMessage(content=prompt)
            ])
            
            # Extract content, handling multiple content blocks
            if hasattr(response, 'content'):
                content = response.content
                # If content is a list of content blocks, extract text from them
                if isinstance(content, list):
                    text_parts = []
                    for block in content:
                        if isinstance(block, dict) and 'text' in block:
                            text_parts.append(block['text'])
                        elif isinstance(block, str):
                            text_parts.append(block)
                        elif hasattr(block, 'text'):
                            text_parts.append(block.text)
                    content = '\n'.join(text_parts) if text_parts else str(content)
            else:
                content = str(response)
            
            formulas = self._parse_llm_formulas(content, observations, "synthesis")
            
            # Apply confidence penalty for synthesized formulas
            for spec in formulas.values():
                spec.confidence = min(spec.confidence, 0.7) * 0.85
            
            return formulas
        
        except Exception as e:
            logger.error(f"[FORMULA EXTRACTOR] Formula synthesis failed: {e}")
            return {}
    
    def _parse_llm_formulas(
        self,
        llm_response: str,
        observations: List[Dict[str, Any]],
        method: str
    ) -> Dict[str, FormulaSpec]:
        """Parse LLM response into FormulaSpec objects.
        
        Args:
            llm_response: LLM response text (or other types to handle gracefully)
            observations: Source observations for IDs
            method: Extraction method ('llm' or 'synthesis')
        
        Returns:
            Dictionary of FormulaSpecs
        """
        import json
        
        formulas = {}
        
        try:
            # Handle case where response might be a list or other unexpected type
            if isinstance(llm_response, list):
                logger.warning("[FORMULA EXTRACTOR] Received list instead of string, converting")
                llm_response = str(llm_response)
            elif not isinstance(llm_response, str):
                logger.warning(f"[FORMULA EXTRACTOR] Received {type(llm_response)} instead of string, converting")
                llm_response = str(llm_response)
            
            # Extract JSON from response (handle markdown code blocks)
            json_text = llm_response
            if "```json" in json_text:
                json_text = json_text.split("```json")[1].split("```")[0]
            elif "```" in json_text:
                json_text = json_text.split("```")[1].split("```")[0]
            
            json_text = json_text.strip()
            data = json.loads(json_text)
            
            # Parse each formula
            for metric_name, spec in data.items():
                if isinstance(spec, dict) and 'formula' in spec:
                    formulas[metric_name] = FormulaSpec(
                        metric_name=metric_name,
                        raw_formula=spec['formula'],
                        variables=spec.get('variables', []),
                        confidence=float(spec.get('confidence', 0.7)),
                        source_obs_ids=[obs.get('id', 'unknown') for obs in observations[:3]],
                        source_text=spec.get('source_text', ''),
                        extraction_method=method
                    )
        
        except json.JSONDecodeError as e:
            logger.error(f"[FORMULA EXTRACTOR] Failed to parse JSON: {e}")
            logger.debug(f"[FORMULA EXTRACTOR] Response was: {llm_response[:500]}")
        except Exception as e:
            logger.error(f"[FORMULA EXTRACTOR] Error parsing formulas: {e}")
        
        return formulas


__all__ = ["FormulaExtractor", "FormulaSpec"]

