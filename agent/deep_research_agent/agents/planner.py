"""
Planner Agent: Strategic planning and task decomposition for research.

Based on deer-flow's planner pattern with iterative refinement and quality assessment.
"""

import json
import copy
from typing import Dict, Any, Optional, List, Literal, Tuple, Sequence
from datetime import datetime
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
# Note: interrupt function needs proper LangGraph context to work
try:
    from langgraph import interrupt
except ImportError:
    # Fallback for test environments
    def interrupt(msg):
        raise RuntimeError("Interrupt not available in test context")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from deep_research_agent.core import get_logger
from deep_research_agent.core.multi_agent_state import EnhancedResearchState, StateManager
from deep_research_agent.core.plan_models import (
    Plan, Step, StepType, StepStatus, PlanQuality, PlanFeedback
)
from deep_research_agent.core.id_generator import PlanIDGenerator
from deep_research_agent.core.instruction_analyzer import InstructionAnalyzer
from deep_research_agent.core.requirement_validator import RequirementValidator
from deep_research_agent.core.requirements import RequirementSet, RequirementExtractionResult
from deep_research_agent.core.presentation_requirements import PresentationRequirements, PresentationType, TableType
from deep_research_agent.core.entity_validation import extract_entities_from_query, EntityValidator, EntityValidationMode, set_global_validator
from deep_research_agent.core.message_utils import get_last_user_message
from deep_research_agent.core.observation_models import observations_to_text_list


logger = get_logger(__name__)


class PlannerAgent:
    """
    Planner agent that generates structured research plans.
    
    Responsibilities:
    - Assess if existing context is sufficient
    - Create structured research plans
    - Decompose complex queries into steps
    - Classify steps (research vs processing)
    - Support iterative refinement
    - Integrate background investigation results
    """
    
    def __init__(self, llm=None, reasoning_llm=None, config=None, event_emitter=None):
        """
        Initialize the planner agent.
        
        Args:
            llm: Language model for standard planning
            reasoning_llm: Optional reasoning model for complex planning
            config: Configuration dictionary
            event_emitter: Optional event emitter for detailed progress tracking
        """
        self.llm = llm
        self.reasoning_llm = reasoning_llm
        self.config = config or {}
        logger.info(f"PLANNER INIT: Received config with keys: {list(self.config.keys()) if self.config else 'None/Empty'}")
        logger.info(f"PLANNER INIT: adaptive_structure config: {self.config.get('adaptive_structure', {})}")
        self.event_emitter = event_emitter  # Optional for detailed event emission
        self.name = "Planner"  # Capital for test compatibility
        
        # Extract planning configuration
        planning_config = self.config.get('planning', {})
        self.enable_iterative_planning = planning_config.get('enable_iterative_planning', True)
        self.max_plan_iterations = planning_config.get('max_plan_iterations', 3)
        self.plan_quality_threshold = planning_config.get('plan_quality_threshold', 0.7)
        self.auto_accept_plan = planning_config.get('auto_accept_plan', True)
        
        # Initialize instruction analysis components
        self.instruction_analyzer = InstructionAnalyzer(llm=self.llm)
        self.requirement_validator = RequirementValidator()
        
        # Memory tracking for circuit breaker
        self._initial_memory_mb = None
    
    def __call__(
        self,
        state: EnhancedResearchState,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate or refine research plan.
        
        Args:
            state: Current research state
            config: Configuration dictionary
            
        Returns:
            Updated state dictionary with planning information
        """
        logger.info("Planner agent generating research plan")
        
        # MEMORY CIRCUIT BREAKER: Check memory health before processing
        memory_ok, memory_msg = self._check_memory_health()
        if not memory_ok:
            logger.warning(f"Memory circuit breaker triggered: {memory_msg}")
            # Accept current plan or generate minimal plan
            if state.get("current_plan"):
                return self._proceed_with_plan(state)
            else:
                # Generate minimal fallback plan
                plan = self._generate_minimal_plan(state["research_topic"])
                state = StateManager.update_plan(state, plan)
                return self._proceed_with_plan(state, plan)
        
        # Check if we've exceeded max iterations
        if state["plan_iterations"] >= state.get("max_plan_iterations", 3):
            logger.warning("Max plan iterations reached, proceeding with current plan")
            return self._proceed_with_plan(state)
        
        # Generate plan
        plan = self._generate_plan(state, config)
        
        # Assess plan quality
        quality = self._assess_plan_quality(plan, state)
        plan.quality_assessment = quality
        
        # Check if plan meets quality threshold
        quality_threshold = config.get("plan_quality_threshold", 0.7)
        
        if quality.overall_score < quality_threshold:
            logger.info(f"Plan quality ({quality.overall_score}) below threshold ({quality_threshold})")
            
            if state["enable_iterative_planning"]:
                # Refine the plan
                return self._refine_plan(state, plan, quality)
            else:
                logger.warning("Iterative planning disabled, proceeding with suboptimal plan")
        
        # Extract and validate requirements from user instructions
        requirements_result = self._extract_and_validate_requirements(state)
        
        # Map requirements to plan steps
        self._map_requirements_to_plan(plan, requirements_result, state)
        
        # Generate report structure if needed
        self._generate_report_structure_if_needed(plan, state)
        
        # Update state with plan
        state = StateManager.update_plan(state, plan)
        
        # Check if human feedback is needed
        if not state.get("auto_accept_plan", True) and state["enable_human_feedback"]:
            return self._request_human_feedback(state, plan)
        
        # Check if we have enough context
        if plan.has_enough_context:
            logger.info("Plan indicates sufficient context, proceeding to report generation")
            updated_state = dict(state)
            updated_state["current_plan"] = plan
            updated_state["plan_iterations"] = state["plan_iterations"] + 1
            
            # Add entities from plan to state
            if hasattr(plan, 'requested_entities') and plan.requested_entities:
                updated_state["requested_entities"] = plan.requested_entities
                logger.info(f"PLANNER: Added {len(plan.requested_entities)} entities to state: {plan.requested_entities}")
            
            return updated_state
        
        # Proceed with research
        return self._proceed_with_plan(state, plan)
    
    def _generate_plan(
        self,
        state: EnhancedResearchState,
        config: Dict[str, Any]
    ) -> Plan:
        """Generate a research plan based on current state."""
        
        # Determine which model to use
        use_reasoning = (
            state.get("enable_deep_thinking", False) or
            self._is_complex_query(state["research_topic"])
        )
        
        llm = self.reasoning_llm if use_reasoning and self.reasoning_llm else self.llm
        
        # Build planning prompt
        prompt = self._build_planning_prompt(state)
        
        # Generate plan
        if llm:
            messages = [
                SystemMessage(content=self._get_system_prompt()),
                HumanMessage(content=prompt)
            ]
            
            # Log the prompt being sent to LLM
            logger.info(f"🔍 LLM_PROMPT [planner]: {prompt[:500]}...")
            
            response = llm.invoke(messages)
            
            # Log the response received from LLM
            logger.info(f"🔍 LLM_RESPONSE [planner]: {response.content[:500]}...")
            
            # ENTITY VALIDATION: Check for hallucinated entities in LLM response
            original_query = state.get("original_user_query") or get_last_user_message(state.get("messages", []))
            requested_entities = extract_entities_from_query(original_query, self.llm)
            response_entities = extract_entities_from_query(response.content, self.llm)
            
            # Check for hallucinated entities
            hallucinated = set(response_entities) - set(requested_entities)
            if hallucinated:
                logger.warning(f"🚨 ENTITY_HALLUCINATION [planner]: LLM added entities not in original query: {hallucinated}")
                logger.warning(f"🚨 REQUESTED: {requested_entities} vs GENERATED: {response_entities}")
            
            plan_dict = self._parse_plan_response(response.content)
        else:
            # Fallback to simple plan generation
            plan_dict = self._generate_simple_plan(state["research_topic"])
        
        # Create Plan object
        plan = self._dict_to_plan(plan_dict, state["research_topic"])
        
        # Add presentation analysis to the plan (SURGICAL FIX)
        background_info = state.get("background_investigation_results")
        if isinstance(background_info, str):
            background_info = [background_info]
        elif background_info and not isinstance(background_info, list):
            background_info = [str(background_info)]
            
        presentation_reqs = self._analyze_presentation_requirements(
            state["research_topic"], 
            background_info
        )
        
        # Add to plan metadata
        plan.presentation_requirements = presentation_reqs.to_dict()
        
        logger.info(f"Presentation analysis: table_needed={presentation_reqs.needs_table}, "
                   f"confidence={presentation_reqs.confidence:.2f}, "
                   f"reasoning={presentation_reqs.table_reasoning[:100]}...")
        
        # ENTITY VALIDATION: Extract requested entities and set up global validator
        try:
            # CRITICAL FIX: Extract entities from ORIGINAL user query, not LLM-generated content
            original_query = state.get("original_user_query") or get_last_user_message(state.get("messages", []))
            logger.info(f"🔍 ENTITY_DEBUG: Extracting entities from original user query: {original_query[:200]}...")
            
            requested_entities = extract_entities_from_query(original_query, self.llm)
            if requested_entities:
                logger.info(f"PLANNER: Extracted {len(requested_entities)} entities from ORIGINAL query: {requested_entities}")
                
                # Get validation mode from config
                validation_mode = EntityValidationMode.STRICT  # Default to strict
                if 'grounding' in self.config:
                    mode_str = self.config['grounding'].get('entity_validation_mode', 'strict').lower()
                    if mode_str == 'lenient':
                        validation_mode = EntityValidationMode.LENIENT
                    elif mode_str == 'moderate':
                        validation_mode = EntityValidationMode.MODERATE
                
                # Create and set global validator
                validator = EntityValidator(requested_entities, validation_mode)
                set_global_validator(validator)
                
                # Add entities to plan for downstream use
                plan.requested_entities = requested_entities
                logger.info(f"PLANNER: Entity validation enabled with mode {validation_mode.value}")
            else:
                logger.warning(f"PLANNER: No entities extracted from original query: {original_query[:100]}")
                plan.requested_entities = []
        except Exception as e:
            logger.error(f"PLANNER: Entity extraction failed: {e}")
            plan.requested_entities = []
        
        logger.info(f"Generated plan with {len(plan.steps)} steps")
        
        # Log detailed plan information for debugging and monitoring
        logger.info(f"Plan Title: {plan.title}")
        logger.info(f"Plan Approach: {plan.thought}")
        for i, step in enumerate(plan.steps, 1):
            logger.info(f"  Step {i}: {step.title}")
            logger.info(f"    - Type: {step.step_type.value}")
            logger.info(f"    - Description: {step.description[:100]}{'...' if len(step.description) > 100 else ''}")
            logger.info(f"    - Needs Search: {step.need_search}")
            if step.search_queries:
                logger.info(f"    - Search Queries: {step.search_queries}")
        
        return plan
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for planner."""
        return """You are a research planning specialist with expertise in query decomposition. Your role is to create detailed, structured research plans.

When creating a plan:
1. Assess whether you have enough context to answer immediately
2. If not, break down the research into logical steps
3. Each step should be specific and actionable
4. Classify steps as 'research' (information gathering) or 'processing' (analysis/computation)
5. Consider dependencies between steps
6. Ensure comprehensive coverage of the topic

CRITICAL: For search queries, follow these principles:
- Break complex requests into focused search queries (<100 chars each)
- Each query should be atomic - searching for ONE specific thing
- Remove all newlines and excessive formatting from queries
- Generate 2-5 queries per research step
- NEVER use the entire user request as a search query

Examples of proper query decomposition:

Example 1 - Multi-entity comparison:
Input: "Compare performance of Tesla Model 3 vs BMW i4 vs Mercedes EQE"
Queries: ["Tesla Model 3 performance specs", "BMW i4 performance review", "Mercedes EQE specifications"]

Example 2 - Multi-aspect analysis:
Input: "Analyze environmental, economic, and social impacts of renewable energy"
Queries: ["renewable energy environmental impact", "renewable energy economic benefits", "renewable energy social effects"]

Example 3 - Complex conditional:
Input: "Best programming languages for beginners with no math background who want to build web apps"
Queries: ["beginner programming languages no math", "easy web development languages", "programming languages web apps beginners"]

Example 4 - Multi-location comparison:
Input: "Compare tax systems across multiple countries for different family scenarios"
Queries: ["Spain income tax rates 2024", "France family tax benefits", "UK personal allowance", "Switzerland tax calculator"]

Output your plan as a JSON object with this structure:
{
    "has_enough_context": false,
    "thought": "Reasoning about the approach",
    "title": "Plan title",
    "steps": [
        {
            "step_id": "step_001",
            "title": "Step title",
            "description": "Detailed description",
            "step_type": "research",
            "need_search": true,
            "search_queries": ["focused query 1", "focused query 2"],
            "depends_on": []
        }
    ]
}"""
    
    def _build_planning_prompt(self, state: EnhancedResearchState) -> str:
        """Build prompt with strict size limits to prevent response explosion."""
        # BALANCED SIZE LIMITING: Keep full research topic for quality while managing other content
        prompt_parts = [
            f"Research Topic: {state['research_topic']}\n"  # Keep full topic - critical for complex queries
        ]
        
        # Add background investigation results if available (NO TRUNCATION)
        if state.get("background_investigation_results"):
            bg_results = state['background_investigation_results']
            # Convert to string if needed
            if not isinstance(bg_results, str):
                bg_results = str(bg_results)
            # NO TRUNCATION - we need full context for comprehensive research
            prompt_parts.append(
                "Background investigation:\n"
                f"{bg_results}\n\n"
            )
        
        # Add previous plan feedback if iterating (MINIMAL)
        if state.get("plan_feedback") and state.get("current_plan"):
            # For refinement, focus on incremental improvement
            current_plan = state['current_plan']
            step_count = len(current_plan.steps) if hasattr(current_plan, 'steps') else 0
            prompt_parts.append(
                f"Current plan has {step_count} steps. "
                "Add at most 2-3 targeted improvements based on gaps.\n\n"
            )
        
        # Add recent observations for context (NO TRUNCATION)
        if state.get("observations"):
            recent_obs = observations_to_text_list(state['observations'][-10:])  # More observations
            obs_text = "\n".join(recent_obs)  # NO TRUNCATION - full observation content
            prompt_parts.append(
                "Recent observations:\n"
                f"{obs_text}\n\n"
            )
        
        # Minimal instructions for incremental refinement
        if state.get("current_plan") and state.get("plan_iterations", 0) > 0:
            prompt_parts.append(
                "Refine the existing plan minimally. Add at most 2-3 steps. "
                "Focus on filling specific gaps, not regenerating."
            )
        else:
            prompt_parts.append(
                "Create a focused research plan with 3-5 key steps. "
                "Keep it concise and actionable."
            )
        
        return "".join(prompt_parts)
    
    def _parse_plan_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into plan dictionary."""
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # Try to parse entire response as JSON
                return json.loads(response)
        except json.JSONDecodeError:
            logger.warning("Failed to parse plan response as JSON, generating fallback plan")
            return self._generate_simple_plan("Research task")
    
    def _generate_simple_plan(self, topic: str) -> Dict[str, Any]:
        """Generate a simple fallback plan with proper search queries."""
        
        # Generate focused search queries instead of using raw topic
        initial_queries = self._generate_fallback_queries(topic, 3)
        detailed_queries = self._generate_fallback_queries(f"{topic} details analysis", 3)
        
        # Import ID generator for consistency
        from deep_research_agent.core.id_generator import PlanIDGenerator
        
        return {
            "has_enough_context": False,
            "thought": f"Need to research {topic}",
            "title": f"Research Plan: {topic[:100]}...",
            "steps": [
                {
                    "step_id": PlanIDGenerator.generate_step_id(1),
                    "title": "Initial Research",
                    "description": f"Gather general information about {topic[:100]}...",
                    "step_type": "research",
                    "need_search": True,
                    "search_queries": initial_queries,
                    "depends_on": []
                },
                {
                    "step_id": PlanIDGenerator.generate_step_id(2), 
                    "title": "Deep Dive",
                    "description": f"Explore specific aspects and details of {topic[:100]}...",
                    "step_type": "research",
                    "need_search": True,
                    "search_queries": detailed_queries,
                    "depends_on": [PlanIDGenerator.generate_step_id(1)]
                },
                {
                    "step_id": PlanIDGenerator.generate_step_id(3),
                    "title": "Synthesis",
                    "description": "Compile and analyze gathered information",
                    "step_type": "processing",
                    "need_search": False,
                    "depends_on": [PlanIDGenerator.generate_step_id(1), PlanIDGenerator.generate_step_id(2)]
                }
            ]
        }
    
    def _generate_fallback_queries(self, text: str, max_queries: int = 3) -> List[str]:
        """Generate fallback queries using simple patterns - lightweight version of researcher's method."""
        import re
        
        # Clean text
        clean_text = ' '.join(text.split())
        
        # If short enough, use as is
        if len(clean_text) < 80:
            return [clean_text]
        
        queries = []
        
        # Extract key entities (capitalized words/phrases)
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', clean_text)
        
        # Extract key topics (common nouns that might be topics)
        topics = re.findall(r'\b(?:analysis|research|study|comparison|benefits|impact|effects|strategies|methods|approaches|solutions)\b', clean_text, re.I)
        
        # Combine entities with topics
        for entity in entities[:2]:
            if len(entity) < 60:
                queries.append(entity)
                if topics:
                    combined = f"{entity} {topics[0]}"
                    if len(combined) < 80:
                        queries.append(combined)
        
        # If no entities found, try to extract key phrases
        if not queries:
            # Look for phrases between prepositions
            phrases = re.findall(r'\b(?:of|about|for|in|on|with)\s+([^,.!?;]+)', clean_text)
            for phrase in phrases[:2]:
                phrase = phrase.strip()
                if 10 < len(phrase) < 80:
                    queries.append(phrase)
        
        # Final fallback: safe truncation
        if not queries:
            truncated = clean_text[:75]
            last_space = truncated.rfind(' ')
            if last_space > 50:
                queries.append(truncated[:last_space] + '...')
            else:
                queries.append(truncated + '...')
        
        return queries[:max_queries]
    
    def _dict_to_plan(self, plan_dict: Dict[str, Any], research_topic: str) -> Plan:
        """Convert dictionary to Plan object."""
        plan_id = f"plan_{uuid4().hex[:8]}"
        
        # Create Step objects
        steps = []
        for step_dict in plan_dict.get("steps", []):
            step = Step(
                step_id=step_dict.get("step_id", f"step_{len(steps)+1:03d}"),
                title=step_dict.get("title", "Research Step"),
                description=step_dict.get("description", ""),
                step_type=StepType(step_dict.get("step_type", "research")),
                need_search=step_dict.get("need_search", True),
                search_queries=step_dict.get("search_queries"),
                depends_on=step_dict.get("depends_on"),
                status=StepStatus.PENDING
            )
            steps.append(step)
        
        # Create Plan
        # CRITICAL FIX: Never set has_enough_context=True if we have research steps
        # This was causing the workflow to skip the researcher and generate empty reports
        has_context = plan_dict.get("has_enough_context", False)
        if steps and len(steps) > 0:
            if has_context:
                logger.warning(
                    f"PLANNER: LLM incorrectly set has_enough_context=True with {len(steps)} steps. "
                    "Forcing to False to ensure research is executed."
                )
            has_context = False  # Force to False when we have steps to execute
        
        plan = Plan(
            plan_id=plan_id,
            title=plan_dict.get("title", f"Research Plan: {research_topic}"),
            research_topic=research_topic,
            thought=plan_dict.get("thought", ""),
            has_enough_context=has_context,
            steps=steps,
            iteration=0
        )
        
        return plan
    
    def _analyze_presentation_requirements(
        self, 
        query: str, 
        background_info: Optional[List[str]] = None
    ) -> PresentationRequirements:
        """
        Analyze query to determine optimal presentation format(s).
        
        This is the SURGICAL FIX - intelligent table requirement analysis
        that replaces primitive pattern matching in InstructionAnalyzer.
        """
        
        if not self.llm:
            return self._fallback_presentation_analysis(query)
        
        system_prompt = """
        You are a research presentation specialist. Analyze this query to determine the optimal presentation format.
        
        Consider:
        - Does this query involve comparing multiple entities? (→ table might help)
        - Does it ask for specific metrics/data points? (→ table might help)  
        - Is it asking for relationships or patterns? (→ table might help)
        - Is it purely explanatory/narrative? (→ table probably not needed)
        - Are there multiple dimensions to compare? (→ table very helpful)
        
        Return JSON:
        {
            "needs_table": true/false,
            "table_reasoning": "detailed explanation why/why not",
            "optimal_table_type": "comparative|summary|matrix|ranking|breakdown|none",
            "suggested_entities": ["entity1", "entity2"],
            "suggested_metrics": ["metric1", "metric2"],
            "primary_presentation": "narrative|table|mixed",
            "confidence": 0.0-1.0,
            "entity_reasoning": "why these entities",
            "metric_reasoning": "why these metrics"
        }
        """
        
        context = f"Query: {query}"
        if background_info:
            # Limit background context to prevent token overflow
            background_summary = ' '.join(background_info[:3])[:500]
            context += f"\nBackground: {background_summary}"
        
        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=context)
            ])
            
            # Parse JSON response
            response_text = response.content.strip()
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "").replace("```", "").strip()
                
            analysis = json.loads(response_text)
            
            return PresentationRequirements(
                needs_table=analysis.get("needs_table", False),
                table_reasoning=analysis.get("table_reasoning", ""),
                optimal_table_type=TableType(analysis.get("optimal_table_type", "none")),
                suggested_entities=analysis.get("suggested_entities", []),
                suggested_metrics=analysis.get("suggested_metrics", []),
                primary_presentation=PresentationType(analysis.get("primary_presentation", "narrative")),
                confidence=analysis.get("confidence", 0.0),
                entity_reasoning=analysis.get("entity_reasoning", ""),
                metric_reasoning=analysis.get("metric_reasoning", ""),
            )
            
        except Exception as e:
            logger.warning(f"Presentation analysis failed: {e}, using fallback")
            return self._fallback_presentation_analysis(query)
    
    def _fallback_presentation_analysis(self, query: str) -> PresentationRequirements:
        """
        Improved fallback analysis (better than current pattern matching).
        
        Uses semantic indicators rather than just looking for "table" keyword.
        """
        query_lower = query.lower()
        
        # Look for comparison indicators
        comparison_words = ['compare', 'versus', 'vs', 'difference', 'between', 'across']
        needs_comparison = any(word in query_lower for word in comparison_words)
        
        # Look for quantitative indicators  
        quantitative_words = ['cost', 'price', 'rate', 'percent', 'amount', 'income', 'tax', 'salary', 'benefit']
        needs_metrics = any(word in query_lower for word in quantitative_words)
        
        # Look for entity multiplicity indicators
        entity_indicators = ['countries', 'companies', 'options', 'alternatives', 'different', 'multiple']
        multiple_entities = any(word in query_lower for word in entity_indicators)
        
        # Determine if table would be valuable
        table_score = 0
        if needs_comparison: table_score += 0.4
        if needs_metrics: table_score += 0.3  
        if multiple_entities: table_score += 0.3
        
        needs_table = table_score >= 0.6
        
        # Try to extract basic entities and metrics
        entities = []
        metrics = []
        
        if needs_table:
            # Simple entity extraction for fallback
            import re
            potential_entities = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', query)
            entities = [e for e in potential_entities if len(e) > 2 and e.lower() not in 
                       ['Compare', 'Different', 'Between', 'Across']][:5]
            
            # Simple metric extraction
            for word in quantitative_words:
                if word in query_lower:
                    metrics.append(word.replace('_', ' ').title())
        
        reasoning = f"Query analysis: comparison={needs_comparison}, metrics={needs_metrics}, multiple_entities={multiple_entities}, score={table_score:.2f}"
        
        return PresentationRequirements(
            needs_table=needs_table,
            table_reasoning=reasoning,
            optimal_table_type=TableType.COMPARATIVE if needs_table else TableType.NONE,
            suggested_entities=entities,
            suggested_metrics=metrics,
            primary_presentation=PresentationType.MIXED if needs_table else PresentationType.NARRATIVE,
            confidence=0.6 if needs_table else 0.8,
            entity_reasoning="Basic entity extraction from query" if entities else "",
            metric_reasoning="Basic metric extraction from query" if metrics else ""
        )
    
    def _assess_plan_quality(self, plan: Plan, state: EnhancedResearchState) -> PlanQuality:
        """Assess the quality of a generated plan."""
        
        # Calculate completeness
        completeness = min(len(plan.steps) / 3, 1.0)  # Expect at least 3 steps
        
        # Calculate feasibility
        feasibility = 1.0
        for step in plan.steps:
            if step.depends_on:
                # Check if dependencies exist
                dep_exists = all(
                    any(s.step_id == dep for s in plan.steps)
                    for dep in step.depends_on
                )
                if not dep_exists:
                    feasibility -= 0.2
        feasibility = max(feasibility, 0.0)
        
        # Calculate clarity
        clarity = sum(
            1.0 if step.description and len(step.description) > 20 else 0.5
            for step in plan.steps
        ) / len(plan.steps) if plan.steps else 0.0
        
        # Calculate coverage
        # Simple heuristic: variety of step types and search queries
        step_types = set(step.step_type for step in plan.steps)
        coverage = min(len(step_types) / 2, 1.0)  # Expect at least 2 types
        
        # Identify issues
        issues = []
        if len(plan.steps) < 2:
            issues.append("Plan has too few steps")
        if not any(step.step_type == StepType.RESEARCH for step in plan.steps):
            issues.append("No research steps in plan")
        if feasibility < 0.8:
            issues.append("Some step dependencies are invalid")
        
        # Generate suggestions
        suggestions = []
        if completeness < 0.7:
            suggestions.append("Add more detailed steps to the plan")
        if coverage < 0.7:
            suggestions.append("Include more diverse research approaches")
        if clarity < 0.8:
            suggestions.append("Provide more detailed descriptions for each step")
        
        quality = PlanQuality(
            completeness_score=completeness,
            feasibility_score=feasibility,
            clarity_score=clarity,
            coverage_score=coverage,
            overall_score=0,  # Will be calculated
            issues=issues,
            suggestions=suggestions
        )
        
        quality.overall_score = quality.calculate_overall_score()
        
        return quality
    
    def _is_complex_query(self, topic: str) -> bool:
        """Determine if a query is complex enough for reasoning model."""
        complexity_indicators = [
            "compare", "analyze", "evaluate", "synthesize",
            "relationship between", "impact of", "implications",
            "trade-offs", "pros and cons", "comprehensive"
        ]
        
        topic_lower = topic.lower()
        return any(indicator in topic_lower for indicator in complexity_indicators)
    
    def _refine_plan(
        self,
        state: EnhancedResearchState,
        plan: Plan,
        quality: PlanQuality
    ) -> Dict[str, Any]:
        """Incrementally refine existing plan instead of regenerating from scratch."""
        logger.info("Incrementally refining existing plan based on quality gaps")
        
        # CIRCUIT BREAKER: Check memory before expensive refinement
        memory_ok, memory_msg = self._check_memory_health()
        if not memory_ok:
            logger.warning(f"Memory protection: {memory_msg}. Accepting current plan.")
            return self._proceed_with_plan(state, plan)
        
        # INCREMENTAL REFINEMENT: Build on existing plan instead of starting over
        refined_plan = self._incrementally_improve_plan(plan, quality, state)
        
        # Create feedback for the incremental improvement
        feedback = PlanFeedback(
            feedback_type="incremental_refinement",
            feedback=f"Added {len(refined_plan.steps) - len(plan.steps)} targeted steps to improve {', '.join(quality.suggestions[:2]) if quality.suggestions else 'quality'}",
            suggestions=quality.suggestions,
            requires_revision=False,  # We've already refined it
            approved=True  # Auto-approve incremental improvements
        )
        
        # Add feedback to state
        if not state.get("plan_feedback"):
            state["plan_feedback"] = []
        state["plan_feedback"].append(feedback)
        
        # Update state with refined plan (not regenerated plan)
        state = StateManager.update_plan(state, refined_plan)
        
        # Proceed with refined plan instead of triggering another generation cycle
        return self._proceed_with_plan(state, refined_plan)
    
    def _request_human_feedback(
        self,
        state: EnhancedResearchState,
        plan: Plan
    ) -> Dict[str, Any]:
        """Request human feedback on the plan."""
        logger.info("Requesting human feedback on plan")
        
        # Format plan for review
        plan_text = self._format_plan_for_review(plan)
        
        # Use interrupt to get human feedback
        try:
            feedback = interrupt(f"Please review the research plan:\n\n{plan_text}")
        except RuntimeError:
            # If we're outside of a LangGraph runnable context (e.g., during unit tests),
            # skip interactive feedback and auto-accept the plan so tests can proceed.
            logger.warning("Interrupt called outside runnable context – auto-accepting plan in tests.")
            return self._proceed_with_plan(state, plan)
        
        # Process feedback
        if feedback.startswith("[EDIT_PLAN]"):
            # Extract edited plan
            edited_plan_text = feedback[len("[EDIT_PLAN]"):].strip()
            
            # Parse edited plan
            try:
                edited_dict = json.loads(edited_plan_text)
                edited_plan = self._dict_to_plan(edited_dict, state["research_topic"])
                
                # Update state with edited plan
                state = StateManager.update_plan(state, edited_plan)
                
                return self._proceed_with_plan(state, edited_plan)
            except json.JSONDecodeError:
                logger.error("Failed to parse edited plan")
                updated_state = dict(state)
                updated_state["plan_iterations"] = state["plan_iterations"] + 1
                return updated_state
        
        elif feedback == "[ACCEPTED]":
            # Plan accepted, proceed
            return self._proceed_with_plan(state, plan)
        
        else:
            # Feedback provided, refine plan
            plan_feedback = PlanFeedback(
                feedback_type="human",
                feedback=feedback,
                requires_revision=True,
                approved=False
            )
            
            if not state.get("plan_feedback"):
                state["plan_feedback"] = []
            state["plan_feedback"].append(plan_feedback)
            
            updated_state = dict(state)
            updated_state["plan_iterations"] = state["plan_iterations"] + 1
            updated_state["plan_feedback"] = state["plan_feedback"]
            return updated_state
    
    def _format_plan_for_review(self, plan: Plan) -> str:
        """Format plan for human review."""
        lines = [
            f"Title: {plan.title}",
            f"Topic: {plan.research_topic}",
            f"Approach: {plan.thought}",
            f"Has Sufficient Context: {plan.has_enough_context}",
            "",
            "Steps:"
        ]
        
        for i, step in enumerate(plan.steps, 1):
            lines.extend([
                f"{i}. {step.title}",
                f"   Type: {step.step_type.value}",
                f"   Description: {step.description}",
                f"   Needs Search: {step.need_search}"
            ])
            if step.depends_on:
                lines.append(f"   Dependencies: {', '.join(step.depends_on)}")
            lines.append("")
        
        if plan.quality_assessment:
            lines.extend([
                "Quality Assessment:",
                f"- Overall Score: {plan.quality_assessment.overall_score:.2f}",
                f"- Completeness: {plan.quality_assessment.completeness_score:.2f}",
                f"- Feasibility: {plan.quality_assessment.feasibility_score:.2f}",
                f"- Clarity: {plan.quality_assessment.clarity_score:.2f}",
                f"- Coverage: {plan.quality_assessment.coverage_score:.2f}"
            ])
        
        return "\n".join(lines)
    
    def _proceed_with_plan(
        self,
        state: EnhancedResearchState,
        plan: Optional[Plan] = None
    ) -> Dict[str, Any]:
        """Proceed with plan execution."""
        if not plan:
            plan = state.get("current_plan")
        
        if not plan or not plan.steps:
            logger.error("No valid plan to proceed with")
            updated_state = dict(state)
            return updated_state
        
        # Record handoff to researcher
        state = StateManager.record_handoff(
            state,
            from_agent=self.name,
            to_agent="researcher",
            reason="Plan ready for execution",
            context={"plan_id": plan.plan_id, "total_steps": len(plan.steps)}
        )
        
        updated_state = dict(state)
        updated_state["current_plan"] = plan
        updated_state["plan_iterations"] = state["plan_iterations"] + 1
        
        # Add entities from plan to state
        if hasattr(plan, 'requested_entities') and plan.requested_entities:
            updated_state["requested_entities"] = plan.requested_entities
            logger.info(f"PLANNER: Added {len(plan.requested_entities)} entities to state: {plan.requested_entities}")

        return updated_state

    def _assert_unique_step_ids(self, steps: Sequence[Step]) -> None:
        """Ensure planner output does not contain duplicate (normalized) step IDs."""

        seen: Dict[str, Step] = {}
        duplicates: List[Dict[str, Any]] = []

        for step in steps:
            normalized_id = PlanIDGenerator.normalize_id(step.step_id)
            if normalized_id in seen:
                duplicates.append(
                    {
                        "normalized_id": normalized_id,
                        "duplicate_step_id": step.step_id,
                        "duplicate_title": step.title,
                        "existing_step_id": seen[normalized_id].step_id,
                        "existing_title": seen[normalized_id].title,
                    }
                )
            else:
                seen[normalized_id] = step

        if duplicates:
            logger.error(
                "Planner produced duplicate step identifiers",
                duplicates=duplicates,
            )
            raise ValueError("Planner generated duplicate step identifiers after renumbering")

    def _generate_report_structure_if_needed(self, plan: Plan, state: EnhancedResearchState):
        """Create a dynamic report template when adaptive structure is enabled."""

        try:
            from deep_research_agent.core.report_styles import ReportStyle
            from deep_research_agent.core.json_utils import robust_json_loads
            from deep_research_agent.core.template_generator import (
                ReportTemplateGenerator,
                SectionContentType,
                DynamicSection,
            )

            report_style = state.get("report_style")
            is_default_style = (
                report_style == ReportStyle.DEFAULT
                or str(report_style).upper() == "DEFAULT"
                or (
                    hasattr(report_style, "value")
                    and str(report_style.value).upper() == "DEFAULT"
                )
            )

            if not is_default_style:
                logger.info(
                    "PLANNER: Dynamic template generation skipped (non-default style)",
                    extra={"report_style": report_style},
                )
                return

            config = self.config or {}
            adaptive_config: Dict[str, Any] = {}
            direct_config = config.get("adaptive_structure")
            if isinstance(direct_config, dict):
                adaptive_config = direct_config
            elif isinstance(config.get("config"), dict):
                nested = config["config"].get("adaptive_structure")
                if isinstance(nested, dict):
                    adaptive_config = nested

            if not adaptive_config.get("enable_adaptive_structure", False):
                logger.info("PLANNER: Adaptive structure disabled in configuration")
                return

            query = state.get("research_topic", "")
            if not query:
                logger.info("PLANNER: Missing research topic; cannot generate template")
                return

            structure_prompt = (
                f"Design a concise yet comprehensive outline for this deep research request:\n\n"
                f"{query}\n\n"
                "Return JSON with a \"sections\" array. Each section entry must include:\n"
                "  - title\n"
                "  - purpose\n"
                "  - section_type (research | synthesis | hybrid)\n"
                "  - priority (integer, lower renders earlier)\n"
                "  - requires_search (boolean)\n"
                "  - style_hints (array of optional guidance strings)\n\n"
                "Also include \"include_appendix\": true | false if an appendix should be recommended."
            )

            messages = [
                SystemMessage(content="You are an expert report architect designing adaptive outlines."),
                HumanMessage(content=structure_prompt),
            ]

            structure_response = self.llm.invoke(messages)
            if not structure_response:
                logger.warning("PLANNER: LLM returned empty response for structure generation")
                return

            structure_payload = robust_json_loads(structure_response.content)
            if not structure_payload:
                logger.warning("PLANNER: Failed to parse structure JSON from LLM output")
                return

            raw_sections = structure_payload.get("sections", [])
            if not isinstance(raw_sections, list) or not raw_sections:
                logger.warning("PLANNER: Structure JSON did not include sections")
                return

            dynamic_sections: List[DynamicSection] = []

            for idx, section in enumerate(raw_sections, start=1):
                title = section.get("title") or f"Section {idx}"
                section_type_key = str(section.get("section_type", "research")).lower()
                hints = tuple(section.get("style_hints", []) or [])
                priority = int(section.get("priority", idx * 10))
                lowered_title = title.lower()
                if "compare" in lowered_title or " vs " in lowered_title:
                    content_type = SectionContentType.COMPARISON
                elif any(
                    token in lowered_title for token in ("timeline", "history", "trend")
                ):
                    content_type = SectionContentType.TIMELINE
                else:
                    content_type = SectionContentType.ANALYSIS

                dynamic_sections.append(
                    DynamicSection(
                        title=title,
                        purpose=section.get("purpose", ""),
                        priority=priority,
                        content_type=content_type,
                        hints=hints,
                    )
                )

            # Store the lightweight dynamic sections for downstream template usage
            plan.dynamic_sections = dynamic_sections

            generator = ReportTemplateGenerator()
            include_appendix = bool(structure_payload.get("include_appendix", False))
            plan.report_template = generator.build_template(
                title=plan.title or query,
                sections=dynamic_sections,
                include_appendix=include_appendix,
            )

            if not plan.structure_metadata:
                plan.structure_metadata = {}
            plan.structure_metadata.update(
                {
                    "dynamic_section_count": len(dynamic_sections),
                    "template_generated_at": datetime.now().isoformat(),
                    "appendix_requested": include_appendix,
                }
            )

            plan.suggested_report_structure = [section.title for section in dynamic_sections]

            logger.info(
                "PLANNER: Dynamic template generated",
                extra={
                    "sections": [section.title for section in dynamic_sections],
                    "include_appendix": include_appendix,
                },
            )

        except Exception as exc:
            logger.error(f"Error generating dynamic report structure: {exc}")
            logger.exception(exc)
    def _extract_and_validate_requirements(self, state: EnhancedResearchState) -> RequirementExtractionResult:
        """Extract and validate requirements from user instructions using LLM-based analysis."""
        try:
            research_topic = state.get("research_topic", "")
            logger.info(f"PLANNER: 🔍 Extracting requirements from instruction: {research_topic[:100]}...")
            
            # Extract requirements using InstructionAnalyzer
            requirements_result = self.instruction_analyzer.extract_requirements(research_topic)
            
            logger.info(f"PLANNER: ✅ Requirements extracted with {requirements_result.confidence_score:.2f} confidence")
            logger.info(f"PLANNER: 📊 Found {len(requirements_result.requirements.output_formats)} output formats, "
                       f"{len(requirements_result.requirements.required_data_points)} data points, "
                       f"{len(requirements_result.requirements.constraints)} constraints")
            
            # Validate requirements
            validation_result = self.requirement_validator.validate(
                requirements_result.requirements, 
                enhance=True
            )
            
            logger.info(f"PLANNER: ✅ Requirements validation complete. Valid: {validation_result.is_valid}, "
                       f"Issues: {len(validation_result.issues)}")
            
            # Log validation issues
            for issue in validation_result.get_errors():
                logger.error(f"PLANNER: ❌ Requirement error: {issue.message}")
            
            for issue in validation_result.get_warnings():
                logger.warning(f"PLANNER: ⚠️ Requirement warning: {issue.message}")
            
            # Use enhanced requirements if available
            enhanced_requirements = validation_result.enhanced_requirements or requirements_result.requirements
            
            # Update the requirements result with validation information
            requirements_result.requirements = enhanced_requirements
            requirements_result.validation_errors = [issue.message for issue in validation_result.get_errors()]
            requirements_result.warnings.extend([issue.message for issue in validation_result.get_warnings()])
            
            # Store in state for later use by Reporter
            state["extracted_requirements"] = requirements_result
            
            return requirements_result
            
        except Exception as e:
            logger.error(f"PLANNER: 💥 Failed to extract/validate requirements: {e}")
            # Create fallback requirements
            fallback_requirements = RequirementSet(
                output_formats=[],
                original_instruction=state.get("research_topic", "")
            )
            return RequirementExtractionResult(
                requirements=fallback_requirements,
                extraction_method="fallback",
                confidence_score=0.3,
                fallback_applied=True,
                validation_errors=[f"Requirement extraction failed: {e}"]
            )
    
    def _map_requirements_to_plan(
        self, 
        plan: Plan, 
        requirements_result: RequirementExtractionResult, 
        state: EnhancedResearchState
    ):
        """Map extracted requirements to plan steps and ensure coverage."""
        try:
            requirements = requirements_result.requirements
            
            logger.info("PLANNER: 🗺️ Mapping requirements to plan steps")
            
            # Store requirements in plan for Reporter access
            plan.extracted_requirements = requirements
            plan.requirement_confidence = requirements_result.confidence_score
            
            # Analyze what data is needed for each requirement
            data_needs = self._analyze_data_needs(requirements)
            logger.info(f"PLANNER: 📋 Identified {len(data_needs)} distinct data needs")
            
            # Check if current plan steps cover all data needs
            coverage_gaps = self._check_requirement_coverage(plan, data_needs)
            
            if coverage_gaps:
                logger.info(f"PLANNER: ⚠️ Found {len(coverage_gaps)} coverage gaps, adding steps")
                self._add_requirement_steps(plan, coverage_gaps, requirements)
            else:
                logger.info("PLANNER: ✅ Current plan covers all requirements")
            
            # Add requirement validation step if many data points
            if len(requirements.required_data_points) > 5:
                self._add_validation_step(plan, requirements)
            
            # Update plan metadata for Reporter
            plan.success_criteria = requirements.success_criteria
            plan.complexity_assessment = requirements.complexity_level
            plan.estimated_total_steps = requirements.estimated_research_steps
            
            logger.info(f"PLANNER: 💾 Plan enhanced with requirement mapping. "
                       f"Total steps: {len(plan.steps)}, Complexity: {requirements.complexity_level}")
            
        except Exception as e:
            logger.error(f"PLANNER: 💥 Failed to map requirements to plan: {e}")
            # Store basic fallback mapping
            plan.extracted_requirements = requirements_result.requirements
            plan.requirement_confidence = requirements_result.confidence_score
    
    def _analyze_data_needs(self, requirements: RequirementSet) -> List[Dict[str, Any]]:
        """Analyze what data needs to be collected to satisfy requirements."""
        data_needs = []
        
        # Extract data needs from required data points
        for data_point in requirements.required_data_points:
            data_needs.append({
                "type": "data_point",
                "name": data_point.name,
                "description": data_point.description,
                "critical": data_point.is_critical,
                "search_terms": self._generate_search_terms(data_point.name, data_point.description)
            })
        
        # Extract data needs from table specifications
        for output_format in requirements.output_formats:
            if output_format.table_spec:
                table_spec = output_format.table_spec
                for required_point in table_spec.required_data_points:
                    data_needs.append({
                        "type": "table_data",
                        "name": required_point,
                        "description": f"Data for table: {required_point}",
                        "critical": True,
                        "search_terms": self._generate_search_terms(required_point, f"table data {required_point}")
                    })
        
        # Remove duplicates by name
        seen_names = set()
        unique_data_needs = []
        for need in data_needs:
            if need["name"] not in seen_names:
                unique_data_needs.append(need)
                seen_names.add(need["name"])
        
        return unique_data_needs
    
    def _generate_search_terms(self, name: str, description: str) -> List[str]:
        """Generate search terms for a data need."""
        # Convert underscores to spaces
        clean_name = name.replace("_", " ")
        
        # Generate 2-3 search terms
        terms = [clean_name]
        
        # Add description-based term if different
        if description and clean_name.lower() not in description.lower():
            terms.append(description[:50])  # Limit length
        
        # Add specific search variations
        if "tax" in clean_name.lower():
            terms.append(f"{clean_name} rate calculation")
        elif "income" in clean_name.lower():
            terms.append(f"{clean_name} after tax")
        elif "cost" in clean_name.lower():
            terms.append(f"{clean_name} average price")
        
        return terms[:3]  # Limit to 3 terms
    
    def _check_requirement_coverage(self, plan: Plan, data_needs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Check if current plan steps cover all data needs."""
        coverage_gaps = []
        
        # Get all search queries from current plan
        all_plan_queries = []
        for step in plan.steps:
            if step.search_queries:
                all_plan_queries.extend([q.lower() for q in step.search_queries])
        
        # Check each data need for coverage
        for data_need in data_needs:
            covered = False
            
            # Check if any search terms are covered by existing queries
            for search_term in data_need["search_terms"]:
                search_term_lower = search_term.lower()
                for plan_query in all_plan_queries:
                    # Check for overlap in keywords
                    search_words = set(search_term_lower.split())
                    plan_words = set(plan_query.split())
                    if len(search_words & plan_words) >= 2:  # At least 2 word overlap
                        covered = True
                        break
                if covered:
                    break
            
            if not covered:
                coverage_gaps.append(data_need)
        
        return coverage_gaps
    
    def _add_requirement_steps(self, plan: Plan, coverage_gaps: List[Dict[str, Any]], requirements: RequirementSet):
        """Add steps to plan to cover requirement gaps."""
        
        # Group gaps by type for efficient step creation
        critical_gaps = [gap for gap in coverage_gaps if gap.get("critical", False)]
        non_critical_gaps = [gap for gap in coverage_gaps if not gap.get("critical", False)]
        
        # Add critical data collection step if needed
        if critical_gaps:
            step_id = f"req_critical_{len(plan.steps) + 1:03d}"
            critical_queries = []
            for gap in critical_gaps[:3]:  # Limit to 3 critical gaps per step
                critical_queries.extend(gap["search_terms"][:2])  # 2 terms per gap
            
            critical_step = Step(
                step_id=step_id,
                title="Collect Critical Required Data",
                description=f"Gather critical data points: {', '.join([gap['name'] for gap in critical_gaps[:3]])}",
                step_type=StepType.RESEARCH,
                need_search=True,
                search_queries=critical_queries,
                depends_on=[],
                requirement_mapping=[gap["name"] for gap in critical_gaps[:3]]
            )
            plan.steps.append(critical_step)
            logger.info(f"PLANNER: ➕ Added critical requirement step: {step_id}")
        
        # Add supplementary data step if needed
        if non_critical_gaps and len(non_critical_gaps) <= 3:
            step_id = f"req_supplementary_{len(plan.steps) + 1:03d}"
            supplementary_queries = []
            for gap in non_critical_gaps:
                supplementary_queries.extend(gap["search_terms"][:1])  # 1 term per gap
            
            supplementary_step = Step(
                step_id=step_id,
                title="Collect Supplementary Data",
                description=f"Gather additional data points: {', '.join([gap['name'] for gap in non_critical_gaps])}",
                step_type=StepType.RESEARCH,
                need_search=True,
                search_queries=supplementary_queries,
                depends_on=[],
                requirement_mapping=[gap["name"] for gap in non_critical_gaps]
            )
            plan.steps.append(supplementary_step)
            logger.info(f"PLANNER: ➕ Added supplementary requirement step: {step_id}")
    
    def _add_validation_step(self, plan: Plan, requirements: RequirementSet):
        """Add data validation step for complex requirements."""
        step_id = f"req_validation_{len(plan.steps) + 1:03d}"
        
        validation_step = Step(
            step_id=step_id,
            title="Validate Collected Data",
            description="Cross-check and validate all collected data points for consistency and accuracy",
            step_type=StepType.PROCESSING,
            need_search=False,
            search_queries=[],
            depends_on=[step.step_id for step in plan.steps if step.step_type == StepType.RESEARCH],
            requirement_mapping=["data_validation"]
        )
        
        plan.steps.append(validation_step)
        logger.info(f"PLANNER: ➕ Added data validation step: {step_id}")
    
    def _check_memory_health(self) -> Tuple[bool, str]:
        """Check memory health with trend analysis."""
        if not PSUTIL_AVAILABLE:
            return True, "psutil not available"
        
        try:
            process = psutil.Process()
            current_mb = process.memory_info().rss / 1024 / 1024
            
            # Initialize baseline memory if first call
            if self._initial_memory_mb is None:
                self._initial_memory_mb = current_mb
                return True, f"Initial memory: {current_mb:.0f}MB"
            
            # Check absolute limit AND growth rate
            growth_factor = current_mb / self._initial_memory_mb
            
            # Balanced thresholds: Allow complex research while preventing extreme memory usage
            if current_mb > 3000:
                return False, f"Memory exceeds 3GB limit: {current_mb:.0f}MB"
            
            if growth_factor > 3:
                return False, f"Memory grew {growth_factor:.1f}x from baseline"
            
            # Warning zone
            if current_mb > 1000 or growth_factor > 2:
                logger.warning(f"Memory warning: {current_mb:.0f}MB (growth: {growth_factor:.1f}x)")
            
            return True, f"Memory OK: {current_mb:.0f}MB (growth: {growth_factor:.1f}x)"
            
        except Exception as e:
            logger.warning(f"Error checking memory: {e}")
            return True, "memory check failed"
    
    def _incrementally_improve_plan(self, plan: Plan, quality: PlanQuality, state: Dict) -> Plan:
        """Surgically improve existing plan without regeneration."""
        # Deep copy to avoid modifying original
        improved_plan = copy.deepcopy(plan)
        
        # Track how many modifications we make
        modifications = 0
        max_modifications = 3  # Limit changes to prevent explosion
        
        # Only add TARGETED improvements based on specific gaps
        if quality.completeness_score < 0.7 and modifications < max_modifications:
            # Add ONE missing coverage step
            new_step = self._generate_coverage_step(improved_plan, state)
            if new_step:
                improved_plan.steps.append(new_step)
                modifications += 1
                logger.info("Added coverage step to improve completeness")
        
        if quality.clarity_score < 0.7 and modifications < max_modifications:
            # Enhance existing step descriptions (no new steps)
            enhanced = 0
            for step in improved_plan.steps:
                if len(step.description) < 50 and enhanced < 2:  # Limit enhancements
                    step.description = self._enhance_description(step, state)
                    enhanced += 1
                    modifications += 1
            if enhanced > 0:
                logger.info(f"Enhanced {enhanced} step descriptions for clarity")
        
        if quality.coverage_score < 0.7 and modifications < max_modifications:
            # Add alternative search queries to EXISTING steps (no new steps)
            enhanced = 0
            for step in improved_plan.steps:
                if step.need_search and len(step.search_queries or []) < 3 and enhanced < 2:
                    if not step.search_queries:
                        step.search_queries = []
                    new_queries = self._generate_alternative_queries(step, max_new=2)
                    step.search_queries.extend(new_queries)
                    enhanced += 1
                    modifications += 1
            if enhanced > 0:
                logger.info(f"Added alternative queries to {enhanced} steps")
        
        # Critical: Update iteration count to prevent infinite loops
        if hasattr(improved_plan, 'iteration'):
            improved_plan.iteration = (improved_plan.iteration or 0) + 1
        
        # Log the incremental changes
        logger.info(
            f"Incremental refinement: {len(plan.steps)} → {len(improved_plan.steps)} steps "
            f"({modifications} modifications)"
        )
        
        return improved_plan
    
    def _generate_coverage_step(self, plan: Plan, state: Dict) -> Optional[Step]:
        """Generate a single step to improve coverage."""
        try:
            # Analyze what's missing from current plan
            existing_topics = set()
            for step in plan.steps:
                if step.search_queries:
                    for query in step.search_queries:
                        existing_topics.update(query.lower().split())
            
            # Find uncovered aspects of the research topic
            topic_words = set(state.get("research_topic", "").lower().split())
            uncovered = topic_words - existing_topics
            
            if not uncovered:
                return None
            
            # Create targeted step for uncovered aspects
            step_id = f"step_{len(plan.steps) + 1:03d}"
            uncovered_terms = " ".join(list(uncovered)[:3])  # Limit to 3 terms
            
            return Step(
                step_id=step_id,
                title=f"Research {uncovered_terms}",
                description=f"Fill knowledge gap about {uncovered_terms}",
                step_type=StepType.RESEARCH,
                need_search=True,
                search_queries=[uncovered_terms[:80]],  # Single focused query
                depends_on=[plan.steps[-1].step_id] if plan.steps else [],
                status=StepStatus.PENDING
            )
        except Exception as e:
            logger.warning(f"Error generating coverage step: {e}")
            return None
    
    def _enhance_description(self, step: Step, state: Dict) -> str:
        """Enhance a step description for clarity."""
        if not step.description:
            step.description = ""
        
        # Add context about why this step is important
        topic = state.get("research_topic", "the topic")
        enhanced = f"{step.description} This step explores {step.title.lower()} "
        enhanced += f"to provide comprehensive coverage of {topic[:50]}."
        
        # Keep it concise
        return enhanced[:150]
    
    def _generate_alternative_queries(self, step: Step, max_new: int = 2) -> List[str]:
        """Generate alternative search queries for better coverage."""
        if not step.search_queries:
            return []
        
        alternatives = []
        for query in step.search_queries[:1]:  # Based on first query only
            # Generate variations
            words = query.split()
            if len(words) > 2:
                # Reorder variation
                alternatives.append(" ".join(words[1:] + words[:1]))
            if len(alternatives) >= max_new:
                break
        
        # Ensure queries are concise
        return [q[:80] for q in alternatives[:max_new]]
    
    def _generate_minimal_plan(self, topic: str) -> Plan:
        """Generate minimal plan when memory is constrained."""
        plan_id = f"plan_{uuid4().hex[:8]}"
        
        # Just 2 essential steps
        steps = [
            Step(
                step_id="step_001",
                title="Initial Research",
                description="Gather core information",
                step_type=StepType.RESEARCH,
                need_search=True,
                search_queries=[topic[:80]],  # Single query
                depends_on=[],
                status=StepStatus.PENDING
            ),
            Step(
                step_id="step_002",
                title="Synthesis",
                description="Compile findings",
                step_type=StepType.PROCESSING,
                need_search=False,
                depends_on=["step_001"],
                status=StepStatus.PENDING
            )
        ]
        
        return Plan(
            plan_id=plan_id,
            title=f"Minimal Plan: {topic[:50]}",
            research_topic=topic,
            thought="Memory-constrained minimal plan",
            has_enough_context=False,
            steps=steps,
            iteration=0
        )
