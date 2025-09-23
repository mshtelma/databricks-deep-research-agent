"""
Fact Checker Agent - Simplified and Reliable Implementation

This agent verifies the factual accuracy of research findings using a clean,
linear approach that always returns a valid dictionary.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from databricks_langchain import ChatDatabricks
from langchain_core.messages import HumanMessage, SystemMessage

from deep_research_agent.core.grounding import (
    FactualityReport,
    Claim,
    ClaimType,
    GroundingStatus,
    VerificationLevel
)
from deep_research_agent.core.entity_validation import get_global_validator, validate_content_global
from deep_research_agent.core.observation_models import observation_to_text
# from deep_research_agent.core.multi_agent_state import EnhancedResearchState  # Not needed

logger = logging.getLogger(__name__)


class FactCheckerAgent:
    """
    Simplified Fact Checker Agent that verifies factual accuracy of research.
    
    Key improvements:
    - Always returns a valid dictionary
    - Linear flow without complex branching
    - No undefined variable errors
    - Clear error handling
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, llm=None, event_emitter=None, embedding_manager=None):
        """Initialize the fact checker with configuration."""
        self.config = config or {}
        self.event_emitter = event_emitter
        self.embedding_manager = embedding_manager
        
        # Model configuration
        model_config = self.config.get('models', {}).get('default', {})
        self.model_endpoint = model_config.get('endpoint', 'databricks-gpt-oss-120b')
        self.temperature = model_config.get('temperature', 0.3)  # Lower for fact checking
        self.max_tokens = model_config.get('max_tokens', 2000)
        
        # Fact checker configuration
        agent_config = self.config.get('agents', {}).get('fact_checker', {})
        self.verification_level = agent_config.get('verification_level', 'moderate')
        self.enable_contradiction_detection = agent_config.get('enable_contradiction_detection', True)
        
        # Grounding configuration
        grounding_config = self.config.get('grounding', {})
        self.factuality_threshold = grounding_config.get('factuality_threshold', 0.6)
        
        # Use provided LLM or initialize new one
        if llm:
            self.llm = llm
            logger.info("Using provided LLM for FactCheckerAgent")
        else:
            try:
                self.llm = ChatDatabricks(
                    endpoint=self.model_endpoint,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                logger.info(f"Initialized FactCheckerAgent with endpoint: {self.model_endpoint}")
            except Exception as e:
                logger.error(f"Failed to initialize LLM: {e}")
                self.llm = None
    
    def __call__(self, state: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main entry point - always returns a valid dictionary.
        
        CRITICAL: This method MUST return a dict, never a Command object.
        The workflow will handle any Command creation if needed.
        
        Args:
            state: Current research state dictionary
            config: Optional configuration dictionary (passed by workflow but not used)
            
        Returns:
            Dict[str, Any]: Dictionary with fact checking results (NEVER a Command)
        """
        logger.info("=" * 80)
        logger.info("FACT CHECKER AGENT STARTING")
        logger.info("=" * 80)
        
        # Initialize default return structure - this is ALWAYS returned
        result = self._create_default_result()
        
        try:
            # Extract content to verify
            content = self._extract_content(state)
            if not content:
                logger.warning("No content to fact-check, checking if this is expected")
                # Check if researcher failed or no observations were generated
                if not state.get('observations') and not state.get('search_results'):
                    logger.info("No observations or search results available - researcher may have failed")
                    result['factuality_score'] = 0.5  # Neutral score when no data
                    result['confidence_level'] = 'low'
                    result['needs_revision'] = False  # Can't revise what doesn't exist
                    result['grounding_analysis'] = self._create_default_grounding_analysis()
                    result['grounding_analysis']['confidence_score'] = 0.5
                    return result
                # If we have some data but couldn't extract content, that's unexpected
                logger.warning("Have state data but couldn't extract content for fact checking")
                return result
            
            # ENTITY VALIDATION: Validate content against requested entities
            entity_validation_result = validate_content_global(content, context="fact_checker")
            if not entity_validation_result.is_valid:
                logger.warning(f"FACT_CHECKER: Entity validation failed. Violations: {entity_validation_result.violations}")
                # Add violation to state for tracking
                if 'entity_violations' not in result:
                    result['entity_violations'] = []
                result['entity_violations'].append({
                    'agent': 'fact_checker',
                    'violations': list(entity_validation_result.violations),
                    'content_preview': content[:200] + "..." if len(content) > 200 else content,
                    'timestamp': datetime.now().isoformat()
                })
                
                # For strict validation, reduce factuality score
                validator = get_global_validator()
                if validator and validator.mode.value == 'strict':
                    logger.warning("FACT_CHECKER: Strict entity validation failed, reducing factuality score")
                    result['factuality_score'] = max(0.3, result.get('factuality_score', 0.7) - 0.4)
                    result['needs_revision'] = True
                    result['confidence_level'] = 'low'
                    result['revision_reason'] = f"Content mentions forbidden entities: {', '.join(entity_validation_result.violations)}"
            else:
                logger.info(f"FACT_CHECKER: Entity validation passed. Coverage: {entity_validation_result.coverage_score:.2f}")
            
            # Extract sources for verification
            sources = self._extract_sources(state)
            
            # Create factuality report
            factuality_report = self._create_factuality_report(content, sources)
            
            # Update result with report data
            result = self._update_result_from_report(result, factuality_report)
            
            # Update state fields that workflow expects
            result['factuality_report'] = factuality_report
            result['factuality_score'] = factuality_report.overall_factuality_score if factuality_report else 0.7
            
            # Ensure all expected fields are present
            if 'grounding_analysis' not in result or result['grounding_analysis'] is None:
                result['grounding_analysis'] = self._create_default_grounding_analysis()
            
            # Log summary
            self._log_summary(result)
            
        except Exception as e:
            logger.error(f"Error in fact checking: {e}", exc_info=True)
            result['error'] = str(e)
            result['factuality_score'] = 0.5  # Neutral score on error
            result['grounding_analysis'] = self._create_default_grounding_analysis()
        
        # Final validation - ensure we're returning a dict
        if not isinstance(result, dict):
            logger.error(f"Result is not a dict: {type(result)}, creating new default")
            result = self._create_default_result()
        
        # CRITICAL: Ensure we NEVER return a Command object
        # If somehow we have a Command, extract its update dict
        if hasattr(result, '__class__') and result.__class__.__name__ == 'Command':
            logger.error("CRITICAL: Result is a Command object, extracting update dict")
            if hasattr(result, 'update') and isinstance(getattr(result, 'update'), dict):
                result = getattr(result, 'update')
            else:
                result = self._create_default_result()
        
        # Final type assertion
        assert isinstance(result, dict), f"Result must be dict, got {type(result)}"
        
        logger.info("=" * 80)
        logger.info("FACT CHECKER AGENT COMPLETED")
        logger.info(f"Returning dict with keys: {list(result.keys())}")
        logger.info("=" * 80)
        
        return result
    
    def _create_default_result(self) -> Dict[str, Any]:
        """Create a default result structure that's always valid."""
        return {
            'factuality_report': None,
            'factuality_score': 0.7,  # Default moderate confidence
            'grounding_analysis': None,  # Will be set as needed
            'contradictions': [],
            'verified_claims': [],
            'unverified_claims': [],
            'confidence_level': 'moderate',
            'needs_revision': False,
            'revision_suggestions': [],
            'timestamp': datetime.now().isoformat(),
            'verification_level': self.verification_level
        }
    
    def _create_default_grounding_analysis(self) -> Dict[str, Any]:
        """Create a default grounding analysis."""
        return {
            'total_claims': 0,
            'verified_claims': 0,
            'unverified_claims': 0,
            'contradiction_count': 0,
            'confidence_score': 0.7,
            'verification_level': self.verification_level
        }
    
    def _extract_content(self, state: Dict[str, Any]) -> str:
        """Extract content to verify from state.
        
        Note: Fact checker runs BEFORE reporter, so final_report doesn't exist yet.
        We need to check observations and research findings instead.
        """
        # Check observations first (these are what researcher produces)
        observations = state.get('observations', [])
        if observations:
            # Join all observations into content
            content = '\n'.join(observation_to_text(obs) for obs in observations if obs)
            if content.strip():
                logger.debug(f"Using {len(observations)} observations for fact checking")
                return content
        
        # Check research_observations (alternative field name)
        research_obs = state.get('research_observations', [])
        if research_obs:
            content = '\n'.join(observation_to_text(obs) for obs in research_obs if obs)
            if content.strip():
                logger.debug(f"Using {len(research_obs)} research_observations for fact checking")
                return content
        
        # Try other possible locations (in case workflow has already generated something)
        content_sources = [
            state.get('synthesis', ''),
            state.get('current_draft', ''),
            state.get('research_findings', ''),
            state.get('final_report', ''),  # Unlikely to exist but check anyway
        ]
        
        for content in content_sources:
            if content and isinstance(content, str) and len(content.strip()) > 10:
                logger.debug(f"Using alternative content source for fact checking")
                return content
        
        # If still no content, try to extract from search results
        search_results = state.get('search_results', [])
        if search_results:
            # Create content from search result snippets
            snippets = []
            for result in search_results[:5]:  # Use first 5 results
                if isinstance(result, dict):
                    snippet = result.get('snippet', '') or result.get('content', '')
                    if snippet:
                        snippets.append(snippet)
            if snippets:
                content = '\n'.join(snippets)
                logger.debug(f"Using {len(snippets)} search result snippets for fact checking")
                return content
        
        logger.debug("No content found for fact checking in any expected location")
        return ''
    
    def _extract_sources(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract sources from state for verification."""
        sources = []
        
        # Try to get sources from various locations
        if 'search_results' in state and isinstance(state['search_results'], list):
            for result in state['search_results']:
                if isinstance(result, dict):
                    sources.append({
                        'title': result.get('title', 'Unknown'),
                        'url': result.get('url', ''),
                        'snippet': result.get('snippet', ''),
                        'content': result.get('content', '')
                    })
        
        if 'sources' in state and isinstance(state['sources'], list):
            sources.extend(state['sources'])
        
        if 'citations' in state and isinstance(state['citations'], list):
            for citation in state['citations']:
                if isinstance(citation, dict):
                    sources.append({
                        'title': citation.get('title', 'Unknown'),
                        'url': citation.get('source', ''),
                        'snippet': citation.get('snippet', ''),
                        'content': citation.get('content', '')
                    })
        
        return sources
    
    def _create_factuality_report(self, content: str, sources: List[Dict[str, Any]]) -> FactualityReport:
        """
        Create a factuality report for the content.
        
        This is simplified - no complex LLM loops or undefined variables.
        """
        # Extract claims from content
        claims = self._extract_claims(content)
        
        # Verify claims against sources
        verified_claims = []
        unverified_claims = []
        
        for claim in claims:
            if self._verify_claim(claim, sources):
                verified_claims.append(claim)
            else:
                unverified_claims.append(claim)
        
        # Calculate factuality score
        total_claims = len(claims)
        if total_claims > 0:
            factuality_score = len(verified_claims) / total_claims
        else:
            factuality_score = 0.7  # Default when no claims
        
        # Check for contradictions if enabled
        contradictions = []
        if self.enable_contradiction_detection:
            contradictions = self._check_contradictions(claims, sources)
        
        # Create grounding analysis
        grounding_analysis = {
            'total_claims': total_claims,
            'verified_claims': len(verified_claims),
            'unverified_claims': len(unverified_claims),
            'contradiction_count': len(contradictions),
            'confidence_score': factuality_score,
            'verification_level': self.verification_level
        }
        
        # Create report - using the actual FactualityReport schema
        report = FactualityReport(
            total_claims=total_claims,
            grounded_claims=len(verified_claims),
            partially_grounded_claims=0,  # Simplified - not distinguishing partial
            ungrounded_claims=len(unverified_claims),
            contradicted_claims=0,  # Will be set based on contradictions
            overall_factuality_score=factuality_score,
            confidence_score=factuality_score,
            grounding_results=[],  # Would need GroundingResult objects
            contradictions=[],  # Would need Contradiction objects
            hallucination_risks=[],
            recommendations=self._generate_revision_suggestions(
                unverified_claims, contradictions
            )
        )
        
        return report
    
    def _extract_claims(self, content: str) -> List[Claim]:
        """Extract factual claims from content."""
        claims = []
        
        if not self.llm:
            logger.warning("LLM not initialized, using basic claim extraction")
            # Basic extraction - split into sentences
            sentences = content.split('.')
            for i, sentence in enumerate(sentences[:20]):  # Limit to first 20
                if len(sentence.strip()) > 20:
                    claims.append(Claim(
                        text=sentence.strip() + '.',
                        claim_type=ClaimType.FACTUAL,
                        start_position=0,
                        end_position=len(sentence),
                        confidence=0.5
                    ))
            return claims
        
        try:
            prompt = f"""Extract the main factual claims from this content.
Return each claim as a separate line.
Focus on statements that can be verified as true or false.
Limit to the 10 most important claims.

Content:
{content[:3000]}

Claims:"""
            
            messages = [
                SystemMessage(content="You are a fact extraction specialist."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            # Handle structured responses properly
            from deep_research_agent.core.llm_response_parser import extract_text_from_response
            content = extract_text_from_response(response)
            claim_texts = content.strip().split('\n')
            
            for text in claim_texts:
                text = text.strip()
                if text and len(text) > 10:
                    # Remove bullet points or numbers
                    text = text.lstrip('•-*0123456789. ')
                    if text:
                        claims.append(Claim(
                            text=text,
                            claim_type=ClaimType.FACTUAL,
                            start_position=0,
                            end_position=len(text),
                            confidence=0.7
                        ))
            
        except Exception as e:
            logger.error(f"Error extracting claims: {e}")
            # Fallback to basic extraction
            sentences = content.split('.')[:10]
            for sentence in sentences:
                if len(sentence.strip()) > 20:
                    claims.append(Claim(
                        text=sentence.strip() + '.',
                        claim_type=ClaimType.FACTUAL,
                        start_position=0,
                        end_position=len(sentence),
                        confidence=0.5
                    ))
        
        return claims
    
    def _verify_claim(self, claim: Claim, sources: List[Dict[str, Any]]) -> bool:
        """Verify if a claim is supported by sources."""
        if not sources:
            return False
        
        claim_text_lower = claim.text.lower()
        
        # Check each source for supporting evidence
        for source in sources:
            source_content = ' '.join([
                str(source.get('snippet', '')),
                str(source.get('content', ''))
            ]).lower()
            
            # Simple keyword matching
            # In production, this could use embeddings or more sophisticated matching
            keywords = [word for word in claim_text_lower.split() 
                       if len(word) > 4 and word not in ['that', 'this', 'with', 'from', 'have']]
            
            if len(keywords) > 0:
                matching_keywords = sum(1 for keyword in keywords if keyword in source_content)
                if matching_keywords / len(keywords) > 0.5:
                    return True
        
        return False
    
    def _check_contradictions(self, claims: List[Claim], sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Check for contradictions between claims and sources."""
        contradictions = []
        
        # Simple contradiction detection
        # In production, this would use more sophisticated NLI models
        for i, claim1 in enumerate(claims):
            for j, claim2 in enumerate(claims[i+1:], i+1):
                if self._are_contradictory(claim1.text, claim2.text):
                    contradictions.append({
                        'claim1': claim1.text,
                        'claim2': claim2.text,
                        'severity': 'medium',
                        'explanation': 'Potential contradiction detected between claims'
                    })
        
        return contradictions[:5]  # Limit to 5 contradictions
    
    def _are_contradictory(self, text1: str, text2: str) -> bool:
        """Simple contradiction detection."""
        # Very basic - check for opposite words
        opposites = [
            ('increase', 'decrease'),
            ('higher', 'lower'),
            ('more', 'less'),
            ('positive', 'negative'),
            ('true', 'false'),
            ('yes', 'no')
        ]
        
        text1_lower = text1.lower()
        text2_lower = text2.lower()
        
        for word1, word2 in opposites:
            if word1 in text1_lower and word2 in text2_lower:
                return True
            if word2 in text1_lower and word1 in text2_lower:
                return True
        
        return False
    
    def _generate_revision_suggestions(self, 
                                      unverified_claims: List[Claim],
                                      contradictions: List[Dict[str, Any]]) -> List[str]:
        """Generate suggestions for improving factual accuracy."""
        suggestions = []
        
        if unverified_claims:
            suggestions.append(f"Verify or provide sources for {len(unverified_claims)} unverified claims")
            
            # Add specific suggestions for first few claims
            for claim in unverified_claims[:3]:
                suggestions.append(f"Find supporting evidence for: {claim.text[:100]}...")
        
        if contradictions:
            suggestions.append(f"Resolve {len(contradictions)} contradictions found in the content")
            
            for contradiction in contradictions[:2]:
                suggestions.append(f"Clarify contradiction: {contradiction.get('explanation', 'Unknown')}")
        
        if not suggestions:
            suggestions.append("Content appears factually sound, consider adding more specific citations")
        
        return suggestions
    
    def _update_result_from_report(self, result: Dict[str, Any], report: FactualityReport) -> Dict[str, Any]:
        """Update the result dictionary with report data."""
        result['factuality_report'] = report
        result['factuality_score'] = report.overall_factuality_score
        result['grounding_analysis'] = {
            'total_claims': report.total_claims,
            'verified_claims': report.grounded_claims,
            'unverified_claims': report.ungrounded_claims,
            'contradiction_count': len(report.contradictions),
            'confidence_score': report.confidence_score,
            'verification_level': str(report.verification_level)
        }
        result['contradictions'] = report.contradictions
        result['verified_claims'] = []  # We don't have the actual claim objects anymore
        result['unverified_claims'] = []  # Same here
        result['needs_revision'] = report.overall_factuality_score < self.factuality_threshold
        result['revision_suggestions'] = report.recommendations
        
        # Set confidence level based on score
        if report.overall_factuality_score >= 0.8:
            result['confidence_level'] = 'high'
        elif report.overall_factuality_score >= 0.6:
            result['confidence_level'] = 'moderate'
        else:
            result['confidence_level'] = 'low'
        
        return result
    
    def _log_summary(self, result: Dict[str, Any]):
        """Log a summary of the fact checking results."""
        logger.info(f"Fact checking complete:")
        logger.info(f"  - Factuality score: {result['factuality_score']:.2f}")
        logger.info(f"  - Confidence level: {result['confidence_level']}")
        logger.info(f"  - Needs revision: {result['needs_revision']}")
        
        if result.get('grounding_analysis'):
            ga = result['grounding_analysis']
            if isinstance(ga, dict):
                logger.info(f"  - Total claims: {ga.get('total_claims', 0)}")
                logger.info(f"  - Verified: {ga.get('verified_claims', 0)}")
                logger.info(f"  - Unverified: {ga.get('unverified_claims', 0)}")
                logger.info(f"  - Contradictions: {ga.get('contradiction_count', 0)}")