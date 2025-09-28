"""
Flexible matching between sections and steps.

This module provides robust matching that handles various ID formats and section titles,
ensuring that research steps can always be found even when section names don't match exactly.
"""

from typing import Optional, List
from difflib import SequenceMatcher

from . import get_logger
from .id_generator import PlanIDGenerator

logger = get_logger(__name__)


class FlexibleMatcher:
    """Match sections to steps using multiple strategies."""
    
    def __init__(self):
        self.similarity_threshold = 0.6  # 60% similarity for fuzzy matching
    
    def find_matching_step(self, section_identifier: str, plan) -> Optional:
        """
        Find matching step using multiple strategies.
        
        Args:
            section_identifier: Could be title like "Executive Summary", ID like "section_001", etc.
            plan: Research plan with steps
            
        Returns:
            Matching Step object or None
        """
        if not plan or not hasattr(plan, 'steps') or not plan.steps:
            logger.warning("Plan has no steps to match against")
            return None
            
        steps = plan.steps
        
        # Strategy 1: Direct ID match
        step = self._match_by_id(section_identifier, steps)
        if step:
            logger.info(f"Matched '{section_identifier}' by ID to step '{step.step_id}'")
            return step
        
        # Strategy 2: Title similarity match
        step = self._match_by_title_similarity(section_identifier, steps)
        if step:
            logger.info(f"Matched '{section_identifier}' by title similarity to step '{step.step_id}': '{step.title}'")
            return step
        
        # Strategy 3: Substring match
        step = self._match_by_substring(section_identifier, steps)
        if step:
            logger.info(f"Matched '{section_identifier}' by substring to step '{step.step_id}': '{step.title}'")
            return step
        
        # Strategy 4: Index-based semantic match
        step = self._match_by_semantic_index(section_identifier, steps)
        if step:
            logger.info(f"Matched '{section_identifier}' by semantic index to step '{step.step_id}': '{step.title}'")
            return step
        
        # Strategy 5: Position-based fallback
        step = self._match_by_position(section_identifier, steps)
        if step:
            logger.info(f"Matched '{section_identifier}' by position to step '{step.step_id}': '{step.title}'")
            return step
        
        # If nothing matches, log warning and return first step as fallback
        logger.warning(f"Could not match section '{section_identifier}' to any step. Available steps: {[s.title for s in steps]}")
        if steps:
            logger.info(f"Using first step '{steps[0].step_id}' as fallback")
            return steps[0]
        
        return None
    
    def _match_by_id(self, section_identifier: str, steps: List) -> Optional:
        """Match by ID, handling various formats."""
        # Try direct ID match first
        for step in steps:
            if hasattr(step, 'step_id') and step.step_id == section_identifier:
                return step
        
        # Try normalized ID match
        try:
            normalized_section = PlanIDGenerator.normalize_id(section_identifier)
            for step in steps:
                if hasattr(step, 'step_id') and step.step_id == normalized_section:
                    return step
                
                # Also try normalizing step ID
                normalized_step = PlanIDGenerator.normalize_id(step.step_id)
                if normalized_step == normalized_section:
                    return step
        except Exception as e:
            logger.debug(f"Error in ID normalization: {e}")
            
        return None
    
    def _match_by_title_similarity(self, section_identifier: str, steps: List) -> Optional:
        """Match by title using fuzzy similarity."""
        best_match = None
        best_score = 0.0
        
        for step in steps:
            if not hasattr(step, 'title') or not step.title:
                continue
                
            # Compare with step title
            score = SequenceMatcher(
                None, 
                section_identifier.lower().strip(), 
                step.title.lower().strip()
            ).ratio()
            
            if score > best_score and score >= self.similarity_threshold:
                best_score = score
                best_match = step
        
        return best_match
    
    def _match_by_substring(self, section_identifier: str, steps: List) -> Optional:
        """Match by substring containment."""
        section_lower = section_identifier.lower().strip()
        
        for step in steps:
            if not hasattr(step, 'title') or not step.title:
                continue
                
            step_title_lower = step.title.lower().strip()
            
            # Check if either contains the other
            if (section_lower in step_title_lower or 
                step_title_lower in section_lower):
                return step
                
            # Check word-level overlap
            section_words = set(section_lower.split())
            step_words = set(step_title_lower.split())
            
            # If more than half the words overlap, consider it a match
            if section_words and step_words:
                overlap = section_words & step_words
                min_words = min(len(section_words), len(step_words))
                if len(overlap) > min_words * 0.5:
                    return step
        
        return None
    
    def _match_by_semantic_index(self, section_identifier: str, steps: List) -> Optional:
        """Match by semantic meaning and expected index."""
        try:
            expected_index = PlanIDGenerator.extract_index_from_title(section_identifier)
            if expected_index is not None and 1 <= expected_index <= len(steps):
                return steps[expected_index - 1]  # Convert to 0-based index
        except Exception as e:
            logger.debug(f"Error in semantic index matching: {e}")
            
        return None
    
    def _match_by_position(self, section_identifier: str, steps: List) -> Optional:
        """Match by position as last resort."""
        if not steps:
            return None
            
        # Extract any number from the section identifier
        import re
        number_match = re.search(r'(\d+)', section_identifier)
        if number_match:
            try:
                index = int(number_match.group(1)) - 1  # Convert to 0-based
                if 0 <= index < len(steps):
                    return steps[index]
            except (ValueError, IndexError):
                pass
        
        # If section identifier suggests it's the first/last
        section_lower = section_identifier.lower()
        if any(word in section_lower for word in ['executive', 'summary', 'introduction', 'overview']):
            return steps[0]  # First step
        elif any(word in section_lower for word in ['conclusion', 'appendix', 'references']):
            return steps[-1]  # Last step
        
        return None
    
    def validate_matching_results(self, matches: List[tuple]) -> List[str]:
        """Validate that section-step matching makes sense."""
        issues = []
        
        used_steps = set()
        for section_id, step in matches:
            if step is None:
                issues.append(f"No step found for section '{section_id}'")
            elif step.step_id in used_steps:
                issues.append(f"Step '{step.step_id}' matched to multiple sections")
            else:
                used_steps.add(step.step_id)
        
        return issues