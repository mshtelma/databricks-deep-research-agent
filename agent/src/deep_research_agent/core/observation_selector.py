"""
Observation Selector for intelligent selection and grouping of research observations.

This module provides semantic matching and relevance scoring to ensure the right
observations are used for each report section, preventing information loss and
improving report quality.
"""

import logging
import re
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ObservationRelevance:
    """Tracks relevance scoring for an observation."""
    observation: Any
    index: int
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    entity_score: float = 0.0
    total_score: float = 0.0
    matched_keywords: List[str] = field(default_factory=list)
    matched_entities: List[str] = field(default_factory=list)


class ObservationSelector:
    """
    Intelligently selects and groups observations for report sections.
    Uses multiple scoring methods for relevance including semantic similarity,
    keyword matching, and entity recognition.
    """
    
    def __init__(self, embedding_manager: Optional[Any] = None):
        """
        Initialize the observation selector.
        
        Args:
            embedding_manager: Optional embedding manager for semantic similarity
        """
        self.embedding_manager = embedding_manager
        self.cache = {}
        
    def select_observations_for_section(
        self,
        section_title: str,
        section_purpose: str,
        all_observations: List[Any],
        max_observations: int = 50,
        min_relevance: float = 0.2,
        use_semantic: bool = True,
        ensure_entity_diversity: bool = False,
        diversity_entities: Optional[List[str]] = None
    ) -> List[Any]:
        """
        Select relevant observations for a specific section.

        Args:
            section_title: Title of the report section
            section_purpose: Purpose/description of the section
            all_observations: All available observations
            max_observations: Maximum observations to return (default 50 for ~30k tokens)
            min_relevance: Minimum relevance score (0-1)
            use_semantic: Whether to use semantic similarity scoring
            ensure_entity_diversity: Force at least 1 observation per key entity
            diversity_entities: List of entities to ensure diversity across (e.g., country names)

        Returns:
            List of relevant observations sorted by relevance
        """
        if not all_observations:
            logger.warning(f"No observations available for section '{section_title}'")
            return []
            
        logger.info(f"Selecting observations for '{section_title}' from {len(all_observations)} available")
        
        # Adaptive min_relevance based on section type
        adaptive_min_relevance = self._get_adaptive_min_relevance(section_title, min_relevance)
        
        # Score all observations
        scored_observations = []
        
        # Extract keywords from section info
        section_keywords = self._extract_keywords(f"{section_title} {section_purpose}")
        
        for idx, obs in enumerate(all_observations):
            relevance = self._score_observation_relevance(
                observation=obs,
                index=idx,
                section_title=section_title,
                section_purpose=section_purpose,
                section_keywords=section_keywords,
                use_semantic=use_semantic and self.embedding_manager is not None
            )
            
            scored_observations.append(relevance)  # Keep ALL observations for fallback logic
        
        # Sort by total score
        scored_observations.sort(key=lambda x: x.total_score, reverse=True)
        
        # Primary selection: observations above adaptive threshold
        primary_selected = [r for r in scored_observations if r.total_score >= adaptive_min_relevance]
        
        # Fallback logic: ensure every section gets some observations
        if not primary_selected and scored_observations:
            logger.warning(f"No observations met threshold {adaptive_min_relevance:.3f} for '{section_title}', applying fallback")
            # Take top 20% of observations or minimum 3, whichever is higher
            fallback_count = max(3, len(scored_observations) // 5)
            primary_selected = scored_observations[:fallback_count]
            logger.info(f"Fallback: selected top {len(primary_selected)} observations for '{section_title}'")
        
        # Take top N observations from primary selection
        selected = primary_selected[:max_observations]

        # DIVERSITY ENFORCEMENT: Ensure key entities are represented
        if ensure_entity_diversity and diversity_entities:
            selected = self._ensure_entity_diversity(
                selected=selected,
                all_scored=scored_observations,
                diversity_entities=diversity_entities,
                max_observations=max_observations
            )

        # Log selection statistics
        if selected:
            avg_score = sum(r.total_score for r in selected) / len(selected)
            logger.info(
                f"Selected {len(selected)} observations for '{section_title}' "
                f"(threshold: {adaptive_min_relevance:.3f}, avg score: {avg_score:.3f}, "
                f"min: {selected[-1].total_score:.3f}, max: {selected[0].total_score:.3f})"
            )
            
            # Log token estimation (rough: ~300 tokens per observation)
            estimated_tokens = len(selected) * 300
            logger.info(f"Estimated tokens for section '{section_title}': ~{estimated_tokens:,} tokens")
            
            if estimated_tokens > 30000:
                logger.warning(
                    f"Section '{section_title}' may use {estimated_tokens:,} tokens. "
                    f"Model supports 130k context window - plenty of room available."
                )
        else:
            logger.error(f"CRITICAL: No observations selected for '{section_title}' - this should never happen with fallback logic!")
        
        # Return observations (not the wrapper objects)
        return [r.observation for r in selected]
    
    def _score_observation_relevance(
        self,
        observation: Any,
        index: int,
        section_title: str,
        section_purpose: str,
        section_keywords: List[str],
        use_semantic: bool = True
    ) -> ObservationRelevance:
        """Score a single observation for relevance to a section."""
        
        relevance = ObservationRelevance(observation=observation, index=index)
        
        # Extract observation content
        obs_content = self._extract_observation_content(observation)
        obs_entities = self._extract_observation_entities(observation)
        
        # 1. Keyword scoring (40% weight)
        relevance.keyword_score, relevance.matched_keywords = self._score_keyword_match(
            obs_content, section_keywords
        )
        
        # 2. Entity scoring (30% weight) 
        relevance.entity_score, relevance.matched_entities = self._score_entity_match(
            obs_entities, section_title, section_keywords
        )
        
        # 3. Semantic scoring (30% weight) - if available
        if use_semantic and self.embedding_manager:
            try:
                relevance.semantic_score = self._score_semantic_similarity(
                    obs_content, f"{section_title} {section_purpose}"
                )
            except Exception as e:
                logger.debug(f"Semantic scoring failed: {e}")
                relevance.semantic_score = 0.0
        
        # Adaptive weights based on section type
        section_lower = section_title.lower()
        is_meta_section = any(meta in section_lower for meta in [
            'executive summary', 'summary', 'overview', 'introduction',
            'key findings', 'findings', 'conclusions', 'conclusion'
        ])
        
        if is_meta_section:
            # Meta-sections benefit more from position/recency and less from exact matching
            weights = {
                'keyword': 0.25 if not use_semantic else 0.2,
                'entity': 0.2,
                'semantic': 0.25 if use_semantic else 0.0,
                'bonus': 0.5  # Higher bonus for meta-sections
            }
        else:
            # Regular sections use standard weighting
            weights = {
                'keyword': 0.4 if not use_semantic else 0.35,
                'entity': 0.3,
                'semantic': 0.35 if use_semantic else 0.0,
                'bonus': 0.3
            }
        
        # Add position bonus (more recent observations slightly preferred)
        position_bonus = 0.2 * (1.0 - index / 1000)  # Small decay over position
        
        relevance.total_score = (
            weights['keyword'] * relevance.keyword_score +
            weights['entity'] * relevance.entity_score +
            weights['semantic'] * relevance.semantic_score +
            weights['bonus'] * position_bonus
        )
        
        return relevance
    
    def _get_adaptive_min_relevance(self, section_title: str, default_min_relevance: float) -> float:
        """
        Get adaptive minimum relevance threshold based on section type.
        Meta-sections like Executive Summary need lower thresholds.
        """
        section_lower = section_title.lower()
        
        # Meta-sections that synthesize information from other sections
        meta_sections = [
            'executive summary', 'summary', 'overview', 'introduction',
            'key findings', 'findings', 'conclusions', 'conclusion',
            'recommendations', 'next steps', 'action items'
        ]
        
        # Analysis sections that need specific content
        analysis_sections = [
            'analysis', 'comparison', 'comparative analysis', 'methodology',
            'technical', 'appendix', 'details'
        ]
        
        # Check if this is a meta-section
        for meta in meta_sections:
            if meta in section_lower:
                # Much lower threshold for meta-sections - they should get broad content
                return max(0.05, default_min_relevance * 0.25)
        
        # Check if this is an analysis section
        for analysis in analysis_sections:
            if analysis in section_lower:
                # Slightly lower threshold for analysis sections
                return max(0.1, default_min_relevance * 0.75)
        
        # Default threshold for specific content sections
        return default_min_relevance
    
    def _extract_observation_content(self, observation: Any) -> str:
        """Extract text content from an observation."""
        # Handle StructuredObservation
        if hasattr(observation, 'content'):
            return observation.content
        # Handle dict
        elif isinstance(observation, dict):
            return observation.get('content', str(observation))
        # Fallback to string
        else:
            return str(observation)
    
    def _extract_observation_entities(self, observation: Any) -> List[str]:
        """Extract entity tags from an observation."""
        # Handle StructuredObservation
        if hasattr(observation, 'entity_tags'):
            return observation.entity_tags or []
        # Handle dict
        elif isinstance(observation, dict):
            return observation.get('entity_tags', [])
        # Try to extract from content
        else:
            content = str(observation)
            # Simple entity extraction - look for capitalized words, countries, etc.
            entities = []
            # Country names (extend this list)
            countries = ['Spain', 'France', 'Germany', 'Poland', 'Bulgaria', 'Switzerland', 'United Kingdom', 'UK']
            for country in countries:
                if country.lower() in content.lower():
                    entities.append(country)
            return entities
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        # Simple keyword extraction - split and filter
        words = re.findall(r'\b\w+\b', text.lower())
        # Filter out common words
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'as', 'is', 'was', 'are', 'were'}
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        return keywords
    
    def _score_keyword_match(self, content: str, keywords: List[str]) -> Tuple[float, List[str]]:
        """Score observation based on keyword matches."""
        if not keywords:
            return 0.0, []
            
        content_lower = content.lower()
        matched = []
        
        for keyword in keywords:
            if keyword in content_lower:
                matched.append(keyword)
        
        # Score based on percentage of keywords matched
        score = len(matched) / len(keywords) if keywords else 0.0
        
        # Bonus for exact phrase matches
        for i in range(len(keywords) - 1):
            phrase = f"{keywords[i]} {keywords[i+1]}"
            if phrase in content_lower:
                score += 0.1
                
        # Cap at 1.0
        score = min(1.0, score)
        
        return score, matched
    
    def _score_entity_match(self, obs_entities: List[str], section_title: str, keywords: List[str]) -> Tuple[float, List[str]]:
        """Score observation based on entity matches."""
        if not obs_entities:
            return 0.0, []
            
        section_text = f"{section_title} {' '.join(keywords)}".lower()
        matched = []
        
        for entity in obs_entities:
            if entity.lower() in section_text:
                matched.append(entity)
        
        # Higher score if multiple entities match
        if len(matched) >= 2:
            score = 1.0
        elif len(matched) == 1:
            score = 0.6
        else:
            score = 0.0
            
        return score, matched
    
    def _score_semantic_similarity(self, obs_content: str, section_context: str) -> float:
        """Score observation using semantic similarity."""
        if not self.embedding_manager:
            return 0.0
            
        try:
            # Get embeddings
            obs_embedding = self.embedding_manager.get_embedding(obs_content)
            section_embedding = self.embedding_manager.get_embedding(section_context)
            
            # Compute cosine similarity
            similarity = np.dot(obs_embedding, section_embedding) / (
                np.linalg.norm(obs_embedding) * np.linalg.norm(section_embedding)
            )
            
            # Normalize to 0-1 range
            return (similarity + 1) / 2
            
        except Exception as e:
            logger.debug(f"Semantic similarity calculation failed: {e}")
            return 0.0
    
    def group_observations_by_topic(
        self,
        observations: List[Any],
        section_title: str,
        max_groups: int = 10
    ) -> Dict[str, List[Any]]:
        """
        Group observations by topic/entity for better organization.
        
        Args:
            observations: List of observations to group
            section_title: Title of the section (for context)
            max_groups: Maximum number of groups to create
            
        Returns:
            Dictionary mapping topic/entity to list of observations
        """
        groups = defaultdict(list)
        
        for obs in observations:
            # Extract primary topic/entity
            entities = self._extract_observation_entities(obs)
            
            if entities:
                # Use first entity as primary grouping key
                primary_topic = entities[0]
            else:
                # Fallback: try to extract from content
                content = self._extract_observation_content(obs)
                primary_topic = self._extract_primary_topic(content)
            
            groups[primary_topic].append(obs)
        
        # If too many groups, consolidate smaller ones
        if len(groups) > max_groups:
            groups = self._consolidate_groups(groups, max_groups)
        
        logger.info(
            f"Grouped {len(observations)} observations into {len(groups)} topics "
            f"for section '{section_title}'"
        )
        
        return dict(groups)
    
    def _extract_primary_topic(self, content: str) -> str:
        """Extract primary topic from content."""
        # Simple heuristic: look for country names first
        countries = ['Spain', 'France', 'Germany', 'Poland', 'Bulgaria', 'Switzerland', 'United Kingdom', 'UK']
        content_lower = content.lower()

        for country in countries:
            if country.lower() in content_lower:
                return country

        # Look for other key terms
        key_terms = ['tax', 'income', 'RSU', 'salary', 'childcare', 'housing', 'rent', 'benefit']
        for term in key_terms:
            if term.lower() in content_lower:
                return term

        # Default
        return 'general'

    def extract_key_entities_from_topic(self, research_topic: str) -> List[str]:
        """
        Extract key entities (countries, companies, products, etc.) from research topic.

        This helps identify which entities should have diverse representation in observations.

        Args:
            research_topic: The research question/topic

        Returns:
            List of entity names found in the topic
        """
        # Extended country list for comprehensive matching
        countries = [
            'Spain', 'France', 'Germany', 'Poland', 'Bulgaria', 'Switzerland',
            'United Kingdom', 'UK', 'USA', 'United States', 'Canada', 'Australia',
            'Italy', 'Netherlands', 'Belgium', 'Austria', 'Sweden', 'Norway',
            'Denmark', 'Finland', 'Portugal', 'Greece', 'Ireland', 'Japan',
            'China', 'India', 'Brazil', 'Mexico', 'Argentina', 'Chile'
        ]

        topic_lower = research_topic.lower()
        found_entities = []

        # Find countries
        for country in countries:
            if country.lower() in topic_lower:
                found_entities.append(country)

        # De-duplicate (e.g., "UK" and "United Kingdom")
        # Keep longer names
        deduplicated = []
        for entity in found_entities:
            # Check if this entity is a substring of another found entity
            is_substring = False
            for other in found_entities:
                if entity != other and entity.lower() in other.lower():
                    is_substring = True
                    break

            if not is_substring:
                deduplicated.append(entity)

        logger.info(
            f"[ENTITY EXTRACTION] Found {len(deduplicated)} key entities in topic: "
            f"{', '.join(deduplicated)}"
        )

        return deduplicated
    
    def _consolidate_groups(self, groups: Dict[str, List[Any]], max_groups: int) -> Dict[str, List[Any]]:
        """Consolidate smaller groups into 'other' category."""
        # Sort groups by size
        sorted_groups = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)

        # Keep top groups, merge rest into 'other'
        consolidated = {}
        other = []

        for i, (topic, obs_list) in enumerate(sorted_groups):
            if i < max_groups - 1:
                consolidated[topic] = obs_list
            else:
                other.extend(obs_list)

        if other:
            consolidated['other'] = other

        return consolidated

    def _ensure_entity_diversity(
        self,
        selected: List[ObservationRelevance],
        all_scored: List[ObservationRelevance],
        diversity_entities: List[str],
        max_observations: int
    ) -> List[ObservationRelevance]:
        """
        Ensure observations about each key entity are represented in the selection.

        This prevents scenarios where all observations are about one country (e.g., Spain)
        when the research covers multiple countries.

        Args:
            selected: Initially selected observations
            all_scored: All scored observations (sorted by score)
            diversity_entities: List of entities to ensure diversity (e.g., country names)
            max_observations: Maximum observations to return

        Returns:
            Modified selection with guaranteed diversity
        """
        logger.info(
            f"[DIVERSITY] Ensuring representation of {len(diversity_entities)} entities: "
            f"{', '.join(diversity_entities)}"
        )

        # Track which entities are already represented in selected observations
        represented_entities = set()
        entity_coverage = {entity: [] for entity in diversity_entities}

        for rel in selected:
            obs_entities = self._extract_observation_entities(rel.observation)
            for entity in diversity_entities:
                # Case-insensitive matching
                if any(entity.lower() in obs_entity.lower() for obs_entity in obs_entities):
                    represented_entities.add(entity)
                    entity_coverage[entity].append(rel)

        missing_entities = set(diversity_entities) - represented_entities

        if not missing_entities:
            logger.info(
                f"[DIVERSITY] All {len(diversity_entities)} entities already represented. "
                f"Coverage: {', '.join(f'{e}({len(entity_coverage[e])})' for e in diversity_entities)}"
            )
            return selected

        logger.warning(
            f"[DIVERSITY] Missing {len(missing_entities)} entities in initial selection: "
            f"{', '.join(missing_entities)}"
        )

        # Find best observations for missing entities from all_scored
        diversity_additions = []
        for entity in missing_entities:
            # Find observations mentioning this entity
            candidates = []
            for rel in all_scored:
                if rel in selected:
                    continue  # Already selected

                obs_entities = self._extract_observation_entities(rel.observation)
                if any(entity.lower() in obs_entity.lower() for obs_entity in obs_entities):
                    candidates.append(rel)

            if candidates:
                # Take best scoring observation for this entity
                best = candidates[0]  # Already sorted by score
                diversity_additions.append(best)
                entity_coverage[entity].append(best)
                logger.info(
                    f"[DIVERSITY] Added observation for '{entity}' (score: {best.total_score:.3f})"
                )
            else:
                logger.warning(f"[DIVERSITY] No observations found mentioning '{entity}'")

        # Combine selected + diversity additions
        combined = selected + diversity_additions

        # If we exceed max_observations, keep diversity additions and trim from bottom of original selection
        if len(combined) > max_observations:
            # Keep ALL diversity additions (they're critical for coverage)
            # Trim from the lowest-scoring original selections
            trim_count = len(combined) - max_observations
            logger.info(
                f"[DIVERSITY] Trimming {trim_count} lowest-scoring observations "
                f"to stay within limit ({max_observations})"
            )
            selected_trimmed = selected[:-trim_count] if trim_count < len(selected) else []
            combined = selected_trimmed + diversity_additions

        # Log final coverage
        final_coverage = {}
        for entity in diversity_entities:
            count = sum(
                1 for rel in combined
                if any(
                    entity.lower() in obs_entity.lower()
                    for obs_entity in self._extract_observation_entities(rel.observation)
                )
            )
            final_coverage[entity] = count

        logger.info(
            f"[DIVERSITY] Final coverage: "
            f"{', '.join(f'{e}({final_coverage[e]})' for e in diversity_entities)}"
        )

        return combined