"""
Core utility classes for deep_research_agent.

This module provides reusable utilities to reduce code duplication across agents:
- StateExtractor: Consolidate state.get() patterns
- ListMergeManager: Consolidate list merging and deduplication logic
- ConfigAccessor: Unify configuration access patterns

These utilities replace 100+ instances of duplicate code patterns throughout the codebase.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)

T = TypeVar('T')


class StateExtractor:
    """
    Consolidates state extraction patterns used across all agents.

    Replaces 10+ instances of state.get() patterns scattered throughout
    planner, researcher, fact_checker, and reporter agents.

    Example usage:
        # Before:
        observations = state.get("observations", [])
        search_results = state.get("search_results", [])
        citations = state.get("citations", [])

        # After:
        extractor = StateExtractor(state)
        observations = extractor.observations()
        search_results = extractor.search_results()
        citations = extractor.citations()
    """

    def __init__(self, state: Dict[str, Any]):
        """Initialize with state dict."""
        self.state = state

    def observations(self) -> List[Any]:
        """Extract observations from state."""
        return self.state.get("observations") or []

    def search_results(self) -> List[Any]:
        """Extract search results from state."""
        return self.state.get("search_results") or []

    def citations(self) -> List[Any]:
        """Extract citations from state."""
        return self.state.get("citations") or []

    def agent_handoffs(self) -> List[Dict[str, str]]:
        """Extract agent handoffs from state."""
        return self.state.get("agent_handoffs") or []

    def current_plan(self) -> Optional[Any]:
        """Extract current plan from state."""
        return self.state.get("current_plan")

    def unified_plan(self) -> Optional[Any]:
        """Extract unified plan from state."""
        return self.state.get("unified_plan")

    def current_step(self) -> Optional[Any]:
        """Extract current step from state."""
        return self.state.get("current_step")

    def research_topic(self) -> str:
        """Extract research topic from state."""
        return self.state.get("research_topic") or ""

    def final_report(self) -> str:
        """Extract final report from state."""
        return self.state.get("final_report") or ""

    def calculation_results(self) -> Optional[Dict[str, Any]]:
        """Extract calculation results from state."""
        return self.state.get("calculation_results")

    def entities(self) -> List[Any]:
        """Extract entities from state."""
        return self.state.get("entities") or []

    def verified_claims(self) -> List[Any]:
        """Extract verified claims from state."""
        return self.state.get("verified_claims") or []

    def query_constraints(self) -> Optional[Any]:
        """Extract query constraints for metric filtering."""
        return self.state.get("query_constraints")

    def grounding_result(self) -> Optional[Any]:
        """Extract grounding verification results."""
        return self.state.get("grounding_result")

    def factuality_report(self) -> Optional[Any]:
        """Extract factuality assessment."""
        return self.state.get("factuality_report")

    def get_full_research_context(self) -> Dict[str, Any]:
        """
        Extract complete research context in one call.

        Returns:
            Dict containing observations, search_results, citations, and plan
        """
        return {
            "observations": self.observations(),
            "search_results": self.search_results(),
            "citations": self.citations(),
            "plan": self.current_plan(),
            "unified_plan": self.unified_plan(),
            "research_topic": self.research_topic(),
            "entities": self.entities()
        }

    def get_execution_context(self) -> Dict[str, Any]:
        """
        Extract execution context for step execution.

        Returns:
            Dict containing current_step, observations, and plan
        """
        return {
            "current_step": self.current_step(),
            "observations": self.observations(),
            "plan": self.current_plan(),
            "unified_plan": self.unified_plan(),
            "search_results": self.search_results()
        }

    def get_calculation_context(self) -> Dict[str, Any]:
        """
        Extract complete calculation context.

        Returns:
            Dict containing unified_plan, calculation_results, query_constraints, and observations
        """
        return {
            "unified_plan": self.unified_plan(),
            "calculation_results": self.calculation_results(),
            "query_constraints": self.query_constraints(),
            "observations": self.observations(),
            "entities": self.entities()
        }


class ListMergeManager:
    """
    Consolidates list merging and deduplication logic.

    Replaces ~75 lines of duplicate code across:
    - multi_agent_state.py (merge_lists reducer)
    - state_validator.py (_merge_lists method)
    - researcher.py (observation merging)
    - reporter.py (citation merging)

    Example usage:
        merger = ListMergeManager(max_items=1000)
        merged = merger.merge_lists(
            left=existing_observations,
            right=new_observations,
            deduplicate=True,
            key_fn=lambda obs: obs.get('content', '')
        )
    """

    def __init__(self, max_items: int = 1000):
        """
        Initialize list merge manager.

        Args:
            max_items: Maximum items to keep in merged lists
        """
        self.max_items = max_items

    def merge_lists(
        self,
        left: Union[List[T], T, None] = None,
        right: Union[List[T], T, None] = None,
        deduplicate: bool = False,
        key_fn: Optional[Callable[[T], Any]] = None
    ) -> List[T]:
        """
        Merge two lists with optional deduplication and size limiting.

        Args:
            left: First list (or single item, or None)
            right: Second list (or single item, or None)
            deduplicate: Whether to remove duplicates
            key_fn: Function to extract deduplication key from items

        Returns:
            Merged list, truncated to max_items if needed
        """
        left_list = self._ensure_list(left)
        right_list = self._ensure_list(right)
        merged = left_list + right_list

        if deduplicate and key_fn:
            merged = self._deduplicate(merged, key_fn)

        if len(merged) > self.max_items:
            logger.warning(
                f"Merged list has {len(merged)} items, truncating to {self.max_items}"
            )
            merged = merged[-self.max_items:]

        return merged

    @staticmethod
    def _ensure_list(obj: Union[List[T], T, None]) -> List[T]:
        """
        Convert input to list.

        Args:
            obj: Object that might be None, single item, or list

        Returns:
            List containing items
        """
        if obj is None:
            return []
        if isinstance(obj, list):
            return obj
        return [obj]

    @staticmethod
    def _deduplicate(items: List[T], key_fn: Callable[[T], Any]) -> List[T]:
        """
        Remove duplicates while preserving order.

        Args:
            items: List of items to deduplicate
            key_fn: Function to extract unique key from each item

        Returns:
            Deduplicated list
        """
        seen = set()
        result = []

        for item in items:
            try:
                key = key_fn(item)
                if key not in seen:
                    seen.add(key)
                    result.append(item)
            except Exception as e:
                logger.warning(f"Error extracting key from item: {e}, keeping item anyway")
                result.append(item)

        return result

    def merge_observations(
        self,
        left: Union[List[Any], None],
        right: Union[List[Any], None]
    ) -> List[Any]:
        """
        Merge observation lists with content-based deduplication.

        Args:
            left: Existing observations
            right: New observations

        Returns:
            Merged and deduplicated observations
        """
        def get_obs_key(obs: Any) -> str:
            """Extract deduplication key from observation."""
            if isinstance(obs, dict):
                return obs.get("content", "") or obs.get("id", "")
            if hasattr(obs, "content"):
                return obs.content or getattr(obs, "id", "")
            return str(obs)

        return self.merge_lists(
            left=left,
            right=right,
            deduplicate=True,
            key_fn=get_obs_key
        )

    def merge_citations(
        self,
        left: Union[List[Any], None],
        right: Union[List[Any], None]
    ) -> List[Any]:
        """
        Merge citation lists with URL-based deduplication.

        Args:
            left: Existing citations
            right: New citations

        Returns:
            Merged and deduplicated citations
        """
        def get_citation_key(citation: Any) -> str:
            """Extract deduplication key from citation."""
            if isinstance(citation, dict):
                return citation.get("url", "") or citation.get("source", "")
            if hasattr(citation, "url"):
                return citation.url or getattr(citation, "source", "")
            return str(citation)

        return self.merge_lists(
            left=left,
            right=right,
            deduplicate=True,
            key_fn=get_citation_key
        )


class ConfigAccessor:
    """
    Unifies configuration access patterns across all agents.

    Replaces ~40 instances of config.get() patterns scattered throughout:
    - All 5 agents (coordinator, planner, researcher, fact_checker, reporter)
    - workflow_nodes_enhanced.py
    - enhanced_research_agent.py

    Example usage:
        # Before:
        planning_config = self.config.get('planning', {})
        enable_iterative = planning_config.get('enable_iterative_planning', True)
        max_iterations = planning_config.get('max_plan_iterations', 3)

        # After:
        accessor = ConfigAccessor(self.config)
        enable_iterative = accessor.get_bool('enable_iterative_planning', 'planning', True)
        max_iterations = accessor.get_int('max_plan_iterations', 'planning', 3)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize configuration accessor.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

    def get_agent_config(self, agent_name: str) -> Dict[str, Any]:
        """
        Get agent-specific configuration.

        Args:
            agent_name: Name of agent (e.g., 'planner', 'researcher')

        Returns:
            Agent configuration dict
        """
        return self.config.get('agents', {}).get(agent_name, {})

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get configuration section.

        Args:
            section: Section name (e.g., 'planning', 'research', 'grounding')

        Returns:
            Section configuration dict
        """
        return self.config.get(section, {})

    def get_bool(
        self,
        key: str,
        section: Optional[str] = None,
        default: bool = False
    ) -> bool:
        """
        Get boolean configuration value.

        Args:
            key: Configuration key
            section: Optional section name
            default: Default value if key not found

        Returns:
            Boolean configuration value
        """
        cfg = self.get_section(section) if section else self.config
        return cfg.get(key, default)

    def get_int(
        self,
        key: str,
        section: Optional[str] = None,
        default: int = 0
    ) -> int:
        """
        Get integer configuration value.

        Args:
            key: Configuration key
            section: Optional section name
            default: Default value if key not found

        Returns:
            Integer configuration value
        """
        cfg = self.get_section(section) if section else self.config
        return cfg.get(key, default)

    def get_float(
        self,
        key: str,
        section: Optional[str] = None,
        default: float = 0.0
    ) -> float:
        """
        Get float configuration value.

        Args:
            key: Configuration key
            section: Optional section name
            default: Default value if key not found

        Returns:
            Float configuration value
        """
        cfg = self.get_section(section) if section else self.config
        return cfg.get(key, default)

    def get_str(
        self,
        key: str,
        section: Optional[str] = None,
        default: str = ""
    ) -> str:
        """
        Get string configuration value.

        Args:
            key: Configuration key
            section: Optional section name
            default: Default value if key not found

        Returns:
            String configuration value
        """
        cfg = self.get_section(section) if section else self.config
        return cfg.get(key, default)

    def get_list(
        self,
        key: str,
        section: Optional[str] = None,
        default: Optional[List[Any]] = None
    ) -> List[Any]:
        """
        Get list configuration value.

        Args:
            key: Configuration key
            section: Optional section name
            default: Default value if key not found

        Returns:
            List configuration value
        """
        cfg = self.get_section(section) if section else self.config
        return cfg.get(key, default or [])

    # Convenience methods for common config patterns

    def get_planning_config(self) -> 'PlanningConfig':
        """Get planning configuration as structured object."""
        planning = self.get_section('planning')
        return PlanningConfig(
            enable_iterative_planning=planning.get('enable_iterative_planning', True),
            max_plan_iterations=planning.get('max_plan_iterations', 3),
            quality_threshold=planning.get('quality_threshold', 0.7),
            enable_deep_thinking=planning.get('enable_deep_thinking', False)
        )

    def get_research_config(self) -> 'ResearchConfig':
        """Get research configuration as structured object."""
        research = self.get_section('research')
        return ResearchConfig(
            max_steps=research.get('max_steps', 10),
            max_steps_per_execution=research.get('max_steps_per_execution', 5),
            enable_reflexion=research.get('enable_reflexion', True)
        )

    def get_grounding_config(self) -> 'GroundingConfig':
        """Get grounding configuration as structured object."""
        grounding = self.get_section('grounding')
        return GroundingConfig(
            enable_grounding=grounding.get('enable_grounding', True),
            verification_level=grounding.get('verification_level', 'moderate'),
            enable_contradiction_detection=grounding.get('enable_contradiction_detection', True)
        )


@dataclass
class PlanningConfig:
    """Structured planning configuration."""
    enable_iterative_planning: bool = True
    max_plan_iterations: int = 3
    quality_threshold: float = 0.7
    enable_deep_thinking: bool = False


@dataclass
class ResearchConfig:
    """Structured research configuration."""
    max_steps: int = 10
    max_steps_per_execution: int = 5
    enable_reflexion: bool = True


@dataclass
class GroundingConfig:
    """Structured grounding configuration."""
    enable_grounding: bool = True
    verification_level: str = "moderate"
    enable_contradiction_detection: bool = True


# Export all utilities
__all__ = [
    'StateExtractor',
    'ListMergeManager',
    'ConfigAccessor',
    'PlanningConfig',
    'ResearchConfig',
    'GroundingConfig',
]
