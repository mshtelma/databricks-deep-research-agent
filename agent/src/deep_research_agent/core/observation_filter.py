"""
Observation filtering utilities for section-specific content selection.

This module provides utilities for filtering observations based on:
- Section-to-step mapping (from plan.dynamic_sections)
- Content availability (full_content vs snippet-only)
- Direct step_id membership (no string matching)

Extracted from reporter.py to provide reusable observation filtering logic.
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class ObservationFilter:
    """
    Filter observations for section-specific content.

    Provides two-stage filtering:
    1. Filter by step_id (section-to-step mapping)
    2. Filter by full_content availability

    Example:
        filter = ObservationFilter()

        filtered_obs = filter.filter_for_section(
            section_name="Tax Comparison",
            all_observations=observations,
            state=state
        )

        if filtered_obs is None:
            # No observations with content - skip section
            pass
    """

    def __init__(self, fallback_limit: int = 30):
        """
        Initialize observation filter.

        Args:
            fallback_limit: Number of observations to return when no filtering is possible
        """
        self.fallback_limit = fallback_limit

    def filter_for_section(
        self,
        section_name: str,
        all_observations: List[Dict[str, Any]],
        state: Dict[str, Any]
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Filter observations by step_id using direct references from DynamicSection.

        Uses section.step_ids for direct lookup (NO STRING MATCHING).

        Args:
            section_name: The section we're generating
            all_observations: All observations from research
            state: Current research state with plan

        Returns:
            - List of filtered observations relevant to this section
            - None if no observations with full_content (signals skip section)
            - Fallback subset if filtering not possible
        """
        plan = state.get("current_plan")

        if not plan:
            logger.warning(f"No plan found in state, returning subset of observations")
            return self._get_fallback_observations(all_observations)

        # Find the section by name using direct reference (no string matching)
        matching_section = self._find_section_by_name(plan, section_name)

        if not matching_section:
            logger.error(f"❌ Section '{section_name}' not found in plan.dynamic_sections!")
            return self._get_fallback_observations(all_observations)

        # Use direct step_ids from section (NO STRING MATCHING!)
        if not matching_section.step_ids:
            logger.warning(f"⚠️ Section '{section_name}' has no step_ids assigned")
            return self._get_fallback_observations(all_observations)

        logger.info(f"🎯 Section '{section_name}' filtering by step_ids: {matching_section.step_ids}")

        # Stage 1: Filter observations by step_id
        step_filtered = self._filter_by_step_ids(
            all_observations,
            matching_section.step_ids,
            section_name
        )

        # Stage 2: Filter by full_content availability
        content_filtered = self._filter_by_full_content(
            step_filtered,
            section_name
        )

        # Return None if no observations with content (signals skip section)
        if not content_filtered:
            logger.warning(
                f"❌ Section '{section_name}' has NO observations with fetched content. "
                f"This means web fetching failed for all sources. Section will be skipped."
            )
            return None

        # Debug: Log first observation
        self._log_first_observation(content_filtered, section_name)

        return content_filtered

    def _find_section_by_name(self, plan: Any, section_name: str) -> Optional[Any]:
        """
        Find section in plan by exact title match.

        Args:
            plan: Research plan with dynamic_sections
            section_name: Section title to find

        Returns:
            Matching section or None
        """
        if not hasattr(plan, 'dynamic_sections') or not plan.dynamic_sections:
            return None

        for section in plan.dynamic_sections:
            if section.title == section_name:
                return section

        return None

    def _filter_by_step_ids(
        self,
        observations: List[Dict[str, Any]],
        step_ids: List[str],
        section_name: str
    ) -> List[Dict[str, Any]]:
        """
        Filter observations by step_id membership.

        Args:
            observations: All observations
            step_ids: Valid step_ids for this section
            section_name: Section name for logging

        Returns:
            Filtered observations matching step_ids
        """
        filtered_observations = []
        observations_without_step_id = 0

        for obs in observations:
            # Get step_id from observation (dict or object)
            obs_step_id = self._get_step_id(obs)

            # Direct membership check - no string matching!
            if obs_step_id:
                if obs_step_id in step_ids:
                    filtered_observations.append(obs)
            else:
                observations_without_step_id += 1

        if observations_without_step_id > 0:
            logger.warning(f"  ⚠️ {observations_without_step_id} observations missing step_id")

        logger.info(
            f"✅ Filtered {len(filtered_observations)}/{len(observations)} "
            f"observations by step_id for section '{section_name}'"
        )

        return filtered_observations

    def _filter_by_full_content(
        self,
        observations: List[Dict[str, Any]],
        section_name: str
    ) -> List[Dict[str, Any]]:
        """
        Filter observations to only those with full_content.

        Args:
            observations: Observations to filter
            section_name: Section name for logging

        Returns:
            Observations with full_content only
        """
        with_content = []
        snippet_only = 0

        for obs in observations:
            has_full = self._has_full_content(obs)

            if has_full:
                with_content.append(obs)
            else:
                snippet_only += 1

        if snippet_only > 0:
            logger.info(f"⏭️  Filtered out {snippet_only} snippet-only observations for '{section_name}'")

        logger.info(f"✅ Section '{section_name}': {len(with_content)} observations with full_content")

        return with_content

    def _get_step_id(self, obs: Any) -> Optional[str]:
        """
        Get step_id from observation (dict or object).

        Args:
            obs: Observation (dict or object)

        Returns:
            step_id or None
        """
        if isinstance(obs, dict):
            return obs.get("step_id")
        else:
            return getattr(obs, "step_id", None)

    def _has_full_content(self, obs: Any) -> bool:
        """
        Check if observation has full_content.

        Args:
            obs: Observation (dict or object)

        Returns:
            True if has full_content
        """
        if isinstance(obs, dict):
            return bool(obs.get("full_content"))
        else:
            return bool(getattr(obs, "full_content", None))

    def _get_fallback_observations(
        self,
        observations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Get fallback subset of observations when filtering not possible.

        Args:
            observations: All observations

        Returns:
            First N observations (fallback_limit)
        """
        return observations[:self.fallback_limit]

    def _log_first_observation(
        self,
        observations: List[Dict[str, Any]],
        section_name: str
    ):
        """
        Log first observation for debugging.

        Args:
            observations: Filtered observations
            section_name: Section name
        """
        if not observations:
            return

        first_obs = observations[0]

        if isinstance(first_obs, dict):
            content_preview = str(first_obs.get("content", ""))[:100]
            step_id = first_obs.get("step_id", "NONE")
        else:
            content_preview = str(getattr(first_obs, "content", ""))[:100]
            step_id = getattr(first_obs, "step_id", "NONE")

        logger.info(f"  📝 First obs (step_id={step_id}): {content_preview}...")


# Convenience function for backward compatibility
def filter_observations_for_section(
    section_name: str,
    all_observations: List[Dict[str, Any]],
    state: Dict[str, Any],
    fallback_limit: int = 30
) -> Optional[List[Dict[str, Any]]]:
    """
    Convenience function to filter observations for a section.

    Args:
        section_name: Section to filter for
        all_observations: All observations
        state: Research state with plan
        fallback_limit: Fallback limit

    Returns:
        Filtered observations or None
    """
    filter = ObservationFilter(fallback_limit=fallback_limit)
    return filter.filter_for_section(section_name, all_observations, state)
