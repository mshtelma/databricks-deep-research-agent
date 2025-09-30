"""
Debug utilities for state inspection and troubleshooting.

Provides utilities to inspect the state of the multi-agent research system,
particularly for debugging section research mapping and ID-based tracking.
"""

import logging
from typing import Dict, Any, List, Optional
import json

logger = logging.getLogger(__name__)

from .id_generator import PlanIDGenerator
from .section_models import SectionResearchResult


def dump_section_state(state: Dict[str, Any], operation: str = "unknown") -> None:
    """
    Dump complete section state for debugging.
    
    Args:
        state: Current state dictionary
        operation: Description of current operation (for context)
    """
    logger.info(f"[DEBUG_STATE] ==> {operation.upper()} <==")
    
    # 1. Check current plan and its sections
    plan = state.get("current_plan")
    if plan:
        section_specs = getattr(plan, 'section_specifications', None)
        if section_specs:
            logger.info(f"[DEBUG_STATE] Plan has {len(section_specs)} section specifications:")
            for i, spec in enumerate(section_specs):
                section_id = getattr(spec, 'id', 'NO_ID')
                normalized_id = (
                    PlanIDGenerator.normalize_id(section_id)
                    if isinstance(section_id, str)
                    else section_id
                )
                logger.info(
                    f"  [{i}] ID={section_id} (normalized={normalized_id}), Title='{spec.title}'"
                )
        else:
            dynamic_sections = getattr(plan, 'dynamic_sections', []) or []
            if dynamic_sections:
                logger.info(f"[DEBUG_STATE] Plan has {len(dynamic_sections)} dynamic sections:")
                for i, section in enumerate(dynamic_sections):
                    logger.info(
                        f"  [{i}] Title='{section.title}', priority={getattr(section, 'priority', 'n/a')}, content_type={getattr(section, 'content_type', 'analysis')}"
                    )
            else:
                logger.info("[DEBUG_STATE] Plan has no structured sections defined")
    else:
        logger.info("[DEBUG_STATE] No plan available in state")
    
    # 2. Check section research results
    section_research = state.get("section_research_results", {})
    logger.info(f"[DEBUG_STATE] Section research results: {len(section_research)} entries")
    for section_id, data in section_research.items():
        if isinstance(data, SectionResearchResult):
            title = data.metadata.get("section_title", section_id)
            has_research = bool(data.synthesis)
            has_synthesis = has_research
            logger.info(f"  [{section_id}] Title='{title}', has_research={has_research}, has_synthesis={has_synthesis}")
        elif isinstance(data, dict):
            title = data.get('title', 'Unknown')
            has_research = bool(data.get('research'))
            has_synthesis = bool(data.get('research', {}).get('synthesis'))
            logger.info(f"  [{section_id}] Title='{title}', has_research={has_research}, has_synthesis={has_synthesis}")
        else:
            logger.info(f"  [{section_id}] Data type: {type(data)}")
    
    # 3. Check observations
    observations = state.get("observations", [])
    logger.info(f"[DEBUG_STATE] Total observations: {len(observations)}")
    
    # 4. Check if there are any mismatches
    plan_ids = set()
    section_specs = getattr(plan, 'section_specifications', None) if plan else None
    if section_specs:
        plan_ids = {
            PlanIDGenerator.normalize_id(getattr(spec, 'id', 'NO_ID'))
            for spec in section_specs
        }
    
    research_ids = {
        PlanIDGenerator.normalize_id(section_id)
        if isinstance(section_id, str)
        else section_id
        for section_id in section_research.keys()
    }

    missing_research = plan_ids - research_ids
    orphaned_research = research_ids - plan_ids
    
    if plan_ids:
        if missing_research:
            logger.warning(f"[DEBUG_STATE] Plan sections missing research: {missing_research}")
        if orphaned_research:
            logger.warning(f"[DEBUG_STATE] Orphaned research (no plan section): {orphaned_research}")
        if plan_ids == research_ids:
            logger.info(f"[DEBUG_STATE] ✓ Perfect ID alignment: {len(plan_ids)} sections")
    
    logger.info(f"[DEBUG_STATE] ==> END {operation.upper()} <==")


def validate_section_id_consistency(state: Dict[str, Any]) -> List[str]:
    """
    Validate that all sections have proper IDs and detect issues.
    
    Returns:
        List of validation error messages (empty if all good)
    """
    errors = []
    
    plan = state.get("current_plan")
    if not plan:
        errors.append("No current_plan in state")
        return errors
    
    section_specs = getattr(plan, 'section_specifications', None)
    if not section_specs:
        # No legacy section specs to validate under the dynamic template flow
        return errors
    
    # Check each section specification
    for i, spec in enumerate(section_specs):
        # Check if ID exists
        if not hasattr(spec, 'id'):
            errors.append(f"Section {i} ('{spec.title}') missing 'id' attribute")
            continue
        
        # Check if ID is non-empty
        if not spec.id:
            errors.append(f"Section {i} ('{spec.title}') has empty ID")
            continue
        
        # Check ID format
        normalized_id = PlanIDGenerator.normalize_id(spec.id)
        if not normalized_id.startswith("step_"):
            errors.append(
                f"Section {i} ('{spec.title}') has malformed ID: '{spec.id}' (expected step_XXX format)"
            )
        elif spec.id != normalized_id:
            errors.append(
                f"Section {i} ('{spec.title}') uses non-normalized ID '{spec.id}'. "
                f"Consider using '{normalized_id}' for consistency."
            )
    
    # Check for duplicate IDs
    ids = [getattr(spec, 'id', None) for spec in section_specs if hasattr(spec, 'id') and spec.id]
    if len(ids) != len(set(ids)):
        duplicates = [id for id in ids if ids.count(id) > 1]
        errors.append(f"Duplicate section IDs found: {set(duplicates)}")
    
    return errors


def inspect_section_research_mapping(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Inspect the mapping between sections and their research results.
    
    Returns:
        Dictionary with detailed mapping analysis
    """
    analysis = {
        "plan_sections": {},
        "research_results": {},
        "mapping_status": {},
        "issues": []
    }
    
    # Get plan sections
    plan = state.get("current_plan")
    section_specs = getattr(plan, 'section_specifications', None) if plan else None
    if section_specs:
        for spec in section_specs:
            raw_id = getattr(spec, 'id', 'NO_ID')
            section_id = (
                PlanIDGenerator.normalize_id(raw_id)
                if isinstance(raw_id, str)
                else raw_id
            )
            analysis["plan_sections"][section_id] = {
                "title": spec.title,
                "has_id": hasattr(spec, 'id'),
                "id_value": raw_id,
                "normalized_id": section_id
            }
    else:
        dynamic_sections = getattr(plan, 'dynamic_sections', []) if plan else []
        for idx, section in enumerate(dynamic_sections, start=1):
            section_id = f"dynamic_section_{idx:02d}"
            analysis["plan_sections"][section_id] = {
                "title": getattr(section, 'title', f'Dynamic Section {idx}'),
                "has_id": False,
                "id_value": None,
                "normalized_id": section_id
            }

    # Get research results
    section_research = state.get("section_research_results", {})
    for raw_id, data in section_research.items():
        section_id = (
            PlanIDGenerator.normalize_id(raw_id)
            if isinstance(raw_id, str)
            else raw_id
        )
        if isinstance(data, SectionResearchResult):
            analysis["research_results"][section_id] = {
                "title": data.metadata.get("section_title", section_id),
                "has_research": bool(data.synthesis),
                "has_synthesis": bool(data.synthesis),
                "research_keys": list(data.to_dict().keys())
            }
        elif isinstance(data, dict):
            analysis["research_results"][section_id] = {
                "title": data.get('title', 'Unknown'),
                "has_research": bool(data.get('research')),
                "has_synthesis": bool(data.get('research', {}).get('synthesis')),
                "research_keys": list(data.get('research', {}).keys())
            }
        else:
            analysis["research_results"][section_id] = {
                "data_type": str(type(data)),
                "is_dict": False
            }
    
    # Analyze mapping
    plan_ids = set(analysis["plan_sections"].keys())
    research_ids = set(analysis["research_results"].keys())
    
    for section_id in plan_ids:
        if section_id in research_ids:
            analysis["mapping_status"][section_id] = "mapped"
        else:
            analysis["mapping_status"][section_id] = "missing_research"
            analysis["issues"].append(f"Section {section_id} has no research results")
    
    for section_id in research_ids:
        if section_id not in plan_ids:
            analysis["mapping_status"][section_id] = "orphaned_research"
            analysis["issues"].append(f"Research {section_id} has no corresponding plan section")
    
    return analysis


def log_section_analysis(state: Dict[str, Any], prefix: str = "ANALYSIS") -> None:
    """
    Log comprehensive section analysis for debugging.
    
    Args:
        state: Current state dictionary
        prefix: Log prefix for grouping messages
    """
    # Validate IDs
    errors = validate_section_id_consistency(state)
    if errors:
        for error in errors:
            logger.error(f"[{prefix}] ID_VALIDATION_ERROR: {error}")
    else:
        logger.info(f"[{prefix}] ✓ All section IDs are valid")
    
    # Analyze mapping
    analysis = inspect_section_research_mapping(state)
    
    # Log summary
    plan_count = len(analysis["plan_sections"])
    research_count = len(analysis["research_results"])
    mapped_count = len([s for s in analysis["mapping_status"].values() if s == "mapped"])
    
    logger.info(f"[{prefix}] Summary: {plan_count} plan sections, {research_count} research results, {mapped_count} mapped")
    
    # Log issues
    for issue in analysis["issues"]:
        logger.warning(f"[{prefix}] MAPPING_ISSUE: {issue}")
    
    # Log detailed status if there are issues
    if analysis["issues"]:
        logger.info(f"[{prefix}] Detailed mapping status:")
        for section_id, status in analysis["mapping_status"].items():
            logger.info(f"  {section_id}: {status}")


def format_section_summary(state: Dict[str, Any]) -> str:
    """
    Format a concise summary of section state for logging.
    
    Returns:
        Human-readable summary string
    """
    plan = state.get("current_plan")
    section_research = state.get("section_research_results", {})
    
    if not plan:
        return "No plan available"

    section_specs = getattr(plan, 'section_specifications', None)
    dynamic_sections = getattr(plan, 'dynamic_sections', []) if hasattr(plan, 'dynamic_sections') else []

    if section_specs:
        plan_count = len(section_specs)
    elif dynamic_sections:
        plan_count = len(dynamic_sections)
    else:
        return "Plan defines no structured sections"
    research_count = len(section_research)
    
    # Count sections with research
    mapped_count = 0
    if section_specs:
        for spec in section_specs:
            section_id = getattr(spec, 'id', None)
            if section_id and section_id in section_research:
                mapped_count += 1
        return f"{plan_count} sections planned, {research_count} research results, {mapped_count} mapped"

    return f"{plan_count} dynamic sections planned, {research_count} research results"


def validate_section_research_state(state: Dict[str, Any], checkpoint_name: str) -> None:
    """
    Validate section_research_results is properly maintained at a checkpoint.
    
    Args:
        state: Current state dictionary
        checkpoint_name: Name of the checkpoint for logging
    """
    import logging
    logger = logging.getLogger(__name__)
    
    section_research = state.get("section_research_results", {})
    logger.info(f"[CHECKPOINT:{checkpoint_name}] section_research_results has {len(section_research)} sections")
    
    if len(section_research) == 0:
        logger.error(f"[CHECKPOINT:{checkpoint_name}] CRITICAL: section_research_results is empty!")
        
    # Log all section IDs
    logger.info(f"[CHECKPOINT:{checkpoint_name}] Section IDs: {list(section_research.keys())}")
    
    # Validate each section has required fields
    for section_id, section_data in section_research.items():
        # Convert SectionResearchResult objects to dicts if needed
        if hasattr(section_data, 'to_dict'):
            section_dict = section_data.to_dict()
        elif isinstance(section_data, dict):
            section_dict = section_data
        else:
            logger.error(f"[CHECKPOINT:{checkpoint_name}] Section {section_id} data is neither dict nor SectionResearchResult: {type(section_data)}")
            continue

        # Validate section structure (legacy format)
        if not section_dict.get("research") and not section_dict.get("synthesis"):
            logger.warning(f"[CHECKPOINT:{checkpoint_name}] Section {section_id} has no research/synthesis data")

        if not section_dict.get("id") and section_id:
            logger.debug(f"[CHECKPOINT:{checkpoint_name}] Section {section_id} uses key as ID (no separate ID field)")

        if not section_dict.get("title"):
            logger.debug(f"[CHECKPOINT:{checkpoint_name}] Section {section_id} has no title")


def log_state_transition(state: Dict[str, Any], from_node: str, to_node: str) -> None:
    """
    Log state transition between workflow nodes with section tracking.
    
    Args:
        state: Current state dictionary  
        from_node: Name of the node we're transitioning from
        to_node: Name of the node we're transitioning to
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Log basic transition
    logger.info(f"[STATE_TRANSITION] {from_node} → {to_node}")
    
    # Log section research state
    section_research = state.get("section_research_results", {})
    section_count = len(section_research)
    logger.info(f"[STATE_TRANSITION] section_research_results: {section_count} sections: {list(section_research.keys())}")
    
    # Log other key state
    obs_count = len(state.get("observations", []))
    search_count = len(state.get("search_results", []))
    logger.info(f"[STATE_TRANSITION] observations: {obs_count}, search_results: {search_count}")
    
    # Validate critical state is not empty where expected
    if to_node == "reporter" and section_count == 0:
        logger.error(f"[STATE_TRANSITION] CRITICAL: Entering reporter with no section research!")
    
    if obs_count == 0 and from_node in ["researcher", "background_investigation"]:
        logger.warning(f"[STATE_TRANSITION] No observations from {from_node} - might indicate research failure")
