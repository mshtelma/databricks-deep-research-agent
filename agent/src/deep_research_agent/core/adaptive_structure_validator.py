"""
Adaptive structure validator for report generation.

Validates and recovers from invalid adaptive report structures
to ensure robust report generation even when plan suggests invalid structures.
"""

import logging
from typing import List, Dict, Any, Optional

from .template_generator import DynamicSection, SectionContentType

logger = logging.getLogger(__name__)


class AdaptiveStructureValidator:
    """
    Validates adaptive report structures from research plans.

    Ensures that suggested_report_structure or dynamic_sections are:
    - Non-empty
    - Properly formatted
    - Have required fields
    - Meet minimum requirements for report generation

    Provides fallback to default structure when validation fails.
    """

    # Default structure to use when validation fails
    DEFAULT_STRUCTURE = [
        DynamicSection(title="Executive Summary", purpose="High-level overview of key findings", priority=10),
        DynamicSection(title="Introduction", purpose="Background and context", priority=20),
        DynamicSection(title="Key Findings", purpose="Main research results and insights", priority=30),
        DynamicSection(title="Analysis", purpose="Detailed analysis of findings", priority=40),
        DynamicSection(title="Recommendations", purpose="Actionable recommendations based on findings", priority=50),
        DynamicSection(title="Conclusion", purpose="Summary and final thoughts", priority=60),
    ]

    MIN_SECTIONS = 2  # Minimum number of sections required
    MAX_SECTIONS = 15  # Maximum number of sections allowed

    @classmethod
    def validate_and_recover(
        cls,
        state: Dict[str, Any],
        fallback_to_default: bool = True
    ) -> Optional[List[DynamicSection]]:
        """
        Validate adaptive structure from state and recover if invalid.

        Args:
            state: Research state containing current_plan
            fallback_to_default: If True, return DEFAULT_STRUCTURE on failure;
                                if False, return None on failure

        Returns:
            List of DynamicSection objects,
            or None if invalid and fallback_to_default=False
        """
        current_plan = state.get('current_plan')

        # Try to extract structure from plan
        structure = cls._extract_structure(current_plan)

        # Validate structure
        validation_result = cls._validate_structure(structure)

        if validation_result is None:
            # Structure is invalid
            logger.warning(
                "Adaptive structure validation failed. "
                f"Using {'default structure' if fallback_to_default else 'None'}."
            )
            return cls.DEFAULT_STRUCTURE if fallback_to_default else None

        logger.info(
            f"Adaptive structure validated successfully: {len(validation_result)} sections"
        )
        return validation_result

    @classmethod
    def _extract_structure(cls, current_plan: Any) -> Optional[List[Any]]:
        """
        Extract structure from current_plan.

        Tries multiple attributes:
        - dynamic_sections (preferred)
        - suggested_report_structure (legacy)

        Returns:
            List of sections or None if not found
        """
        if not current_plan:
            logger.debug("No current_plan provided")
            return None

        # Try dynamic_sections first (preferred)
        if hasattr(current_plan, 'dynamic_sections'):
            structure = current_plan.dynamic_sections
            if structure:
                logger.debug(f"Found dynamic_sections: {len(structure)} sections")
                return structure

        # Try suggested_report_structure (legacy)
        if hasattr(current_plan, 'suggested_report_structure'):
            structure = current_plan.suggested_report_structure
            if structure:
                logger.debug(f"Found suggested_report_structure: {len(structure)} sections")
                return structure

        logger.debug("No structure found in current_plan")
        return None

    @classmethod
    def _validate_structure(cls, structure: Optional[List[Any]]) -> Optional[List[DynamicSection]]:
        """
        Validate structure format and content.

        Args:
            structure: Structure to validate

        Returns:
            Normalized structure (list of DynamicSection objects),
            or None if validation fails
        """
        # Check if structure exists
        if structure is None:
            logger.warning("Structure validation failed: None")
            return None

        # Check if structure is a list
        if not isinstance(structure, list):
            logger.warning(
                f"Structure validation failed: Not a list (type={type(structure).__name__})"
            )
            return None

        # Check if structure has minimum sections
        if len(structure) < cls.MIN_SECTIONS:
            logger.warning(
                f"Structure validation failed: Too few sections "
                f"({len(structure)} < {cls.MIN_SECTIONS})"
            )
            return None

        # Check if structure has too many sections
        if len(structure) > cls.MAX_SECTIONS:
            logger.warning(
                f"Structure validation failed: Too many sections "
                f"({len(structure)} > {cls.MAX_SECTIONS}), truncating"
            )
            structure = structure[:cls.MAX_SECTIONS]

        # Normalize and validate each section
        normalized = []
        for i, section in enumerate(structure):
            normalized_section = cls._normalize_section(section, i)
            if normalized_section is None:
                logger.warning(f"Section {i} validation failed, skipping")
                continue
            normalized.append(normalized_section)

        # Check if we have enough valid sections after normalization
        if len(normalized) < cls.MIN_SECTIONS:
            logger.warning(
                f"Structure validation failed: Too few valid sections after normalization "
                f"({len(normalized)} < {cls.MIN_SECTIONS})"
            )
            return None

        logger.info(f"Structure validation passed: {len(normalized)} sections")
        return normalized

    @classmethod
    def _parse_content_type(cls, value: Any) -> SectionContentType:
        """
        Parse content_type from various formats.

        Args:
            value: Content type value (can be enum, string, or None)

        Returns:
            SectionContentType enum value
        """
        if isinstance(value, SectionContentType):
            return value
        if isinstance(value, str):
            try:
                # Handle both "ANALYSIS" and "SectionContentType.ANALYSIS" formats
                if value.startswith('SectionContentType.'):
                    value = value.split('.')[1]
                return SectionContentType[value.upper()]
            except (KeyError, AttributeError):
                logger.warning(f"Invalid content_type string '{value}', using ANALYSIS")
                return SectionContentType.ANALYSIS
        return SectionContentType.ANALYSIS

    @classmethod
    def _normalize_section(cls, section: Any, index: int) -> Optional[DynamicSection]:
        """
        Normalize a section to standard format.

        Args:
            section: Section to normalize (can be dict, DynamicSection, or string)
            index: Section index for error reporting

        Returns:
            Normalized DynamicSection object,
            or None if normalization fails
        """
        # Handle DynamicSection object (already in correct format)
        if isinstance(section, DynamicSection):
            return section

        # Handle dict format
        if isinstance(section, dict):
            title = section.get('title', '').strip()
            purpose = section.get('purpose', '').strip()

            if not title:
                logger.warning(f"Section {index}: Missing or empty 'title'")
                return None

            # Extract all DynamicSection attributes from dict if available
            content_type = cls._parse_content_type(section.get('content_type'))
            
            return DynamicSection(
                title=title,
                purpose=purpose or f"Section about {title.lower()}",
                priority=section.get('priority', 100),
                content_type=content_type,
                hints=tuple(section.get('hints', [])),
                content_bullets=tuple(section.get('content_bullets', [])),
                step_ids=tuple(section.get('step_ids', [])),
            )

        # Handle object format (with attributes)
        if hasattr(section, 'title'):
            title = getattr(section, 'title', '').strip()
            purpose = getattr(section, 'purpose', '').strip()

            if not title:
                logger.warning(f"Section {index}: Missing or empty 'title' attribute")
                return None

            # Preserve all attributes from the original object
            content_type = cls._parse_content_type(getattr(section, 'content_type', None))
            
            return DynamicSection(
                title=title,
                purpose=purpose or f"Section about {title.lower()}",
                priority=getattr(section, 'priority', 100),
                content_type=content_type,
                hints=tuple(getattr(section, 'hints', [])),
                content_bullets=tuple(getattr(section, 'content_bullets', [])),
                step_ids=tuple(getattr(section, 'step_ids', [])),
            )

        # Handle string format (just a title)
        if isinstance(section, str):
            title = section.strip()

            if not title:
                logger.warning(f"Section {index}: Empty string")
                return None

            return DynamicSection(
                title=title,
                purpose=f"Section about {title.lower()}",
                priority=100,
            )

        # Unknown format
        logger.warning(
            f"Section {index}: Unknown format (type={type(section).__name__})"
        )
        return None

    @classmethod
    def get_section_titles(cls, structure: List[DynamicSection]) -> List[str]:
        """
        Extract section titles from normalized structure.

        Args:
            structure: Normalized structure from validate_and_recover()

        Returns:
            List of section titles
        """
        return [section.title for section in structure]

    @classmethod
    def validate_section_content(
        cls,
        section_title: str,
        content: str,
        min_length: int = 50
    ) -> bool:
        """
        Validate that a generated section has sufficient content.

        Args:
            section_title: Title of the section
            content: Generated content for the section
            min_length: Minimum content length in characters

        Returns:
            True if content is valid, False otherwise
        """
        if not content or not content.strip():
            logger.warning(f"Section '{section_title}': Empty content")
            return False

        if len(content.strip()) < min_length:
            logger.warning(
                f"Section '{section_title}': Content too short "
                f"({len(content.strip())} < {min_length} chars)"
            )
            return False

        return True
