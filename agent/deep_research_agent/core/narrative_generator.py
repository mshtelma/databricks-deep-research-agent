"""
Narrative generator with word count enforcement and structured formatting.

Generates narratives that strictly adhere to word limits, formatting requirements,
and structural specifications from extracted requirements.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .requirements import NarrativeSpecification, RequiredDataPoint

logger = logging.getLogger(__name__)


class NarrativeStyle(str, Enum):
    """Supported narrative styles."""
    PROFESSIONAL = "professional"
    ACADEMIC = "academic"
    CASUAL = "casual"
    EXECUTIVE = "executive"
    TECHNICAL = "technical"


@dataclass
class NarrativeSection:
    """Represents a section of the narrative."""
    title: str
    content: str
    word_count: int
    data_points_covered: List[str]
    citations: List[str] = None
    
    def __post_init__(self):
        if self.citations is None:
            self.citations = []


@dataclass
class GeneratedNarrative:
    """Generated narrative with metadata."""
    title: str
    sections: List[NarrativeSection]
    total_word_count: int
    style: str
    meets_requirements: bool
    metadata: Dict[str, Any]
    
    def get_full_text(self) -> str:
        """Get the complete narrative text."""
        lines = []
        
        if self.title:
            lines.append(f"# {self.title}")
            lines.append("")
        
        for section in self.sections:
            if section.title:
                lines.append(f"## {section.title}")
                lines.append("")
            lines.append(section.content)
            lines.append("")
        
        return "\n".join(lines)
    
    def get_word_count(self) -> int:
        """Get actual word count of the narrative."""
        return self._count_words(self.get_full_text())
    
    def _count_words(self, text: str) -> int:
        """Count words in text, excluding markdown headers."""
        # Remove markdown headers
        text = re.sub(r'^#+\s+.*$', '', text, flags=re.MULTILINE)
        # Count words
        words = re.findall(r'\b\w+\b', text)
        return len(words)


class NarrativeGenerator:
    """
    Generates structured narratives with word count enforcement.
    
    Capabilities:
    1. Strict word count adherence
    2. Multiple narrative styles
    3. Section-based organization
    4. Data point integration
    5. Citation management
    6. Requirement validation
    """
    
    def __init__(self):
        """Initialize the narrative generator."""
        self.style_templates = self._initialize_style_templates()
        self.word_count_buffer = 0.95  # Use 95% of word limit for safety
    
    def generate_narrative(
        self,
        narrative_spec: NarrativeSpecification,
        research_data: List[Dict[str, Any]],
        required_data_points: List[RequiredDataPoint] = None,
        title: str = None
    ) -> GeneratedNarrative:
        """
        Generate narrative according to specification.
        
        Args:
            narrative_spec: Narrative specification from requirements
            research_data: Collected research data
            required_data_points: Required data points to include
            title: Optional title for the narrative
            
        Returns:
            GeneratedNarrative with structured content
        """
        logger.info(f"NARRATIVE_GENERATOR: Generating {narrative_spec.style} narrative")
        if narrative_spec.word_limit:
            logger.info(f"NARRATIVE_GENERATOR: Word limit: {narrative_spec.word_limit}")
        
        try:
            # Plan narrative structure
            structure_plan = self._plan_narrative_structure(
                narrative_spec, required_data_points
            )
            
            # Allocate word budget across sections
            word_budget = self._allocate_word_budget(narrative_spec, structure_plan)
            
            # Generate sections
            sections = []
            for section_plan, word_allocation in zip(structure_plan, word_budget):
                section = self._generate_section(
                    section_plan, research_data, word_allocation, narrative_spec.style
                )
                sections.append(section)
            
            # Create narrative
            narrative = GeneratedNarrative(
                title=title or "Research Summary",
                sections=sections,
                total_word_count=sum(s.word_count for s in sections),
                style=narrative_spec.style,
                meets_requirements=True,  # Will be validated
                metadata={
                    "word_limit": narrative_spec.word_limit,
                    "required_sections": narrative_spec.required_sections,
                    "numbered_format": narrative_spec.numbered_format
                }
            )
            
            # Apply formatting
            narrative = self._apply_formatting(narrative, narrative_spec)
            
            # Validate requirements
            self._validate_narrative_requirements(narrative, narrative_spec, required_data_points)
            
            # Enforce word count if necessary
            if narrative_spec.word_limit:
                narrative = self._enforce_word_count(narrative, narrative_spec.word_limit)
            
            logger.info(f"NARRATIVE_GENERATOR: Generated narrative with {narrative.total_word_count} words")
            return narrative
            
        except Exception as e:
            logger.error(f"NARRATIVE_GENERATOR: Failed to generate narrative: {e}")
            return self._generate_fallback_narrative(research_data, narrative_spec, title)
    
    def _plan_narrative_structure(
        self,
        narrative_spec: NarrativeSpecification,
        required_data_points: List[RequiredDataPoint] = None
    ) -> List[Dict[str, Any]]:
        """Plan the structure of the narrative."""
        
        structure_plan = []
        
        # Use required sections if specified
        if narrative_spec.required_sections:
            for section_title in narrative_spec.required_sections:
                structure_plan.append({
                    "title": section_title,
                    "type": "content",
                    "priority": "high",
                    "data_points": []
                })
        else:
            # Generate default structure based on style
            if narrative_spec.style == NarrativeStyle.ACADEMIC:
                structure_plan = [
                    {"title": "Introduction", "type": "intro", "priority": "high", "data_points": []},
                    {"title": "Methodology", "type": "method", "priority": "medium", "data_points": []},
                    {"title": "Findings", "type": "content", "priority": "high", "data_points": []},
                    {"title": "Discussion", "type": "analysis", "priority": "high", "data_points": []},
                    {"title": "Conclusion", "type": "conclusion", "priority": "medium", "data_points": []}
                ]
            elif narrative_spec.style == NarrativeStyle.EXECUTIVE:
                structure_plan = [
                    {"title": "Executive Summary", "type": "summary", "priority": "high", "data_points": []},
                    {"title": "Key Findings", "type": "content", "priority": "high", "data_points": []},
                    {"title": "Recommendations", "type": "recommendations", "priority": "high", "data_points": []}
                ]
            else:  # Professional, casual, technical
                structure_plan = [
                    {"title": "Overview", "type": "intro", "priority": "high", "data_points": []},
                    {"title": "Key Information", "type": "content", "priority": "high", "data_points": []},
                    {"title": "Summary", "type": "conclusion", "priority": "medium", "data_points": []}
                ]
        
        # Assign data points to sections
        if required_data_points:
            self._assign_data_points_to_sections(structure_plan, required_data_points)
        
        logger.info(f"NARRATIVE_GENERATOR: Planned {len(structure_plan)} sections")
        return structure_plan
    
    def _assign_data_points_to_sections(
        self, 
        structure_plan: List[Dict[str, Any]], 
        required_data_points: List[RequiredDataPoint]
    ):
        """Assign required data points to narrative sections."""
        
        # Group data points by category or importance
        critical_points = [dp for dp in required_data_points if dp.is_critical]
        non_critical_points = [dp for dp in required_data_points if not dp.is_critical]
        
        # Find content sections (high priority sections that can hold data)
        content_sections = [s for s in structure_plan if s["type"] in ["content", "findings"]]
        
        if content_sections:
            # Distribute critical data points across content sections
            for i, data_point in enumerate(critical_points):
                section_idx = i % len(content_sections)
                content_sections[section_idx]["data_points"].append(data_point.name)
            
            # Add non-critical points to first content section
            if non_critical_points and content_sections:
                content_sections[0]["data_points"].extend([dp.name for dp in non_critical_points])
    
    def _allocate_word_budget(
        self,
        narrative_spec: NarrativeSpecification,
        structure_plan: List[Dict[str, Any]]
    ) -> List[int]:
        """Allocate word budget across sections."""
        
        if not narrative_spec.word_limit:
            # No word limit, use reasonable defaults
            return [200 for _ in structure_plan]
        
        # Calculate total available words (with buffer)
        available_words = int(narrative_spec.word_limit * self.word_count_buffer)
        
        # Calculate weights based on section priority and type
        weights = []
        for section_plan in structure_plan:
            if section_plan["priority"] == "high":
                weight = 3
            elif section_plan["priority"] == "medium":
                weight = 2
            else:
                weight = 1
            
            # Adjust based on section type
            if section_plan["type"] == "content":
                weight *= 1.5  # Content sections get more words
            elif section_plan["type"] == "intro":
                weight *= 0.7  # Intro sections are shorter
            elif section_plan["type"] == "conclusion":
                weight *= 0.8  # Conclusion sections are shorter
            
            weights.append(weight)
        
        # Allocate words proportionally
        total_weight = sum(weights)
        allocations = []
        
        for weight in weights:
            allocation = int((weight / total_weight) * available_words)
            allocation = max(allocation, 50)  # Minimum 50 words per section
            allocations.append(allocation)
        
        # Adjust if total exceeds limit
        total_allocated = sum(allocations)
        if total_allocated > available_words:
            # Scale down proportionally
            scale_factor = available_words / total_allocated
            allocations = [int(alloc * scale_factor) for alloc in allocations]
        
        logger.info(f"NARRATIVE_GENERATOR: Word allocation: {allocations} (total: {sum(allocations)})")
        return allocations
    
    def _generate_section(
        self,
        section_plan: Dict[str, Any],
        research_data: List[Dict[str, Any]],
        word_allocation: int,
        style: str
    ) -> NarrativeSection:
        """Generate a single narrative section."""
        
        logger.debug(f"NARRATIVE_GENERATOR: Generating section '{section_plan['title']}' ({word_allocation} words)")
        
        # Extract relevant data for this section
        relevant_data = self._extract_relevant_data(research_data, section_plan)
        
        # Generate content based on section type and style
        content = self._generate_section_content(
            section_plan, relevant_data, word_allocation, style
        )
        
        # Count words in generated content
        actual_word_count = self._count_words(content)
        
        # Trim if necessary
        if actual_word_count > word_allocation:
            content = self._trim_content(content, word_allocation)
            actual_word_count = self._count_words(content)
        
        return NarrativeSection(
            title=section_plan["title"],
            content=content,
            word_count=actual_word_count,
            data_points_covered=section_plan.get("data_points", []),
            citations=self._extract_citations(relevant_data)
        )
    
    def _extract_relevant_data(
        self, 
        research_data: List[Dict[str, Any]], 
        section_plan: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Extract research data relevant to this section."""
        
        if not section_plan.get("data_points"):
            # No specific data points, return sample of all data
            return research_data[:3]  # Limit to prevent overwhelming
        
        relevant_data = []
        data_points = section_plan["data_points"]
        
        for data_item in research_data:
            content = str(data_item.get("content", "")).lower()
            
            # Check if this data item mentions any required data points
            for data_point in data_points:
                data_point_clean = data_point.replace("_", " ").lower()
                if data_point_clean in content:
                    relevant_data.append(data_item)
                    break  # Avoid duplicates
        
        # If no specific matches, add some general data
        if not relevant_data:
            relevant_data = research_data[:2]
        
        return relevant_data
    
    def _generate_section_content(
        self,
        section_plan: Dict[str, Any],
        relevant_data: List[Dict[str, Any]],
        word_allocation: int,
        style: str
    ) -> str:
        """Generate content for a specific section."""
        
        section_type = section_plan["type"]
        section_title = section_plan["title"]
        
        if section_type == "intro":
            return self._generate_intro_content(relevant_data, word_allocation, style)
        elif section_type == "content" or section_type == "findings":
            return self._generate_content_section(relevant_data, word_allocation, style)
        elif section_type == "analysis":
            return self._generate_analysis_content(relevant_data, word_allocation, style)
        elif section_type == "conclusion":
            return self._generate_conclusion_content(relevant_data, word_allocation, style)
        elif section_type == "summary":
            return self._generate_summary_content(relevant_data, word_allocation, style)
        elif section_type == "recommendations":
            return self._generate_recommendations_content(relevant_data, word_allocation, style)
        else:
            # Generic content generation
            return self._generate_generic_content(relevant_data, word_allocation, style)
    
    def _generate_intro_content(
        self, 
        relevant_data: List[Dict[str, Any]], 
        word_allocation: int, 
        style: str
    ) -> str:
        """Generate introduction section content."""
        
        if style == NarrativeStyle.ACADEMIC:
            intro = "This research examines "
        elif style == NarrativeStyle.EXECUTIVE:
            intro = "This analysis provides "
        else:
            intro = "This report covers "
        
        # Extract key topics from data
        topics = []
        for data_item in relevant_data[:2]:
            content = str(data_item.get("content", ""))[:200]
            # Extract key phrases
            key_phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[a-z]+){0,2}\b', content)
            topics.extend(key_phrases[:3])
        
        if topics:
            unique_topics = list(dict.fromkeys(topics))[:3]  # Remove duplicates, limit to 3
            topics_text = ", ".join(unique_topics[:2])
            if len(unique_topics) > 2:
                topics_text += f", and {unique_topics[2]}"
            
            intro += f"{topics_text.lower()}. "
        
        # Add context based on available data
        if relevant_data:
            intro += f"Based on analysis of {len(relevant_data)} key sources, "
            intro += "this section provides foundational context and establishes "
            intro += "the framework for understanding the key findings presented in subsequent sections."
        
        return self._trim_content(intro, word_allocation)
    
    def _generate_content_section(
        self, 
        relevant_data: List[Dict[str, Any]], 
        word_allocation: int, 
        style: str
    ) -> str:
        """Generate main content section."""
        
        paragraphs = []
        
        for data_item in relevant_data:
            content = str(data_item.get("content", ""))
            
            # Extract key information
            sentences = re.split(r'[.!?]+', content)
            key_sentences = [s.strip() for s in sentences if len(s.strip()) > 20][:3]
            
            if key_sentences:
                paragraph = ". ".join(key_sentences) + "."
                paragraphs.append(paragraph)
        
        content_text = "\n\n".join(paragraphs)
        
        if not content_text.strip():
            content_text = "Based on the research conducted, several key findings emerge that are relevant to this analysis. "
            content_text += "The data indicates important patterns and trends that contribute to our understanding of the topic. "
            content_text += "These findings provide valuable insights that inform the overall conclusions of this research."
        
        return self._trim_content(content_text, word_allocation)
    
    def _generate_analysis_content(
        self, 
        relevant_data: List[Dict[str, Any]], 
        word_allocation: int, 
        style: str
    ) -> str:
        """Generate analysis section content."""
        
        analysis = ""
        
        if style == NarrativeStyle.ACADEMIC:
            analysis = "The analysis reveals several significant patterns in the data. "
        else:
            analysis = "Analysis of the collected information shows "
        
        # Extract numeric data for analysis
        numeric_mentions = []
        for data_item in relevant_data:
            content = str(data_item.get("content", ""))
            numbers = re.findall(r'\d+(?:\.\d+)?%?|\$\d+(?:,\d{3})*(?:\.\d{2})?', content)
            numeric_mentions.extend(numbers[:2])  # Limit per item
        
        if numeric_mentions:
            analysis += f"Key metrics include {', '.join(numeric_mentions[:3])}. "
        
        analysis += "These findings suggest important implications for understanding the broader context. "
        analysis += "The patterns observed align with expected outcomes while also revealing some unexpected trends that warrant further consideration."
        
        return self._trim_content(analysis, word_allocation)
    
    def _generate_conclusion_content(
        self, 
        relevant_data: List[Dict[str, Any]], 
        word_allocation: int, 
        style: str
    ) -> str:
        """Generate conclusion section content."""
        
        if style == NarrativeStyle.ACADEMIC:
            conclusion = "In conclusion, this research demonstrates "
        elif style == NarrativeStyle.EXECUTIVE:
            conclusion = "In summary, the key takeaways are "
        else:
            conclusion = "To conclude, "
        
        conclusion += "that the analyzed information provides valuable insights into the topic under investigation. "
        conclusion += "The findings contribute to our understanding and offer a foundation for future analysis. "
        
        if len(relevant_data) > 1:
            conclusion += f"Based on {len(relevant_data)} sources analyzed, "
            conclusion += "the evidence supports the main conclusions presented throughout this report."
        
        return self._trim_content(conclusion, word_allocation)
    
    def _generate_summary_content(
        self, 
        relevant_data: List[Dict[str, Any]], 
        word_allocation: int, 
        style: str
    ) -> str:
        """Generate executive summary content."""
        
        summary = "This summary presents the most critical findings from the research conducted. "
        
        # Extract key points
        key_points = []
        for data_item in relevant_data[:3]:
            content = str(data_item.get("content", ""))
            # Find sentences with numbers or key indicators
            sentences = re.split(r'[.!?]+', content)
            important_sentences = [
                s.strip() for s in sentences 
                if any(indicator in s.lower() for indicator in ['shows', 'indicates', 'demonstrates', 'reveals']) 
                and len(s.strip()) > 20
            ]
            key_points.extend(important_sentences[:2])
        
        if key_points:
            for point in key_points[:3]:
                summary += f"{point}. "
        
        summary += "These findings provide the essential information needed for decision-making and strategic planning."
        
        return self._trim_content(summary, word_allocation)
    
    def _generate_recommendations_content(
        self, 
        relevant_data: List[Dict[str, Any]], 
        word_allocation: int, 
        style: str
    ) -> str:
        """Generate recommendations section content."""
        
        recommendations = "Based on the analysis conducted, the following recommendations emerge:\n\n"
        
        # Generate 3-4 recommendations based on data
        rec_starters = [
            "Consider implementing",
            "It is advisable to",
            "Organizations should",
            "Future efforts might focus on"
        ]
        
        for i, starter in enumerate(rec_starters[:3]):
            recommendations += f"{i+1}. {starter} strategies that address the key findings identified in this research.\n\n"
        
        recommendations += "These recommendations provide a framework for action based on the evidence collected and analyzed."
        
        return self._trim_content(recommendations, word_allocation)
    
    def _generate_generic_content(
        self, 
        relevant_data: List[Dict[str, Any]], 
        word_allocation: int, 
        style: str
    ) -> str:
        """Generate generic content for unspecified section types."""
        
        content = "This section presents information relevant to the research topic. "
        
        if relevant_data:
            content += f"Based on {len(relevant_data)} sources analyzed, "
            content += "several important points emerge that contribute to our understanding. "
            
            # Add some content from the data
            for data_item in relevant_data[:2]:
                item_content = str(data_item.get("content", ""))[:100]
                if item_content.strip():
                    content += f"{item_content.strip()}. "
        
        content += "This information provides context for the broader analysis presented in this report."
        
        return self._trim_content(content, word_allocation)
    
    def _count_words(self, text: str) -> int:
        """Count words in text."""
        # Remove markdown and clean text
        clean_text = re.sub(r'[#*`\[\]()]+', '', text)
        words = re.findall(r'\b\w+\b', clean_text)
        return len(words)
    
    def _trim_content(self, content: str, word_limit: int) -> str:
        """Trim content to word limit."""
        words = re.findall(r'\S+', content)
        if len(words) <= word_limit:
            return content
        
        # Trim to word limit
        trimmed_words = words[:word_limit]
        trimmed_text = ' '.join(trimmed_words)
        
        # Try to end on a complete sentence
        last_period = trimmed_text.rfind('.')
        if last_period > len(trimmed_text) * 0.8:  # If period is in last 20%
            trimmed_text = trimmed_text[:last_period + 1]
        else:
            # Add ellipsis if no good sentence ending
            trimmed_text += "..."
        
        return trimmed_text
    
    def _extract_citations(self, relevant_data: List[Dict[str, Any]]) -> List[str]:
        """Extract citations from relevant data."""
        citations = []
        for data_item in relevant_data:
            if 'url' in data_item:
                citations.append(data_item['url'])
        return citations
    
    def _apply_formatting(
        self, 
        narrative: GeneratedNarrative, 
        narrative_spec: NarrativeSpecification
    ) -> GeneratedNarrative:
        """Apply formatting based on specification."""
        
        if narrative_spec.numbered_format:
            # Add numbers to section titles
            for i, section in enumerate(narrative.sections, 1):
                if not section.title.startswith(f"{i}."):
                    section.title = f"{i}. {section.title}"
        
        return narrative
    
    def _validate_narrative_requirements(
        self,
        narrative: GeneratedNarrative,
        narrative_spec: NarrativeSpecification,
        required_data_points: List[RequiredDataPoint] = None
    ):
        """Validate that narrative meets requirements."""
        
        issues = []
        
        # Check word count
        if narrative_spec.word_limit:
            if narrative.total_word_count > narrative_spec.word_limit:
                issues.append(f"Exceeds word limit: {narrative.total_word_count} > {narrative_spec.word_limit}")
        
        # Check required sections
        if narrative_spec.required_sections:
            section_titles = [s.title for s in narrative.sections]
            for required_section in narrative_spec.required_sections:
                if not any(required_section.lower() in title.lower() for title in section_titles):
                    issues.append(f"Missing required section: {required_section}")
        
        # Check data point coverage
        if required_data_points:
            all_covered_points = []
            for section in narrative.sections:
                all_covered_points.extend(section.data_points_covered)
            
            for data_point in required_data_points:
                if data_point.is_critical and data_point.name not in all_covered_points:
                    # Check if mentioned in content
                    full_text = narrative.get_full_text().lower()
                    data_point_clean = data_point.name.replace("_", " ").lower()
                    if data_point_clean not in full_text:
                        issues.append(f"Missing critical data point: {data_point.name}")
        
        # Update narrative metadata
        narrative.meets_requirements = len(issues) == 0
        narrative.metadata["validation_issues"] = issues
        
        if issues:
            logger.warning(f"NARRATIVE_GENERATOR: Validation issues: {issues}")
        else:
            logger.info("NARRATIVE_GENERATOR: Narrative validation passed")
    
    def _enforce_word_count(
        self, 
        narrative: GeneratedNarrative, 
        word_limit: int
    ) -> GeneratedNarrative:
        """Enforce strict word count limit by trimming if necessary."""
        
        current_count = narrative.get_word_count()
        
        if current_count <= word_limit:
            return narrative  # Already within limit
        
        logger.info(f"NARRATIVE_GENERATOR: Enforcing word limit: {current_count} -> {word_limit}")
        
        # Calculate reduction needed
        words_to_remove = current_count - word_limit
        
        # Remove words proportionally from sections, starting with lowest priority
        section_priorities = []
        for section in narrative.sections:
            if "conclusion" in section.title.lower() or "summary" in section.title.lower():
                priority = 1  # Low priority for trimming
            elif "intro" in section.title.lower():
                priority = 2
            else:
                priority = 3  # High priority sections trimmed last
            section_priorities.append(priority)
        
        # Sort sections by priority for trimming (lowest first)
        sections_with_priority = list(zip(narrative.sections, section_priorities))
        sections_with_priority.sort(key=lambda x: x[1])
        
        words_removed = 0
        for section, priority in sections_with_priority:
            if words_removed >= words_to_remove:
                break
            
            # Calculate how many words to remove from this section
            words_needed = min(words_to_remove - words_removed, section.word_count // 3)
            if words_needed > 0:
                new_word_limit = section.word_count - words_needed
                section.content = self._trim_content(section.content, new_word_limit)
                section.word_count = self._count_words(section.content)
                words_removed += words_needed
        
        # Update total word count
        narrative.total_word_count = sum(s.word_count for s in narrative.sections)
        
        logger.info(f"NARRATIVE_GENERATOR: Word count enforced: {narrative.total_word_count}")
        return narrative
    
    def _generate_fallback_narrative(
        self,
        research_data: List[Dict[str, Any]],
        narrative_spec: NarrativeSpecification,
        title: str = None
    ) -> GeneratedNarrative:
        """Generate fallback narrative when main generation fails."""
        
        logger.warning("NARRATIVE_GENERATOR: Generating fallback narrative")
        
        # Create simple summary from research data
        content = "Based on the research conducted, several key points emerge:\n\n"
        
        for i, data_item in enumerate(research_data[:3], 1):
            item_content = str(data_item.get("content", ""))[:150]
            content += f"{i}. {item_content}\n\n"
        
        content += "This summary provides an overview of the key findings from the available research data."
        
        # Apply word limit if specified
        if narrative_spec.word_limit:
            content = self._trim_content(content, narrative_spec.word_limit)
        
        section = NarrativeSection(
            title="Summary",
            content=content,
            word_count=self._count_words(content),
            data_points_covered=[]
        )
        
        return GeneratedNarrative(
            title=title or "Research Summary",
            sections=[section],
            total_word_count=section.word_count,
            style=narrative_spec.style,
            meets_requirements=False,
            metadata={
                "is_fallback": True,
                "word_limit": narrative_spec.word_limit
            }
        )
    
    def _initialize_style_templates(self) -> Dict[str, Dict[str, str]]:
        """Initialize style-specific templates and phrases."""
        return {
            NarrativeStyle.PROFESSIONAL: {
                "intro_starter": "This analysis examines",
                "conclusion_starter": "In conclusion",
                "transition": "Furthermore",
                "emphasis": "It is important to note"
            },
            NarrativeStyle.ACADEMIC: {
                "intro_starter": "This research investigates",
                "conclusion_starter": "The findings demonstrate",
                "transition": "Moreover",
                "emphasis": "Significantly"
            },
            NarrativeStyle.EXECUTIVE: {
                "intro_starter": "This executive summary presents",
                "conclusion_starter": "Key takeaways include",
                "transition": "Additionally",
                "emphasis": "Critically important"
            },
            NarrativeStyle.CASUAL: {
                "intro_starter": "Let's look at",
                "conclusion_starter": "To wrap up",
                "transition": "Also",
                "emphasis": "What's really important"
            },
            NarrativeStyle.TECHNICAL: {
                "intro_starter": "This technical analysis covers",
                "conclusion_starter": "The analysis concludes",
                "transition": "Subsequently",
                "emphasis": "A key consideration"
            }
        }