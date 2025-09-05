"""
Reporter Agent: Report synthesis and formatting specialist.

Generates styled reports from research findings with proper citations.
"""

from typing import Dict, Any, Optional, List, Literal
from datetime import datetime

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langgraph.types import Command

from deep_research_agent.core import get_logger, Citation
from deep_research_agent.core.multi_agent_state import EnhancedResearchState, StateManager
from deep_research_agent.core.report_styles import (
    ReportStyle,
    STYLE_CONFIGS,
    StyleTemplate,
    ReportFormatter
)
from deep_research_agent.core.grounding import HallucinationPrevention


logger = get_logger(__name__)


class ReporterAgent:
    """
    Reporter agent that generates formatted reports from research findings.
    
    Responsibilities:
    - Compile all observations
    - Apply style-specific formatting
    - Structure final report
    - Ensure citation compliance
    - Integrate grounding markers if enabled
    """
    
    def __init__(self, llm=None, config=None):
        """
        Initialize the reporter agent.
        
        Args:
            llm: Language model for report generation
            config: Configuration dictionary  
        """
        self.llm = llm
        self.config = config or {}
        self.name = "Reporter"  # Capital for test compatibility
        self.formatter = ReportFormatter()
        
        # Extract report configuration
        report_config = self.config.get('report', {})
        self.default_style = ReportStyle(report_config.get('default_style', 'professional'))
        self.include_citations = report_config.get('include_citations', True)
        self.include_grounding_markers = report_config.get('include_grounding_markers', True)
        self.hallucination_prevention = HallucinationPrevention()
    
    def __call__(
        self,
        state: EnhancedResearchState,
        config: Dict[str, Any]
    ) -> Command[Literal["end"]]:
        """
        Generate final report from research findings.
        
        Args:
            state: Current research state
            config: Configuration dictionary
            
        Returns:
            Command to end workflow with final report
        """
        logger.info("Reporter agent generating final report")
        
        # Get report style
        report_style = state.get("report_style", ReportStyle.PROFESSIONAL)
        logger.info(f"Using report style: {report_style}")
        
        # Get style configuration
        style_config = STYLE_CONFIGS[report_style]
        
        # Compile research findings
        compiled_findings = self._compile_findings(state)
        
        # Generate report sections
        report_sections = self._generate_report_sections(
            compiled_findings,
            style_config,
            state
        )
        
        # Apply style formatting
        formatted_report = self._apply_style_formatting(
            report_sections,
            report_style,
            state
        )
        
        # Add citations and references
        final_report = self._add_citations_and_references(
            formatted_report,
            state.get("citations", []),
            report_style
        )
        
        # Add grounding markers if enabled
        if state.get("enable_grounding") and state.get("grounding_results"):
            final_report = self._add_grounding_markers(
                final_report,
                state["grounding_results"]
            )
        
        # Add metadata
        report_with_metadata = self._add_report_metadata(
            final_report,
            state,
            report_style
        )
        
        # Update state with final report
        state["final_report"] = report_with_metadata
        state["report_sections"] = report_sections
        
        # Record completion
        state = StateManager.finalize_state(state)
        
        logger.info("Report generation completed")
        
        return Command(
            goto="end",
            update={
                "final_report": report_with_metadata,
                "report_sections": report_sections
            }
        )
    
    def _compile_findings(self, state: EnhancedResearchState) -> Dict[str, Any]:
        """Compile all research findings from state."""
        
        # Get observations
        observations = state.get("observations", [])
        
        # Get completed steps for structure
        plan = state.get("current_plan")
        completed_steps = []
        if plan:
            completed_steps = [
                step for step in plan.steps
                if step.status == "completed"
            ]
        
        # Get citations
        citations = state.get("citations", [])
        
        # Get background investigation if available
        background = state.get("background_investigation_results")
        
        # Get reflections if available
        reflections = state.get("reflections", [])
        
        # Compile into structured format
        compiled = {
            "research_topic": state.get("research_topic", ""),
            "background_context": background,
            "observations": observations,
            "completed_steps": completed_steps,
            "citations": citations,
            "reflections": reflections,
            "total_sources": len(citations),
            "confidence_score": state.get("confidence_score", 0.0),
            "factuality_score": state.get("factuality_score", 0.0)
        }
        
        logger.info(f"Compiled {len(observations)} observations from {len(completed_steps)} steps")
        
        return compiled
    
    def _generate_report_sections(
        self,
        findings: Dict[str, Any],
        style_config,
        state: EnhancedResearchState
    ) -> Dict[str, str]:
        """Generate report sections based on style configuration."""
        
        sections = {}
        
        for section_name in style_config.structure:
            logger.info(f"Generating section: {section_name}")
            
            # Get section template
            section_template = StyleTemplate.get_section_template(
                style_config.style,
                section_name
            )
            
            # Generate section content
            section_content = self._generate_section_content(
                section_name,
                findings,
                section_template,
                style_config.style
            )
            
            sections[section_name] = section_content
        
        return sections
    
    def _generate_section_content(
        self,
        section_name: str,
        findings: Dict[str, Any],
        template: str,
        style: ReportStyle
    ) -> str:
        """Generate content for a specific report section."""
        
        # Build prompt for section generation
        prompt = self._build_section_prompt(
            section_name,
            findings,
            template,
            style
        )
        
        if self.llm:
            messages = [
                SystemMessage(content=StyleTemplate.get_style_prompt(style)),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            content = response.content
        else:
            # Fallback content generation
            content = self._generate_fallback_section(
                section_name,
                findings,
                style
            )
        
        return content
    
    def _build_section_prompt(
        self,
        section_name: str,
        findings: Dict[str, Any],
        template: str,
        style: ReportStyle
    ) -> str:
        """Build prompt for generating a report section."""
        
        prompt_parts = [
            f"Generate the '{section_name}' section for a {style.value} style report.",
            "",
            "Research Topic: " + findings["research_topic"],
            ""
        ]
        
        # Add relevant findings based on section
        if section_name in ["Introduction", "Background", "Background Context"]:
            if findings.get("background_context"):
                prompt_parts.append("Background Information:")
                prompt_parts.append(findings["background_context"][:1000])
        
        elif section_name in ["Findings", "Analysis", "Main Discoveries", "Key Findings"]:
            prompt_parts.append("Key Observations:")
            for i, obs in enumerate(findings["observations"][:10], 1):
                prompt_parts.append(f"{i}. {obs}")
        
        elif section_name in ["Conclusion", "Summary", "Key Takeaways"]:
            if findings.get("reflections"):
                prompt_parts.append("Research Reflections:")
                prompt_parts.append(findings["reflections"][-1])
        
        elif section_name in ["References", "Bibliography", "Citations"]:
            prompt_parts.append(f"Total Sources: {findings['total_sources']}")
        
        prompt_parts.append("")
        prompt_parts.append("Section Template Guidelines:")
        prompt_parts.append(template)
        
        return "\n".join(prompt_parts)
    
    def _generate_fallback_section(
        self,
        section_name: str,
        findings: Dict[str, Any],
        style: ReportStyle
    ) -> str:
        """Generate fallback content for a section."""
        
        if section_name in ["Introduction", "Overview"]:
            return f"This report presents research findings on {findings['research_topic']}."
        
        elif section_name in ["Findings", "Key Findings"]:
            content = "Key findings from the research:\n\n"
            for i, obs in enumerate(findings["observations"][:5], 1):
                content += f"{i}. {obs}\n"
            return content
        
        elif section_name in ["Conclusion", "Summary"]:
            return (
                f"This research on {findings['research_topic']} revealed "
                f"{len(findings['observations'])} key insights based on "
                f"{findings['total_sources']} sources."
            )
        
        elif section_name in ["References", "Bibliography"]:
            return "See citations section for complete references."
        
        else:
            return f"[{section_name} content to be added]"
    
    def _apply_style_formatting(
        self,
        sections: Dict[str, str],
        style: ReportStyle,
        state: EnhancedResearchState
    ) -> str:
        """Apply style-specific formatting to report sections."""
        
        formatted_parts = []
        
        # Add report title
        title = f"# Research Report: {state.get('research_topic', 'Research Findings')}\n\n"
        formatted_parts.append(title)
        
        # Add metadata if not social media style
        if style not in [ReportStyle.SOCIAL_MEDIA]:
            metadata = self._generate_metadata_section(state, style)
            if metadata:
                formatted_parts.append(metadata + "\n\n")
        
        # Format each section
        for section_name, content in sections.items():
            # Add section header
            header = self.formatter.format_section_header(section_name, style)
            formatted_parts.append(header)
            
            # Apply style-specific formatting to content
            formatted_content = self.formatter.apply_style_formatting(content, style)
            formatted_parts.append(formatted_content)
            formatted_parts.append("\n")
        
        return "".join(formatted_parts)
    
    def _generate_metadata_section(
        self,
        state: EnhancedResearchState,
        style: ReportStyle
    ) -> Optional[str]:
        """Generate metadata section for report."""
        
        if style == ReportStyle.ACADEMIC:
            return (
                f"**Date:** {datetime.now().strftime('%B %d, %Y')}\n"
                f"**Sources Consulted:** {len(state.get('citations', []))}\n"
                f"**Research Confidence:** {(state.get('confidence_score') or 0):.1%}"
            )
        elif style in [ReportStyle.PROFESSIONAL, ReportStyle.EXECUTIVE]:
            return (
                f"**Prepared:** {datetime.now().strftime('%B %d, %Y')}\n"
                f"**Data Sources:** {len(state.get('citations', []))}\n"
                f"**Confidence Level:** {(state.get('confidence_score') or 0):.1%}"
            )
        elif style == ReportStyle.TECHNICAL:
            return (
                f"**Generated:** {datetime.now().isoformat()}\n"
                f"**Sources:** {len(state.get('citations', []))}\n"
                f"**Observations:** {len(state.get('observations', []))}\n"
                f"**Factuality Score:** {(state.get('factuality_score') or 0):.2f}"
            )
        
        return None
    
    def _add_citations_and_references(
        self,
        report: str,
        citations: List[Citation],
        style: ReportStyle
    ) -> str:
        """Add citations and references section to report."""
        
        if not citations:
            return report
        
        # Format citations according to style
        formatted_citations = []
        for i, citation in enumerate(citations, 1):
            citation_dict = {
                "number": i,
                "title": citation.title,
                "url": citation.source,
                "author": "Unknown",  # Would need to extract from source
                "date": datetime.now().strftime("%Y")
            }
            formatted = self.formatter.format_citation(citation_dict, style)
            formatted_citations.append(formatted)
        
        # Add references section
        references_header = self.formatter.format_section_header("References", style)
        
        if style == ReportStyle.ACADEMIC:
            # APA style bibliography
            references = references_header + "\n".join(formatted_citations)
        elif style == ReportStyle.TECHNICAL:
            # Numbered references
            references = references_header
            for i, citation in enumerate(formatted_citations, 1):
                references += f"[{i}] {citation}\n"
        elif style == ReportStyle.SOCIAL_MEDIA:
            # Simple links
            references = "\n\n📚 Sources:\n"
            for citation in citations[:3]:  # Limit for social media
                references += f"• {citation.title[:50]}... [{citation.source}]\n"
        else:
            # Standard format
            references = references_header
            for citation in formatted_citations:
                references += f"• {citation}\n"
        
        # Append to report
        return report + "\n\n" + references
    
    def _add_grounding_markers(
        self,
        report: str,
        grounding_results: List
    ) -> str:
        """Add grounding markers to indicate factuality status."""
        
        # Add confidence summary at the beginning
        if grounding_results:
            # Create a simple grounding summary
            grounded_count = sum(
                1 for r in grounding_results
                if r.status == "grounded"
            )
            total_claims = len(grounding_results)
            
            grounding_summary = (
                f"\n📊 **Factuality Assessment**: "
                f"{grounded_count}/{total_claims} claims verified\n\n"
            )
            
            # Insert after title
            parts = report.split("\n\n", 1)
            if len(parts) == 2:
                report = parts[0] + "\n" + grounding_summary + parts[1]
            else:
                report = grounding_summary + report
        
        # Optionally add inline markers (if detailed marking needed)
        # This would require more sophisticated text processing
        
        return report
    
    def _add_report_metadata(
        self,
        report: str,
        state: EnhancedResearchState,
        style: ReportStyle
    ) -> str:
        """Add final metadata to report."""
        
        # Generate footer based on style
        footer_parts = []
        
        if style != ReportStyle.SOCIAL_MEDIA:
            footer_parts.append("\n" + "="*50 + "\n")
            
            # Add generation information
            footer_parts.append(
                f"Generated by Deep Research Agent\n"
                f"Report Style: {style.value}\n"
                f"Generation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            
            # Add quality metrics if available
            if state.get("factuality_score") is not None:
                footer_parts.append(
                    f"Factuality Score: {state['factuality_score']:.2f}\n"
                )
            
            if state.get("confidence_score") is not None:
                footer_parts.append(
                    f"Confidence Score: {state['confidence_score']:.2f}\n"
                )
            
            # Add research metrics
            plan = state.get("current_plan")
            if plan:
                footer_parts.append(
                    f"Research Steps Completed: {plan.completed_steps}/{len(plan.steps)}\n"
                )
            
            footer_parts.append("="*50)
        
        if footer_parts:
            report += "\n" + "".join(footer_parts)
        
        return report