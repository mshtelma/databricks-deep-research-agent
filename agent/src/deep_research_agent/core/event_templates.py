"""
Event templates for generating human-readable descriptions and titles.

This module provides templates and utilities for creating user-friendly
descriptions of intermediate events emitted during agent execution.
"""

from typing import Dict, Any, Optional
from .types import IntermediateEventType, EventCategory


class EventTemplates:
    """Templates for generating human-readable event descriptions."""
    
    # Title templates for different event types
    TITLES = {
        # Search events
        IntermediateEventType.QUERY_GENERATED: "Generated search query",
        IntermediateEventType.QUERY_EXECUTING: "Executing search",
        IntermediateEventType.SEARCH_RESULTS_FOUND: "Found search results",
        IntermediateEventType.TOOL_CALL_START: "Starting {tool_name}",
        IntermediateEventType.TOOL_CALL_COMPLETE: "Completed {tool_name}",
        IntermediateEventType.SOURCE_DISCOVERED: "Discovered new source",
        IntermediateEventType.SEARCH_STRATEGY: "Planning search strategy",
        
        # Reflection events
        IntermediateEventType.REASONING_REFLECTION: "Reflecting on approach",
        IntermediateEventType.AGENT_THINKING: "Analyzing situation",
        IntermediateEventType.HYPOTHESIS_FORMED: "Formed hypothesis",
        IntermediateEventType.CONFIDENCE_UPDATE: "Updated confidence",
        IntermediateEventType.KNOWLEDGE_GAP_IDENTIFIED: "Identified knowledge gap",
        
        # Analysis events
        IntermediateEventType.SOURCE_EVALUATION: "Evaluating source relevance",
        IntermediateEventType.SOURCE_ANALYZED: "Analyzed source content",
        IntermediateEventType.CONTEXT_ESTABLISHED: "Established context",
        IntermediateEventType.QUALITY_ASSESSMENT: "Assessed information quality",
        
        # Synthesis events
        IntermediateEventType.PARTIAL_SYNTHESIS: "Drew partial conclusion",
        IntermediateEventType.SYNTHESIS_PROGRESS: "Synthesizing information",
        IntermediateEventType.SYNTHESIS_STRATEGY: "Planning synthesis approach",
        IntermediateEventType.SECTION_GENERATION: "Generating report section",
        IntermediateEventType.CITATION_LINKING: "Linking citations",
        IntermediateEventType.REPORT_GENERATION: "Generating final report",
        
        # Planning events
        IntermediateEventType.PLAN_CONSIDERATION: "Considering research plan",
        IntermediateEventType.PLAN_CREATED: "Created research plan",
        IntermediateEventType.PLAN_UPDATED: "Updated research plan",
        IntermediateEventType.STEP_GENERATED: "Generated plan step",
        IntermediateEventType.PLAN_QUALITY_ASSESSMENT: "Assessed plan quality",
        IntermediateEventType.PLAN_REVISION: "Revising plan",
        IntermediateEventType.INVESTIGATION_START: "Starting investigation",
        
        # Verification events
        IntermediateEventType.CLAIM_IDENTIFIED: "Identified claim to verify",
        IntermediateEventType.VERIFICATION_ATTEMPT: "Verifying claim",
        IntermediateEventType.CONTRADICTION_FOUND: "Found contradiction",
        IntermediateEventType.GROUNDING_START: "Starting fact verification",
        IntermediateEventType.GROUNDING_COMPLETE: "Completed fact verification",
        IntermediateEventType.GROUNDING_CONTRADICTION: "Found factual contradiction",
        IntermediateEventType.CONFIDENCE_ADJUSTMENT: "Adjusted confidence level",
        
        # Coordination events
        IntermediateEventType.AGENT_HANDOFF: "Transferring to {to_agent}",
        IntermediateEventType.AGENT_START: "Starting {agent} agent",
        IntermediateEventType.AGENT_COMPLETE: "Completed {agent} agent",
        IntermediateEventType.STAGE_TRANSITION: "Moving to {stage}",
        
        # Error events
        IntermediateEventType.TOOL_CALL_ERROR: "Error with {tool_name}",
    }
    
    # Detailed description templates
    DESCRIPTIONS = {
        # Search events
        IntermediateEventType.QUERY_GENERATED: "Generated search query: '{query}' to explore {purpose}",
        IntermediateEventType.QUERY_EXECUTING: "Searching for '{query}' using {search_provider}",
        IntermediateEventType.SEARCH_RESULTS_FOUND: "Found {result_count} results for '{query}' with average relevance {avg_relevance:.1%}",
        IntermediateEventType.TOOL_CALL_START: "Executing {tool_name} with parameters: {parameters}",
        IntermediateEventType.TOOL_CALL_COMPLETE: "Completed {tool_name} successfully, found {result_count} results",
        IntermediateEventType.SOURCE_DISCOVERED: "Discovered source: '{title}' from {domain} (relevance: {relevance:.1%})",
        IntermediateEventType.SEARCH_STRATEGY: "Planning to search for {query_count} queries focusing on {focus_areas}",
        
        # Reflection events
        IntermediateEventType.REASONING_REFLECTION: "Considering {options} because {reasoning}. Current confidence: {confidence:.1%}",
        IntermediateEventType.AGENT_THINKING: "Analyzing {subject} to determine {goal}",
        IntermediateEventType.HYPOTHESIS_FORMED: "Hypothesis: {hypothesis} (confidence: {confidence:.1%})",
        IntermediateEventType.CONFIDENCE_UPDATE: "Confidence {direction} to {new_confidence:.1%} based on {evidence}",
        IntermediateEventType.KNOWLEDGE_GAP_IDENTIFIED: "Need more information about {topic} to {purpose}",
        
        # Analysis events
        IntermediateEventType.SOURCE_EVALUATION: "Evaluating '{title}': {relevance:.1%} relevant because {reasoning}",
        IntermediateEventType.SOURCE_ANALYZED: "Extracted {insights_count} key insights from '{title}'",
        IntermediateEventType.CONTEXT_ESTABLISHED: "Established context for {topic} with {source_count} sources",
        IntermediateEventType.QUALITY_ASSESSMENT: "Information quality: {quality:.1%} based on {criteria}",
        
        # Synthesis events
        IntermediateEventType.PARTIAL_SYNTHESIS: "Partial finding: {conclusion} based on {source_count} sources",
        IntermediateEventType.SYNTHESIS_PROGRESS: "Synthesizing {section} using insights from {source_count} sources",
        IntermediateEventType.SYNTHESIS_STRATEGY: "Will organize findings into {sections} focusing on {approach}",
        IntermediateEventType.SECTION_GENERATION: "Generating section: {section_title} with {citations_count} citations",
        IntermediateEventType.CITATION_LINKING: "Linked {citation_count} citations to support {claim}",
        IntermediateEventType.REPORT_GENERATION: "Generating {report_style} report with {sections_count} sections",
        
        # Planning events
        IntermediateEventType.PLAN_CONSIDERATION: "Considering {approach} because {reasoning}",
        IntermediateEventType.PLAN_CREATED: "Created {step_count}-step research plan focusing on {objectives}",
        IntermediateEventType.PLAN_UPDATED: "Updated plan: {changes} (quality: {quality:.1%})",
        IntermediateEventType.STEP_GENERATED: "Step {step_number}: {description} (estimated time: {duration})",
        IntermediateEventType.PLAN_QUALITY_ASSESSMENT: "Plan quality: {quality:.1%} - {assessment}",
        IntermediateEventType.PLAN_REVISION: "Revising plan because {reason}",
        IntermediateEventType.INVESTIGATION_START: "Starting background investigation on {topic}",
        
        # Verification events
        IntermediateEventType.CLAIM_IDENTIFIED: "Claim to verify: '{claim}' from {source}",
        IntermediateEventType.VERIFICATION_ATTEMPT: "Verifying '{claim}' using {method}",
        IntermediateEventType.CONTRADICTION_FOUND: "Contradiction: {claim} conflicts with {evidence}",
        IntermediateEventType.GROUNDING_START: "Starting fact verification with {level} rigor",
        IntermediateEventType.GROUNDING_COMPLETE: "Verification complete: {factuality:.1%} factual accuracy",
        IntermediateEventType.GROUNDING_CONTRADICTION: "Factual issue found: {contradiction}",
        IntermediateEventType.CONFIDENCE_ADJUSTMENT: "Adjusted confidence from {old_confidence:.1%} to {new_confidence:.1%} due to {reason}",
        
        # Coordination events
        IntermediateEventType.AGENT_HANDOFF: "Transferring from {from_agent} to {to_agent}: {reason}",
        IntermediateEventType.AGENT_START: "Starting {agent} agent to {purpose}",
        IntermediateEventType.AGENT_COMPLETE: "Completed {agent} agent: {summary}",
        IntermediateEventType.STAGE_TRANSITION: "Transitioning from {from_stage} to {to_stage}",
        
        # Error events
        IntermediateEventType.TOOL_CALL_ERROR: "Error executing {tool_name}: {error_message}",
    }
    
    # Icon mapping for different event categories
    CATEGORY_ICONS = {
        EventCategory.SEARCH: "🔍",
        EventCategory.REFLECTION: "💭", 
        EventCategory.ANALYSIS: "📊",
        EventCategory.SYNTHESIS: "🧩",
        EventCategory.PLANNING: "📋",
        EventCategory.VERIFICATION: "✅",
        EventCategory.COORDINATION: "🤝",
        EventCategory.ERROR: "⚠️",
    }
    
    # Color schemes for different categories
    CATEGORY_COLORS = {
        EventCategory.SEARCH: {"bg": "bg-blue-50", "text": "text-blue-700", "border": "border-blue-200"},
        EventCategory.REFLECTION: {"bg": "bg-purple-50", "text": "text-purple-700", "border": "border-purple-200"},
        EventCategory.ANALYSIS: {"bg": "bg-orange-50", "text": "text-orange-700", "border": "border-orange-200"},
        EventCategory.SYNTHESIS: {"bg": "bg-green-50", "text": "text-green-700", "border": "border-green-200"},
        EventCategory.PLANNING: {"bg": "bg-indigo-50", "text": "text-indigo-700", "border": "border-indigo-200"},
        EventCategory.VERIFICATION: {"bg": "bg-red-50", "text": "text-red-700", "border": "border-red-200"},
        EventCategory.COORDINATION: {"bg": "bg-gray-50", "text": "text-gray-700", "border": "border-gray-200"},
        EventCategory.ERROR: {"bg": "bg-yellow-50", "text": "text-yellow-700", "border": "border-yellow-200"},
    }

    @classmethod
    def get_title(cls, event_type: IntermediateEventType, data: Dict[str, Any] = None) -> str:
        """
        Get human-readable title for an event.
        
        Args:
            event_type: The type of event
            data: Event data for template interpolation
            
        Returns:
            Human-readable title
        """
        template = cls.TITLES.get(event_type, "Agent activity")
        
        if data:
            try:
                return template.format(**data)
            except (KeyError, ValueError):
                # Fall back to unformatted template if data doesn't match
                return template
        
        return template
    
    @classmethod
    def get_description(cls, event_type: IntermediateEventType, data: Dict[str, Any] = None) -> str:
        """
        Get detailed description for an event.
        
        Args:
            event_type: The type of event
            data: Event data for template interpolation
            
        Returns:
            Detailed human-readable description
        """
        template = cls.DESCRIPTIONS.get(event_type, "Processing...")
        
        if data:
            try:
                return template.format(**data)
            except (KeyError, ValueError):
                # Fall back to basic description if data doesn't match
                if 'action' in data:
                    return f"Processing: {data['action']}"
                elif 'description' in data:
                    return data['description']
        
        return template
    
    @classmethod
    def get_icon(cls, category: EventCategory) -> str:
        """Get icon for event category."""
        return cls.CATEGORY_ICONS.get(category, "🤖")
    
    @classmethod
    def get_colors(cls, category: EventCategory) -> Dict[str, str]:
        """Get color scheme for event category."""
        return cls.CATEGORY_COLORS.get(category, {
            "bg": "bg-gray-50", 
            "text": "text-gray-700", 
            "border": "border-gray-200"
        })
    
    @classmethod
    def format_confidence(cls, confidence: Optional[float]) -> str:
        """Format confidence score for display."""
        if confidence is None:
            return "Unknown"
        return f"{confidence:.1%}"
    
    @classmethod
    def format_reasoning(cls, reasoning: Optional[str], max_length: int = 100) -> str:
        """Format reasoning text for display."""
        if not reasoning:
            return ""
        
        if len(reasoning) <= max_length:
            return reasoning
        
        return reasoning[:max_length-3] + "..."
    
    @classmethod
    def get_priority_from_event_type(cls, event_type: IntermediateEventType) -> int:
        """Get default priority for event type (higher = more important)."""
        priority_mapping = {
            # High priority - key decisions and findings
            IntermediateEventType.HYPOTHESIS_FORMED: 9,
            IntermediateEventType.PARTIAL_SYNTHESIS: 9,
            IntermediateEventType.CONTRADICTION_FOUND: 9,
            IntermediateEventType.PLAN_CREATED: 8,
            IntermediateEventType.GROUNDING_COMPLETE: 8,
            
            # Medium-high priority - important activities
            IntermediateEventType.SEARCH_STRATEGY: 7,
            IntermediateEventType.SOURCE_EVALUATION: 7,
            IntermediateEventType.REASONING_REFLECTION: 7,
            IntermediateEventType.CONFIDENCE_UPDATE: 6,
            IntermediateEventType.KNOWLEDGE_GAP_IDENTIFIED: 6,
            
            # Medium priority - regular activities
            IntermediateEventType.QUERY_GENERATED: 5,
            IntermediateEventType.SEARCH_RESULTS_FOUND: 5,
            IntermediateEventType.SOURCE_ANALYZED: 5,
            IntermediateEventType.AGENT_HANDOFF: 5,
            
            # Lower priority - routine operations
            IntermediateEventType.QUERY_EXECUTING: 3,
            IntermediateEventType.TOOL_CALL_START: 3,
            IntermediateEventType.SYNTHESIS_PROGRESS: 3,
            
            # Low priority - basic operations
            IntermediateEventType.AGENT_START: 2,
            IntermediateEventType.STAGE_TRANSITION: 2,
        }
        
        return priority_mapping.get(event_type, 4)  # Default medium priority