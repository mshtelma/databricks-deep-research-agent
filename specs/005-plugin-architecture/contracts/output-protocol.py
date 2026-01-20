"""
Output Type Protocol Definitions
================================

This file defines the output type customization protocol for extending
the Deep Research Agent's synthesis output.

Plugins can define custom structured output types (e.g., MeetingPrepOutput)
instead of the default SynthesisReport.

Location: src/deep_research/plugins/output.py
"""

from typing import Protocol, Any, Type
from dataclasses import dataclass, field
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Default Output Type
# ---------------------------------------------------------------------------

class SynthesisReport(BaseModel):
    """
    Default output type for research reports.

    This is what the synthesizer produces when no custom output type
    is configured.
    """

    title: str = Field(..., description="Report title")

    executive_summary: str = Field(
        ...,
        description="2-3 sentence summary of findings"
    )

    sections: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Report sections with heading and content"
    )

    key_findings: list[str] = Field(
        default_factory=list,
        description="Bullet-point key findings"
    )

    sources: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Sources consulted during research"
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (timing, iterations, etc.)"
    )


# ---------------------------------------------------------------------------
# Synthesizer Configuration
# ---------------------------------------------------------------------------

@dataclass
class SynthesizerConfig:
    """
    Configuration for the synthesizer agent when producing a specific
    output type.
    """

    output_type: str
    """Identifier for the output type (e.g., 'synthesis_report', 'meeting_prep')."""

    model_tier: str = "complex"
    """Model tier to use for synthesis."""

    max_tokens: int = 8000
    """Maximum tokens for synthesis output."""

    temperature: float = 0.7
    """Temperature for synthesis generation."""

    system_prompt_addition: str | None = None
    """Additional instructions to append to synthesizer system prompt."""

    output_format: str = "json"
    """Output format: 'json' for structured, 'markdown' for prose."""

    validation_strict: bool = True
    """Whether to fail on output validation errors."""


# ---------------------------------------------------------------------------
# Output Type Provider Protocol
# ---------------------------------------------------------------------------

class OutputTypeProvider(Protocol):
    """
    Protocol for plugins that define custom output types.

    Implement this to produce domain-specific structured outputs
    instead of the default SynthesisReport.
    """

    def get_output_schema(self) -> Type[BaseModel]:
        """
        Return the Pydantic model for the custom output type.

        The model should:
        - Inherit from pydantic.BaseModel
        - Define all required fields with Field() annotations
        - Include a 'sources' field for citation tracking

        Returns:
            Pydantic model class (not an instance)

        Example:
            def get_output_schema(self) -> Type[BaseModel]:
                return MeetingPrepOutput
        """
        ...

    def get_synthesizer_config(
        self,
        ctx: "ResearchContext",
    ) -> SynthesizerConfig:
        """
        Return synthesizer configuration for this output type.

        Args:
            ctx: Research context for conditional configuration

        Returns:
            SynthesizerConfig with output-specific settings
        """
        ...

    def get_synthesizer_prompt(
        self,
        ctx: "ResearchContext",
    ) -> str | None:
        """
        Return custom synthesizer prompt for this output type.

        If None, the default synthesizer prompt is used with the
        output schema injected.

        Args:
            ctx: Research context for conditional prompting

        Returns:
            Custom prompt string, or None for default

        Example:
            def get_synthesizer_prompt(self, ctx):
                return '''
                Generate a meeting preparation document following
                the MEDDPICC framework. Include:
                - Executive summary
                - Key insights with sources
                - Discovery questions by category
                - Competitive positioning
                '''
        """
        ...


# ---------------------------------------------------------------------------
# Output Type Registry
# ---------------------------------------------------------------------------

class OutputTypeRegistry:
    """
    Registry for output types.

    Maps output type identifiers to their schemas and configurations.
    """

    def __init__(self) -> None:
        self._types: dict[str, Type[BaseModel]] = {
            "synthesis_report": SynthesisReport,
        }
        self._providers: dict[str, OutputTypeProvider] = {}

    def register(
        self,
        output_type: str,
        schema: Type[BaseModel],
        provider: OutputTypeProvider | None = None,
    ) -> None:
        """Register an output type with its schema."""
        self._types[output_type] = schema
        if provider:
            self._providers[output_type] = provider

    def get_schema(self, output_type: str) -> Type[BaseModel] | None:
        """Get the schema for an output type."""
        return self._types.get(output_type)

    def get_provider(self, output_type: str) -> OutputTypeProvider | None:
        """Get the provider for an output type."""
        return self._providers.get(output_type)

    def list_types(self) -> list[str]:
        """List all registered output types."""
        return list(self._types.keys())


# ---------------------------------------------------------------------------
# Example: MeetingPrepOutput (sapresalesbot pattern)
# ---------------------------------------------------------------------------

# This is an example of a custom output type for meeting preparation.
# In a real plugin, this would be defined in the plugin's codebase.

"""
from pydantic import BaseModel, Field
from typing import Any
from enum import Enum

class InsightSource(str, Enum):
    COMPANY_NEWS = "company_news"
    COMPETITIVE_INTEL = "competitive_intel"
    ATTENDEE_RESEARCH = "attendee_research"
    INDUSTRY_CONTEXT = "industry_context"
    SFDC_CONTEXT = "sfdc_context"

class KeyInsight(BaseModel):
    insight: str
    source: InsightSource
    relevance: str | None = None
    source_refs: list[str] = Field(default_factory=list)

class Question(BaseModel):
    text: str
    context: str | None = None
    target_persona: str | None = None
    context_source_refs: list[str] = Field(default_factory=list)

class DiscoveryQuestions(BaseModel):
    business: list[Question] = Field(default_factory=list)
    technical: list[Question] = Field(default_factory=list)
    qualification: list[Question] = Field(default_factory=list)
    competitive: list[Question] = Field(default_factory=list)

class MeetingPlan(BaseModel):
    point_of_view: str
    recommended_products: list[str] = Field(default_factory=list)
    key_differentiators: list[str]
    talking_points: list[str] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)

class CaseStudy(BaseModel):
    title: str
    url: str | None = None
    company: str | None = None
    industry: str | None = None
    summary: str
    relevance: str | None = None

class AttendeeBrief(BaseModel):
    name: str
    title: str
    linkedin_url: str | None = None
    background: str | None = None
    approach: str

class MeetingPrepOutput(BaseModel):
    '''Custom output type for meeting preparation (sapresalesbot pattern).'''

    id: str
    account_id: str
    account_name: str
    use_case_id: str | None = None
    use_case_name: str | None = None

    executive_summary: str
    executive_summary_source_refs: list[str] = Field(default_factory=list)

    meeting_plan: MeetingPlan
    key_insights: list[KeyInsight]
    discovery_questions: DiscoveryQuestions
    attendee_briefs: list[AttendeeBrief] = Field(default_factory=list)
    case_studies: list[CaseStudy] = Field(default_factory=list)
    competitive_positioning: str | None = None
    landmines: list[str] = Field(default_factory=list)

    sources: list[dict[str, Any]] = Field(default_factory=list)
    research_iterations: int | None = None
"""


# ---------------------------------------------------------------------------
# Example: MeetingPrepOutputProvider
# ---------------------------------------------------------------------------

"""
class MeetingPrepOutputProvider:
    '''OutputTypeProvider for meeting preparation.'''

    def get_output_schema(self) -> Type[BaseModel]:
        return MeetingPrepOutput

    def get_synthesizer_config(self, ctx: ResearchContext) -> SynthesizerConfig:
        return SynthesizerConfig(
            output_type="meeting_prep",
            model_tier="complex",
            max_tokens=12000,
            temperature=0.7,
            output_format="json",
        )

    def get_synthesizer_prompt(self, ctx: ResearchContext) -> str:
        return '''
        You are an expert pre-sales consultant preparing meeting materials.

        Generate a comprehensive meeting preparation document using the
        MEDDPICC framework:

        M - Metrics: What success metrics does the customer care about?
        E - Economic Buyer: Who controls the budget?
        D - Decision Criteria: What factors will drive the decision?
        D - Decision Process: What is the buying process?
        P - Paper Process: What procurement steps are needed?
        I - Identify Pain: What problems are they trying to solve?
        C - Champion: Who is advocating internally?
        C - Competition: Who else are they considering?

        Include:
        1. Executive summary (2-3 sentences)
        2. Meeting plan with point of view and differentiators
        3. Key insights organized by source type
        4. Discovery questions (~20) across 4 categories
        5. Attendee briefs with recommended approaches
        6. Relevant case studies (aim for 2-3)
        7. Competitive positioning guidance
        8. Potential landmines to avoid

        CRITICAL: Cite all sources using the provided source IDs.
        '''
"""
