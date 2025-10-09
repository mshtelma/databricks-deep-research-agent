"""
Pydantic models for structured synthesis output.

Using Databricks structured output capabilities for guaranteed valid responses.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional


class FactExtraction(BaseModel):
    """
    Structured output for extracting facts from a single search result.

    Simpler than ResearchSynthesis - just extracts facts from one source.
    """
    facts: List[str] = Field(
        description="List of specific factual statements extracted from the source. Each fact should be complete, substantial, and self-contained."
    )

    @validator('facts')
    def validate_facts_quality(cls, v):
        """Allow empty facts - caller will use full_content as fallback."""
        if not v:
            return []  # Empty is OK - upstream has fallback strategy

        # Filter out meta-descriptions and too-short facts
        valid_facts = []
        for fact in v:
            # Skip if too short
            if len(fact) < 30:
                continue

            # Skip meta-descriptions
            meta_phrases = [
                "information about", "details on", "provides information",
                "contains details", "describes", "overview of"
            ]
            fact_lower = fact.lower()
            if any(phrase in fact_lower for phrase in meta_phrases):
                continue

            valid_facts.append(fact)

        # Return what we have (can be empty) - no fake fallbacks
        return valid_facts

    @validator('facts', each_item=True)
    def validate_individual_fact(cls, v):
        """Validate each fact has substance."""
        # Ensure reasonable length
        if len(v) < 20:
            raise ValueError(f"Fact too short: {v}")

        # Ensure it's not just punctuation or numbers
        if not any(c.isalpha() for c in v):
            raise ValueError(f"Fact lacks text content: {v}")

        return v


class QualityMetrics(BaseModel):
    """Quality metrics for research synthesis."""
    completeness: float = Field(
        default=0.8,
        description="How complete the extraction is (0.0 to 1.0)"
    )
    data_points_extracted: int = Field(
        default=0,
        description="Number of data points found"
    )
    reliability: float = Field(
        default=0.8,
        description="Reliability of sources (0.0 to 1.0)"
    )
    coverage: float = Field(
        default=0.7,
        description="Topic coverage (0.0 to 1.0)"
    )


class ResearchCitation(BaseModel):
    """Citation for a research source."""
    title: str = Field(description="Title of the source")
    url: str = Field(default="", description="URL of the source")
    relevance_score: float = Field(
        default=0.8,
        description="Relevance score (0.0 to 1.0)"
    )
    snippet: Optional[str] = Field(
        default=None,
        description="Key excerpt from source"
    )


class ResearchSynthesis(BaseModel):
    """
    Structured output for research synthesis.

    This model ensures we always get valid, substantive observations
    and synthesis from the LLM using Databricks structured output.
    """

    observations: List[str] = Field(
        # FIXED: Removed min_items=1 as Databricks doesn't support minItems in JSON schemas
        description="List of specific, factual observations extracted from search results. Each should be a complete, informative statement."
    )

    synthesis: str = Field(
        # FIXED: Removed min_length=50 - Databricks doesn't support minLength in JSON schemas
        # Validation handled by @validator below
        description="Comprehensive narrative synthesis that connects and contextualizes the observations into a coherent analysis"
    )

    extracted_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured data organized by logical categories (e.g., statistics, dates, entities, comparisons)"
    )

    citations: List[ResearchCitation] = Field(
        default_factory=list,
        description="Source citations for the information extracted"
    )

    quality_metrics: Optional[QualityMetrics] = Field(
        default=None,
        description="Quality assessment of the synthesis"
    )

    @validator('observations')
    def validate_observations_quality(cls, v):
        """
        Ensure observations contain actual research content, not meta-commentary.
        """
        if not v:
            raise ValueError("At least one observation is required")

        # Meta-commentary indicators that suggest LLM confusion
        meta_indicators = [
            "system says", "must output", "json only", "cannot fulfill",
            "as an ai", "i cannot", "format required", "conflict",
            "instructions say", "been asked to", "i must follow",
            "output format", "json format", "formatting", "i should"
        ]

        valid_observations = []
        for obs in v:
            obs_lower = obs.lower()

            # Skip meta-commentary
            if any(indicator in obs_lower for indicator in meta_indicators):
                continue

            # Ensure minimum quality
            if len(obs) < 20:
                continue

            # Keep substantive observations
            valid_observations.append(obs)

        if not valid_observations:
            # If all observations were filtered, create a minimal valid one
            # This should rarely happen with structured output
            valid_observations = ["Research data extracted from provided sources"]

        return valid_observations

    @validator('observations', each_item=True)
    def validate_individual_observation(cls, v):
        """Validate each observation has substance."""
        # Ensure reasonable length
        if len(v) < 15:
            raise ValueError(f"Observation too short: {v}")

        # Ensure it's not just punctuation or numbers
        if not any(c.isalpha() for c in v):
            raise ValueError(f"Observation lacks text content: {v}")

        return v

    @validator('observations', each_item=True)
    def reject_meta_descriptions(cls, v):
        """Reject vague meta-descriptions instead of concrete facts."""
        # Reject meta-descriptions
        meta_phrases = [
            "detailed description of", "information about", "data on",
            "contains information", "provides details", "has a system",
            "includes data about", "covers the topic of", "overview of"
        ]

        v_lower = v.lower()
        if any(phrase in v_lower for phrase in meta_phrases):
            raise ValueError(f"Observation is meta-description, not a fact: {v[:100]}")

        # Require concrete content (numbers OR specific verbs OR specific entities)
        has_number = any(c.isdigit() for c in v)
        specific_verbs = ['is', 'are', 'was', 'were', 'has', 'have', 'had',
                         'increased', 'decreased', 'equals', 'ranges', 'contains',
                         'features', 'requires', 'supports', 'measures']
        has_specific_verb = any(verb in v_lower.split() for verb in specific_verbs)

        if not (has_number or has_specific_verb):
            raise ValueError(f"Observation lacks concrete facts (no numbers or specific verbs): {v[:100]}")

        return v

    @validator('synthesis')
    def validate_synthesis_content(cls, v):
        """Ensure synthesis contains actual research content."""
        if len(v) < 50:
            raise ValueError("Synthesis too short to be meaningful")

        # Check for meta-commentary in synthesis
        meta_phrases = [
            "system says", "output json", "format conflict",
            "cannot provide", "unable to", "as an ai model",
            "json only", "must output"
        ]

        v_lower = v.lower()
        if any(phrase in v_lower for phrase in meta_phrases):
            raise ValueError("Synthesis contains meta-commentary instead of research content")

        # Ensure it has some substance (contains both letters and spaces)
        if v.count(' ') < 5:
            raise ValueError("Synthesis lacks proper sentence structure")

        return v

    class Config:
        """Pydantic configuration."""
        # Allow extra fields for forward compatibility
        extra = "allow"
        # Provide clear error messages
        str_strip_whitespace = True
        # Example for documentation
        json_schema_extra = {
            "example": {
                "observations": [
                    "The 2024 tax rate for high earners increased to 37%",
                    "Standard deduction rose to $14,600 for single filers",
                    "401(k) contribution limit increased to $23,000"
                ],
                "synthesis": "The 2024 tax changes reflect adjustments for inflation, with the standard deduction increasing by 5.4% to $14,600 for single filers. The seven tax brackets remain but have been adjusted upward, with the top rate of 37% applying to incomes over $609,350.",
                "extracted_data": {
                    "tax_rates": {
                        "top_rate": 0.37,
                        "brackets": 7
                    },
                    "deductions": {
                        "standard_single": 14600,
                        "standard_married": 29200
                    },
                    "retirement": {
                        "401k_limit": 23000,
                        "catch_up": 7500
                    }
                },
                "citations": [
                    {
                        "title": "IRS 2024 Tax Tables",
                        "url": "https://irs.gov/tables",
                        "relevance_score": 0.95
                    }
                ],
                "quality_metrics": {
                    "completeness": 0.85,
                    "data_points_extracted": 8,
                    "reliability": 0.9,
                    "coverage": 0.75
                }
            }
        }