"""Typed schema for LLM-driven file extraction.

One universal extractor replaces per-format regex: the LLM reads the first
few thousand chars of any attached file (CRM export, call notes, deck
text, free-form markdown, spreadsheet, PDF) and returns a structured
``FileExtraction``. Callers upsert the fields into ``chat_memory_files``,
``chat_memory_entities``, and ``chat_memory_findings`` (as step-0
findings with ``origin=FILE``).

Design principles (user directive — see memory
``feedback_memory_is_chat_scoped.md`` and the "remove regex completely"
requirement):

- **Format-agnostic**: no filename-driven branching, no profile registry.
  The model is asked to *classify* the file itself via ``file_purpose``.
- **Typed**: Pydantic schema enforces structure; prompt-injection
  imperatives inside the file body cannot produce valid JSON that
  subverts the pipeline.
- **Bounded**: capped ``entities`` / ``key_facts`` counts so a
  pathological file can't generate a 10 k-entity list.
- **Confidence-aware**: every ``key_fact`` carries a confidence the LLM
  assigns; low-confidence facts are still useful context but downstream
  consumers know to defer to higher-confidence signals.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

EntityTypeLit = Literal[
    "account", "person", "product", "date", "competitor", "location", "other"
]
"""String values mirror ``deep_research.models.enums.EntityType``.
Kept as a Literal here so the LLM returns the same token set the DB
columns expect; the app-side enum is imported by the service to validate
round-trip."""

ConfidenceLit = Literal["high", "medium", "low"]

FactCategoryLit = Literal[
    "industry",
    "stage",
    "owner",
    "next_step",
    "opportunity",
    "timeline",
    "blocker",
    "history",
    "decision",
    "action_item",
    "attendee_role",
    "competitor_note",
    "other",
]
"""Structured categorisation that lets downstream projections (e.g.
``AccountBrief.stage``) find the right field without fuzzy matching.
Kept deliberately short; anything outside the list falls into
``other`` and still carries its ``content``."""


class ExtractedEntity(BaseModel):
    """One entity the LLM identified in the file."""

    name: str = Field(
        description="Canonical name as it should appear in the entity registry.",
    )
    entity_type: EntityTypeLit = Field(
        description=(
            "Which registry bucket this entity belongs to. Use 'account' for "
            "the customer/prospect company, 'person' for named individuals, "
            "'product' for SKUs, 'competitor' for competing vendors, "
            "'location' for geographies, and 'other' when none fit."
        )
    )
    role: str | None = Field(
        default=None,
        description=(
            "Optional role or title (e.g. 'Account Executive', 'VP Data "
            "Platform'). Useful when the entity is a person or a product "
            "variant."
        ),
    )
    aliases: list[str] = Field(
        default_factory=list,
        description=(
            "Alternative surface forms of the name as they appear in the "
            "file (e.g. short names, stock tickers). Max 5."
        ),
    )


class ExtractedFact(BaseModel):
    """One structured fact the LLM extracted from the file."""

    content: str = Field(
        description=(
            "Compact natural-language statement of the fact. "
            "Example: 'Stage: Technical Evaluation' or "
            "'Open opportunity: $2.3M ARR data platform modernisation'."
        )
    )
    category: FactCategoryLit = Field(
        description=(
            "High-level category so downstream projections (e.g. the "
            "sapresalesbot AccountBrief) can pick the right field. "
            "If nothing fits use 'other'."
        ),
    )
    confidence: ConfidenceLit = Field(
        default="medium",
        description=(
            "How certain the model is that the fact was stated explicitly "
            "in the file. 'high' for direct quotes, 'medium' for clear "
            "paraphrases, 'low' for inferences."
        ),
    )
    related_entity: str | None = Field(
        default=None,
        description=(
            "Name of a related entity (must match one of the extracted "
            "entities above). Use this to link a fact to its owner, e.g. "
            "'Stage: Technical Evaluation' related_entity='Sagacity Corp'."
        ),
    )


class FileExtraction(BaseModel):
    """Structured output the LLM produces per uploaded file."""

    file_purpose: str = Field(
        description=(
            "Short phrase describing what kind of file this is from a "
            "sales-research perspective. Examples: 'CRM account export', "
            "'call notes from discovery meeting', 'sales deck slides', "
            "'spreadsheet of open opportunities', 'free-form account "
            "history'. If unclear, say 'unclassified'."
        )
    )
    one_line_summary: str = Field(
        description=(
            "One-sentence summary agents will see in the attached-context "
            "appendix. Under 180 chars. Include the primary account name "
            "if identifiable."
        ),
        max_length=240,
    )
    entities: list[ExtractedEntity] = Field(
        default_factory=list,
        description="Structured entities found in the file. Max 20.",
        max_length=20,
    )
    key_facts: list[ExtractedFact] = Field(
        default_factory=list,
        description=(
            "Structured facts extracted from the file. Max 25. Focus on "
            "sales-relevant signals: account stage, owner, industry, open "
            "opportunities, next steps, blockers, competitive threats, "
            "decisions made in meetings, action items."
        ),
        max_length=25,
    )
    notes: str = Field(
        default="",
        description=(
            "Optional free-form note when the model wants to flag "
            "something (e.g. 'file appears truncated', 'file language is "
            "Spanish', 'contains PII — redacted email addresses'). Leave "
            "empty in the normal case."
        ),
    )
