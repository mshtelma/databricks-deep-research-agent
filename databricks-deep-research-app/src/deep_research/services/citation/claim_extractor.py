"""Extract verifiable claims from structured output.

Auto-discovers verifiable fields by finding text+source_refs pairs.
No hardcoded schema knowledge - works with ANY Pydantic model that follows
the convention of pairing text fields with source_refs fields.

Discovery patterns:
1. field + field_source_refs (e.g., executive_summary + executive_summary_source_refs)
2. Object with text field + source_refs sibling (e.g., {insight: "...", source_refs: [...]})
3. Nested objects and arrays are walked recursively
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from deep_research.core.logging_utils import get_logger

logger = get_logger(__name__)

# Minimum text length to consider for verification (skip short fields)
MIN_TEXT_LENGTH = 50


@dataclass
class ExtractedClaim:
    """A claim extracted from structured output with its source references."""

    text: str
    source_refs: list[str]  # Source indices like ["1", "2"]
    field_path: str  # JSON path for applying corrections (e.g., "key_insights[0].insight")
    priority: str = "medium"  # "high", "medium", "low" based on field importance
    char_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.char_count = len(self.text)


class StructuredClaimExtractor:
    """Extract verifiable claims from ANY Pydantic structured output.

    Auto-discovers text+source_refs pairs by walking the model.
    No schema-specific knowledge required.
    """

    def __init__(self, min_text_length: int = MIN_TEXT_LENGTH) -> None:
        """Initialize extractor.

        Args:
            min_text_length: Minimum characters for a text field to be verified.
                             Shorter fields are skipped as they're likely labels/titles.
        """
        self._min_length = min_text_length

    def extract(self, output: BaseModel) -> list[ExtractedClaim]:
        """Extract all verifiable claims from structured output.

        Walks the entire model, finding text fields that have associated
        source_refs fields nearby. Works with any schema.

        Args:
            output: Any Pydantic model instance.

        Returns:
            List of ExtractedClaim objects for verification.
        """
        data = output.model_dump()
        claims: list[ExtractedClaim] = []

        self._walk_and_extract(data, "", claims)

        logger.info(
            "CLAIMS_EXTRACTED",
            total=len(claims),
            avg_length=int(sum(c.char_count for c in claims) / len(claims)) if claims else 0,
        )

        return claims

    def _walk_and_extract(
        self,
        obj: Any,
        path: str,
        claims: list[ExtractedClaim],
    ) -> None:
        """Recursively walk object to find text+source_refs pairs."""

        if isinstance(obj, dict):
            # First pass: find source_refs fields in this dict
            source_refs_fields = {
                k: v for k, v in obj.items()
                if k.endswith("_source_refs") or k == "source_refs"
            }

            # Second pass: find text fields and match with source_refs
            for field_name, value in obj.items():
                field_path = f"{path}.{field_name}" if path else field_name

                # Skip source_refs fields themselves
                if field_name.endswith("_source_refs") or field_name == "source_refs":
                    continue

                # Check if this is a verifiable text field
                if isinstance(value, str) and len(value) >= self._min_length:
                    refs = self._find_source_refs_for_field(field_name, source_refs_fields)
                    if refs is not None:  # Only extract if we found source_refs
                        claims.append(ExtractedClaim(
                            text=value,
                            source_refs=[str(r) for r in refs] if refs else [],
                            field_path=field_path,
                            priority=self._infer_priority(field_name, len(value)),
                        ))

                # Check if this is a list of strings (like key_differentiators)
                elif isinstance(value, list) and value and all(isinstance(v, str) for v in value):
                    combined = " | ".join(v for v in value if v)
                    if len(combined) >= self._min_length:
                        refs = self._find_source_refs_for_field(field_name, source_refs_fields)
                        if refs is not None:
                            claims.append(ExtractedClaim(
                                text=combined,
                                source_refs=[str(r) for r in refs] if refs else [],
                                field_path=field_path,
                                priority="medium",
                            ))

                # Recurse into nested objects
                elif isinstance(value, dict):
                    self._walk_and_extract(value, field_path, claims)

                # Recurse into arrays
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            self._walk_and_extract(item, f"{field_path}[{i}]", claims)

        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                item_path = f"{path}[{i}]"
                self._walk_and_extract(item, item_path, claims)

    def _find_source_refs_for_field(
        self,
        field_name: str,
        source_refs_fields: dict[str, Any],
    ) -> list[str] | None:
        """Find source_refs for a given field.

        Checks for:
        1. field_source_refs (e.g., executive_summary -> executive_summary_source_refs)
        2. source_refs (generic, used when object has one source_refs for all fields)

        Returns None if no source_refs found (field shouldn't be verified).
        Returns [] if source_refs exists but is empty (field should be verified).
        """
        # Pattern 1: field_source_refs
        specific_refs = f"{field_name}_source_refs"
        if specific_refs in source_refs_fields:
            refs = source_refs_fields[specific_refs]
            return refs if isinstance(refs, list) else []

        # Pattern 2: generic source_refs in same object
        if "source_refs" in source_refs_fields:
            refs = source_refs_fields["source_refs"]
            return refs if isinstance(refs, list) else []

        # No source_refs found - don't verify this field
        return None

    def _infer_priority(self, field_name: str, text_length: int) -> str:
        """Infer priority based on field name and length.

        High priority: summary, overview, key findings
        Low priority: context, notes, short fields
        Medium: everything else
        """
        name_lower = field_name.lower()

        # High priority indicators
        if any(kw in name_lower for kw in ["summary", "overview", "insight", "point_of_view", "positioning"]):
            return "high"

        # Low priority indicators
        if any(kw in name_lower for kw in ["context", "note", "comment"]):
            return "low"

        # Length-based: longer text = higher priority
        if text_length > 500:
            return "high"
        if text_length < 150:
            return "low"

        return "medium"
