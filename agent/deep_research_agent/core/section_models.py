"""Structured models for section-level research results."""

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from deep_research_agent.core import Citation
from deep_research_agent.core.observation_models import StructuredObservation


@dataclass(frozen=True)
class SectionResearchResult:
    """Immutable container for the outcome of a section-level research step."""

    synthesis: str
    observations: Tuple[StructuredObservation, ...] = field(default_factory=tuple)
    citations: Tuple[Citation, ...] = field(default_factory=tuple)
    search_results: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)
    extracted_data: Mapping[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the research result into JSON-safe primitives."""
        return {
            "synthesis": self.synthesis,
            "observations": [obs.to_dict() for obs in self.observations],
            "citations": [citation.to_dict() for citation in self.citations],
            "search_results": [dict(result) for result in self.search_results],
            "extracted_data": dict(self.extracted_data),
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SectionResearchResult":
        """Reconstruct a research result from a legacy dictionary payload."""
        # Handle observations with proper type checking
        raw_observations = payload.get("observations", [])
        observations = []
        
        for item in raw_observations:
            if isinstance(item, StructuredObservation):
                observations.append(item)
            elif isinstance(item, dict):
                # Only call from_dict if it's actually a dict
                try:
                    observations.append(StructuredObservation.from_dict(item))
                except Exception:
                    # If dict parsing fails, convert as string
                    observations.append(StructuredObservation.from_string(str(item)))
            elif isinstance(item, str):
                # Handle string observations
                observations.append(StructuredObservation.from_string(item))
            elif item is not None:
                # Handle any other non-None type
                observations.append(StructuredObservation.from_string(str(item)))
        
        observations = tuple(observations)

        citations = tuple(
            citation
            if isinstance(citation, Citation)
            else Citation(
                source=citation.get("source", ""),
                url=citation.get("url"),
                title=citation.get("title"),
                snippet=citation.get("snippet"),
                relevance_score=citation.get("relevance_score", 0.0),
            )
            for citation in payload.get("citations", [])
        )

        search_results = tuple(
            result if isinstance(result, dict) else getattr(result, "to_dict", lambda: {} )()
            for result in payload.get("search_results", [])
        )

        metadata = payload.get("metadata", {})
        if isinstance(metadata, dict):
            metadata_value = metadata
        else:
            metadata_value = dict(metadata)

        return cls(
            synthesis=payload.get("synthesis", ""),
            observations=observations,
            citations=citations,
            search_results=search_results,
            extracted_data=payload.get("extracted_data", {}),
            confidence=payload.get("confidence", 0.0),
            metadata=metadata_value,
        )


def replace_section_research_result(
    result: SectionResearchResult,
    *,
    synthesis: Optional[str] = None,
    observations: Optional[Iterable[StructuredObservation]] = None,
    citations: Optional[Iterable[Citation]] = None,
    search_results: Optional[Iterable[Mapping[str, Any]]] = None,
    extracted_data: Optional[Mapping[str, Any]] = None,
    confidence: Optional[float] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> SectionResearchResult:
    """Return a new ``SectionResearchResult`` with updated fields.

    This helper keeps the dataclass immutable while allowing callers to
    amend individual fields without mutating the original instance.
    """

    return replace(
        result,
        synthesis=result.synthesis if synthesis is None else synthesis,
        observations=result.observations if observations is None else tuple(observations),
        citations=result.citations if citations is None else tuple(citations),
        search_results=result.search_results if search_results is None else tuple(dict(r) for r in search_results),
        extracted_data=result.extracted_data if extracted_data is None else dict(extracted_data),
        confidence=result.confidence if confidence is None else confidence,
        metadata=result.metadata if metadata is None else dict(metadata),
    )
