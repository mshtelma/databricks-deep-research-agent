"""
Grounding and factuality checking framework for research outputs.

Implements multi-layer verification to prevent hallucinations and ensure
all claims are properly grounded in source material.
"""

import re
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import hashlib
from datetime import datetime

from pydantic import BaseModel, Field
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from deep_research_agent.core import get_logger, SearchResult


logger = get_logger(__name__)


class ClaimType(str, Enum):
    """Types of claims that can be extracted from text."""
    FACTUAL = "factual"
    STATISTICAL = "statistical"
    OPINION = "opinion"
    PREDICTION = "prediction"
    DEFINITION = "definition"
    QUOTATION = "quotation"


class GroundingStatus(str, Enum):
    """Status of grounding for a claim."""
    GROUNDED = "grounded"
    PARTIALLY_GROUNDED = "partially_grounded"
    UNGROUNDED = "ungrounded"
    CONTRADICTED = "contradicted"
    UNCERTAIN = "uncertain"


class VerificationLevel(str, Enum):
    """Levels of verification strictness."""
    STRICT = "strict"  # Reject any ungrounded claims
    MODERATE = "moderate"  # Flag ungrounded claims but allow
    LENIENT = "lenient"  # Minimal checking


@dataclass
class Claim:
    """Represents an extracted claim from text."""
    text: str
    claim_type: ClaimType
    start_position: int
    end_position: int
    entities: List[str] = None
    numbers: List[str] = None
    dates: List[str] = None
    confidence: float = 1.0
    
    def __hash__(self):
        """Make claim hashable for deduplication."""
        return hash(self.text)


@dataclass
class GroundingResult:
    """Result of grounding a single claim."""
    claim: Claim
    status: GroundingStatus
    supporting_sources: List[Tuple[SearchResult, float]]  # (source, relevance_score)
    contradicting_sources: List[Tuple[SearchResult, float]]
    confidence_score: float
    explanation: str
    suggested_revision: Optional[str] = None


@dataclass
class Contradiction:
    """Represents a contradiction between sources."""
    claim: str
    source1: SearchResult
    source2: SearchResult
    severity: float  # 0-1, how severe the contradiction is
    explanation: str
    resolution_strategy: str


class FactualityReport(BaseModel):
    """Complete factuality assessment report."""
    total_claims: int = Field(description="Total number of claims extracted")
    grounded_claims: int = Field(description="Number of fully grounded claims")
    partially_grounded_claims: int = Field(description="Number of partially grounded claims")
    ungrounded_claims: int = Field(description="Number of ungrounded claims")
    contradicted_claims: int = Field(description="Number of contradicted claims")
    
    overall_factuality_score: float = Field(description="Overall factuality score (0-1)")
    confidence_score: float = Field(description="Confidence in the assessment (0-1)")
    
    grounding_results: List[GroundingResult] = Field(default_factory=list)
    contradictions: List[Contradiction] = Field(default_factory=list)
    
    hallucination_risks: List[str] = Field(default_factory=list, description="Identified hallucination risks")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations for improvement")
    
    verification_timestamp: datetime = Field(default_factory=datetime.now)
    verification_level: VerificationLevel = Field(default=VerificationLevel.MODERATE)


class ClaimExtractor:
    """Extracts claims from text for verification."""
    
    # Patterns for different claim types
    FACTUAL_PATTERNS = [
        r"(?:is|are|was|were|will be|has been|have been)\s+[\w\s]+",
        r"(?:according to|research shows|studies indicate|data suggests)",
        r"(?:proved|proven|demonstrated|confirmed|established)"
    ]
    
    STATISTICAL_PATTERNS = [
        r"\d+(?:\.\d+)?%",
        r"\d+(?:,\d{3})*(?:\.\d+)?(?:\s+(?:million|billion|thousand))?",
        r"(?:increase|decrease|growth|decline).*?\d+",
        r"(?:majority|minority|half|quarter|third)"
    ]
    
    QUOTE_PATTERNS = [
        r'"[^"]+?"',
        r"'[^']+?'",
        r"(?:said|stated|claimed|argued|noted|wrote)"
    ]
    
    def extract_claims(self, text: str) -> List[Claim]:
        """Extract all claims from text."""
        claims = []
        sentences = self._split_into_sentences(text)
        
        for sentence in sentences:
            sentence_claims = self._extract_from_sentence(sentence, text)
            claims.extend(sentence_claims)
        
        # Deduplicate claims
        unique_claims = list(set(claims))
        logger.info(f"Extracted {len(unique_claims)} unique claims from text")
        
        return unique_claims
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitter - could be enhanced with spaCy or NLTK
        sentences = re.split(r'[.!?]\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _extract_from_sentence(self, sentence: str, full_text: str) -> List[Claim]:
        """Extract claims from a single sentence."""
        claims = []
        start_pos = full_text.find(sentence)
        
        # Check for factual claims
        if any(re.search(pattern, sentence, re.IGNORECASE) for pattern in self.FACTUAL_PATTERNS):
            claims.append(Claim(
                text=sentence,
                claim_type=ClaimType.FACTUAL,
                start_position=start_pos,
                end_position=start_pos + len(sentence),
                entities=self._extract_entities(sentence),
                confidence=0.9
            ))
        
        # Check for statistical claims
        stats = []
        for pattern in self.STATISTICAL_PATTERNS:
            matches = re.findall(pattern, sentence)
            if matches:
                stats.extend(matches)
        
        if stats:
            claims.append(Claim(
                text=sentence,
                claim_type=ClaimType.STATISTICAL,
                start_position=start_pos,
                end_position=start_pos + len(sentence),
                numbers=stats,
                confidence=0.95
            ))
        
        # Check for quotations
        quotes = re.findall(r'"([^"]+)"', sentence)
        for quote in quotes:
            quote_start = full_text.find(quote)
            claims.append(Claim(
                text=quote,
                claim_type=ClaimType.QUOTATION,
                start_position=quote_start,
                end_position=quote_start + len(quote),
                confidence=1.0
            ))
        
        return claims
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text."""
        # Simple capitalized word extraction - could be enhanced with NER
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        return entities


class GroundingChecker:
    """Verifies claims against source material."""
    
    def __init__(self, verification_level: VerificationLevel = VerificationLevel.MODERATE):
        self.verification_level = verification_level
        self.claim_extractor = ClaimExtractor()
    
    def verify_content(
        self, 
        content: str, 
        sources: List[SearchResult],
        check_contradictions: bool = True
    ) -> FactualityReport:
        """
        Verify all claims in content against sources.
        
        Args:
            content: Text content to verify
            sources: Source documents to verify against
            check_contradictions: Whether to check for contradictions
            
        Returns:
            Complete factuality report
        """
        logger.info("Starting content verification")
        
        # Extract claims
        claims = self.claim_extractor.extract_claims(content)
        
        # Verify each claim
        grounding_results = []
        for claim in claims:
            result = self.verify_claim(claim, sources)
            grounding_results.append(result)
        
        # Check for contradictions if requested
        contradictions = []
        if check_contradictions:
            contradictions = self.detect_contradictions(sources)
        
        # Generate report
        report = self._generate_report(grounding_results, contradictions)
        
        # Add recommendations based on verification level
        report.recommendations = self._generate_recommendations(report)
        
        return report
    
    def verify_claim(self, claim: Claim, sources: List[SearchResult]) -> GroundingResult:
        """
        Verify a single claim against sources.
        
        Args:
            claim: Claim to verify
            sources: Source documents
            
        Returns:
            Grounding result for the claim
        """
        supporting_sources = []
        contradicting_sources = []
        
        for source in sources:
            relevance_score = self._calculate_relevance(claim.text, source.content)
            
            if relevance_score > 0.7:
                # Check if source supports or contradicts
                if self._supports_claim(claim, source):
                    supporting_sources.append((source, relevance_score))
                elif self._contradicts_claim(claim, source):
                    contradicting_sources.append((source, relevance_score))
        
        # Determine grounding status
        status = self._determine_status(
            supporting_sources, 
            contradicting_sources,
            claim.claim_type
        )
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            supporting_sources,
            contradicting_sources,
            claim.claim_type
        )
        
        # Generate explanation
        explanation = self._generate_explanation(
            status,
            supporting_sources,
            contradicting_sources
        )
        
        # Suggest revision if needed
        suggested_revision = None
        if status in [GroundingStatus.UNGROUNDED, GroundingStatus.CONTRADICTED]:
            suggested_revision = self._suggest_revision(claim, supporting_sources)
        
        return GroundingResult(
            claim=claim,
            status=status,
            supporting_sources=supporting_sources,
            contradicting_sources=contradicting_sources,
            confidence_score=confidence,
            explanation=explanation,
            suggested_revision=suggested_revision
        )
    
    def detect_contradictions(self, sources: List[SearchResult]) -> List[Contradiction]:
        """Detect contradictions between sources."""
        contradictions = []
        
        # Compare each pair of sources
        for i, source1 in enumerate(sources):
            for source2 in sources[i+1:]:
                conflicts = self._find_conflicts(source1, source2)
                contradictions.extend(conflicts)
        
        return contradictions
    
    def _calculate_relevance(self, claim_text: str, source_content: str) -> float:
        """Calculate relevance between claim and source content."""
        # Simple keyword overlap - could be enhanced with embeddings
        claim_words = set(claim_text.lower().split())
        source_words = set(source_content.lower().split())
        
        if not claim_words:
            return 0.0
        
        overlap = len(claim_words.intersection(source_words))
        relevance = overlap / len(claim_words)
        
        return min(relevance, 1.0)
    
    def _supports_claim(self, claim: Claim, source: SearchResult) -> bool:
        """Check if source supports the claim."""
        # Look for claim text or similar content in source
        claim_lower = claim.text.lower()
        source_lower = source.content.lower()
        
        # Check for direct mention
        if claim_lower in source_lower:
            return True
        
        # Check for key entities and numbers
        if claim.entities:
            entity_matches = sum(1 for e in claim.entities if e.lower() in source_lower)
            if entity_matches / len(claim.entities) > 0.5:
                return True
        
        if claim.numbers:
            number_matches = sum(1 for n in claim.numbers if n in source.content)
            if number_matches / len(claim.numbers) > 0.5:
                return True
        
        return False
    
    def _contradicts_claim(self, claim: Claim, source: SearchResult) -> bool:
        """Check if source contradicts the claim."""
        # Look for contradiction indicators
        contradiction_phrases = [
            "however", "contrary to", "actually", "in fact",
            "not true", "false", "incorrect", "disputed"
        ]
        
        source_lower = source.content.lower()
        claim_lower = claim.text.lower()
        
        # Check if source mentions claim context with contradiction
        for phrase in contradiction_phrases:
            if phrase in source_lower and any(
                word in source_lower for word in claim_lower.split()[:3]
            ):
                return True
        
        return False
    
    def _determine_status(
        self,
        supporting_sources: List[Tuple[SearchResult, float]],
        contradicting_sources: List[Tuple[SearchResult, float]],
        claim_type: ClaimType
    ) -> GroundingStatus:
        """Determine grounding status based on sources."""
        if contradicting_sources and len(contradicting_sources) >= len(supporting_sources):
            return GroundingStatus.CONTRADICTED
        elif supporting_sources:
            if len(supporting_sources) >= 2 or (
                supporting_sources and supporting_sources[0][1] > 0.9
            ):
                return GroundingStatus.GROUNDED
            else:
                return GroundingStatus.PARTIALLY_GROUNDED
        elif claim_type == ClaimType.OPINION:
            return GroundingStatus.PARTIALLY_GROUNDED
        else:
            return GroundingStatus.UNGROUNDED
    
    def _calculate_confidence(
        self,
        supporting_sources: List[Tuple[SearchResult, float]],
        contradicting_sources: List[Tuple[SearchResult, float]],
        claim_type: ClaimType
    ) -> float:
        """Calculate confidence in grounding assessment."""
        if not supporting_sources and not contradicting_sources:
            return 0.1
        
        # Base confidence on source quality and agreement
        support_score = sum(s[1] for s in supporting_sources) if supporting_sources else 0
        contradict_score = sum(s[1] for s in contradicting_sources) if contradicting_sources else 0
        
        total_score = support_score + contradict_score
        if total_score == 0:
            return 0.1
        
        confidence = support_score / total_score
        
        # Adjust for claim type
        if claim_type == ClaimType.STATISTICAL:
            confidence *= 0.9  # Stats need exact matches
        elif claim_type == ClaimType.OPINION:
            confidence *= 1.1  # Opinions are less strict
        
        return min(confidence, 1.0)
    
    def _generate_explanation(
        self,
        status: GroundingStatus,
        supporting_sources: List[Tuple[SearchResult, float]],
        contradicting_sources: List[Tuple[SearchResult, float]]
    ) -> str:
        """Generate explanation for grounding result."""
        if status == GroundingStatus.GROUNDED:
            return f"Claim is well-grounded with {len(supporting_sources)} supporting source(s)"
        elif status == GroundingStatus.PARTIALLY_GROUNDED:
            return f"Claim has limited support from {len(supporting_sources)} source(s)"
        elif status == GroundingStatus.UNGROUNDED:
            return "No supporting evidence found in provided sources"
        elif status == GroundingStatus.CONTRADICTED:
            return f"Claim is contradicted by {len(contradicting_sources)} source(s)"
        else:
            return "Uncertain grounding status"
    
    def _suggest_revision(
        self,
        claim: Claim,
        supporting_sources: List[Tuple[SearchResult, float]]
    ) -> Optional[str]:
        """Suggest revision for ungrounded or contradicted claim."""
        if claim.claim_type == ClaimType.STATISTICAL:
            return "Consider verifying numerical data or adding 'approximately' qualifier"
        elif claim.claim_type == ClaimType.FACTUAL:
            return "Add qualifying language like 'may', 'possibly', or 'according to some sources'"
        elif supporting_sources:
            source = supporting_sources[0][0]
            return f"Revise to align with source: '{source.title}'"
        else:
            return "Consider removing or finding supporting sources for this claim"
    
    def _find_conflicts(
        self,
        source1: SearchResult,
        source2: SearchResult
    ) -> List[Contradiction]:
        """Find conflicts between two sources."""
        contradictions = []
        
        # Extract comparable statements from both sources
        # This is a simplified version - could be enhanced with NLP
        numbers1 = re.findall(r'\d+(?:\.\d+)?%?', source1.content)
        numbers2 = re.findall(r'\d+(?:\.\d+)?%?', source2.content)
        
        # Check for conflicting numbers about same topic
        if numbers1 and numbers2 and numbers1 != numbers2:
            # Check if sources discuss same topic
            common_words = set(source1.content.lower().split()).intersection(
                set(source2.content.lower().split())
            )
            if len(common_words) > 10:  # Arbitrary threshold
                contradictions.append(Contradiction(
                    claim="Conflicting numerical data",
                    source1=source1,
                    source2=source2,
                    severity=0.7,
                    explanation=f"Sources provide different numbers: {numbers1[:3]} vs {numbers2[:3]}",
                    resolution_strategy="Prefer more recent or authoritative source"
                ))
        
        return contradictions
    
    def _generate_report(
        self,
        grounding_results: List[GroundingResult],
        contradictions: List[Contradiction]
    ) -> FactualityReport:
        """Generate complete factuality report."""
        total_claims = len(grounding_results)
        grounded = sum(1 for r in grounding_results if r.status == GroundingStatus.GROUNDED)
        partially = sum(1 for r in grounding_results if r.status == GroundingStatus.PARTIALLY_GROUNDED)
        ungrounded = sum(1 for r in grounding_results if r.status == GroundingStatus.UNGROUNDED)
        contradicted = sum(1 for r in grounding_results if r.status == GroundingStatus.CONTRADICTED)
        
        # Calculate overall score
        if total_claims == 0:
            overall_score = 0.0
        else:
            overall_score = (grounded + 0.5 * partially) / total_claims
        
        # Calculate confidence
        confidences = [r.confidence_score for r in grounding_results]
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        # Identify hallucination risks
        hallucination_risks = []
        for result in grounding_results:
            if result.status == GroundingStatus.UNGROUNDED:
                if result.claim.claim_type == ClaimType.STATISTICAL:
                    hallucination_risks.append(f"Unverified statistics: {result.claim.text[:50]}...")
                elif result.claim.claim_type == ClaimType.FACTUAL:
                    hallucination_risks.append(f"Unsupported fact: {result.claim.text[:50]}...")
        
        return FactualityReport(
            total_claims=total_claims,
            grounded_claims=grounded,
            partially_grounded_claims=partially,
            ungrounded_claims=ungrounded,
            contradicted_claims=contradicted,
            overall_factuality_score=overall_score,
            confidence_score=avg_confidence,
            grounding_results=grounding_results,
            contradictions=contradictions,
            hallucination_risks=hallucination_risks,
            verification_level=self.verification_level
        )
    
    def _generate_recommendations(self, report: FactualityReport) -> List[str]:
        """Generate recommendations based on report."""
        recommendations = []
        
        if report.ungrounded_claims > 0:
            recommendations.append(
                f"Review and source {report.ungrounded_claims} ungrounded claim(s)"
            )
        
        if report.contradicted_claims > 0:
            recommendations.append(
                f"Resolve {report.contradicted_claims} contradicted claim(s)"
            )
        
        if report.overall_factuality_score < 0.8:
            recommendations.append(
                "Consider additional research to improve factual grounding"
            )
        
        if report.contradictions:
            recommendations.append(
                "Address source contradictions by preferring authoritative sources"
            )
        
        if self.verification_level == VerificationLevel.STRICT and report.ungrounded_claims > 0:
            recommendations.append(
                "STRICT MODE: Remove all ungrounded claims before publication"
            )
        
        return recommendations


class HallucinationPrevention:
    """Utilities for preventing hallucinations in generated content."""
    
    @staticmethod
    def generate_revision_suggestions(ungrounded_claims: List[GroundingResult]) -> List[str]:
        """Generate suggestions for revising ungrounded claims."""
        suggestions = []
        
        for result in ungrounded_claims:
            claim = result.claim
            if result.status == GroundingStatus.UNGROUNDED:
                if claim.claim_type == ClaimType.STATISTICAL:
                    suggestions.append(f"Add source for statistical claim: '{claim.text[:50]}...'")
                elif claim.claim_type == ClaimType.FACTUAL:
                    suggestions.append(f"Verify or qualify factual claim: '{claim.text[:50]}...'")
                elif claim.claim_type == ClaimType.PREDICTION:
                    suggestions.append(f"Add qualifier to prediction: '{claim.text[:50]}...'")
                else:
                    suggestions.append(f"Review and support claim: '{claim.text[:50]}...'")
            elif result.status == GroundingStatus.CONTRADICTED:
                suggestions.append(f"Resolve contradiction in: '{claim.text[:50]}...'")
            
            # Add specific suggestion if available
            if result.suggested_revision:
                suggestions.append(result.suggested_revision)
        
        return suggestions
    
    @staticmethod
    def add_grounding_markers(content: str, grounding_results: List[GroundingResult]) -> str:
        """Add inline markers showing grounding status of claims."""
        marked_content = content
        
        # Sort by position to maintain order
        sorted_results = sorted(
            grounding_results,
            key=lambda r: r.claim.start_position,
            reverse=True
        )
        
        for result in sorted_results:
            if result.status == GroundingStatus.GROUNDED:
                marker = " ✓"
            elif result.status == GroundingStatus.PARTIALLY_GROUNDED:
                marker = " ⚠"
            elif result.status == GroundingStatus.UNGROUNDED:
                marker = " ❌"
            elif result.status == GroundingStatus.CONTRADICTED:
                marker = " ⚡"
            else:
                marker = " ?"
            
            # Insert marker after the claim
            insert_pos = result.claim.end_position
            marked_content = (
                marked_content[:insert_pos] +
                marker +
                marked_content[insert_pos:]
            )
        
        return marked_content
    
    @staticmethod
    def generate_confidence_summary(report: FactualityReport) -> str:
        """Generate a summary of confidence in the content."""
        confidence_level = "HIGH" if report.overall_factuality_score > 0.8 else \
                          "MEDIUM" if report.overall_factuality_score > 0.6 else "LOW"
        
        summary = f"""
FACTUALITY ASSESSMENT:
- Overall Score: {report.overall_factuality_score:.2f}
- Confidence Level: {confidence_level}
- Grounded Claims: {report.grounded_claims}/{report.total_claims}
- Potential Issues: {len(report.hallucination_risks)}
- Contradictions Found: {len(report.contradictions)}

VERIFICATION LEVEL: {report.verification_level.value}
"""
        
        if report.hallucination_risks:
            summary += "\nHALLUCINATION RISKS:\n"
            for risk in report.hallucination_risks[:3]:
                summary += f"- {risk}\n"
        
        if report.recommendations:
            summary += "\nRECOMMENDATIONS:\n"
            for rec in report.recommendations[:3]:
                summary += f"- {rec}\n"
        
        return summary