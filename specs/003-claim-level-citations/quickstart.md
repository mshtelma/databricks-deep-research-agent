# Quickstart: Claim-Level Citation Granularity

**Feature**: 003-claim-level-citations
**Date**: 2025-12-25
**Last Updated**: 2025-12-25 (Improved with 2024-2025 SOTA Research)

## Overview

This feature adds claim-level citation granularity to the Deep Research Agent, enabling users to trace every factual assertion to specific evidence. The implementation follows an **improved 6-stage pipeline** based on 2024-2025 SOTA research:

| Stage | Approach | Key Benefit |
|-------|----------|-------------|
| 1. Evidence Pre-Selection | Extract spans BEFORE generation | Evidence-first constraint |
| 2. Interleaved Generation | ReClaim-style reference→claim | 90% citation accuracy |
| 3. Confidence Classification | HaluGate-style routing | 72% overhead savings |
| 4. Isolated Verification | CoVe in ISOLATION | Prevents bias |
| 5. Citation Correction | CiteFix post-processing | Fixes wrong citations |
| 6. Numeric QA Verification | QAFactEval approach | Catches semantic errors |

## Prerequisites

- Python 3.11+
- PostgreSQL/Lakebase database
- Existing Deep Research Agent setup (see 001-deep-research-agent)
- Node.js 18+ for frontend development

## Quick Setup

### 1. Database Migration

Run the Alembic migration to add claim-level citation tables:

```bash
# Generate migration (if not already created)
uv run alembic revision -m "add_claim_citation_tables_v2"

# Apply migration
uv run alembic upgrade head
```

### 2. Backend Configuration

Add claim verification settings to `config/app.yaml`:

```yaml
# Claim verification configuration (2024-2025 SOTA)
agents:
  citation_verification:
    # Stage 1: Evidence Pre-Selection
    evidence_extraction_max_spans_per_source: 10

    # Stage 2: Interleaved Generation (ReClaim)
    interleaved_max_claims_per_section: 20

    # Stage 3: Confidence Classification (HaluGate)
    high_confidence_threshold: 0.85
    low_confidence_threshold: 0.50

    # Stage 4: Isolated Verification (CoVe)
    verification_model_tier: simple  # Use fast model for high-conf
    verification_analytical_tier: analytical  # Use for low-conf

    # Stage 5: Citation Correction (CiteFix)
    enable_citation_correction: true

    # Stage 6: Numeric QA Verification (QAFactEval)
    enable_numeric_qa_verification: true

    # General settings
    max_parallel_verifications: 5
    verification_timeout_seconds: 3
    unsupported_warning_threshold: 0.20
    contradicted_warning_threshold: 0.05
    abstain_on_no_evidence: true
```

### 3. Test the Feature

#### Backend Unit Tests

```bash
# Run all citation service tests
uv run pytest tests/unit/services/citations/ -v

# Run specific stage tests
uv run pytest tests/unit/services/citations/test_evidence_preselection.py -v
uv run pytest tests/unit/services/citations/test_interleaved_synthesis.py -v
uv run pytest tests/unit/services/citations/test_confidence_classifier.py -v
uv run pytest tests/unit/services/citations/test_isolated_verification.py -v
uv run pytest tests/unit/services/citations/test_citation_correction.py -v
uv run pytest tests/unit/services/citations/test_numeric_verification.py -v

# Run API endpoint tests
uv run pytest tests/unit/api/test_claims_api.py -v
```

#### Frontend Tests

```bash
cd frontend

# Run component tests
npm run test -- --filter=EvidenceCard
npm run test -- --filter=InlineCitation
npm run test -- --filter=VerificationSummary
npm run test -- --filter=ContradictionWarning
```

#### E2E Tests

```bash
# Run claim-level citation E2E tests
make e2e -- --grep="claim-level"
```

## Development Workflow

### Backend Development

1. **Start the development server**:
   ```bash
   make dev
   ```

2. **Key files to work on** (organized by stage):

   **Stage 1-2: Generation Pipeline**
   - `src/services/citations/evidence_preselection.py` - Extract & rank evidence
   - `src/services/citations/interleaved_synthesis.py` - ReClaim-style generation

   **Stage 3-4: Verification Pipeline**
   - `src/services/citations/confidence_classifier.py` - HaluGate-style routing
   - `src/services/citations/isolated_verification.py` - CoVe in isolation

   **Stage 5-6: Post-Processing**
   - `src/services/citations/citation_correction.py` - CiteFix corrections
   - `src/services/citations/numeric_verification.py` - QAFactEval for numbers

   **Supporting Files**
   - `src/agent/nodes/synthesizer.py` - Integration point
   - `src/api/v1/claims.py` - API endpoints
   - `src/models/claim.py` - SQLAlchemy models
   - `src/agent/prompts/citation_prompts.py` - All prompts

3. **API testing**:
   ```bash
   # Get claims for a message
   curl -X GET "http://localhost:8000/api/v1/messages/{message_id}/claims" \
     -H "Authorization: Bearer $TOKEN"

   # Get verification summary (with four-tier verdicts)
   curl -X GET "http://localhost:8000/api/v1/messages/{message_id}/verification-summary" \
     -H "Authorization: Bearer $TOKEN"

   # Get citation corrections
   curl -X GET "http://localhost:8000/api/v1/claims/{claim_id}/corrections" \
     -H "Authorization: Bearer $TOKEN"

   # Get correction metrics
   curl -X GET "http://localhost:8000/api/v1/messages/{message_id}/correction-metrics" \
     -H "Authorization: Bearer $TOKEN"

   # Export provenance data
   curl -X GET "http://localhost:8000/api/v1/messages/{message_id}/provenance" \
     -H "Authorization: Bearer $TOKEN"
   ```

### Frontend Development

1. **Start the development server**:
   ```bash
   make dev-frontend
   ```

2. **Key components to implement**:
   - `frontend/src/components/citations/InlineCitation.tsx` - Citation markers
   - `frontend/src/components/citations/EvidenceCard.tsx` - Evidence popover (4 verdict types)
   - `frontend/src/components/citations/NumericClaimChip.tsx` - Numeric chips
   - `frontend/src/components/citations/VerificationSummary.tsx` - Summary panel
   - `frontend/src/components/citations/ContradictionWarning.tsx` - **NEW**: Warning for contradicted claims

3. **Extend MarkdownRenderer**:
   ```tsx
   // Add custom remark plugin for citation markers
   import { citationPlugin } from '@/lib/markdown/citationPlugin';

   <MarkdownRenderer
     content={content}
     remarkPlugins={[citationPlugin]}
   />
   ```

### SSE Event Handling

New streaming events to handle in the frontend:

```typescript
// In useStreamingQuery.ts

case 'claim_generated':
  // NEW: Stage 2 - claim generated with initial citation
  const genEvent = data as ClaimGeneratedEvent;
  addGeneratedClaim(genEvent);
  break;

case 'claim_verified':
  // Stage 4 - claim verified (4 possible verdicts)
  const verifyEvent = data as ClaimVerifiedEvent;
  updateClaimVerdict(verifyEvent);
  break;

case 'citation_corrected':
  // NEW: Stage 5 - citation was corrected
  const correctEvent = data as CitationCorrectedEvent;
  updateClaimCitation(correctEvent);
  break;

case 'numeric_claim_detected':
  // Stage 6 - numeric claim detected with QA verification
  const numericEvent = data as NumericClaimDetectedEvent;
  addNumericClaim(numericEvent);
  break;

case 'verification_summary':
  // Final - aggregate stats with 4 tiers
  const summaryEvent = data as VerificationSummaryEvent;
  setVerificationSummary(summaryEvent);
  break;
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      Synthesizer Agent                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ STAGE 1: EVIDENCE PRE-SELECTION                          │   │
│  │ • Extract all citable spans from sources                 │   │
│  │ • Rank by relevance, detect numeric content              │   │
│  │ • ~800ms using SIMPLE tier                               │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│  ┌──────────────────────────┴───────────────────────────────┐   │
│  │ STAGE 2: INTERLEAVED GENERATION (ReClaim)                │   │
│  │ • Select reference → Generate claim → Repeat             │   │
│  │ • 90% citation accuracy (vs 60% generate-then-cite)      │   │
│  │ • ~1500ms, streamed                                      │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│  ┌──────────────────────────┴───────────────────────────────┐   │
│  │ STAGE 3: CONFIDENCE CLASSIFICATION (HaluGate)            │   │
│  │ • HIGH: Direct quotes → Quick verify                     │   │
│  │ • LOW: Hedged claims → Full CoVe verify                  │   │
│  │ • ~200ms batch                                           │   │
│  └──────────────────────────────────────────────────────────┘   │
│                    /                       \                     │
│  ┌────────────────┴──────┐ ┌───────────────┴──────────────────┐ │
│  │ HIGH CONFIDENCE       │ │ LOW CONFIDENCE                   │ │
│  │ • SIMPLE tier         │ │ • ANALYTICAL tier                │ │
│  │ • ~300ms              │ │ • ISOLATED context               │ │
│  │ • Saves 72%           │ │ • ~1000ms                        │ │
│  └───────────────────────┘ └──────────────────────────────────┘ │
│                    \                       /                     │
│  ┌──────────────────────────┴───────────────────────────────┐   │
│  │ STAGE 4: CITATION CORRECTION (CiteFix)                   │   │
│  │ • Verify citation actually grounds claim                 │   │
│  │ • KEEP / REPLACE / REMOVE / ADD_ALTERNATE                │   │
│  │ • ~500ms (only non-SUPPORTED)                            │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│  ┌──────────────────────────┴───────────────────────────────┐   │
│  │ STAGE 5: NUMERIC QA VERIFICATION (QAFactEval)            │   │
│  │ • Generate questions about numeric values                │   │
│  │ • Answer from claim AND evidence separately              │   │
│  │ • Divergent answers = potential hallucination            │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│  ┌──────────────────────────┴───────────────────────────────┐   │
│  │ STAGE 6: FINAL ASSEMBLY & STREAMING                      │   │
│  │ • Assemble verified claims with corrected citations      │   │
│  │ • Stream SSE events to frontend                          │   │
│  │ • Support abstention when evidence insufficient          │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Database Layer                             │
├─────────────────────────────────────────────────────────────────┤
│  claims → evidence_spans → citations → numeric_claims           │
│         → citation_corrections (NEW)                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Frontend Layer                             │
├─────────────────────────────────────────────────────────────────┤
│  InlineCitation[1] → EvidenceCard (4 verdict types)             │
│  NumericClaimChip → NumericClaimPopover (QA results)            │
│  ContradictionWarning (NEW - purple warning)                    │
│  VerificationSummary (supported/partial/unsupported/contradicted)│
└─────────────────────────────────────────────────────────────────┘
```

## Key Implementation Patterns

### 1. Evidence Pre-Selection (Stage 1)

```python
# src/services/citations/evidence_preselection.py
from src.services.llm import LLMClient

class EvidencePreSelector:
    """Extract and rank evidence spans BEFORE generation."""

    async def select_evidence_spans(
        self,
        sources: list[Source],
        query: str,
        observations: list[str],
    ) -> list[RankedEvidence]:
        """Extract citable spans with relevance ranking."""
        # 1. Extract candidate spans from each source
        # 2. Rank by relevance to query and observations
        # 3. Detect numeric content for special handling
        # 4. Return top-K ranked evidence pool
```

### 2. Interleaved Generation (Stage 2 - ReClaim)

```python
# src/services/citations/interleaved_synthesis.py
class InterleavedSynthesizer:
    """ReClaim-style reference→claim generation."""

    async def synthesize_with_interleaving(
        self,
        query: str,
        evidence_pool: list[RankedEvidence],
    ) -> AsyncGenerator[InterleavedClaim, None]:
        """Generate claims constrained by evidence."""
        # For each claim:
        #   1. SELECT best evidence from pool
        #   2. GENERATE claim constrained by that evidence
        #   3. YIELD claim + citation pair
```

### 3. Isolated Verification (Stage 4 - CoVe Critical)

```python
# src/services/citations/isolated_verification.py
class IsolatedVerifier:
    """CoVe-style verification in ISOLATION."""

    async def verify_with_isolation(
        self,
        claim: ClassifiedClaim,
        evidence: RankedEvidence,
    ) -> VerificationResult:
        """Verify claim with NO access to generation context.

        CRITICAL: Verification context must NOT include:
        - Why the claim was generated
        - Other claims in the response
        - Synthesis reasoning or prompts
        - Any generation context

        Only includes:
        - The claim text to verify
        - The evidence quote
        """
        # Use ANALYTICAL tier for low-confidence claims
        response = await self.llm.complete(
            prompt=ISOLATED_VERIFICATION_PROMPT.format(
                claim=claim.text,
                evidence=evidence.quote_text,
            ),
            tier="analytical" if claim.confidence == "low" else "simple",
            response_format=VerificationResponse,
        )
        return VerificationResult(
            verdict=response.verdict,  # 4 possible values
            reasoning=response.reasoning,
        )
```

### 4. Citation Correction (Stage 5 - CiteFix)

```python
# src/services/citations/citation_correction.py
class CitationCorrector:
    """Post-process to correct incorrect citations."""

    async def correct_citations(
        self,
        verified_claims: list[VerifiedClaim],
        full_evidence_pool: list[RankedEvidence],
    ) -> list[CitationCorrection]:
        """Fix citations that don't actually support claims."""
        for claim in verified_claims:
            # Check if citation actually grounds the claim
            if not await self._citation_entails(claim):
                # Search for better citation
                better = await self._find_better_citation(
                    claim, full_evidence_pool
                )
                if better:
                    yield CitationCorrection(type="replace", ...)
                else:
                    yield CitationCorrection(type="remove", ...)
```

### 5. Numeric QA Verification (Stage 6 - QAFactEval)

```python
# src/services/citations/numeric_verification.py
class NumericClaimVerifier:
    """QA-based verification for numeric claims."""

    async def verify_numeric_claim(
        self,
        claim: InterleavedClaim,
        evidence: RankedEvidence,
    ) -> NumericVerificationResult:
        """Verify numbers using QA approach."""
        # 1. Generate questions about the numeric value
        questions = await self._generate_questions(claim)

        for question in questions:
            # 2. Answer from claim AND evidence SEPARATELY
            claim_answer = await self._answer_from_claim(question, claim)
            evidence_answer = await self._answer_from_evidence(question, evidence)

            # 3. Compare (after normalization)
            match = self._answers_match(claim_answer, evidence_answer)

            yield QAVerificationResult(
                question=question,
                claim_answer=claim_answer,
                evidence_answer=evidence_answer,
                match=match,
            )
```

## Four-Tier Verdict System

| Verdict | Definition | UI Color | Action |
|---------|------------|----------|--------|
| **SUPPORTED** | Claim FULLY entailed by evidence | Green | Show normally |
| **PARTIAL** | Some aspects supported, others unstated | Amber | Note gaps |
| **UNSUPPORTED** | No evidence basis (not contradicted) | Red | Flag for review |
| **CONTRADICTED** | Evidence DIRECTLY opposes claim | Purple | Strong warning |

## Verification Criteria

Before merging, verify:

1. **Evidence Pre-Selection**: Spans extracted before generation
2. **Interleaved Generation**: 90%+ initial citation accuracy
3. **Confidence Routing**: High-conf claims use SIMPLE tier
4. **Isolated Verification**: No access to generation context
5. **Citation Correction**: Wrong citations are fixed/removed
6. **Numeric QA**: Numbers verified with question-answering
7. **Four Verdicts**: CONTRADICTED shown with purple warning
8. **Abstention**: Claims abstained when evidence insufficient
9. **Performance**: <2s total verification overhead
10. **Provenance Export**: Complete claim-to-evidence mappings

## Common Issues

### Verification Latency Too High

If verification is adding >2s latency:
1. Check confidence classification is routing correctly
2. Ensure high-confidence claims use SIMPLE tier
3. Verify parallel verification is enabled
4. Review `max_parallel_verifications` setting

### Evidence Spans Not Matching

If evidence spans are incorrect:
1. Check evidence pre-selection is running before generation
2. Verify relevance ranking is working
3. Review evidence extraction prompts

### Citation Correction False Positives

If too many citations are being corrected:
1. Increase `high_confidence_threshold`
2. Review entailment check prompts
3. Add more examples to correction prompts

### Contradicted Claims Confusion

If users are confused by contradicted claims:
1. Ensure purple warning is visible
2. Add clear explanation text
3. Show contradicting evidence quote

### Numeric QA Failures

If numeric verification is failing incorrectly:
1. Check normalization is handling different formats
2. Review question generation prompts
3. Add more numeric format examples

## References

- [Feature Specification](./spec.md)
- [Research Document](./research.md) - 2024-2025 SOTA synthesis
- [Data Model](./data-model.md)
- [API Contract](./contracts/openapi.yaml)
- [Implementation Plan](./plan.md)

### Key Papers

- [FActScore](https://arxiv.org/abs/2305.14251) - Min et al., EMNLP 2023
- [Chain-of-Verification](https://arxiv.org/abs/2309.11495) - Dhuliawala et al., 2023
- [ReClaim](https://arxiv.org/abs/2407.01796) - Ground Every Sentence, 2024
- [CiteFix](https://arxiv.org/html/2504.15629v2) - Citation Correction, 2024
- [HaluGate](https://blog.vllm.ai/2025/12/14/halugate.html) - vLLM 2025
- [Semantic Entropy](https://www.nature.com/articles/s41586-024-07421-0) - Nature 2024
