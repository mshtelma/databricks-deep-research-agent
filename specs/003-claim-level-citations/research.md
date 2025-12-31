# Research: Claim-Level Citation Granularity

**Feature**: 003-claim-level-citations  
**Date**: 2025-12-25  
**Status**: Research plan (SOTA-informed)  
**Last Updated**: 2025-12-25  

## How to read this document

This file is the **research plan and verification strategy** for preventing hallucinations / ungrounded claims in our deep research agent.

- **If you want requirements & UX**: start with `specs/003-claim-level-citations/spec.md`
- **If you want implementation tasks**: use `specs/003-claim-level-citations/plan.md`
- **If you want the paper summaries**: see `specs/003-claim-level-citations/sota_research.md`
- **If you want dataset options**: see `specs/003-claim-level-citations/sota_eval_datasets.md`

This file focuses on: **threat model → design invariants → pipeline plan → evaluation commitments**.

## Executive summary (what we’re actually trying to guarantee)

We cannot guarantee “no hallucinations” in the philosophical sense (sources can be wrong; extraction can fail; evidence can be ambiguous). What we *can* guarantee is a **product-level invariants**:

- **Invariant A (No-claim-without-evidence)**: every user-visible factual claim is either (a) backed by at least one evidence span or (b) explicitly marked as not verifiable / abstained.
- **Invariant B (No-fake-citations)**: a citation is never shown unless the cited evidence span is judged to *support that specific claim*.
- **Invariant C (Verification is isolated from generation)**: the verifier never sees the generator’s reasoning or the full draft (prevents copying the original mistake).
- **Invariant D (Numbers get provenance)**: numeric claims must show the **original quoted number(s)**, **normalization**, and if derived, the **calculation chain**.

These invariants are the practical mechanism by which we reduce ungrounded claims to near-zero and make residual errors *auditable*.

## Definitions (to keep the plan precise)

- **Claim**: a user-visible assertion that can be checked against evidence (not instructions, not hedged opinions).
- **Atomic claim**: a single checkable proposition (FACTSCORE-style).
- **Verifiable claim**: an atomic claim that contains enough context (entities/time/conditions) to be meaningfully checked (VeriScore insight).
- **Evidence span**: the minimal quote (with offsets/location) used to justify a claim (not “the whole document”).
- **Citation**: the link from a claim → one or more evidence spans.
- **Derived numeric claim**: a numeric claim computed from multiple quoted inputs (must store inputs + formula).

## Threat model: how “hallucination” shows up in real RAG agents

We explicitly treat “hallucination” as multiple failure modes (because each needs a different mitigation):

- **Unsupported / extrinsic**: claim cannot be found in available evidence (no clear support).
- **Contradicted / intrinsic**: evidence explicitly refutes the claim.
- **Citation error (misattribution)**: claim may be true in the corpus, but the citation points to the wrong source/span. CiteFix reports this can dominate “unverifiable facts”.
- **Over-synthesis**: claim is a plausible inference across sources, but not explicitly stated anywhere (should be labeled as an inference or avoided).
- **Numeric drift**: wrong unit, wrong period, wrong entity, rounding beyond tolerance, or mixing denominators.
- **Context loss**: claim extracted/generated without the qualifiers that make it true (scope, timeframe, population).

## SOTA-backed plan (minimal, implementable, testable)

The plan is a **6-stage evidence-grounding pipeline** designed around three core ideas from the papers:

- **Evidence-first generation** (ReClaim): select evidence before generating each claim.
- **Isolated verification** (CoVe): verify claims without access to the original draft/reasoning.
- **Citation correction** (CiteFix): explicitly repair misattributions after generation.

### Stage overview (inputs/outputs + what success looks like)

| Stage | Purpose | Inputs | Outputs | Primary “done” criterion |
|------:|---------|--------|---------|--------------------------|
| 1 | Build an evidence pool | sources + query | ranked evidence spans | high Recall@K for “citable” spans |
| 2 | Generate claims constrained by evidence | evidence pool | interleaved (claim, cite) | near-100% claim→cite coverage |
| 3 | Route verification effort | claims + evidence | confidence buckets | calibration (ECE) good enough to save compute safely |
| 4 | Verify claims (isolated) | claim + cited span(s) | verdict + rationale | high agreement with human labels |
| 5 | Correct citations | verified claims + full pool | KEEP/REPLACE/REMOVE/ADD | higher citation precision without harming recall |
| 6 | Numeric-specific verification | numeric claims + spans | numeric verdict + provenance | numeric claims show quote + normalization + derivation |

### Verdict taxonomy (user-facing meaning)

We keep verdicts separable because users make different decisions based on them:

- **SUPPORTED**: evidence entails the claim.
- **PARTIAL**: evidence supports some facets but not all (missing qualifiers; incomplete).
- **UNSUPPORTED**: evidence exists but does not support the claim as stated.
- **CONTRADICTED**: evidence refutes the claim.
- **NEI / INSUFFICIENT EVIDENCE**: no relevant evidence retrieved (optional; can be merged if UX demands).

### Non-negotiable implementation constraints

- **Verification must be isolated** (CoVe’s key insight). Even “small” leaks (showing the full paragraph) can reintroduce bias/copying.
- **Claims must be rewritten until verifiable** (VeriScore insight). If a claim needs context to be checkable, we either add that context into the claim or mark it as not verifiable.
- **Explicitly represent derivations**. If the agent computes, it must show the computation graph; otherwise we treat it as ungrounded.

## Evaluation commitments (how we will prove this works)

We will not rely on “it seems better”. We will measure:

- **Citation precision/recall** on citation benchmarks (e.g., ALCE) for stages 2 and 5.
- **Verdict accuracy** on claim-verification benchmarks (FEVER-style mapping) for stage 4.
- **Calibration** (ECE / reliability curves) using span-level hallucination corpora (e.g., RAGTruth-derived labels) for stage 3.
- **Numeric correctness** using NumerSense as a sanity check plus a purpose-built numeric test suite for our product domains (finance/health/legal).
- **End-to-end product metrics**: unsupported claim rate, contradicted claim rate, NEI rate, citation correction rate, abstention rate, and p95 latency overhead.

The detailed benchmark mapping and per-stage evaluation guidance is preserved below.

---

## Part I: Foundational Research (2022-2023)

### 1. FActScore: Fine-grained Atomic Evaluation

**Paper**: Min et al., EMNLP 2023
**Key Contribution**: Decompose text into atomic claims for individual verification

**Why It Matters**:
- Breaking text into atomic claims achieves higher accuracy than sentence-level verification
- Provides the granularity needed for "evidence cards" UX requirement
- Model-agnostic approach works with any LLM

**Implementation Pattern**:
```
Input: "Company X was founded in 2010 and is headquartered in New York with 5,000 employees"
Output claims:
  - Claim 1: "Company X was founded in 2010" → Evidence span A
  - Claim 2: "Company X is headquartered in New York" → Evidence span B
  - Claim 3: "Company X has 5,000 employees" → Evidence span C (numeric)
```

**Limitations Identified**:
- FActScore verifies against Wikipedia, not retrieved sources
- Doesn't address citation correctness, only claim factuality
- No handling of derived/computed claims

### 2. Chain-of-Verification (CoVe)

**Paper**: Dhuliawala et al., 2023
**Key Contribution**: Self-checking paradigm during generation

**CRITICAL INSIGHT (Often Overlooked)**:
> "CoVe is done such that the model cannot simply copy its earlier mistakes – **the questions are answered in isolation to avoid bias from the initial text**"

**Why This Matters**:
Most implementations miss this key detail. The verification must happen with NO ACCESS TO:
- Why the claim was generated
- Other claims in the response
- Synthesis reasoning or prompts
- Original generation context

**Correct Implementation**:
```python
# Verification context deliberately excludes generation context
class VerificationContext:
    evidence_quote: str          # ✓ Only the evidence
    claim_to_verify: str         # ✓ Only the claim
    # NO ACCESS TO:
    # - Generation reasoning
    # - Other claims
    # - Synthesis prompts
```

### 3. QAFactEval: Question-Answering for Consistency

**Paper**: Fabbri et al., 2022
**Key Contribution**: QA-based verification for factual claims

**Why It's Superior for Numeric Claims**:
1. Generate questions about the numeric value
2. Answer questions from the claim AND from the evidence SEPARATELY
3. If answers diverge → potential hallucination

**Example**:
```
Claim: "Revenue was $3.2 billion in Q4 2024"
Evidence: "The company reported Q4 revenues of $3.2B"

Generated Question: "What was the Q4 2024 revenue?"
Answer from Claim: "$3.2 billion"
Answer from Evidence: "$3.2B"
Result: MATCH (after normalization) → SUPPORTED
```

### 4. RARR: Research and Revision

**Paper**: Gao et al., 2023
**Key Contribution**: Post-hoc retrieval and revision pipeline

**Limitation for Our Use Case**:
- Adds latency (user sees unchecked content first)
- But the principle of "revision" is valuable for citation correction

### 5. LLM-Augmenter

**Paper**: Peng et al., 2023
**Key Contribution**: Iterative RAG with feedback loops

**Limitation for Our Use Case**:
- Too many API calls for real-time synthesis
- But the feedback mechanism informs our confidence classification

---

## Part II: 2024-2025 Breakthrough Research

### 6. ReClaim: Reference-First Generation (2024)

**Paper**: arXiv:2407.01796
**Key Contribution**: Interleaved generation that grounds every sentence

**Paradigm Shift**:
- Traditional: Generate → Verify → Cite
- ReClaim: Select Reference → Generate Claim → Repeat

**Performance**:
- 90% citation accuracy (vs ~60% for generate-then-verify)
- Reduces verification burden by ensuring claims are evidence-constrained

**Implementation Pattern**:
```
For each claim to generate:
  1. SELECT best evidence span from pool
  2. GENERATE claim constrained by that evidence
  3. EMIT claim + citation pair
  4. REPEAT
```

### 7. CiteFix: Post-Process Citation Correction (2024)

**Paper**: arXiv:2504.15629v2
**Key Finding**: "Incorrect citations outnumber hallucinations"

**Why This Changes Everything**:
Even with careful generation, citations often don't actually support the claims they're attributed to. We need a correction stage.

**Correction Types**:
| Type | Action | Frequency |
|------|--------|-----------|
| KEEP | Citation is correct | ~60% |
| REPLACE | Find better citation from pool | ~25% |
| REMOVE | No valid citation exists | ~10% |
| ADD_ALTERNATE | Multiple valid citations | ~5% |

**Implementation**:
```python
async def correct_citations(
    verified_claims: list[VerifiedClaim],
    full_evidence_pool: list[RankedEvidence],
) -> list[CitationCorrection]:
    for claim in verified_claims:
        # Check if current citation actually grounds the claim
        if not entails(claim.citation.evidence, claim.text):
            # Search full pool for better citation
            better_cite = find_best_citation(claim, full_evidence_pool)
            if better_cite:
                yield CitationCorrection(type="replace", ...)
            else:
                yield CitationCorrection(type="remove", ...)
```

### 8. HaluGate: Confidence Pre-Classification (2025)

**Source**: vLLM Blog (2025-12-14)
**Key Contribution**: Token-level hallucination detection with confidence routing

**Efficiency Breakthrough**:
Pre-classifying claims by confidence level saves **72% verification overhead**:

| Confidence | Verification Type | Latency |
|------------|-------------------|---------|
| HIGH (>0.85) | Quick single-pass | ~300ms |
| MEDIUM (0.50-0.85) | Standard verify | ~500ms |
| LOW (<0.50) | Full CoVe with isolation | ~1000ms |

**Confidence Indicators**:
- HIGH: Direct quotes, exact matches, copied verbatim
- MEDIUM: Paraphrased facts, summarized data
- LOW: Hedged statements ("approximately", "may"), comparative claims, synthetic computations

### 9. EVER: External Verification (2023)

**Key Contribution**: Distinguishes internal vs external hallucinations

**Four-Tier Verdict System**:
| Verdict | Definition | UI Treatment |
|---------|------------|--------------|
| **SUPPORTED** | Claim FULLY entailed by evidence | Green indicator |
| **PARTIAL** | Some aspects supported, others unstated | Amber indicator |
| **UNSUPPORTED** | No evidence basis (not contradicted) | Red indicator |
| **CONTRADICTED** | Evidence DIRECTLY opposes claim | Purple warning |

**Why "Contradicted" Matters**:
- "Not supported" = claim may be true but we can't verify
- "Contradicted" = claim is demonstrably false
- Users need this distinction for decision-making

### 10. Semantic Entropy (Nature 2024)

**Paper**: Nature (2024) - "Detecting Hallucinations in Large Language Models Using Semantic Entropy"
**Key Contribution**: Meaning-level uncertainty detection

**How It Works**:
1. Sample multiple responses for the same prompt
2. Cluster responses by semantic equivalence
3. High entropy (many clusters) = low confidence
4. Low entropy (one cluster) = high confidence

**Application**: Can inform our confidence pre-classification by checking if the model is "certain" about a claim.

### 11. SAFE/LongFact (DeepMind 2024)

**Repository**: google-deepmind/long-form-factuality
**Key Contribution**: Search-augmented factuality for long-form content

**Relevance**:
- Designed for verifying long research reports (our use case)
- Uses search to retrieve supporting evidence
- Provides calibrated factuality scores

### 12. VeriScore/VeriFastScore (2024)

**Paper**: arXiv:2406.19276
**Key Contribution**: Single-pass factuality evaluation

**Why It's Fast**:
- Evaluates entire response in one pass
- Achieves 0.95+ correlation with multi-pass methods
- Can be used as a final "sanity check" on synthesis output

---

## Part III: Improved Architecture

Based on the 2024-2025 research, we propose a **6-stage pipeline**:

```
┌─────────────────────────────────────────────────────────────┐
│ STAGE 1: EVIDENCE PRE-SELECTION                             │
│ - Extract all citable spans from sources BEFORE generation  │
│ - Rank by relevance to query, detect numeric content        │
│ - ~800ms total using SIMPLE tier                            │
│ - Produces: RankedEvidence[]                                │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────┴───────────────────────────────┐
│ STAGE 2: INTERLEAVED GENERATION (ReClaim-style)             │
│ - For each paragraph:                                       │
│   - Select best reference → Generate claim → Repeat         │
│ - Claims constrained by pre-selected evidence               │
│ - 90% initial citation accuracy                             │
│ - ~1500ms, streamed as generated                            │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────┴───────────────────────────────┐
│ STAGE 3: CONFIDENCE PRE-CLASSIFICATION (HaluGate-inspired)  │
│ - HIGH (>0.85): Direct quotes, exact matches                │
│ - MEDIUM (0.50-0.85): Paraphrased facts                     │
│ - LOW (<0.50): Hedged, comparative, synthetic claims        │
│ - ~200ms batch classification                               │
└─────────────────────────────────────────────────────────────┘
                    /                       \
┌──────────────────┴─────────┐ ┌────────────┴──────────────────┐
│ HIGH CONFIDENCE            │ │ LOW CONFIDENCE                │
│ - Single-pass verification │ │ - Full CoVe with ISOLATION    │
│ - SIMPLE tier model        │ │ - ANALYTICAL tier model       │
│ - ~300ms per claim         │ │ - ~1000ms per claim           │
│ - Saves 72% overhead       │ │ - No access to gen context    │
└────────────────────────────┘ └───────────────────────────────┘
                    \                       /
┌─────────────────────────────┴───────────────────────────────┐
│ STAGE 4: CITATION CORRECTION (CiteFix-style)                │
│ - Verify each citation actually grounds the claim           │
│ - Search full evidence pool for better citations            │
│ - Actions: KEEP, REPLACE, REMOVE, ADD_ALTERNATE             │
│ - ~500ms (only for non-SUPPORTED claims)                    │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────┴───────────────────────────────┐
│ STAGE 5: NUMERIC QA VERIFICATION (QAFactEval-style)         │
│ - For each numeric claim:                                   │
│   - Generate questions about the numeric value              │
│   - Answer from claim AND evidence SEPARATELY               │
│   - Divergent answers = potential hallucination             │
│ - ~800ms per numeric claim, parallelized                    │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────┴───────────────────────────────┐
│ STAGE 6: FINAL ASSEMBLY & STREAMING                         │
│ - Assemble verified claims with corrected citations         │
│ - Stream with real-time verification events                 │
│ - Emit verification summary with warning if needed          │
│ - Support abstention when evidence insufficient             │
└─────────────────────────────────────────────────────────────┘
```

### Stage 1: Evidence Pre-Selection (Detailed)

**Purpose**: Extract all citable spans from sources BEFORE generation to enable attribute-first synthesis.

**Algorithm (Pseudo-code)**:
```
ALGORITHM: Evidence Pre-Selection

INPUT: sources[], query, observations[]
OUTPUT: RankedEvidence[]

1. FOR each source IN sources:
   a. SEGMENT source.content into candidate_spans (sentence boundaries)
   b. FILTER spans by length (50-500 chars)
   c. DETECT numeric content in each span
   d. FOR each span:
      - COMPUTE relevance_score = similarity(span, query)
      - IF span.has_numeric: relevance_score += numeric_content_boost
      - ADD (span, relevance_score, source_metadata) to candidates

2. SORT candidates BY relevance_score DESC

3. SELECT top_k candidates (default: max_spans_per_source × source_count)

4. RETURN RankedEvidence[]
```

**Implementation Pattern (Python)**:
```python
from dataclasses import dataclass
from typing import AsyncGenerator

@dataclass
class RankedEvidence:
    """A citable evidence span with ranking metadata."""
    quote: str
    source_id: str
    source_title: str
    source_url: str
    relevance_score: float
    has_numeric: bool
    position_in_source: tuple[int, int]  # start, end offsets

class EvidencePreSelector:
    """Stage 1: Extract citable spans from sources before generation.

    TOGGLE: enable_evidence_preselection
    FALLBACK: When disabled, returns full source documents as single spans.
    """

    def __init__(self, config: CitationVerificationConfig, llm: LLMClient):
        self.config = config
        self.llm = llm

    async def select_evidence_spans(
        self,
        sources: list[SourceInfo],
        query: str,
        observations: list[str],
    ) -> list[RankedEvidence]:
        if not self.config.enable_evidence_preselection:
            # FALLBACK: Use full documents
            return self._sources_as_evidence(sources)

        candidates: list[RankedEvidence] = []

        for source in sources:
            spans = self._segment_into_spans(source.content)
            for span in spans:
                if not self._is_valid_span(span):
                    continue

                relevance = await self._compute_relevance(span, query, observations)
                has_numeric = self._detect_numeric(span)

                if has_numeric:
                    relevance += self.config.evidence_preselection.numeric_content_boost

                if relevance >= self.config.evidence_preselection.relevance_threshold:
                    candidates.append(RankedEvidence(
                        quote=span.text,
                        source_id=source.id,
                        source_title=source.title,
                        source_url=source.url,
                        relevance_score=relevance,
                        has_numeric=has_numeric,
                        position_in_source=(span.start, span.end),
                    ))

        # Sort and limit
        candidates.sort(key=lambda x: x.relevance_score, reverse=True)
        max_total = (
            self.config.evidence_preselection.max_spans_per_source * len(sources)
        )
        return candidates[:max_total]
```

**Critical Review from SOTA**:
- **ReClaim** achieves 90% citation accuracy partly due to high-quality evidence pre-selection
- **CiteFix** found keyword-based matching outperforms pure semantic for some LLMs
- **VeriScore** recommends sliding window approach for context preservation

**Proposed Solution for Gap**: Offer multiple ranking strategies via `relevance_computation_method: semantic | keyword | hybrid`

---

### Stage 2: Interleaved Generation (Detailed)

**Purpose**: Generate claims constrained by pre-selected evidence (ReClaim pattern).

**Algorithm (Pseudo-code)**:
```
ALGORITHM: Interleaved Reference-Claim Generation (ReClaim)

INPUT: query, evidence_pool[], section_outline
OUTPUT: (claim, evidence)[] pairs, streamed

1. FOR each section IN section_outline:
   a. INITIALIZE remaining_evidence = evidence_pool

   b. WHILE section not complete:
      i.   SELECT best_evidence from remaining_evidence
           - Score by relevance to current section context
           - Prioritize unused evidence

      ii.  GENERATE claim CONSTRAINED BY best_evidence
           - Prompt includes ONLY the selected evidence quote
           - Claim must be derivable from evidence

      iii. VALIDATE claim-evidence entailment (quick check)
           - If fails: RETRY with next-best evidence

      iv.  YIELD (claim, best_evidence)

      v.   REMOVE best_evidence from remaining_evidence
           - Prevents over-citation of single source

      vi.  CHECK if section content sufficient
           - If yes: BREAK to next section
```

**Implementation Pattern (Python)**:
```python
class InterleavedSynthesizer:
    """Stage 2: ReClaim-style interleaved reference→claim generation.

    TOGGLE: enable_interleaved_generation
    FALLBACK: When disabled, uses traditional generate-then-cite (60% accuracy).

    Key insight from ReClaim paper:
    > "By selecting evidence FIRST, we ensure every claim has a valid citation
    > at generation time, rather than hoping to find one afterwards."
    """

    async def synthesize(
        self,
        query: str,
        evidence_pool: list[RankedEvidence],
        section_outline: list[str],
        llm: LLMClient,
        config: CitationVerificationConfig,
    ) -> AsyncGenerator[InterleavedClaim, None]:
        if not config.enable_interleaved_generation:
            # FALLBACK: Traditional generate-then-cite
            async for claim in self._generate_then_cite(query, evidence_pool, llm):
                yield claim
            return

        remaining_evidence = list(evidence_pool)

        for section in section_outline:
            claims_in_section = 0

            while claims_in_section < config.interleaved_generation.max_claims_per_section:
                # Step 1: SELECT best evidence for this section
                best_evidence = self._select_best_evidence(
                    section_context=section,
                    remaining_evidence=remaining_evidence,
                    min_similarity=config.interleaved_generation.min_evidence_similarity,
                )

                if best_evidence is None:
                    break  # No more relevant evidence for this section

                # Step 2: GENERATE claim constrained by evidence
                claim = await self._generate_constrained_claim(
                    query=query,
                    section=section,
                    evidence=best_evidence,
                    llm=llm,
                )

                # Step 3: Quick entailment check
                if not await self._quick_entailment_check(claim.text, best_evidence.quote):
                    # Retry with next-best evidence
                    remaining_evidence.remove(best_evidence)
                    continue

                # Step 4: YIELD claim + citation pair
                yield InterleavedClaim(
                    text=claim.text,
                    evidence=best_evidence,
                    section=section,
                    confidence=claim.confidence,
                )

                # Step 5: Remove used evidence
                remaining_evidence.remove(best_evidence)
                claims_in_section += 1

    async def _generate_constrained_claim(
        self,
        query: str,
        section: str,
        evidence: RankedEvidence,
        llm: LLMClient,
    ) -> GeneratedClaim:
        """Generate a claim that is derivable from the given evidence.

        CRITICAL: The prompt ONLY includes the evidence quote.
        The LLM cannot invent facts not in the evidence.
        """
        prompt = f"""Generate a factual claim for a research report.

TOPIC: {section}
RESEARCH QUESTION: {query}

EVIDENCE (you may ONLY use facts from this quote):
"{evidence.quote}"
— Source: {evidence.source_title}

Write ONE clear, factual sentence that:
1. Is fully supported by the evidence above
2. Contributes to answering the research question
3. Could stand alone with a citation

CLAIM:"""

        response = await llm.complete(
            messages=[{"role": "user", "content": prompt}],
            tier=ModelTier.ANALYTICAL,
            max_tokens=200,
        )

        return GeneratedClaim(text=response.content.strip(), confidence=0.8)
```

**Critical Review from SOTA**:
- **ReClaim** achieves 90% citation accuracy vs 60% for generate-then-cite
- **Trade-off**: May reduce creativity and information synthesis
- **CiteFix** notes that even ReClaim needs post-processing for 10% of citations

**When to Disable**:
- Creative synthesis required (brainstorming, ideation)
- Speed is critical and accuracy can be sacrificed
- Evidence pool is small and well-curated

---

### Stage 3: Confidence Pre-Classification (Detailed)

**Purpose**: Route claims to appropriate verification intensity, saving 72% overhead.

**Challenge**: HaluGate requires token-level log probabilities, which Databricks endpoints don't expose.

**Proposed Solution**: Linguistic Confidence Proxy using rule-based signals.

**Algorithm (Pseudo-code)**:
```
ALGORITHM: Linguistic Confidence Estimation

INPUT: claim_text, evidence_quote
OUTPUT: confidence_score (0.0 - 1.0), confidence_level (HIGH/MEDIUM/LOW)

SIGNALS:
  HIGH_CONFIDENCE_SIGNALS (+0.3 each):
    - Direct quote markers ("" or '')
    - Exact numeric match between claim and evidence
    - Named entity exact match (after normalization)
    - Temporal expression exact match

  MEDIUM_CONFIDENCE_SIGNALS (+0.1 each):
    - High semantic similarity (>0.85) between claim and evidence
    - Source explicitly mentioned in claim
    - Common factual patterns (e.g., "X was founded in Y")

  LOW_CONFIDENCE_SIGNALS (-0.2 each):
    - Hedging words: "approximately", "around", "may", "might", "possibly", "likely"
    - Comparative claims: "more than", "less than", "similar to", "compared to"
    - Derived/computed values: "totaling", "combined", "on average", "sum of"
    - Subjective qualifiers: "significant", "substantial", "notable", "major"
    - Superlatives without source: "best", "largest", "most"

COMPUTATION:
  base_score = 0.5
  FOR each signal IN detected_signals(claim_text, evidence_quote):
    base_score += signal.weight
  confidence_score = clamp(base_score, 0.0, 1.0)

  IF confidence_score >= high_threshold:  # default 0.85
    RETURN (confidence_score, HIGH)
  ELIF confidence_score >= low_threshold:  # default 0.50
    RETURN (confidence_score, MEDIUM)
  ELSE:
    RETURN (confidence_score, LOW)
```

**Implementation Pattern (Python)**:
```python
import re
from enum import Enum

class ConfidenceLevel(str, Enum):
    HIGH = "high"      # >0.85: Quick single-pass verification
    MEDIUM = "medium"  # 0.50-0.85: Standard verification
    LOW = "low"        # <0.50: Full CoVe with isolation

class LinguisticConfidenceClassifier:
    """Stage 3: Classify claims by confidence using linguistic signals.

    TOGGLE: enable_confidence_classification
    FALLBACK: When disabled, all claims get LOW confidence (full verification).

    This is a PROXY for HaluGate's token-level confidence, since Databricks
    serving endpoints don't expose log probabilities. See research.md for
    detailed justification and signal weights.
    """

    # Compiled patterns for efficiency
    HEDGING_PATTERNS = re.compile(
        r'\b(approximately|around|about|may|might|possibly|likely|perhaps|'
        r'probably|seemingly|apparently|reportedly|allegedly)\b',
        re.IGNORECASE
    )
    COMPARATIVE_PATTERNS = re.compile(
        r'\b(more than|less than|greater than|fewer than|similar to|'
        r'compared to|relative to|versus|vs\.?)\b',
        re.IGNORECASE
    )
    DERIVED_PATTERNS = re.compile(
        r'\b(totaling|combined|on average|average of|sum of|total of|'
        r'calculated|estimated|projected)\b',
        re.IGNORECASE
    )
    QUOTE_MARKERS = re.compile(r'["""\']([^"""\']+)["""\']')

    def __init__(self, config: CitationVerificationConfig):
        self.config = config
        self.high_threshold = config.confidence_classification.high_threshold
        self.low_threshold = config.confidence_classification.low_threshold

    async def classify_batch(
        self,
        claims: list[InterleavedClaim],
    ) -> list[ClassifiedClaim]:
        """Classify multiple claims in batch for efficiency."""
        if not self.config.enable_confidence_classification:
            # FALLBACK: All claims get LOW confidence → full verification
            return [
                ClassifiedClaim(
                    claim=c,
                    confidence_score=0.3,
                    confidence_level=ConfidenceLevel.LOW,
                )
                for c in claims
            ]

        results = []
        for claim in claims:
            score = self._compute_confidence(claim.text, claim.evidence.quote)
            level = self._score_to_level(score)
            results.append(ClassifiedClaim(
                claim=claim,
                confidence_score=score,
                confidence_level=level,
            ))
        return results

    def _compute_confidence(self, claim_text: str, evidence_quote: str) -> float:
        """Compute linguistic confidence score."""
        score = 0.5  # Base score

        # HIGH signals (+0.3 each)
        if self.QUOTE_MARKERS.search(claim_text):
            score += self.config.confidence_classification.quote_match_bonus

        if self._has_exact_numeric_match(claim_text, evidence_quote):
            score += 0.3

        if self._has_entity_exact_match(claim_text, evidence_quote):
            score += 0.2

        # LOW signals (-0.2 each)
        hedging_count = len(self.HEDGING_PATTERNS.findall(claim_text))
        score -= hedging_count * self.config.confidence_classification.hedging_word_penalty

        if self.COMPARATIVE_PATTERNS.search(claim_text):
            score -= 0.2

        if self.DERIVED_PATTERNS.search(claim_text):
            score -= 0.2

        return max(0.0, min(1.0, score))

    def _score_to_level(self, score: float) -> ConfidenceLevel:
        if score >= self.high_threshold:
            return ConfidenceLevel.HIGH
        elif score >= self.low_threshold:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
```

**Critical Review from SOTA**:
- **HaluGate** reports 72% overhead savings with token-level confidence
- Our linguistic proxy is less accurate but works without log probabilities
- **Semantic Entropy** (Nature 2024) offers an alternative: sample multiple responses

**Proposed Enhancement**: Add `embedding_similarity` method that computes claim-evidence similarity score as confidence proxy.

---

### Stage 4: Isolated Verification (CRITICAL - Detailed)

**Purpose**: Verify claims in ISOLATION to prevent bias propagation.

> **CRITICAL INSIGHT FROM CoVe PAPER**:
> "CoVe is done such that the model cannot simply copy its earlier mistakes –
> the questions are answered in isolation to avoid bias from the initial text"

This is the **most-often-missed detail** in verification implementations. The verification context MUST NOT include generation reasoning.

**What MUST be Excluded from Verification Context**:
1. The original synthesis system prompt
2. Other claims in the response
3. The query (to prevent recall bias)
4. Any generation reasoning or chain-of-thought
5. The section outline or plan

**What IS Included in Verification Context**:
1. The claim text to verify
2. The evidence quote
3. Verification instructions only

**Algorithm (Pseudo-code)**:
```
ALGORITHM: Isolated Claim Verification (CoVe-style)

INPUT: claim, evidence, confidence_level
OUTPUT: VerificationResult (verdict, reasoning, corrected_claim?)

1. CREATE isolated_context:
   - claim_text: claim.text
   - evidence_quote: evidence.quote
   - NO other context (query, other claims, generation reasoning)

2. SELECT verification_intensity BY confidence_level:
   - HIGH: Single-pass NLI check (SIMPLE tier)
   - MEDIUM: Standard verification with reasoning (ANALYTICAL tier)
   - LOW: Full CoVe with verification questions (ANALYTICAL tier)

3. FOR LOW confidence (full CoVe):
   a. GENERATE verification questions about the claim
   b. ANSWER each question using ONLY the evidence (in isolation)
   c. COMPARE answers to what claim asserts
   d. IDENTIFY any mismatches

4. DETERMINE verdict using EVER five-tier system:
   - SUPPORTED: All claim parts entailed by evidence
   - PARTIAL: Some parts supported, others unstated
   - UNSUPPORTED: No evidence supports (but doesn't contradict)
   - CONTRADICTED: Evidence directly opposes claim
   - NEI: Not Enough Information in evidence

5. IF verdict != SUPPORTED AND config.enable_verification_retrieval:
   a. SEARCH for additional evidence (web search)
   b. RETRY verification with expanded evidence
   c. UPDATE verdict if better evidence found

6. RETURN VerificationResult
```

**Implementation Pattern (Python)**:
```python
class IsolatedVerifier:
    """Stage 4: CoVe-style verification in ISOLATION.

    CANNOT BE DISABLED - This is the core of the verification system.

    CRITICAL DESIGN PRINCIPLE:
    The verification context MUST NOT include:
    - Generation reasoning or chain-of-thought
    - Other claims from the response
    - The synthesis system prompt
    - The original query (to prevent recall bias)
    - Any context about WHY the claim was generated

    ONLY includes:
    - The claim text to verify
    - The evidence quote
    - Instructions for verification

    This isolation prevents the LLM from "copying its earlier mistakes"
    as noted in the CoVe paper (Dhuliawala et al., 2023).
    """

    # Verification prompts - note the minimal context
    QUICK_VERIFICATION_PROMPT = """Determine if the claim is supported by the evidence.

EVIDENCE: "{evidence}"

CLAIM: "{claim}"

Is the claim fully supported by the evidence?
Answer: SUPPORTED, PARTIAL, UNSUPPORTED, or CONTRADICTED"""

    FULL_COVE_PROMPT = """Verify this claim against the evidence.

EVIDENCE: "{evidence}"

CLAIM: "{claim}"

Step 1: List the key assertions in the claim.
Step 2: For each assertion, check if the evidence supports it.
Step 3: Determine the overall verdict.

Analysis:"""

    async def verify(
        self,
        claim: ClassifiedClaim,
        llm: LLMClient,
        config: CitationVerificationConfig,
    ) -> VerificationResult:
        """Verify a claim in isolation from generation context."""

        # Select verification intensity based on confidence
        if claim.confidence_level == ConfidenceLevel.HIGH:
            return await self._quick_verify(claim, llm)
        elif claim.confidence_level == ConfidenceLevel.MEDIUM:
            return await self._standard_verify(claim, llm)
        else:  # LOW
            return await self._full_cove_verify(claim, llm, config)

    async def _quick_verify(
        self,
        claim: ClassifiedClaim,
        llm: LLMClient,
    ) -> VerificationResult:
        """Quick single-pass verification for HIGH confidence claims."""
        # ISOLATION: Only claim and evidence in context
        prompt = self.QUICK_VERIFICATION_PROMPT.format(
            evidence=claim.claim.evidence.quote,
            claim=claim.claim.text,
        )

        response = await llm.complete(
            messages=[{"role": "user", "content": prompt}],
            tier=ModelTier.SIMPLE,  # Fast model for high-confidence
            max_tokens=50,
        )

        verdict = self._parse_verdict(response.content)
        return VerificationResult(
            verdict=verdict,
            reasoning="Quick verification (high confidence claim)",
            verification_method="quick",
        )

    async def _full_cove_verify(
        self,
        claim: ClassifiedClaim,
        llm: LLMClient,
        config: CitationVerificationConfig,
    ) -> VerificationResult:
        """Full CoVe verification for LOW confidence claims.

        Follows the factored execution pattern from CoVe paper:
        1. Generate verification questions
        2. Answer each question in ISOLATION (no access to claim)
        3. Compare answers to claim assertions
        """
        # Step 1: Generate verification questions about the claim
        questions = await self._generate_verification_questions(
            claim.claim.text,
            llm,
        )

        # Step 2: Answer each question using ONLY the evidence
        # CRITICAL: Do NOT include the claim when answering
        answers = []
        for question in questions:
            answer = await self._answer_from_evidence_only(
                question=question,
                evidence=claim.claim.evidence.quote,
                llm=llm,
            )
            answers.append((question, answer))

        # Step 3: Compare answers to what the claim asserts
        verdict, reasoning = await self._compare_answers_to_claim(
            claim=claim.claim.text,
            qa_pairs=answers,
            llm=llm,
        )

        # Step 4: Optional additional retrieval for weak evidence
        if (
            verdict in (Verdict.UNSUPPORTED, Verdict.NEI)
            and config.enable_verification_retrieval
        ):
            enhanced_result = await self._verify_with_additional_retrieval(
                claim, llm, config
            )
            if enhanced_result.verdict == Verdict.SUPPORTED:
                return enhanced_result

        return VerificationResult(
            verdict=verdict,
            reasoning=reasoning,
            verification_method="full_cove",
            qa_pairs=answers,
        )

    async def _answer_from_evidence_only(
        self,
        question: str,
        evidence: str,
        llm: LLMClient,
    ) -> str:
        """Answer a verification question using ONLY the evidence.

        CRITICAL: The claim is NOT included in this prompt.
        This prevents the model from simply agreeing with the claim.
        """
        prompt = f"""Answer this question based ONLY on the evidence provided.
If the evidence doesn't contain the answer, say "NOT FOUND IN EVIDENCE".

EVIDENCE: "{evidence}"

QUESTION: {question}

ANSWER (based only on evidence above):"""

        response = await llm.complete(
            messages=[{"role": "user", "content": prompt}],
            tier=ModelTier.ANALYTICAL,
            max_tokens=100,
        )

        return response.content.strip()
```

**Five-Tier Verdict System (EVER + NEI)**:

```python
class Verdict(str, Enum):
    """Five-tier verdict system combining EVER framework with NEI.

    SUPPORTED: All parts of claim fully entailed by evidence
    PARTIAL: Some aspects supported, others unstated (not contradicted)
    UNSUPPORTED: Evidence exists but doesn't support the claim
    CONTRADICTED: Evidence directly opposes one or more claim parts
    NEI: Not Enough Information - no relevant evidence exists

    The distinction between UNSUPPORTED and NEI is important:
    - UNSUPPORTED: We have evidence about this topic, but it doesn't support the claim
    - NEI: We have no evidence about this topic at all
    """
    SUPPORTED = "supported"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"
    NEI = "nei"  # Not Enough Information
```

**Decision Tree for Verdict Classification**:
```
                    Evidence exists for claim topic?
                           /           \
                         YES            NO
                          |              |
                Evidence mentions claim?   RETURN NEI
                   /           \       (Not Enough Information)
                 YES            NO
                  |              |
            Evidence supports?  RETURN UNSUPPORTED
               /    |    \
             FULL  PARTIAL  NO
              |      |       \
         SUPPORTED PARTIAL  Evidence contradicts?
                                /        \
                              YES         NO
                               |           |
                         CONTRADICTED  UNSUPPORTED
```

**Critical Review from SOTA**:
- **CoVe** emphasizes isolation as the key to preventing error propagation
- **EVER** shows real-time verification prevents "snowballing" errors
- **EVER** achieves 94.5% FACTSCORE with retrieval augmentation

**Proposed Solutions**:
1. Use different model tier for verification than synthesis (prevents context leakage)
2. Add optional `enable_verification_retrieval` for additional evidence search
3. Implement NEI verdict to distinguish "no evidence" from "contradicting evidence"

---

### Stage 5: Citation Correction (Detailed)

**Purpose**: Post-process citations to fix incorrect attributions (CiteFix insight).

> **KEY FINDING FROM CITEFIX**:
> "Approximately 80% of unverifiable facts were not hallucinations but citation errors—
> the model generated correct information but attributed it to the wrong source."

**Correction Types**:
| Type | Action | Frequency | When to Apply |
|------|--------|-----------|---------------|
| KEEP | Citation is correct | ~60% | Evidence entails claim |
| REPLACE | Find better citation | ~25% | Better evidence exists in pool |
| REMOVE | No valid citation exists | ~10% | No evidence supports claim |
| ADD_ALTERNATE | Multiple valid citations | ~5% | Additional sources support claim |

**Algorithm (Pseudo-code)**:
```
ALGORITHM: Citation Correction (CiteFix-style)

INPUT: verified_claims[], full_evidence_pool[], correction_method
OUTPUT: CorrectionAction[] for each claim

FOR each claim IN verified_claims:
  IF claim.verdict == SUPPORTED:
    YIELD CorrectionAction(type=KEEP, claim=claim)
    CONTINUE

  # Claim is PARTIAL, UNSUPPORTED, or CONTRADICTED
  # Search for better citation

  FOR each evidence IN full_evidence_pool:
    IF evidence == claim.current_evidence:
      CONTINUE  # Skip current citation

    similarity = compute_similarity(claim.text, evidence.quote, correction_method)

    IF similarity > correction_threshold:
      # Found better citation
      IF claim.verdict == CONTRADICTED:
        # Current citation contradicts - definitely replace
        YIELD CorrectionAction(type=REPLACE, claim=claim, new_evidence=evidence)
      ELIF similarity > claim.current_evidence.similarity:
        # New evidence is better
        IF config.allow_alternate_citations:
          YIELD CorrectionAction(type=ADD_ALTERNATE, claim=claim, new_evidence=evidence)
        ELSE:
          YIELD CorrectionAction(type=REPLACE, claim=claim, new_evidence=evidence)
      BREAK

  ELSE:
    # No better citation found
    IF claim.verdict in (UNSUPPORTED, NEI):
      YIELD CorrectionAction(type=REMOVE, claim=claim)
    ELSE:
      YIELD CorrectionAction(type=KEEP, claim=claim)  # Keep partial
```

**Implementation Pattern (Python) - Pluggable Strategy**:
```python
from typing import Protocol

class CitationCorrectionStrategy(Protocol):
    """Protocol for pluggable citation correction methods.

    CiteFix paper found different LLMs work better with different methods:
    - Model A: Keyword + Semantic Context works best (+21% MQLA)
    - Model B: Fine-tuned BERT Score works best (+21% MQLA)
    - Model C: Both approaches yield +15.5-15.8% MQLA

    We implement multiple strategies and make them configurable.
    """

    async def find_best_citation(
        self,
        claim: str,
        current_evidence: RankedEvidence,
        evidence_pool: list[RankedEvidence],
    ) -> tuple[RankedEvidence | None, float]:
        """Find best matching evidence and return with confidence score."""
        ...

class KeywordCorrector(CitationCorrectionStrategy):
    """CiteFix Method 1: Keyword-based matching."""

    async def find_best_citation(self, claim, current, pool):
        claim_tokens = set(claim.lower().split())
        best_match = None
        best_score = 0.0

        for evidence in pool:
            if evidence == current:
                continue
            evidence_tokens = set(evidence.quote.lower().split())
            overlap = len(claim_tokens & evidence_tokens)
            score = overlap / max(len(claim_tokens), 1)

            if score > best_score:
                best_score = score
                best_match = evidence

        return best_match, best_score

class KeywordSemanticHybridCorrector(CitationCorrectionStrategy):
    """CiteFix Method 2: Keyword + Semantic Context (λ=0.8).

    This is the recommended default based on CiteFix experiments.
    """

    def __init__(self, lambda_weight: float = 0.8, embedding_model=None):
        self.lambda_weight = lambda_weight
        self.embedding_model = embedding_model

    async def find_best_citation(self, claim, current, pool):
        best_match = None
        best_score = 0.0

        for evidence in pool:
            if evidence == current:
                continue

            keyword_score = self._keyword_overlap(claim, evidence.quote)
            semantic_score = await self._semantic_similarity(claim, evidence.quote)

            combined = (
                self.lambda_weight * keyword_score
                + (1 - self.lambda_weight) * semantic_score
            )

            if combined > best_score:
                best_score = combined
                best_match = evidence

        return best_match, best_score

class CitationCorrector:
    """Stage 5: Post-process to correct incorrect citations.

    TOGGLE: enable_citation_correction
    FALLBACK: When disabled, initial citations used as-is (25% may be incorrect).
    """

    STRATEGIES = {
        CorrectionMethod.KEYWORD: KeywordCorrector,
        CorrectionMethod.KEYWORD_SEMANTIC_HYBRID: KeywordSemanticHybridCorrector,
        # Add more strategies as needed
    }

    def __init__(self, config: CitationVerificationConfig):
        self.config = config
        strategy_class = self.STRATEGIES.get(
            config.citation_correction.correction_method,
            KeywordSemanticHybridCorrector,
        )
        self.strategy = strategy_class(
            lambda_weight=config.citation_correction.lambda_weight
        )

    async def correct_citations(
        self,
        verified_claims: list[VerifiedClaim],
        full_evidence_pool: list[RankedEvidence],
    ) -> list[CitationCorrection]:
        """Correct citations for claims that aren't fully supported."""
        if not self.config.enable_citation_correction:
            # FALLBACK: Keep all citations as-is
            return [
                CitationCorrection(type=CorrectionType.KEEP, claim=c)
                for c in verified_claims
            ]

        corrections = []

        for claim in verified_claims:
            if claim.verdict == Verdict.SUPPORTED:
                corrections.append(CitationCorrection(
                    type=CorrectionType.KEEP,
                    claim=claim,
                ))
                continue

            # Find better citation
            better_evidence, score = await self.strategy.find_best_citation(
                claim=claim.text,
                current_evidence=claim.evidence,
                evidence_pool=full_evidence_pool,
            )

            if better_evidence and score > self.config.citation_correction.correction_threshold:
                corrections.append(CitationCorrection(
                    type=CorrectionType.REPLACE,
                    claim=claim,
                    new_evidence=better_evidence,
                    confidence=score,
                ))
            elif claim.verdict in (Verdict.UNSUPPORTED, Verdict.NEI):
                corrections.append(CitationCorrection(
                    type=CorrectionType.REMOVE,
                    claim=claim,
                ))
            else:
                corrections.append(CitationCorrection(
                    type=CorrectionType.KEEP,
                    claim=claim,
                ))

        return corrections
```

**Critical Review from SOTA**:
- **CiteFix** shows citation correction improves MQLA by 15-21%
- Different LLMs benefit from different correction methods
- Fine-tuned BERT Score requires training data but outperforms for some models

**Proposed Solution**: Make correction method configurable via `correction_method` parameter, with A/B testing to find optimal per-model.

---

### Stage 6: Numeric QA Verification (Detailed)

**Purpose**: Verify numeric claims using QA-based approach (QAFactEval).

> **WHY QA IS SUPERIOR TO REGEX**:
> Regex catches patterns but can't verify semantic correctness.
> QA-based verification catches:
> - Wrong entity (revenue vs. profit)
> - Wrong time period (Q4 2024 vs. Q4 2023)
> - Wrong unit (millions vs. billions)
> - Calculation errors (sum computed incorrectly)

**Algorithm (Pseudo-code)**:
```
ALGORITHM: Numeric QA Verification (QAFactEval-style)

INPUT: numeric_claim, evidence
OUTPUT: NumericVerificationResult

1. EXTRACT numeric components from claim:
   - value: The number itself
   - unit: Currency, percentage, count, etc.
   - entity: What the number describes
   - time_period: When the number applies (if any)
   - derivation_type: DIRECT (quoted) or DERIVED (computed)

2. GENERATE verification questions:
   Q1: "What is the [entity]?"  (e.g., "What is the Q4 2024 revenue?")
   Q2: "What unit is used?"
   Q3: "What time period does this apply to?"

3. FOR each question:
   a. ANSWER from claim (claim_answer)
   b. ANSWER from evidence SEPARATELY (evidence_answer)
   c. NORMALIZE both answers (currency, abbreviations, rounding)
   d. COMPARE normalized answers

4. COMPUTE match_score:
   - Exact match after normalization: 1.0
   - Within rounding tolerance: 0.9
   - Same order of magnitude: 0.5
   - Different values: 0.0

5. IF match_score >= threshold:
   RETURN SUPPORTED with normalized values
   ELSE:
   RETURN CONTRADICTED with discrepancy details
```

**Implementation Pattern (Python)**:
```python
@dataclass
class NumericComponents:
    """Extracted components of a numeric claim."""
    raw_value: str           # "3.2B", "$500M"
    normalized_value: float  # 3200000000, 500000000
    unit: str               # "USD", "users", "percent"
    entity: str             # "revenue", "active users"
    time_period: str | None # "Q4 2024", "FY2023"
    derivation_type: str    # "direct" or "derived"

class NumericClaimVerifier:
    """Stage 6: QA-based verification for numeric claims.

    TOGGLE: enable_numeric_qa_verification
    FALLBACK: When disabled, numeric claims use standard text verification.

    Based on QAFactEval (Fabbri et al., 2022):
    > "QA-based verification catches semantic errors that regex misses,
    > such as wrong entity, wrong time period, or calculation errors."
    """

    # Normalization patterns
    ABBREVIATIONS = {
        'b': 1e9, 'bn': 1e9, 'billion': 1e9,
        'm': 1e6, 'mn': 1e6, 'million': 1e6,
        'k': 1e3, 'thousand': 1e3,
        '%': 0.01, 'percent': 0.01,
    }

    async def verify_numeric_claim(
        self,
        claim: InterleavedClaim,
        llm: LLMClient,
        config: CitationVerificationConfig,
    ) -> NumericVerificationResult:
        """Verify a numeric claim using QA-based approach."""
        if not config.enable_numeric_qa_verification:
            # FALLBACK: Use standard text verification
            return await self._standard_verification(claim, llm)

        # Step 1: Extract numeric components
        claim_components = await self._extract_numeric_components(
            claim.text, llm
        )
        evidence_components = await self._extract_numeric_components(
            claim.evidence.quote, llm
        )

        # Step 2: Generate and answer verification questions
        qa_results = await self._qa_verification(
            claim_components, evidence_components, llm
        )

        # Step 3: Compute match score
        match_score = self._compute_match_score(
            claim_components,
            evidence_components,
            config.numeric_qa_verification.rounding_tolerance,
        )

        # Step 4: Determine verdict
        if match_score >= 0.9:
            verdict = Verdict.SUPPORTED
        elif match_score >= 0.5:
            verdict = Verdict.PARTIAL
        else:
            verdict = Verdict.CONTRADICTED

        return NumericVerificationResult(
            verdict=verdict,
            claim_value=claim_components,
            evidence_value=evidence_components,
            match_score=match_score,
            qa_results=qa_results,
        )

    def _normalize_value(self, raw: str) -> float:
        """Normalize numeric value handling abbreviations and currency."""
        # Remove currency symbols
        clean = re.sub(r'[$€£¥]', '', raw.lower().strip())

        # Find number and multiplier
        match = re.match(r'([\d,.]+)\s*([a-z%]*)', clean)
        if not match:
            raise ValueError(f"Cannot parse numeric value: {raw}")

        number_str = match.group(1).replace(',', '')
        multiplier_str = match.group(2)

        number = float(number_str)
        multiplier = self.ABBREVIATIONS.get(multiplier_str, 1)

        return number * multiplier

    def _compute_match_score(
        self,
        claim: NumericComponents,
        evidence: NumericComponents,
        rounding_tolerance: float,
    ) -> float:
        """Compute how well claim matches evidence."""
        # Check if values match within tolerance
        if claim.normalized_value == 0 and evidence.normalized_value == 0:
            value_match = 1.0
        elif evidence.normalized_value == 0:
            value_match = 0.0
        else:
            ratio = claim.normalized_value / evidence.normalized_value
            if 1 - rounding_tolerance <= ratio <= 1 + rounding_tolerance:
                value_match = 1.0 - abs(1 - ratio) / rounding_tolerance * 0.1
            elif 0.1 <= ratio <= 10:  # Same order of magnitude
                value_match = 0.5
            else:
                value_match = 0.0

        # Check unit match
        unit_match = 1.0 if claim.unit == evidence.unit else 0.5

        # Check entity match (fuzzy)
        entity_match = self._fuzzy_entity_match(claim.entity, evidence.entity)

        # Weighted combination
        return 0.6 * value_match + 0.2 * unit_match + 0.2 * entity_match
```

**Critical Review from SOTA**:
- **QAFactEval** showed LERC scoring outperforms exact match by ~9 points
- **VeriScore** notes that complex claims may not be decomposable
- **FACTSCORE** found numeric claims in later positions are less accurate

**Proposed Enhancement**: Add `answer_comparison_method: exact_match | f1 | lerc` for configurable answer comparison.

---

## Part IV: Key Research Decisions (Updated)

### 1. Claim Decomposition Strategy (Updated)

**Decision**: Use FActScore-style atomic claim extraction, but with **interleaved generation** (ReClaim pattern).

**Rationale**:
- Original approach: Extract claims AFTER generation → lower citation accuracy
- Improved approach: Generate claims constrained by pre-selected evidence → 90% accuracy

### 2. Verification Pipeline (Significantly Updated)

**Decision**: Multi-stage verification with **confidence-based routing** and **isolated verification**.

**Key Changes from Original**:
1. Add confidence pre-classification (HaluGate)
2. Ensure verification happens in ISOLATION (CoVe critical insight)
3. Add citation correction stage (CiteFix)
4. Add numeric QA verification (QAFactEval)

### 3. Evidence Span Selection (Updated)

**Decision**: Pre-select evidence spans BEFORE generation (attribute-first).

**Enhancement**: Rank evidence by relevance and numeric content to prioritize high-stakes claims.

### 4. Numeric Claim Detection (Significantly Updated)

**Decision**: Replace regex+LLM hybrid with **QA-based verification**.

**Why QA is Superior**:
- Regex catches patterns but can't verify correctness
- QA-based approach catches semantic errors (wrong entity, wrong time period)
- Handles derived/computed values through step-by-step verification

### 5. Verification Verdict Categories (Updated)

**Decision**: Four-tier verdict system (EVER-style).

| Verdict | Definition | UI Color | Action |
|---------|------------|----------|--------|
| SUPPORTED | Claim FULLY entailed by evidence | Green | Show normally |
| PARTIAL | Some aspects supported, others unstated | Amber | Note gaps |
| UNSUPPORTED | No evidence basis (not contradicted) | Red | Flag for review |
| CONTRADICTED | Evidence DIRECTLY opposes claim | Purple | Strong warning |

### 6. Abstention Capability (New)

**Decision**: Support abstention when evidence is insufficient.

**Rationale**:
- Sometimes no evidence exists for a claim
- It's better to abstain than to show "unsupported"
- User trusts system more when it admits uncertainty

---

## Part V: Critical Analysis of Proposed Approach

This section provides an honest assessment of our approach against SOTA research, identifying both strengths and gaps with **concrete proposed solutions**.

### Strengths Aligned with SOTA

| Our Approach | SOTA Support | Evidence | Implementation Status |
|--------------|--------------|----------|----------------------|
| Interleaved generation | ReClaim (2024) | 90% vs 60% citation accuracy | Stage 2 |
| Isolated verification | CoVe (2023) | Prevents bias propagation | Stage 4 (core) |
| Confidence-based routing | HaluGate (2025) | 72% overhead savings | Stage 3 |
| Four-tier verdicts | EVER (2023) | Better user decision support | Stage 4 |
| Citation correction | CiteFix (2024) | Fixes 25% of incorrect citations | Stage 5 |
| QA-based numeric verification | QAFactEval (2023) | Catches semantic errors | Stage 6 |

### Identified Gaps and Proposed Solutions

For each gap identified between our approach and SOTA papers, we provide a **concrete mitigation strategy**:

#### Gap 1: No Token-Level Confidence (HaluGate requires logprobs)

**Problem**: HaluGate uses token-level log probabilities for confidence estimation. Databricks serving endpoints don't expose logprobs by default.

**Risk**: Heuristic confidence estimation is less accurate than token-level signals.

**Proposed Solution: Linguistic Confidence Proxy**

Implement rule-based classifier using linguistic signals (detailed in Stage 3):

```
ALGORITHM: Linguistic Confidence Estimation

HIGH_SIGNALS (+0.3 each):
  - Direct quote markers ("" or '')
  - Exact numeric match claim↔evidence
  - Named entity exact match
  - Temporal expression match

LOW_SIGNALS (-0.2 each):
  - Hedging words ("approximately", "may", "might")
  - Comparative claims ("more than", "less than")
  - Derived values ("totaling", "combined")
  - Subjective qualifiers ("significant", "notable")

Configurable via: confidence_estimation_method: linguistic | embedding_similarity | hybrid
```

**Alternative Enhancement**: Add `embedding_similarity` method that computes claim-evidence cosine similarity as confidence proxy.

---

#### Gap 2: Single Correction Method (CiteFix shows method varies by LLM)

**Problem**: CiteFix found different LLMs work optimally with different citation correction strategies. Using one method may be suboptimal.

**Risk**: Correction accuracy varies by 15-21% depending on method-LLM pairing.

**Proposed Solution: Pluggable Correction Strategy Pattern**

```python
class CitationCorrectionStrategy(Protocol):
    """Protocol for pluggable citation correction methods."""
    async def find_best_citation(self, claim, current, pool) -> tuple[Evidence, float]: ...

# Available strategies (configurable via correction_method)
STRATEGIES = {
    "keyword": KeywordCorrector,                    # Pure keyword overlap
    "semantic": SemanticCorrector,                  # Embedding similarity
    "bert_score": BertScoreCorrector,               # Requires trained model
    "llm_based": LLMBasedCorrector,                 # Uses LLM for entailment
    "keyword_semantic_hybrid": HybridCorrector,     # λ=0.8 (CiteFix optimal)
}
```

**A/B Testing Enhancement**: Track correction success rate per method-model pair to optimize configuration.

---

#### Gap 3: No Retrieval Augmentation (EVER shows +20% with retrieval)

**Problem**: EVER paper shows retrieval-augmented verification improves FACTSCORE by ~20%. Our approach only uses pre-retrieved evidence.

**Risk**: May miss supporting evidence that exists but wasn't in initial retrieval.

**Proposed Solution: Optional Verification Retrieval Toggle**

```yaml
# config/app.yaml
citation_verification:
  enable_verification_retrieval: false  # Default off for latency
  verification_retrieval:
    trigger_on_verdicts: [unsupported, nei]  # Only for weak evidence
    max_additional_searches: 2
    search_timeout_seconds: 3
```

When enabled:
1. Claims with UNSUPPORTED or NEI verdict trigger additional web search
2. Search uses claim text as query
3. Retry verification with expanded evidence pool
4. Update verdict if better evidence found

**Trade-off**: Adds 1-3 seconds latency per unsupported claim. Disable for real-time use cases.

---

#### Gap 4: Missing Extrinsic Hallucination Handling (EVER distinguishes intrinsic/extrinsic)

**Problem**: EVER framework distinguishes intrinsic (contradicts evidence) from extrinsic (no evidence exists) hallucinations. Our original four-tier system conflates these.

**Risk**: May over-penalize claims that are true but unverifiable (extrinsic).

**Proposed Solution: NEI (Not Enough Information) Verdict**

Add fifth verdict category distinct from UNSUPPORTED:

```python
class Verdict(str, Enum):
    SUPPORTED = "supported"      # Evidence fully entails claim
    PARTIAL = "partial"          # Some aspects supported
    UNSUPPORTED = "unsupported"  # Evidence exists, doesn't support
    CONTRADICTED = "contradicted" # Evidence opposes claim
    NEI = "nei"                  # No relevant evidence exists

# Decision tree
if not evidence_exists_for_topic:
    return Verdict.NEI  # Extrinsic - no evidence
elif evidence_contradicts:
    return Verdict.CONTRADICTED  # Intrinsic - contradicts
elif not evidence_supports:
    return Verdict.UNSUPPORTED  # Evidence exists but doesn't support
```

**UI Treatment**:
- NEI: Gray indicator, "Insufficient evidence to verify"
- UNSUPPORTED: Red indicator, "Evidence doesn't support claim"

---

#### Gap 5: No Preference Tuning (EVER-PREF improves 17%)

**Problem**: EVER-PREF shows that DPO fine-tuning on verification pairs improves factuality by 17%. Our approach uses prompt-only methods.

**Risk**: Factuality preferences not internalized in model weights.

**Proposed Solution: Data Collection for Future DPO**

Phase 1 (current implementation):
- Log all verification decisions as (claim, evidence, verdict, reasoning) tuples
- Store in MLflow traces for analysis
- Build dataset of human-verified examples

Phase 2 (future enhancement):
- Train DPO adapter on collected preference pairs
- Deploy fine-tuned verification model
- A/B test against prompt-only baseline

```python
# Data collection (enabled by default)
@mlflow.trace(name="verification_decision")
async def log_verification_decision(
    claim: str,
    evidence: str,
    verdict: Verdict,
    reasoning: str,
    human_override: Verdict | None = None,  # For correction tracking
):
    mlflow.log_dict({
        "claim": claim,
        "evidence": evidence,
        "model_verdict": verdict.value,
        "reasoning": reasoning,
        "human_override": human_override.value if human_override else None,
    }, artifact_file="verification_pair.json")
```

---

#### Gap 6: Claims Often Too Complex (VeriScore insight)

**Problem**: VeriScore found that over-decomposition loses meaning. Some claims require context to be verifiable.

**Risk**: Atomic claims may be too granular, losing semantic coherence.

**Proposed Solution: Verifiable Claim Extraction with Context**

Use VeriScore's sliding window approach:

```
ALGORITHM: Contextual Claim Extraction

1. Segment text into candidate claims (sentence boundaries)
2. FOR each candidate:
   a. CHECK if claim is self-contained (describes single event/state)
   b. IF claim references prior context ("this", "that", "such"):
      - Include minimal preceding context
      - Mark as "context-dependent"
   c. CHECK claim_complexity:
      - Too simple: Merge with adjacent claims
      - Too complex: Further decompose
3. RETURN claims with context windows

Configurable via: claim_complexity_threshold (1-5 scale)
```

---

### Risk Assessment Matrix

| Risk | Likelihood | Impact | Mitigation Status |
|------|------------|--------|-------------------|
| Linguistic confidence less accurate than logprobs | High | Medium | Implemented proxy, configurable method |
| Correction method suboptimal for model | Medium | Medium | Pluggable strategy, A/B testing planned |
| Missing evidence due to no retrieval | Medium | High | Optional toggle, default off |
| NEI/UNSUPPORTED confusion | Low | Medium | Five-tier system implemented |
| DPO training data insufficient | High | Low | Data collection in progress |
| Claims too granular/complex | Medium | Medium | Context windows in extraction |

---

## Part VI: Performance Budget (Updated with Realistic Estimates)

### Latency Targets by Stage

| Stage | Target Latency | Realistic Estimate | Strategy | SOTA Reference |
|-------|----------------|-------------------|----------|----------------|
| Evidence Pre-Selection | <800ms | 500-1000ms | Parallel extraction, SIMPLE tier | N/A |
| Interleaved Generation | <1500ms | 1000-2000ms | Stream as generated | ReClaim: ~1.5s |
| Confidence Classification | <200ms | 100-300ms | Batch LLM call or regex | HaluGate: ~200ms |
| High-Conf Verification | <300ms/claim | 200-400ms | Quick NLI check, SIMPLE tier | N/A |
| Medium-Conf Verification | <500ms/claim | 400-700ms | Standard verification | N/A |
| Low-Conf Verification | <1000ms/claim | 800-1500ms | Full CoVe, ANALYTICAL tier | CoVe: ~1s |
| Citation Correction | <500ms | 300-600ms | Only non-SUPPORTED claims | CiteFix: ~0.5s |
| Numeric QA | <800ms/claim | 500-1000ms | Parallel QA execution | QAFactEval: ~0.8s |
| **Total Overhead** | **<2000ms** | **1500-3000ms** | **Streaming hides latency** | |

### Reference: EVER Paper Latency Data

From EVER paper (full biography verification):
- Biography generation + verification: 115-210 seconds total
- ~50 claims per biography
- **Per-claim verification: ~2-4 seconds**

Our targets are aggressive but achievable through:
1. **Confidence-based routing**: 72% of claims skip full verification
2. **Parallelization**: Verify multiple claims concurrently
3. **Streaming**: Hide verification latency behind generation

### Performance by Toggle Configuration

| Configuration | Expected Latency | Use Case |
|---------------|------------------|----------|
| All toggles ON | 2500-4000ms | Maximum accuracy |
| Retrieval OFF | 1500-2500ms | Standard usage (default) |
| Confidence classification OFF | 3000-5000ms | High-stakes verification |
| Interleaved generation OFF | 2000-3000ms | Creative synthesis |
| Minimal (only isolated verification) | 1000-2000ms | Fast mode |

### Parallelization Strategy

```
Timeline (with parallelization):

T=0ms    ┌─ Evidence Pre-Selection (parallel per source)
         │
T=500ms  └─ All evidence ranked
         │
T=500ms  ┌─ Interleaved Generation (streaming)
         │   ├─ Claim 1 generated → Stream to UI
         │   ├─ Claim 2 generated → Stream to UI
         │   └─ ...continues...
         │
T=2000ms └─ All claims generated
         │
T=2000ms ┌─ Confidence Classification (batch)
         │
T=2200ms └─ Claims routed to verification tiers
         │
T=2200ms ┌─ Verification (parallel by confidence tier)
         │   ├─ HIGH claims: 5 in parallel @ 300ms each
         │   ├─ MEDIUM claims: 3 in parallel @ 500ms each
         │   └─ LOW claims: 2 in parallel @ 1000ms each
         │
T=3200ms └─ All claims verified
         │
T=3200ms ┌─ Citation Correction (parallel, only non-SUPPORTED)
         │
T=3500ms └─ Corrections applied
         │
T=3500ms ┌─ Numeric QA (parallel for numeric claims)
         │
T=4000ms └─ Final assembly, summary emitted

Total wall-clock: ~4000ms (with parallelization)
User-perceived: ~2000ms (due to streaming)
```

---

## Part VII: Per-Stage Evaluation Plan

This section provides a **comprehensive evaluation strategy** that tests each stage/algorithm independently using established public benchmarks. This expanded version includes:

1. **Functional Mapping**: How dataset structures map to pipeline functionality
2. **Metric Intuition**: Why each metric matters for the specific stage
3. **Integration Patterns**: How to generate predictions from our pipeline
4. **Critical Review**: Honest assessment of dataset/metric choices with alternatives

---

### 7.1 Evaluation Philosophy

#### What Are We Actually Testing at Each Stage?

Each stage in the 6-stage pipeline has a **distinct functional purpose**. Evaluation must test that specific function, not general quality:

| Stage | Function | What We Test | What We DON'T Test |
|-------|----------|--------------|-------------------|
| **Stage 1** | Extract citable evidence spans | Span relevance to query | Generation quality |
| **Stage 2** | Generate claims constrained by evidence | Citation correctness at generation time | Verification accuracy |
| **Stage 3** | Predict verification difficulty | Calibration of confidence | Actual correctness |
| **Stage 4** | Verify claims in isolation | Verdict accuracy | Citation selection |
| **Stage 5** | Fix incorrect citations | Correction quality | Verification |
| **Stage 6** | Verify numeric claims via QA | Numeric accuracy | Text claims |

#### Metric Selection Rationale

Metrics are chosen to answer specific questions:

| Question | Metric | Why This Metric |
|----------|--------|-----------------|
| Did we find all relevant evidence? | Recall@K | Missing evidence = missing citations downstream |
| Is the evidence we found actually relevant? | Precision@K | Noise degrades generation quality |
| Are our confidence estimates calibrated? | ECE | 85% confidence should mean 85% correct |
| Do our verdicts match human judgment? | Cohen's Kappa | >0.7 = substantial agreement |
| Did corrections improve quality? | MQLA Δ | Must show net improvement, not harm |

#### Integration Requirements

To evaluate each stage, we must be able to:
1. **Run stages in isolation** (disable downstream stages)
2. **Format outputs** to match benchmark expectations
3. **Map our data structures** to benchmark format
4. **Handle label mismatches** (e.g., FEVER 3-class → our 5-class)

---

### 7.2 Dataset Deep Dive

#### 7.2.1 ALCE (Automatic LLMs' Citation Evaluation)

**Source**: Princeton NLP, EMNLP 2023
**Link**: [GitHub](https://github.com/princeton-nlp/ALCE) | [HuggingFace](https://huggingface.co/datasets/princeton-nlp/ALCE-data)

**Data Structure** (JSONL):
```json
{
  "question": "When did the US break away from England?",
  "docs": [
    {"id": "doc1", "title": "American Revolution", "text": "The Declaration of Independence was signed on July 4, 1776..."},
    {"id": "doc2", "title": "Treaty of Paris", "text": "The Treaty of Paris was signed in 1783..."}
  ],
  "answer": "The US declared independence on July 4, 1776 [1], but the Treaty of Paris in 1783 [2] formally ended the war."
}
```

**Three Subsets**:
| Subset | Size | Question Type | Challenge |
|--------|------|---------------|-----------|
| **ASQA** | ~1,000 | Ambiguous questions | Requires synthesizing multiple perspectives |
| **QAMPARI** | ~1,000 | List-type answers | Requires citing each list item |
| **ELI5** | ~1,000 | Long-form explanations | 50% lack complete citation support in evidence |

**What It Tests**: End-to-end citation quality via NLI-based verification
**Evaluated Dimensions**: Fluency, Correctness, Citation Recall, Citation Precision

**Maps to Our Stages**: Stage 2 (generation), Stage 5 (correction)

**Limitations**:
- Passage-level citations (not span-level like ours)
- NLI-based verification may disagree with our Stage 4 verdicts
- ELI5 subset has inherently incomplete evidence

---

#### 7.2.2 FEVER (Fact Extraction and VERification)

**Source**: Amazon Science, NAACL 2018
**Link**: [fever.ai](https://fever.ai/dataset/fever.html) | [HuggingFace](https://huggingface.co/datasets/fever/fever)

**Data Structure** (JSONL):
```json
{
  "id": 62037,
  "label": "SUPPORTS",
  "claim": "Oliver Reed was a film actor.",
  "evidence": [
    [null, null, "Oliver_Reed", 0],
    [null, null, "Gladiator_-LRB-2000_film-RRB-", 0]
  ]
}
```

**Evidence Format**: `[annotation_id, evidence_id, wikipedia_page, sentence_index]`

**Labels**: SUPPORTS (33%), REFUTES (33%), NOT ENOUGH INFO (33%)

**Scale**: 185,445 claims (largest claim verification dataset)

**What It Tests**: Claim-evidence entailment judgment
**Task Pipeline**: Document retrieval → Sentence selection → Entailment classification

**Maps to Our Stage**: Stage 4 (Isolated Verification)

**Label Mapping to Our 5-Tier System**:
| FEVER Label | Our Verdict | Notes |
|-------------|-------------|-------|
| SUPPORTS | SUPPORTED | Direct mapping |
| REFUTES | CONTRADICTED | Direct mapping |
| NOT ENOUGH INFO | NEI | If `enable_nei_verdict=true` |
| NOT ENOUGH INFO | UNSUPPORTED | If `enable_nei_verdict=false` |
| *(none)* | PARTIAL | No FEVER equivalent; need synthetic |

**Limitations**:
- Wikipedia-style factoids (not research synthesis)
- 3 labels vs our 5 (PARTIAL has no equivalent)
- Evidence pre-selected (doesn't test our Stage 1)

---

#### 7.2.3 FACTS Grounding

**Source**: Google DeepMind, December 2024
**Link**: [Blog](https://deepmind.google/blog/facts-grounding-a-new-benchmark-for-evaluating-the-factuality-of-large-language-models/) | [HuggingFace](https://huggingface.co/datasets/google/FACTS-grounding-public)

**Data Structure**:
```json
{
  "system_instruction": "Answer only based on the provided document. Do not use external knowledge.",
  "user_request": "Summarize the key financial metrics from this annual report.",
  "context_document": "The company reported Q4 revenues of $3.2B, up 15% year-over-year... [up to 32,000 tokens]"
}
```

**Split**: 860 public + 859 private (held out for leaderboard)

**Domains**: Finance, Law, Medicine, Technology, Retail

**Task Types**: Summarization, Q&A, Rewriting (NO creative/math tasks)

**Evaluation Method**: Three LLM judges (Gemini 1.5 Pro, GPT-4o, Claude 3.5 Sonnet)
1. **Eligibility check**: Does response address the request?
2. **Factuality check**: Is response fully grounded in context document?

**Maps to Our Stages**: Stage 1 (evidence selection from long documents), Stage 2 (generation)

**Limitations**:
- No gold evidence spans (only binary factuality)
- Judge-based evaluation (not deterministic)
- Private split requires Kaggle submission

---

#### 7.2.4 RAGTruth

**Source**: ACL 2024
**Link**: [Paper](https://aclanthology.org/2024.acl-long.585/) | [GitHub](https://github.com/ParticleMedia/RAGTruth)

**Data Structure**:
```json
{
  "context": "The restaurant opened in 2018 and has 50 employees...",
  "response": "The restaurant was founded in 2015 and employs about 50 people.",
  "annotations": {
    "case_level": "UNSUPPORTED",
    "word_level": [
      {"span": "founded in 2015", "label": "UNSUPPORTED"},
      {"span": "employs about 50 people", "label": "SUPPORTED"}
    ]
  }
}
```

**Scale**: ~18,000 responses with word-level hallucination annotations

**Tasks**: QA (MS MARCO), Data-to-text (Yelp), Summarization (CNN/DailyMail)

**LLMs Covered**: Llama-2, Mistral, GPT-3.5, GPT-4

**What It Tests**: Word-level hallucination detection in RAG responses

**Maps to Our Stage**: Stage 3 (confidence pre-classification ground truth)

**Key Insight**: Can derive confidence calibration ground truth:
- SUPPORTED spans → should get HIGH confidence
- UNSUPPORTED spans → should get LOW confidence

**Limitations**:
- Word-level, not claim-level annotations
- Need to align spans to our claim decomposition
- No explicit confidence scores (we derive from labels)

---

#### 7.2.5 NumerSense

**Source**: USC INK Lab, EMNLP 2020
**Link**: [Website](https://inklab.usc.edu/NumerSense/) | [GitHub](https://github.com/INK-USC/NumerSense) | [HuggingFace](https://huggingface.co/datasets/INK-USC/numer_sense)

**Data Structure**:
```json
{
  "sentence": "A person has <mask> legs.",
  "target": "two"
}
```

**Scale**: 13.6k probes (10.5k train, 3.1k test)

**Numeric Range**: 0-10 only (masked values)

**Categories**: Objects, Biology, Geometry, Math, Physics, Geography

**What It Tests**: Numeric commonsense reasoning

**Maps to Our Stage**: Stage 6 (Numeric QA Verification)

**Use Case for Us**:
- Test numeric extraction: Can we identify "two" as the answer?
- Test normalization: Does "2" = "two" = "II"?

**Limitations**:
- Commonsense, NOT document-grounded
- Small numeric range (0-10)
- No financial/business numerics

**Better Alternative for Finance**: Fin-Fact (3,369 financial claims)

---

#### 7.2.6 AttrScore

**Source**: OSU NLP Group, EMNLP 2023 Findings
**Link**: [GitHub](https://github.com/OSU-NLP-Group/AttrScore)

**Data Structure**:
```json
{
  "claim": "The movie was released in 2020 and directed by Christopher Nolan.",
  "reference": "Tenet (2020) is a sci-fi film directed by Christopher Nolan.",
  "label": "Attributable"
}
```

**Labels**: Attributable, Extrapolatory, Contradictory

| Label | Definition | Our Mapping |
|-------|------------|-------------|
| **Attributable** | Reference fully supports claim | SUPPORTED |
| **Extrapolatory** | Reference lacks sufficient info | UNSUPPORTED / NEI |
| **Contradictory** | Claim contradicts reference | CONTRADICTED |

**Fine-tuned Models Available**:
- AttrScore-Alpaca (7B)
- AttrScore-FLAN-T5 (3B)

**Maps to Our Stages**: Stage 4 (verification), Stage 5 (correction quality)

**Limitations**:
- 3 classes vs our 5 (no PARTIAL)
- No SUPPORTED sub-levels
- Research paper: "all models struggle with contradictory errors"

---

#### 7.2.7 Evaluation Frameworks

**FActScore** (pip install factscore):
- Decomposes text into atomic facts
- Verifies each fact against Wikipedia (or custom KB)
- Returns % of supported atomic facts
- <2% error vs human judgment
- ~$1/100 sentences with GPT-4

**RAGAS** (pip install ragas):
- Faithfulness: Decomposes answer into claims, verifies against context
- Answer Relevance: Does answer address the question?
- Context Precision/Recall: Quality of retrieved context
- 0.6-0.8 AUROC on hallucination detection

**Vectara HHEM** (pip install vectara-hhem):
- Classification model for hallucination detection
- <50ms latency, runs on CPU
- 1.5x better F1 than GPT-3.5 on RAGTruth

---

### 7.3 Per-Stage Evaluation (Deep Dive)

#### 7.3.1 Stage 1: Evidence Pre-Selection

**What We're Testing**: The ability to extract minimal, relevant evidence spans from source documents BEFORE generation begins.

**Functional Purpose**: Pre-selecting evidence enables attribute-first synthesis (ReClaim pattern), which achieves 90% citation accuracy vs 60% for generate-then-cite.

##### Functional Mapping to FACTS Grounding

```
FACTS Grounding Example:
┌──────────────────────────────────────────────────────────────┐
│ context_document: "The company reported Q4 revenues of $3.2B,│
│                   representing a 15% increase YoY. Operating │
│                   expenses decreased to $1.8B... [20k words]"│
│ user_request: "Summarize the key financial metrics"          │
└──────────────────────────────────────────────────────────────┘
                              ↓
                    OUR STAGE 1 EXTRACTS:
┌──────────────────────────────────────────────────────────────┐
│ RankedEvidence[]:                                            │
│   [1] "Q4 revenues of $3.2B" (relevance: 0.95, has_numeric)  │
│   [2] "15% increase YoY" (relevance: 0.92, has_numeric)      │
│   [3] "Operating expenses decreased to $1.8B" (relevance: 0.88)│
│   ...                                                        │
└──────────────────────────────────────────────────────────────┘
                              ↓
                    EVALUATION QUESTION:
"Did we extract the spans that FACTS Grounding judges would
consider necessary for a factual response?"
```

##### Metric Intuition

| Metric | What It Answers | Why It Matters |
|--------|-----------------|----------------|
| **Recall@K** | "Did we find ALL the important evidence?" | Missing evidence = missing citations in final output. If Recall@10 = 70%, we lose 30% of potential citations. |
| **Precision@K** | "Is the evidence we found actually useful?" | Noise degrades generation. If Precision@10 = 50%, half our evidence pool is distracting. |
| **NDCG** | "Are the BEST evidence spans ranked first?" | Generation uses top-ranked evidence first. Poor ranking = suboptimal citations. |
| **Coverage** | "What % of verifiable facts can we cite?" | Coverage < 100% means some facts will be unsupported. |

##### Integration Pattern

```python
async def evaluate_stage1_with_facts_grounding():
    """
    Evaluate evidence pre-selection using FACTS Grounding.

    Challenge: FACTS Grounding doesn't provide gold evidence spans.
    Solution: Derive gold spans from LLM judge explanations OR
              create synthetic ground truth from document structure.
    """
    from datasets import load_dataset
    facts = load_dataset("google/FACTS-grounding-public")

    results = []
    for example in facts["test"]:
        # Run ONLY Stage 1 (evidence pre-selection)
        config = CitationVerificationConfig(
            enable_evidence_preselection=True,
            enable_interleaved_generation=False,  # STOP HERE
        )

        evidence_spans = await stage1_evidence_preselection(
            sources=[Source(content=example["context_document"])],
            query=example["user_request"],
            config=config.evidence_preselection,
        )

        # Derive gold spans (Option A: Use LLM to identify key facts)
        gold_spans = await extract_key_facts_from_document(
            document=example["context_document"],
            query=example["user_request"],
        )

        # Compute retrieval metrics
        metrics = compute_retrieval_metrics(
            predicted=[e.quote for e in evidence_spans],
            gold=gold_spans,
            k=10,
        )
        results.append(metrics)

    return aggregate_metrics(results)

def compute_retrieval_metrics(predicted: list[str], gold: list[str], k: int) -> dict:
    """Compute Recall@K, Precision@K, F1@K."""
    predicted_k = set(predicted[:k])
    gold_set = set(gold)

    hits = len(predicted_k & gold_set)
    recall = hits / len(gold_set) if gold_set else 0
    precision = hits / len(predicted_k) if predicted_k else 0
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    return {"recall@k": recall, "precision@k": precision, "f1@k": f1}
```

##### Critical Review

| Aspect | Assessment | Mitigation |
|--------|------------|------------|
| ✅ Strength | FACTS Grounding has long documents (32k tokens) matching our use case | - |
| ✅ Strength | Covers finance, law, medicine - our target domains | - |
| ⚠️ Weakness | No gold evidence spans provided | Derive from judge explanations or create synthetic |
| ⚠️ Weakness | Only 860 public examples | Supplement with GaRAGe (35k passages) |
| ⚠️ Weakness | Judge-based evaluation (non-deterministic) | Run multiple times, average results |

**Alternative Dataset**: **GaRAGe** (Amazon Science, ACL 2025)
- 2,366 questions with 35,000+ annotated passages
- Explicit grounding labels (not derived)
- Better for span-level evaluation

---

#### 7.3.2 Stage 2: Interleaved Generation

**What We're Testing**: Reference-first claim generation where each claim is constrained by pre-selected evidence (ReClaim pattern).

**Functional Purpose**: By selecting evidence BEFORE generating each claim, we achieve ~90% citation accuracy vs ~60% for traditional generate-then-cite.

##### Functional Mapping to ALCE

```
ALCE Example (ASQA subset):
┌──────────────────────────────────────────────────────────────┐
│ question: "When did the US break away from England?"         │
│ passages: [                                                  │
│   {"id": "doc1", "text": "The Declaration of Independence    │
│    was signed on July 4, 1776, marking the formal..."},      │
│   {"id": "doc2", "text": "The Treaty of Paris in 1783        │
│    formally ended the Revolutionary War and recognized..."}  │
│ ]                                                            │
└──────────────────────────────────────────────────────────────┘
                              ↓
                    OUR STAGE 2 GENERATES:
┌──────────────────────────────────────────────────────────────┐
│ InterleavedClaim[]:                                          │
│   [1] claim: "The US declared independence on July 4, 1776"  │
│       evidence: doc1, span: "July 4, 1776"                   │
│   [2] claim: "The Treaty of Paris in 1783 formally ended..." │
│       evidence: doc2, span: "Treaty of Paris in 1783"        │
└──────────────────────────────────────────────────────────────┘
                              ↓
                    ALCE EVALUATION:
"For each claim, does the cited passage actually support it?"
Citation Recall: 2/2 = 100% (all claims have citations)
Citation Precision: 2/2 = 100% (all citations are correct)
```

##### Metric Intuition

| Metric | What It Answers | Why It Matters |
|--------|-----------------|----------------|
| **Citation Recall** | "Does every claim have at least one valid citation?" | Recall < 100% means some claims are ungrounded. Users can't verify them. |
| **Citation Precision** | "Are the citations we provide actually correct?" | Low precision = misleading users. Worse than no citation. |
| **Citation F1** | "Balanced measure of recall and precision" | Harmonic mean prevents gaming by over/under-citing. |
| **MAUVE** | "Does the text still read naturally?" | Citation constraints may harm fluency. MAUVE detects this. |

##### Integration Pattern

```python
async def evaluate_stage2_with_alce():
    """
    Evaluate interleaved generation using ALCE benchmark.

    Key insight: ALCE uses passage-level citations [1][2][3].
    We use span-level citations. Need format conversion.
    """
    from datasets import load_dataset
    alce = load_dataset("princeton-nlp/ALCE-data", "asqa")

    results = []
    for example in alce["test"]:
        # Run Stages 1-2 (evidence selection + interleaved generation)
        config = CitationVerificationConfig(
            enable_evidence_preselection=True,
            enable_interleaved_generation=True,
            enable_confidence_classification=False,  # STOP HERE
        )

        # Convert ALCE passages to our Source format
        sources = [
            Source(id=p["id"], content=p["text"], title=p.get("title", ""))
            for p in example["docs"]
        ]

        # Run pipeline
        claims = await run_stages_1_2(
            query=example["question"],
            sources=sources,
            config=config,
        )

        # Convert to ALCE format: "[1] Statement. [2][3] Statement."
        alce_response = format_claims_as_alce(claims, sources)

        # Run ALCE's automatic evaluator
        # (uses NLI to check if citations support claims)
        metrics = await run_alce_eval(
            prediction=alce_response,
            gold=example,
            dataset="asqa",
        )
        results.append(metrics)

    return aggregate_metrics(results)

def format_claims_as_alce(claims: list[InterleavedClaim], sources: list[Source]) -> str:
    """Convert our claims to ALCE format with [n] citation markers."""
    output = []
    for claim in claims:
        # Find source index for citation marker
        source_idx = next(
            i for i, s in enumerate(sources)
            if s.id == claim.evidence.source_id
        )
        output.append(f"[{source_idx + 1}] {claim.text}")
    return " ".join(output)
```

##### Critical Review

| Aspect | Assessment | Mitigation |
|--------|------------|------------|
| ✅ Strength | ALCE is the gold standard for citation evaluation (EMNLP 2023) | - |
| ✅ Strength | Automatic NLI-based evaluation (scalable) | - |
| ✅ Strength | Three diverse subsets (ASQA, QAMPARI, ELI5) | - |
| ⚠️ Weakness | Passage-level citations (we use span-level) | Map spans to passage IDs |
| ⚠️ Weakness | ALCE's NLI may disagree with our Stage 4 verdicts | Accept as different judgment; track correlation |
| ⚠️ Weakness | ELI5 subset has 50% incomplete evidence | Focus on ASQA/QAMPARI for Stage 2 |

**Alternative**: **AttrScore** for 3-way classification (Attributable/Extrapolatory/Contradictory)

---

#### 7.3.3 Stage 3: Confidence Pre-Classification

**What We're Testing**: The ability to predict which claims need full verification (LOW confidence) vs quick verification (HIGH confidence).

**Functional Purpose**: HaluGate (2025) shows confidence-based routing saves **72% verification overhead** by skipping full CoVe for high-confidence claims.

##### Functional Mapping to RAGTruth

```
RAGTruth Example:
┌──────────────────────────────────────────────────────────────┐
│ context: "The restaurant was founded in 2018 by chef Maria  │
│           Garcia and currently employs 45 staff members."    │
│                                                              │
│ response: "The restaurant was established in 2015 and has   │
│            about 50 employees."                              │
│                                                              │
│ word_level_annotations: [                                    │
│   {span: "established in 2015", label: "UNSUPPORTED"},       │
│   {span: "about 50 employees", label: "SUPPORTED"}           │
│ ]                                                            │
└──────────────────────────────────────────────────────────────┘
                              ↓
                    OUR STAGE 3 CLASSIFIES:
┌──────────────────────────────────────────────────────────────┐
│ ClassifiedClaim[]:                                           │
│   [1] "established in 2015" → LOW confidence (0.35)          │
│       Reason: No quote match, hedging detected               │
│   [2] "about 50 employees" → MEDIUM confidence (0.65)        │
│       Reason: Close numeric match (50 vs 45)                 │
└──────────────────────────────────────────────────────────────┘
                              ↓
                    EVALUATION:
"Do our confidence scores correlate with RAGTruth labels?"
- LOW confidence claims should be UNSUPPORTED in RAGTruth
- HIGH confidence claims should be SUPPORTED in RAGTruth
```

##### Metric Intuition

| Metric | What It Answers | Why It Matters |
|--------|-----------------|----------------|
| **ECE** (Expected Calibration Error) | "When we say 85% confident, are we right 85% of the time?" | Poor calibration = unreliable routing. Users can't trust confidence displays. |
| **AUROC** | "Can we distinguish easy claims from hard claims?" | If AUROC ~0.5, confidence is random; no routing benefit. |
| **Brier Score** | "How accurate are probability estimates overall?" | Lower = better probabilistic predictions. |
| **Overhead Savings** | "What % of claims skip full verification?" | Target: 72% (HaluGate). Too low = no efficiency gain. |

##### Integration Pattern

```python
async def evaluate_stage3_calibration():
    """
    Evaluate confidence calibration using Stage 4 outcomes as ground truth.

    Key insight: We don't have explicit confidence labels.
    Ground truth = verification outcome:
    - SUPPORTED → should be HIGH confidence
    - Others → should be LOW confidence
    """
    # Step 1: Run full pipeline on test queries
    all_claims = []
    for query in test_queries:
        claims = await run_full_pipeline(query)
        for claim in claims:
            all_claims.append({
                'confidence_score': claim.confidence_score,
                'confidence_level': claim.confidence_level,
                'actual_verdict': claim.verification_result.verdict,
            })

    # Step 2: Define ground truth
    # SUPPORTED = "easy" (should be HIGH confidence)
    # Others = "hard" (should be LOW confidence)
    ground_truth = [
        1 if c['actual_verdict'] == Verdict.SUPPORTED else 0
        for c in all_claims
    ]
    predictions = [c['confidence_score'] for c in all_claims]

    # Step 3: Compute calibration metrics
    return {
        'ece': expected_calibration_error(predictions, ground_truth, n_bins=10),
        'auroc': roc_auc_score(ground_truth, predictions),
        'brier': brier_score_loss(ground_truth, predictions),
        'overhead_savings': compute_overhead_savings(all_claims),
    }

def expected_calibration_error(
    confidences: list[float],
    accuracies: list[int],
    n_bins: int = 10
) -> float:
    """
    ECE measures how well confidence matches accuracy.

    Perfect calibration: ECE = 0
    Random confidence: ECE ≈ 0.5
    """
    confidences = np.array(confidences)
    accuracies = np.array(accuracies)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i + 1])
        if mask.sum() > 0:
            bin_confidence = confidences[mask].mean()
            bin_accuracy = accuracies[mask].mean()
            ece += (mask.sum() / len(confidences)) * abs(bin_confidence - bin_accuracy)

    return ece

def compute_overhead_savings(claims: list[dict]) -> float:
    """% of claims routed to quick verification (HIGH confidence)."""
    high_conf = sum(1 for c in claims if c['confidence_level'] == 'HIGH')
    return high_conf / len(claims)
```

##### Critical Review

| Aspect | Assessment | Mitigation |
|--------|------------|------------|
| ✅ Strength | RAGTruth has word-level labels (fine-grained) | - |
| ✅ Strength | 18k examples across QA, summarization, data-to-text | - |
| ⚠️ Weakness | No explicit confidence scores (need to derive) | Use verification outcome as proxy |
| ⚠️ Weakness | Our linguistic proxy differs from HaluGate's logprobs | Accept as approximation; measure effectiveness |
| ⚠️ Weakness | Word-level ≠ claim-level | Aggregate word labels to claim level |

**Alternative**: Build calibration dataset from production traffic with human verification

---

#### 7.3.4 Stage 4: Isolated Verification

**What We're Testing**: CoVe-style verification where claims are checked against evidence IN ISOLATION (no generation context leakage).

**Functional Purpose**: The CoVe paper's key insight: "Verification must happen in isolation to prevent the model from copying its earlier mistakes."

##### Functional Mapping to FEVER

```
FEVER Example:
┌──────────────────────────────────────────────────────────────┐
│ claim: "Oliver Reed was a film actor."                       │
│                                                              │
│ evidence: [                                                  │
│   ["Oliver_Reed", sentence_0]: "Oliver Reed was an English   │
│    actor who appeared in dozens of films..."                 │
│   ["Gladiator_(2000_film)", sentence_0]: "Gladiator is a     │
│    2000 epic historical drama film..."                       │
│ ]                                                            │
│                                                              │
│ label: SUPPORTS                                              │
└──────────────────────────────────────────────────────────────┘
                              ↓
                    OUR STAGE 4 VERIFIES:
┌──────────────────────────────────────────────────────────────┐
│ CRITICAL: Verification context includes ONLY:                │
│   - The claim: "Oliver Reed was a film actor."               │
│   - The evidence quotes                                      │
│                                                              │
│ MUST EXCLUDE (isolation requirement):                        │
│   ❌ Original synthesis reasoning                             │
│   ❌ Other claims from the response                           │
│   ❌ The user's research question                             │
│   ❌ Any generation chain-of-thought                          │
│                                                              │
│ Output: VerificationResult(verdict=SUPPORTED)                │
└──────────────────────────────────────────────────────────────┘
                              ↓
                    EVALUATION:
"Does our SUPPORTED verdict match FEVER's SUPPORTS label?"
```

##### Metric Intuition

| Metric | What It Answers | Why It Matters |
|--------|-----------------|----------------|
| **Verdict Accuracy (5-class)** | "What % of verdicts are correct?" | Overall correctness measure. |
| **Macro-F1** | "Is performance balanced across all verdict classes?" | Prevents majority-class gaming (FEVER is 33/33/33). |
| **Cohen's Kappa** | "Do we agree with human annotators?" | >0.7 = substantial agreement (our target). |
| **SUPPORTED Precision** | "When we say SUPPORTED, are we right?" | False positives are dangerous (claiming support that doesn't exist). |
| **CONTRADICTED Recall** | "Do we catch all contradictions?" | Missing contradictions = safety risk. |

##### Integration Pattern

```python
async def evaluate_stage4_with_fever():
    """
    Evaluate isolated verification using FEVER benchmark.

    Key challenge: FEVER has 3 labels, we have 5.
    Solution: Map bidirectionally and track unmapped cases.
    """
    from datasets import load_dataset
    fever = load_dataset("fever", "v1.0")

    predictions = []
    for example in fever["test"]:
        # Convert FEVER evidence to our format
        evidence_text = await get_wikipedia_sentences(example["evidence"])

        # Run Stage 4 IN ISOLATION
        # CRITICAL: No generation context, no other claims
        result = await isolated_verify(
            claim=example["claim"],
            evidence=evidence_text,
            config=IsolatedVerificationConfig(
                exclude_generation_context=True,  # CRITICAL
                model_tier="analytical",
            )
        )

        # Map our verdict to FEVER labels for comparison
        fever_prediction = map_verdict_to_fever(result.verdict)

        predictions.append({
            'predicted_fever': fever_prediction,
            'actual_fever': example["label"],
            'our_verdict': result.verdict.value,
            'reasoning': result.reasoning,
        })

    return compute_fever_metrics(predictions)

def map_verdict_to_fever(verdict: Verdict) -> str:
    """Map our 5-tier verdict to FEVER 3-class labels."""
    mapping = {
        Verdict.SUPPORTED: "SUPPORTS",
        Verdict.CONTRADICTED: "REFUTES",
        Verdict.NEI: "NOT ENOUGH INFO",
        Verdict.UNSUPPORTED: "NOT ENOUGH INFO",  # Closest match
        Verdict.PARTIAL: "SUPPORTS",  # No FEVER equivalent; treat as partial support
    }
    return mapping[verdict]

def map_fever_to_verdict(label: str) -> Verdict:
    """Map FEVER 3-class to our 5-tier (for reference)."""
    mapping = {
        "SUPPORTS": Verdict.SUPPORTED,
        "REFUTES": Verdict.CONTRADICTED,
        "NOT ENOUGH INFO": Verdict.NEI,
    }
    return mapping[label]
```

##### Isolation Test (Critical Ablation)

```python
async def test_isolation_prevents_bias():
    """
    Ablation study: Verify that ISOLATION actually matters.

    Hypothesis: Including generation context causes the LLM to
    "remember" its earlier claims and confirm them even when
    evidence doesn't support them.
    """
    test_cases = [
        {
            "claim": "The company had $10M revenue in Q4 2024",
            "evidence": "Revenue figures for Q4 2024 were not disclosed.",
            "generation_context": "Based on growth trends, I estimate Q4 revenue at $10M...",
            "expected_isolated": Verdict.NEI,  # No evidence
            "expected_with_context": Verdict.SUPPORTED,  # LLM "remembers" $10M
        },
        # Add more adversarial cases...
    ]

    for case in test_cases:
        # Verification WITH context (BAD - should leak bias)
        with_context = await verify_with_context(
            claim=case["claim"],
            evidence=case["evidence"],
            generation_context=case["generation_context"],
        )

        # Verification in ISOLATION (GOOD - no bias)
        isolated = await isolated_verify(
            claim=case["claim"],
            evidence=case["evidence"],
            # NO generation_context parameter
        )

        print(f"With context: {with_context.verdict} (expected: {case['expected_with_context']})")
        print(f"Isolated: {isolated.verdict} (expected: {case['expected_isolated']})")

        # Assert isolation produces correct verdict
        assert isolated.verdict == case["expected_isolated"], \
            f"Isolation failed: got {isolated.verdict}, expected {case['expected_isolated']}"
```

##### Critical Review

| Aspect | Assessment | Mitigation |
|--------|------------|------------|
| ✅ Strength | FEVER is the gold standard (185k claims, human-annotated) | - |
| ✅ Strength | Balanced labels (33/33/33) prevents majority-class gaming | - |
| ⚠️ Weakness | Wikipedia factoids ≠ research synthesis | Supplement with domain-specific test set |
| ⚠️ Weakness | 3 labels vs our 5 (PARTIAL has no equivalent) | Track unmapped cases; create synthetic PARTIAL set |
| ⚠️ Weakness | Evidence pre-selected (doesn't test our retrieval) | FEVER tests Stage 4 only; use FACTS for Stage 1 |

**Alternative**: **ANLI** (~170k examples) for adversarial robustness testing

---

#### 7.3.5 Stage 5: Citation Correction

**What We're Testing**: Post-processing to fix incorrect citations (CiteFix pattern).

**Functional Purpose**: CiteFix (2024) found that "~80% of unverifiable facts were not hallucinations but citation errors." We need to fix citations, not just verify claims.

##### Functional Mapping to ALCE + Synthetic Errors

```
Evaluation Strategy (Synthetic Error Injection):
┌──────────────────────────────────────────────────────────────┐
│ Step 1: Take ALCE example with known-correct citations       │
│                                                              │
│ Original: "The US declared independence on July 4, 1776 [1]" │
│           where [1] = correct passage about Declaration      │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│ Step 2: Inject synthetic citation error (30% error rate)     │
│                                                              │
│ Corrupted: "The US declared independence on July 4, 1776 [3]"│
│            where [3] = passage about the War of 1812 (WRONG) │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│ Step 3: Run Stage 5 citation correction                      │
│                                                              │
│ Corrected: "The US declared independence on July 4, 1776 [1]"│
│            CiteFix finds [1] is better match than [3]        │
└──────────────────────────────────────────────────────────────┘
                              ↓
                    EVALUATION:
"Did correction restore the original correct citation?"
Correction Accuracy = % of corrupted citations fixed correctly
```

##### Metric Intuition

| Metric | What It Answers | Why It Matters |
|--------|-----------------|----------------|
| **MQLA** (Mean Question-Level Attribution) | "Overall citation quality after correction" | CiteFix's primary metric. Shows +15-21% improvement. |
| **Correction Accuracy** | "When we change a citation, is the new one better?" | Must be >50% to justify running correction. |
| **False Positive Rate** | "Did we break correct citations?" | FPR >5% means correction hurts more than helps. |
| **Correction Rate** | "What % of claims needed correction?" | High rate = upstream generation problem. |

##### Integration Pattern

```python
async def evaluate_stage5_citation_correction():
    """
    Evaluate citation correction using ALCE with synthetic errors.

    Key insight: Real citation errors are hard to label.
    Solution: Inject known errors, then measure recovery.
    """
    from datasets import load_dataset
    alce = load_dataset("princeton-nlp/ALCE-data", "asqa")

    results = []
    for example in alce["test"]:
        # Step 1: Get original (correct) claims with citations
        original_claims = parse_alce_response(example["answer"], example["docs"])

        # Step 2: Inject citation errors (30% rate)
        corrupted_claims = inject_citation_errors(
            claims=original_claims,
            evidence_pool=example["docs"],
            error_rate=0.3,
        )

        # Step 3: Run Stage 5 citation correction
        corrections = await stage5_citation_correction(
            verified_claims=corrupted_claims,
            full_evidence_pool=example["docs"],
            config=CitationCorrectionConfig(
                correction_method=CorrectionMethod.KEYWORD_SEMANTIC_HYBRID,
                lambda_weight=0.8,  # CiteFix optimal
            ),
        )

        # Step 4: Apply corrections and measure improvement
        corrected_claims = apply_corrections(corrupted_claims, corrections)

        pre_quality = compute_citation_quality(corrupted_claims, original_claims)
        post_quality = compute_citation_quality(corrected_claims, original_claims)

        results.append({
            'mqla_before': pre_quality,
            'mqla_after': post_quality,
            'mqla_improvement': post_quality - pre_quality,
            'correction_accuracy': count_correct_fixes(corrections, original_claims),
            'false_positive_rate': count_incorrect_fixes(corrections, original_claims),
            'correction_rate': len([c for c in corrections if c.type != CorrectionType.KEEP]) / len(corrections),
        })

    return aggregate_metrics(results)

def inject_citation_errors(
    claims: list[VerifiedClaim],
    evidence_pool: list[dict],
    error_rate: float = 0.3,
) -> list[VerifiedClaim]:
    """Inject citation errors for controlled evaluation."""
    corrupted = []
    for claim in claims:
        if random.random() < error_rate:
            # Replace with random wrong citation
            wrong_evidence = random.choice([
                e for e in evidence_pool
                if e["id"] != claim.evidence.source_id
            ])
            corrupted.append(claim.with_evidence(wrong_evidence))
        else:
            corrupted.append(claim)
    return corrupted
```

##### Critical Review

| Aspect | Assessment | Mitigation |
|--------|------------|------------|
| ✅ Strength | Synthetic error injection gives controlled evaluation | - |
| ✅ Strength | Can measure exact improvement from correction | - |
| ⚠️ Weakness | Synthetic errors may differ from real citation errors | Sample real errors from production for validation |
| ⚠️ Weakness | CiteFix's optimal λ=0.8 is model-dependent | Test multiple values; tune for our models |
| ⚠️ Weakness | Correction may introduce new errors | Track false positive rate strictly |

**Alternative**: Use **AttrScore** to classify before/after quality (Attributable/Extrapolatory/Contradictory)

---

#### 7.3.6 Stage 6: Numeric QA Verification

**What We're Testing**: QA-based verification specifically for numeric claims, catching semantic errors that text matching misses.

**Functional Purpose**: QAFactEval shows QA-based verification catches errors like "wrong entity" (revenue vs profit) or "wrong time period" (Q4 2024 vs Q4 2023) that pure text matching misses.

##### Functional Mapping to NumerSense (and Custom Test Suite)

```
NumerSense Example:
┌──────────────────────────────────────────────────────────────┐
│ sentence: "A person has <mask> legs."                        │
│ target: "two"                                                │
└──────────────────────────────────────────────────────────────┘
                              ↓
                    OUR STAGE 6 PROCESSES:
┌──────────────────────────────────────────────────────────────┐
│ 1. Fill mask: "A person has two legs."                       │
│ 2. Generate QA:                                              │
│    Q: "How many legs does a person have?"                    │
│    A (from claim): "two"                                     │
│    A (from commonsense): "two"                               │
│ 3. Compare: "two" == "two" → MATCH → SUPPORTED               │
└──────────────────────────────────────────────────────────────┘

Custom Financial Test Case:
┌──────────────────────────────────────────────────────────────┐
│ claim: "Apple's Q4 2024 revenue was $95 billion"             │
│ evidence: "Apple reported Q4 2024 profit of $95B"            │
└──────────────────────────────────────────────────────────────┘
                              ↓
                    OUR STAGE 6 DETECTS:
┌──────────────────────────────────────────────────────────────┐
│ Q1: "What financial metric is $95B?"                         │
│   A (from claim): "revenue"                                  │
│   A (from evidence): "profit"                                │
│   → MISMATCH → CONTRADICTED (wrong entity!)                  │
│                                                              │
│ Text matching would say SUPPORTED (both have "$95B")         │
│ QA catches the semantic error.                               │
└──────────────────────────────────────────────────────────────┘
```

##### Metric Intuition

| Metric | What It Answers | Why It Matters |
|--------|-----------------|----------------|
| **Numeric Accuracy** | "Are numeric values correctly verified?" | Core measure. Text matching baseline ~70%; QA should be ~85%+. |
| **Entity Match** | "Is the number associated with the correct entity?" | "Revenue" vs "Profit" vs "Expenses" - critical distinction. |
| **Unit Match** | "Are units correctly identified?" | "$3.2B" ≠ "€3.2B" - currency matters. |
| **Derivation Tracing** | "Can we trace computed values to inputs?" | "Total: $10M" from "$6M + $4M" - must show work. |

##### Integration Pattern

```python
async def evaluate_stage6_numeric_qa():
    """
    Evaluate numeric QA verification.

    NumerSense tests commonsense; we need document-grounded tests.
    Solution: Custom test suite + NumerSense for normalization.
    """
    # Test Suite: Different numeric verification scenarios
    test_cases = [
        # Case 1: Direct quote (should be SUPPORTED)
        {
            "claim": "Revenue was $3.2B in Q4 2024",
            "evidence": "The company reported Q4 2024 revenues of $3.2 billion",
            "expected_verdict": Verdict.SUPPORTED,
            "expected_qa": {
                "value_match": True,
                "entity_match": True,
                "period_match": True,
            },
        },
        # Case 2: Rounded value (should be SUPPORTED with tolerance)
        {
            "claim": "About 500 employees work there",
            "evidence": "The company employs 498 full-time staff",
            "expected_verdict": Verdict.SUPPORTED,
            "expected_qa": {
                "value_match": True,  # Within 5% tolerance
                "entity_match": True,
            },
        },
        # Case 3: Wrong entity (should be CONTRADICTED)
        {
            "claim": "Revenue was $3.2B",
            "evidence": "Profit was $3.2 billion",
            "expected_verdict": Verdict.CONTRADICTED,
            "expected_qa": {
                "value_match": True,
                "entity_match": False,  # revenue ≠ profit
            },
        },
        # Case 4: Computed sum (should be SUPPORTED)
        {
            "claim": "Total annual revenue was $10M",
            "evidence": "Q1: $2M, Q2: $3M, Q3: $2.5M, Q4: $2.5M",
            "expected_verdict": Verdict.SUPPORTED,
            "expected_qa": {
                "derivation_type": "computed",
                "inputs": ["$2M", "$3M", "$2.5M", "$2.5M"],
                "operation": "sum",
            },
        },
        # Case 5: Wrong time period (should be CONTRADICTED)
        {
            "claim": "2024 revenue was $5B",
            "evidence": "The company reported $5B revenue in fiscal year 2023",
            "expected_verdict": Verdict.CONTRADICTED,
            "expected_qa": {
                "value_match": True,
                "period_match": False,  # 2024 ≠ 2023
            },
        },
    ]

    results = []
    for case in test_cases:
        result = await stage6_numeric_qa_verify(
            claim=case["claim"],
            evidence=case["evidence"],
            config=NumericQAConfig(
                rounding_tolerance=0.05,
                answer_comparison_method="lerc",
            ),
        )

        results.append({
            'predicted': result.verdict,
            'expected': case["expected_verdict"],
            'correct': result.verdict == case["expected_verdict"],
            'qa_pairs': result.questions_answered,
        })

    # Compute metrics
    accuracy = sum(1 for r in results if r['correct']) / len(results)

    return {
        'numeric_accuracy': accuracy,
        'by_case_type': group_by_case_type(results),
    }
```

##### Critical Review

| Aspect | Assessment | Mitigation |
|--------|------------|------------|
| ⚠️ Weakness | NumerSense is commonsense, not document-grounded | Create custom financial/business test suite |
| ⚠️ Weakness | NumerSense only tests 0-10 | Custom suite includes millions/billions |
| ⚠️ Weakness | No standard benchmark for document-grounded numerics | Use Fin-Fact for financial domain |
| ✅ Strength | Custom test suite gives controlled evaluation | - |
| ✅ Strength | Can test specific error types (entity, period, unit) | - |

**Better Alternative for Finance**: **Fin-Fact** (3,369 financial claims with expert annotations)

---

### 7.4 End-to-End Evaluation

#### Integrated Metrics

| Metric | Description | Target | Benchmark Source |
|--------|-------------|--------|------------------|
| **FActScore** | % of atomic facts supported by evidence | >90% | Min et al. EMNLP 2023 |
| **VeriScore** | % of verifiable claims supported | >85% | Song et al. EMNLP 2024 |
| **SAFE Score** | Search-augmented factuality evaluation | >85% | DeepMind LongFact 2024 |
| **Citation F1** | Harmonic mean of precision/recall | >85% | ALCE |
| **ALCE Combined** | Fluency + Correctness + Citation | Top quartile | Princeton NLP |

#### Benchmark Suite Execution

```bash
# Full evaluation suite
make eval-citations-all

# Individual benchmarks
make eval-factscore      # Long-form factuality (FActScore)
make eval-alce           # Citation quality (ALCE)
make eval-fever          # Claim verification (FEVER)
make eval-calibration    # Confidence calibration (internal)

# Individual stage evaluation
make eval-stage1         # Evidence pre-selection
make eval-stage2         # Interleaved generation
make eval-stage3         # Confidence classification
make eval-stage4         # Isolated verification
make eval-stage5         # Citation correction
make eval-stage6         # Numeric QA verification
```

#### FActScore Execution

```python
# Install: pip install factscore

from factscore.factscorer import FactScorer

# Initialize with retrieval + ChatGPT (recommended)
scorer = FactScorer(model_name="retrieval+ChatGPT")

# Evaluate our research synthesis output
topics = ["Apple Inc.", "Microsoft Corporation", ...]  # Entity names
generations = [research_output_1, research_output_2, ...]  # Our outputs

# Compute FActScore
out = scorer.get_score(topics, generations)

print(f"FActScore: {out['score']:.2%}")  # Target: >90%
print(f"Num facts: {out['num_facts_per_response']}")
print(f"Respond rate: {out['respond_ratio']:.2%}")
```

#### ALCE Execution

```bash
# Clone ALCE benchmark
git clone https://github.com/princeton-nlp/ALCE
cd ALCE

# Generate predictions for each subset
python generate.py --dataset asqa --output_file predictions_asqa.json
python generate.py --dataset qampari --output_file predictions_qampari.json
python generate.py --dataset eli5 --output_file predictions_eli5.json

# Evaluate
python eval.py --dataset asqa --pred_file predictions_asqa.json
python eval.py --dataset qampari --pred_file predictions_qampari.json
python eval.py --dataset eli5 --pred_file predictions_eli5.json
```

---

### 7.5 Ablation Study Design

#### Toggle Impact Matrix

| Toggle | Test Method | Primary Dataset | Expected Impact | Statistical Test |
|--------|-------------|-----------------|-----------------|------------------|
| `enable_evidence_preselection` | ON vs OFF | ALCE, FACTS | +10-15% citation precision | McNemar's test |
| `enable_interleaved_generation` | ON vs OFF | ALCE | +30% citation accuracy (60%→90%) | χ² test |
| `enable_confidence_classification` | ON vs OFF | Internal | -72% compute overhead | t-test on latency |
| `enable_citation_correction` | ON vs OFF | ALCE + synthetic | +15-21% MQLA | Paired t-test |
| `enable_numeric_qa_verification` | ON vs OFF | Custom numeric | +10-15% numeric accuracy | McNemar's test |
| `enable_nei_verdict` | ON vs OFF | FEVER | Better extrinsic handling | χ² on verdict dist |

#### Ablation Execution Pattern

```python
async def run_ablation_study(toggle: str, dataset: Dataset) -> dict:
    """Run A/B test for a single toggle."""

    # Configuration A: Toggle ON (treatment)
    config_on = CitationVerificationConfig(**{toggle: True})
    results_on = await run_evaluation(dataset, config_on)

    # Configuration B: Toggle OFF (control)
    config_off = CitationVerificationConfig(**{toggle: False})
    results_off = await run_evaluation(dataset, config_off)

    # Statistical significance test
    from scipy import stats

    # McNemar's test for paired binary outcomes
    contingency = compute_contingency_table(results_on, results_off)
    chi2, p_value = stats.mcnemar(contingency, exact=False)

    return {
        'toggle': toggle,
        'metric_on': results_on['primary_metric'],
        'metric_off': results_off['primary_metric'],
        'delta': results_on['primary_metric'] - results_off['primary_metric'],
        'p_value': p_value,
        'significant': p_value < 0.05,
    }
```

---

### 7.6 Human Evaluation Protocol (High-Level)

#### Sample Size and Annotators

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Sample size | 100 responses per round | Balance between coverage and cost |
| Annotators per response | 2 | Majority vote for disagreement |
| Annotator qualification | Domain expert OR trained evaluator | Ensure quality judgments |
| Evaluation cadence | Weekly (development), Monthly (production) | Catch regressions early |

#### Key Metrics

| Metric | Scale | Target | What It Measures |
|--------|-------|--------|------------------|
| **Fleiss Kappa** | 0-1 | >0.7 | Inter-annotator agreement (substantial) |
| **Citation Correctness** | 1-5 Likert | >4.0 average | Does citation support claim? |
| **Verdict Accuracy** | Binary | >85% | Is verdict correct? |
| **Overall Trust** | 1-5 Likert | >4.0 average | Would you trust this citation? |
| **Numeric Accuracy** | Binary | >90% | Are numbers correct? |

#### Annotation Focus Areas

1. **Citation-Claim Alignment**: "Does the cited evidence actually support this specific claim?"
2. **Verdict Correctness**: "Is the verdict (SUPPORTED/PARTIAL/etc.) appropriate?"
3. **Numeric Tracing**: "Can you verify the numeric value from the cited source?"
4. **Evidence Sufficiency**: "Is there enough context in the citation to understand the claim?"

#### Disagreement Resolution

```
If annotators disagree:
1. Record both judgments
2. Third annotator breaks tie
3. Track disagreement rate by claim type
4. High disagreement → refine guidelines or claim is ambiguous
```

## Part VIII: References

### Foundational (2022-2023)

1. **FActScore** - Min et al., EMNLP 2023 - [Paper](https://arxiv.org/abs/2305.14251) | [GitHub](https://github.com/shmsw25/FActScore)
2. **Chain-of-Verification (CoVe)** - Dhuliawala et al., 2023 - [Paper](https://arxiv.org/abs/2309.11495)
3. **QAFactEval** - Fabbri et al., 2022 - QA-based consistency checking
4. **RARR** - Gao et al., 2023 - Research and revision pipeline
5. **LLM-Augmenter** - Peng et al., 2023 - Iterative RAG with feedback
6. **RAGAS** - Es et al., 2023 - Reference-free RAG evaluation

### Breakthrough (2024-2025)

7. **Semantic Entropy** - Nature 2024 - [Paper](https://www.nature.com/articles/s41586-024-07421-0)
8. **HaluGate** - vLLM 2025 - [Blog](https://blog.vllm.ai/2025/12/14/halugate.html)
9. **ReClaim** - 2024 - [Paper](https://arxiv.org/abs/2407.01796)
10. **CiteFix** - 2024 - [Paper](https://arxiv.org/html/2504.15629v2)
11. **SAFE/LongFact** - DeepMind 2024 - [GitHub](https://github.com/google-deepmind/long-form-factuality)
12. **VeriScore/VeriFastScore** - 2024 - [Paper](https://arxiv.org/html/2406.19276)
13. **EVER** - 2023 - External verification with hallucination types
14. **Google Cloud Grounding** - [Docs](https://cloud.google.com/generative-ai-app-builder/docs/check-grounding)

---

## Appendix A: Original Approach Gaps Analysis

### Gap 1: Verification Not Isolated

**Original**: Verification happened in same context as generation
**Problem**: LLM can "copy its earlier mistakes" (CoVe paper)
**Fix**: Isolated verification with no access to generation context

### Gap 2: Generate-Then-Verify Pattern

**Original**: Generate text → Extract claims → Verify
**Problem**: Only ~60% citation accuracy
**Fix**: Interleaved reference→claim generation (ReClaim) → ~90% accuracy

### Gap 3: Trust Initial Citations

**Original**: Citations assumed correct after generation
**Problem**: "Incorrect citations outnumber hallucinations" (CiteFix)
**Fix**: Post-process citation correction stage

### Gap 4: Three-Tier Verdicts

**Original**: Supported / Partial / Unsupported
**Problem**: Conflates "no evidence" with "contradictory evidence"
**Fix**: Add CONTRADICTED verdict (EVER framework)

### Gap 5: Uniform Verification Effort

**Original**: All claims verified with same effort
**Problem**: Wastes compute on high-confidence claims
**Fix**: Confidence pre-classification (HaluGate) → 72% savings

### Gap 6: Regex-Based Numeric Detection

**Original**: Regex + LLM filter for numeric claims
**Problem**: Catches patterns but not semantic errors
**Fix**: QA-based verification (QAFactEval)

### Gap 7: No Abstention

**Original**: Always assigns verdict
**Problem**: Forces verdict when evidence insufficient
**Fix**: Abstention capability for truly unknowable claims

---

## Appendix B: Implementation Toggles and Fallback Behavior

This appendix provides the complete specification for independently toggleable algorithms, following the two-level toggle architecture.

### Complete Toggle Reference

| Toggle | Default | Stage | Fallback When Disabled |
|--------|---------|-------|------------------------|
| `enable_evidence_preselection` | `true` | 1 | Use full source documents as single spans |
| `enable_interleaved_generation` | `true` | 2 | Generate-then-cite pattern (60% accuracy) |
| `enable_confidence_classification` | `true` | 3 | All claims get LOW confidence (full verification) |
| `enable_citation_correction` | `true` | 5 | Keep initial citations as-is (25% may be incorrect) |
| `enable_numeric_qa_verification` | `true` | 6 | Standard text verification for numeric claims |
| `enable_abstention` | `true` | 4 | Force verdict even with insufficient evidence |
| `enable_nei_verdict` | `true` | 4 | Merge NEI into UNSUPPORTED |
| `enable_verification_retrieval` | `false` | 4 | No additional search for weak evidence |

### Fallback Behavior Specification

#### When `enable_evidence_preselection = false`

```python
class EvidencePreSelector:
    async def select_evidence_spans(self, sources, query, observations):
        if not self.config.enable_evidence_preselection:
            # FALLBACK: Return full documents as single spans
            return [
                RankedEvidence(
                    quote=source.content[:5000],  # Truncate for context limits
                    source_id=source.id,
                    source_title=source.title,
                    source_url=source.url,
                    relevance_score=1.0,  # All equally relevant
                    has_numeric=self._contains_numbers(source.content),
                    position_in_source=(0, len(source.content)),
                )
                for source in sources
            ]
        # ... normal pre-selection logic
```

**Impact**:
- Larger context windows used → higher token costs
- Less precise evidence matching
- Document-level rather than span-level citations

---

#### When `enable_interleaved_generation = false`

```python
class InterleavedSynthesizer:
    async def synthesize(self, query, evidence_pool, llm, config):
        if not config.enable_interleaved_generation:
            # FALLBACK: Traditional generate-then-cite
            full_response = await self._generate_full_response(query, evidence_pool, llm)
            claims = await self._extract_claims(full_response, llm)
            for claim in claims:
                best_evidence = self._find_best_evidence(claim, evidence_pool)
                yield InterleavedClaim(
                    text=claim,
                    evidence=best_evidence,
                    confidence=0.5,  # Lower confidence for post-hoc citation
                )
            return
        # ... normal interleaved logic
```

**Impact**:
- Citation accuracy drops from ~90% to ~60%
- More post-verification failures
- More citation corrections needed

---

#### When `enable_confidence_classification = false`

```python
class LinguisticConfidenceClassifier:
    async def classify_batch(self, claims):
        if not self.config.enable_confidence_classification:
            # FALLBACK: All claims get LOW confidence → full verification
            return [
                ClassifiedClaim(
                    claim=claim,
                    confidence_score=0.3,
                    confidence_level=ConfidenceLevel.LOW,
                )
                for claim in claims
            ]
        # ... normal classification logic
```

**Impact**:
- All claims get full CoVe verification
- 72% efficiency savings lost
- Higher latency (~3x slower verification)

---

#### When `enable_citation_correction = false`

```python
class CitationCorrector:
    async def correct_citations(self, verified_claims, evidence_pool):
        if not self.config.enable_citation_correction:
            # FALLBACK: Keep all citations as-is
            return [
                CitationCorrection(type=CorrectionType.KEEP, claim=claim)
                for claim in verified_claims
            ]
        # ... normal correction logic
```

**Impact**:
- ~25% of citations may be incorrect (per CiteFix)
- REPLACE/REMOVE/ADD_ALTERNATE actions never applied
- Lower overall citation precision

---

#### When `enable_numeric_qa_verification = false`

```python
class NumericClaimVerifier:
    async def verify_numeric_claim(self, claim, llm, config):
        if not config.enable_numeric_qa_verification:
            # FALLBACK: Standard text verification
            return await self._standard_text_verification(claim, llm)
        # ... normal QA logic
```

**Impact**:
- Semantic errors in numbers not caught (wrong entity, time period)
- Normalization not applied (3.2B vs $3,200,000,000)
- Derived values not traced to inputs

---

#### When `enable_abstention = false`

```python
class IsolatedVerifier:
    async def verify(self, claim, llm, config):
        result = await self._core_verification(claim, llm)

        if not config.enable_abstention:
            # FALLBACK: Force a verdict even with insufficient evidence
            if result.verdict == Verdict.NEI:
                return VerificationResult(
                    verdict=Verdict.UNSUPPORTED,
                    reasoning=result.reasoning + " (forced verdict, abstention disabled)",
                )
        return result
```

**Impact**:
- May incorrectly mark unverifiable claims as UNSUPPORTED
- User may think claim is wrong when evidence simply doesn't exist
- Lower trust in verification accuracy

---

#### When `enable_nei_verdict = false`

```python
class Verdict(str, Enum):
    # When NEI disabled, only four verdicts available
    SUPPORTED = "supported"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"
    # NEI = "nei"  # Disabled

class IsolatedVerifier:
    def _determine_verdict(self, output):
        if not self.config.enable_nei_verdict:
            # FALLBACK: Map NEI to UNSUPPORTED
            if self._would_be_nei(output):
                return Verdict.UNSUPPORTED
        # ... normal verdict logic
```

**Impact**:
- Loses distinction between "no evidence" and "evidence doesn't support"
- Simpler three-tier UI (may be desired)
- Less granular user decision support

---

### Configuration Access Pattern

```python
# src/agent/config.py
from functools import lru_cache

@lru_cache(maxsize=1)
def get_citation_verification_config() -> CitationVerificationConfig:
    """Get cached citation verification configuration."""
    app_config = get_app_config()
    return app_config.citation_verification

# Usage in services
class EvidencePreSelector:
    def __init__(self):
        self.config = get_citation_verification_config()

    async def select_evidence_spans(self, ...):
        if not self.config.enable_evidence_preselection:
            return self._fallback_full_documents(...)
```

### Testing Toggle Combinations

```python
# tests/unit/services/citations/test_toggle_combinations.py

@pytest.mark.parametrize("toggles,expected_accuracy,expected_latency", [
    # All on (maximum accuracy, higher latency)
    ({"enable_all": True}, 0.90, 4000),

    # Standard usage
    ({"enable_verification_retrieval": False}, 0.85, 2500),

    # Fast mode (minimum verification)
    ({"enable_confidence_classification": False, "enable_citation_correction": False}, 0.70, 1500),

    # Creative mode (no interleaving)
    ({"enable_interleaved_generation": False}, 0.60, 2000),
])
async def test_toggle_combination(toggles, expected_accuracy, expected_latency):
    config = CitationVerificationConfig(**toggles)
    pipeline = CitationPipeline(config)

    result = await pipeline.verify_claims(test_claims)

    assert result.accuracy >= expected_accuracy * 0.9  # 10% tolerance
    assert result.latency_ms <= expected_latency * 1.2  # 20% tolerance
```

