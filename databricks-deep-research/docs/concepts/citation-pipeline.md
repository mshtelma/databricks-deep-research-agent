# Citation Pipeline

> 7-stage verification pipeline for trustworthy, evidence-backed research output.

## Overview
The citation pipeline transforms raw research output into verified, cited claims. It runs 7 stages sequentially, each adding a layer of verification. Stages can be individually enabled/disabled.

## Why Citation Verification?
LLMs hallucinate citations. They invent source URLs, misattribute quotes, and fabricate numbers. The 7-stage pipeline:
1. Finds real evidence from collected sources
2. Generates claims interleaved with citations
3. Classifies confidence (high/medium/low)
4. Verifies each claim against evidence (NLI)
5. Corrects incorrect citations
6. Deep-verifies numeric claims
7. Decomposes atomic facts and retrieves external evidence

## The 7 Stages

### Stage 1: Evidence Pre-selection
- Selects relevant text spans from collected sources
- Enterprise sources (Genie, Vector Search, Knowledge Assistant) bypass quality filtering
- Sources without content or snippet are excluded; snippet-only sources skip quality evaluation
- Configurable: `max_spans_per_source`, `min_span_length`, `max_span_length`, `relevance_threshold`, `numeric_content_boost`, `relevance_computation_method` (semantic/keyword/hybrid)
- Long sources are chunked (`chunk_size`, `chunk_overlap`, `max_chunks_per_source`)
- Output: `RankedEvidence` list sorted by relevance score

### Stage 2: Interleaved Generation (ReClaim Pattern)
- Generates claims interleaved with citation markers
- Each claim is produced alongside its evidence via streaming
- Previous observations are passed as context
- Configurable: `max_claims_per_section`, `min_evidence_similarity`, `retry_on_entailment_failure`, `max_retries`
- Output: `InterleavedClaim` list (each carries `claim_text`, `claim_type`, `evidence`, `citation_key`, `claim_role`)

### Stage 3: Confidence Classification (HaluGate-style)
- Routes claims by confidence: high (skip verification), medium (quick verify), low (deep verify)
- Deterministic fast-path rules applied first:
  - Analysis claims (`ClaimRole.ANALYSIS`) always route to LOW (full grounding verifier)
  - Free-block claims (`ClaimRole.FREE`) always route to LOW
  - Claims with no evidence route to LOW
  - Exact numeric matches (numbers + temporal context match evidence) route to HIGH
  - Claims with material analysis language route to LOW or MEDIUM depending on evidence match
  - Strong evidence match (score >= 0.82, overlap >= 0.45) routes to HIGH
- Falls back to heuristic classifier when no deterministic rule matches
- Configurable: `high_threshold`, `low_threshold`, `quote_match_bonus`, `hedging_word_penalty`, `estimation_method` (linguistic/embedding_similarity/hybrid)
- Output: Claims with `ConfidenceLevel` assigned (HIGH, MEDIUM, LOW)

### Stage 4: Isolated Verification (NLI)
- Verifies each claim against its cited evidence using NLI (entailment check)
- Verdicts: `supported`, `partial`, `unsupported`, `contradicted`
- High-confidence claims use quick verification; low-confidence claims use full isolation
- Configurable: `enable_nei_verdict`, `verification_model_tier`, `quick_verification_tier`
- Output: Claims with `VerificationVerdict`

### Stage 5: Citation Correction
- For claims with wrong citations, finds better matching evidence from the pool
- Actions: `keep`, `replace`, `remove`, `add_alternate`
- Tracks aggregate metrics via `CorrectionMetrics` (total, kept, replaced, removed, added_alternate, correction_rate)
- Configurable: `correction_method` (keyword_semantic_hybrid/keyword_only/semantic_only), `lambda_weight`, `correction_threshold`, `allow_alternate_citations`
- Output: Claims with corrected `citation_keys`

### Stage 6: Numeric QA Verification
- Deep verification of numeric claims (dates, percentages, dollar amounts)
- Compares extracted numbers against source data using QA-based approach
- Parses `NumericValue` (raw text, normalized value, unit, entity, multiplier)
- Each numeric fact is verified via question-answer comparison (`QAVerificationResult`)
- Configurable: `rounding_tolerance`, `answer_comparison_method` (exact_match/f1/lerc), `require_unit_match`, `require_entity_match`
- Output: Numeric claims with `NumericVerificationResult` (overall_match, derivation_type, confidence)

### Stage 7: ARE Verification (Atomic Retrieval-Enhanced)
- **Stage 7a**: Decompose claims into atomic facts (FActScore / SAFE pattern)
- **Stage 7b**: Search for external evidence, verify via entailment, revise unsupported claims
- Triggers on configurable verdicts (default: `unsupported`, `partial`)
- Softening strategies for unverified facts: `hedge` ("reportedly"), `qualify` ("it is believed that"), `parenthetical` ("(unverified)")
- Scientific basis: [ARE](https://arxiv.org/abs/2410.16708), [FActScore](https://arxiv.org/abs/2305.14251), [SAFE](https://arxiv.org/abs/2403.18802)
- Configurable: `max_atomic_facts_per_claim`, `max_searches_per_fact`, `max_external_urls_per_search`, `entailment_threshold`, `internal_search_threshold`, `softening_strategy`, timeouts, model tiers
- Output: Final verified claims with revisions applied

## Pipeline Flow Diagram
```
Sources + Draft Report
       |
       v
+------------------+
| 1. Evidence      | -> RankedEvidence[]
|    Pre-selection  |
+--------+---------+
         |
         v
+------------------+
| 2. Interleaved   | -> InterleavedClaim[]
|    Generation    |
+--------+---------+
         |
         v
+------------------+
| 3. Confidence    | -> Claims with ConfidenceLevel
|    Classification|
+--------+---------+
         |
         v
+------------------+
| 4. Isolated      | -> Claims with VerificationVerdict
|    Verification  |
+--------+---------+
         |
         v
+------------------+
| 5. Citation      | -> Claims with corrected keys
|    Correction    |
+--------+---------+
         |
         v
+------------------+
| 6. Numeric QA    | -> Numeric claims verified
|    Verification  |
+--------+---------+
         |
         v
+------------------+
| 7. ARE           | -> Final verified output
|    Verification  |
+------------------+
```

## Key Types

All types are defined in `citation/types.py`.

### Enums

| Enum | Values | Purpose |
|------|--------|---------|
| `VerificationVerdict` | `supported`, `partial`, `unsupported`, `contradicted` | Four-tier verdict for a claim against evidence |
| `ConfidenceLevel` | `high`, `medium`, `low` | HaluGate-style confidence levels for verification routing |
| `ClaimRole` | `fact`, `analysis`, `free` | Role of generated content in reclaim mode |
| `CorrectionAction` | `keep`, `replace`, `remove`, `add_alternate` | Types of citation corrections |
| `VerificationMethod` | `entailment`, `numeric_qa`, `grounding`, `structural` | Verification strategy used for a claim |

### Data Classes

| Type | Stage | Description |
|------|-------|-------------|
| `EvidenceInfo` | 1 | Pre-selected evidence span with source URL, quote text, relevance score, offsets |
| `RankedEvidence` | 1 | Evidence span with relevance ranking, source metadata, `is_snippet_based` flag |
| `InterleavedClaim` | 2 | Claim generated with evidence constraint -- carries `claim_text`, `claim_type`, `evidence`, `citation_key`, `claim_role` |
| `ClaimInfo` | 2+ | Atomic claim enriched by all stages -- confidence, verdict, correction, citation keys |
| `ConfidenceResult` | 3 | Confidence level + score + indicators + reasoning |
| `VerificationResult` | 4 | Verdict + reasoning + key match + confidence + abstained flag |
| `CorrectionResult` | 5 | Correction action + original/corrected evidence + reasoning |
| `CorrectionMetrics` | 5 | Aggregate counts (kept, replaced, removed, added_alternate) + `correction_rate` |
| `NumericValue` | 6 | Parsed numeric value with unit, entity, multiplier |
| `QAVerificationResult` | 6 | Single QA comparison (question, claim answer, evidence answer, match) |
| `NumericVerificationResult` | 6 | Complete numeric verification (parsed value, QA results, overall match, confidence) |
| `VerificationSummaryInfo` | All | Summary of all verification results including Stage 7 ARE metrics |
| `ContentQuality` | Pre-1 | Source quality evaluation (score, has_specific_facts, is_paywall, word_count) |

### Pydantic Output Models (LLM structured output)

| Model | Used By | Description |
|-------|---------|-------------|
| `EvidenceSpanOutput` | Stage 1 | Single evidence span extracted by LLM (quote, relevance, has_numeric) |
| `VerificationOutput` | Stage 4 | Isolated verification result (verdict, reasoning, key_match, confidence) |
| `BatchVerificationOutput` | Stage 4 | Batched verification for multiple claims at once |
| `CorrectionDecisionOutput` | Stage 5 | Citation correction decision (action, evidence_index, reasoning) |

## CitationConfig

The top-level `CitationConfig` (in `citation/config.py`) controls all pipeline behavior:

```python
class CitationConfig(BaseModel):
    # Master toggle
    enabled: bool = True

    # Synthesis mode
    synthesis_mode: SynthesisMode       # "interleaved" or "react"
    generation_mode: GenerationMode     # "classical", "natural", or "strict"
    react_synthesis: ReactSynthesisConfig

    # Stage toggles (each can be independently disabled)
    enable_evidence_preselection: bool = True
    enable_interleaved_generation: bool = True
    enable_confidence_classification: bool = True
    enable_citation_correction: bool = True
    enable_numeric_qa_verification: bool = True
    enable_verification_retrieval: bool = False   # Stage 7 off by default

    # Per-stage configuration
    evidence_preselection: EvidencePreselectionConfig
    interleaved_generation: InterleavedGenerationConfig
    confidence_classification: ConfidenceClassificationConfig
    isolated_verification: IsolatedVerificationConfig
    citation_correction: CitationCorrectionConfig
    numeric_qa_verification: NumericQAVerificationConfig
    verification_retrieval: VerificationRetrievalConfig

    # Additional configs
    grounding_validation: GroundingValidationConfig
    post_verification: PostVerificationConfig
```

### Generation Modes

| Mode | Behavior | Verification |
|------|----------|-------------|
| `classical` | Free-form prose with `[Title](url)` links | Skips stages 3-6 |
| `natural` | Light-touch `[N]` citations | Full verification pipeline |
| `strict` | Heavy `[N]` constraints, maximum citations | Full verification pipeline |

### Synthesis Modes

| Mode | Behavior |
|------|----------|
| `interleaved` | Evidence dumped into context, LLM generates with `[N]` markers |
| `react` | LLM uses tools to retrieve evidence before each factual claim |

## Pipeline Class

`CitationVerificationPipeline` accepts stage components via constructor injection:

```python
pipeline = CitationVerificationPipeline(
    llm=llm_client,
    evidence_selector=evidence_selector,       # Stage 1
    claim_generator=claim_generator,           # Stage 2
    confidence_classifier=confidence_classifier, # Stage 3
    isolated_verifier=isolated_verifier,       # Stage 4
    citation_corrector=citation_corrector,     # Stage 5
    numeric_verifier=numeric_verifier,         # Stage 6
    verification_retriever=verification_retriever, # Stage 7 (optional)
    content_quality_evaluator=quality_evaluator,   # Pre-Stage 1 (optional)
    search_client=search_client,               # Stage 7 dependency (optional)
    web_crawler=web_crawler,                   # Stage 7 dependency (optional)
    config=citation_config,                    # Pipeline configuration
)
```

Each stage component follows a protocol interface, making stages independently swappable and testable.

## Events Emitted

The pipeline yields `VerificationEvent` objects during execution:

| Event Type | When | Data |
|-----------|------|------|
| `claim_generated` | Stage 2 produces a claim | Claim text, type, evidence, citation key |
| `claim_verified` | Stage 4 verifies a claim | Verdict, confidence, reasoning, method |
| `citation_corrected` | Stage 5 corrects a citation | Action (keep/replace/remove), original and corrected evidence |
| `numeric_claim_detected` | Stage 6 finds a numeric claim | Claim text, parsed value, verification result |
| `verification_summary` | Pipeline completes | Total claims, supported/partial/unsupported/contradicted counts, rates, Stage 7 metrics |

## See Also
- [Citation Verification Guide](../guides/citation-verification.md)
- [Citation Config Reference](../reference/citation-config-reference.md)
- [Events](events.md)
