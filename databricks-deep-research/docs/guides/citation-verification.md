# Citation Verification

> Configure and tune the 7-stage citation verification pipeline.

## Overview
The citation pipeline transforms raw LLM output into verified, evidence-backed claims. Each stage can be individually configured and toggled.

## Enabling Citation Verification
In YAML, set grounding_mode on the synthesizer:
```yaml
- id: synthesize
  type: agent
  config:
    subtype: synthesizer
    grounding_mode: reclaim    # none | classical_lite | reclaim
```

## Grounding Modes
- none: No verification
- classical_lite: Basic citation key mapping
- reclaim: Full 7-stage pipeline

### Soft-warn grounding banner
When the verifier cannot produce real entailment judgments (e.g. evidence was not attached to claims, or the NLI call crashed), the synthesizer ships the LLM-written report prefixed with a `> ⚠️ Grounding warning` banner instead of hard-failing with an "Insufficient Evidence" template. This is controlled by the `CITATION_SOFT_WARN_ENABLED` environment variable (default `true`); set it to `false` to revert to the legacy hard-fail behavior.

## Stage-by-Stage Configuration
For each of the 7 stages, show: purpose, key config fields, example config, when to tune.

### Stage 1: Evidence Pre-selection
```yaml
evidence_preselection:
  max_spans_per_source: 5
  relevance_threshold: 0.3
  chunk_size: 500
  chunk_overlap: 50
```

### Stage 2: Interleaved Generation
```yaml
interleaved_generation:
  max_claims_per_section: 20
  min_evidence_similarity: 0.5
```

### Stage 3: Confidence Classification
```yaml
confidence_classification:
  high_threshold: 0.85
  low_threshold: 0.5
  quote_match_bonus: 0.15
  hedging_word_penalty: 0.1
```

### Stage 4: Isolated Verification
```yaml
isolated_verification:
  enable_nei_verdict: true
  verification_model_tier: bulk_analysis
  quick_verification_tier: bulk_analysis
```

### Stage 5: Citation Correction
```yaml
citation_correction:
  correction_method: keyword_semantic_hybrid  # keyword_only | semantic_only | keyword_semantic_hybrid
  lambda_weight: 0.8
  correction_threshold: 0.6
  allow_alternate_citations: true
```

### Stage 6: Numeric QA
```yaml
numeric_qa_verification:
  rounding_tolerance: 0.05
  answer_comparison_method: f1  # exact_match | f1 | lerc
  require_unit_match: true
  require_entity_match: true
```

### Stage 7: ARE Verification
```yaml
verification_retrieval:
  trigger_on_verdicts: [unsupported, partial]
  max_atomic_facts_per_claim: 5
  max_searches_per_fact: 2
  max_external_urls_per_search: 3
  entailment_threshold: 0.6
  internal_search_threshold: 0.7
  softening_strategy: hedge  # hedge | qualify | parenthetical
  decomposition_timeout_seconds: 10.0
  search_timeout_seconds: 10.0
  crawl_timeout_seconds: 15.0
```

## Toggling Stages
```yaml
citation:
  enable_evidence_preselection: true
  enable_interleaved_generation: true
  enable_confidence_classification: true
  enable_isolated_verification: true
  enable_citation_correction: true
  enable_numeric_qa: false        # Disable for speed
  enable_verification_retrieval: false  # Disable for speed
```

## Performance vs Accuracy
- Fast mode: Only stages 1-3 (evidence + generation + confidence)
- Balanced: Stages 1-5 (add verification + correction)
- Full: All 7 stages (highest accuracy, slowest)

## See Also
- [Citation Pipeline Concept](../concepts/citation-pipeline.md)
- [Citation Config Reference](../reference/citation-config-reference.md)
- [Builtin Agents - Synthesizer](builtin-agents.md#6-synthesizer)
