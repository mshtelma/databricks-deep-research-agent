# Citation Config Reference

> Complete field-by-field reference for CitationConfig and all stage configs.

Source: `databricks_deep_research/citation/config.py`

---

## Enums

### RelevanceMethod

Method for computing evidence relevance scores.

| Value | Description |
|-------|-------------|
| `semantic` | Embedding-based similarity |
| `keyword` | Keyword overlap |
| `hybrid` | Combined semantic + keyword |

### AnswerComparisonMethod

Method for comparing answers in numeric QA verification.

| Value | Description |
|-------|-------------|
| `exact_match` | Exact string match |
| `f1` | Token-level F1 score |
| `lerc` | Learned evaluation metric |

### ConfidenceEstimationMethod

Method for estimating claim confidence levels.

| Value | Description |
|-------|-------------|
| `linguistic` | Detect hedging language, assertion strength |
| `embedding_similarity` | Embedding distance between claim and evidence |
| `hybrid` | Combined linguistic + embedding |

### CorrectionMethod

Method for citation correction.

| Value | Description |
|-------|-------------|
| `keyword_semantic_hybrid` | Combined keyword + semantic matching |
| `keyword_only` | Keyword overlap only |
| `semantic_only` | Embedding similarity only |

### SofteningStrategy

Strategy for softening unverified claims in Stage 7.

| Value | Description |
|-------|-------------|
| `hedge` | Add hedging words ("reportedly", "allegedly") |
| `qualify` | Add qualifying phrases ("it is believed that") |
| `parenthetical` | Add parenthetical markers ("(unverified)") |

### GenerationMode

Generation mode for research reports.

| Value | Verification stages | Description |
|-------|---------------------|-------------|
| `classical` | Skips stages 3-6 | Free-form prose with `[Title](url)` links. Best text quality. |
| `natural` | Full verification | Light-touch `[N]` citations. |
| `strict` | Full verification | Heavy `[N]` constraints. Maximum citations. |

### SynthesisMode

Synthesis approach for report generation.

| Value | Description |
|-------|-------------|
| `interleaved` | Evidence dumped into context, LLM generates with `[N]` markers. |
| `react` | LLM uses tools to retrieve evidence before each factual claim. |

---

## CitationConfig (Top-Level)

Top-level configuration for the 7-stage citation verification pipeline.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `True` | Master toggle for the entire citation pipeline. |
| `max_evidence_chars` | `int` | `3000` (200--10000) | Pipeline-wide cap on evidence quote length, applied at all evidence-truncation sites (evidence selection, claim-generation prompt, single-claim NLI, batch verification, retry verification). Supersedes the deprecated `evidence_preselection.max_span_length`. |
| `synthesis_mode` | `SynthesisMode` | `interleaved` | Synthesis approach: `interleaved` or `react`. |
| `generation_mode` | `GenerationMode` | `strict` | Generation mode: `classical`, `natural`, or `strict`. |
| `react_synthesis` | `ReactSynthesisConfig` | *(defaults)* | Configuration for ReAct-based synthesis mode. |
| `enable_evidence_preselection` | `bool` | `True` | Enable Stage 1: Evidence Pre-Selection. |
| `enable_interleaved_generation` | `bool` | `True` | Enable Stage 2: Interleaved Generation. |
| `enable_confidence_classification` | `bool` | `True` | Enable Stage 3: Confidence Classification. |
| `enable_citation_correction` | `bool` | `True` | Enable Stage 5: Citation Correction. |
| `enable_numeric_qa_verification` | `bool` | `True` | Enable Stage 6: Numeric QA Verification. |
| `enable_verification_retrieval` | `bool` | `False` | Enable Stage 7: ARE Verification Retrieval. |
| `evidence_preselection` | `EvidencePreselectionConfig` | *(defaults)* | Stage 1 config. |
| `interleaved_generation` | `InterleavedGenerationConfig` | *(defaults)* | Stage 2 config. |
| `confidence_classification` | `ConfidenceClassificationConfig` | *(defaults)* | Stage 3 config. |
| `isolated_verification` | `IsolatedVerificationConfig` | *(defaults)* | Stage 4 config. |
| `citation_correction` | `CitationCorrectionConfig` | *(defaults)* | Stage 5 config. |
| `numeric_qa_verification` | `NumericQAVerificationConfig` | *(defaults)* | Stage 6 config. |
| `verification_retrieval` | `VerificationRetrievalConfig` | *(defaults)* | Stage 7 config. |
| `grounding_validation` | `GroundingValidationConfig` | *(defaults)* | Grounding validation for analysis/free blocks. |
| `post_verification` | `PostVerificationConfig` | *(defaults)* | Post-generation verification of structured output. |

> **Note:** There is no explicit `enable_isolated_verification` toggle at the top level. Stage 4 (Isolated Verification) is always available when the pipeline is enabled. Stages with explicit toggles can be individually disabled.

---

## Stage Configs

### EvidencePreselectionConfig

**Stage 1: Evidence Pre-Selection** -- Extract relevant quotes from sources.

| Field | Type | Default | Constraints | Description |
|-------|------|---------|-------------|-------------|
| `max_spans_per_source` | `int` | `10` | 1--50 | Maximum evidence spans to extract per source. |
| `min_span_length` | `int` | `50` | >= 10 | Minimum character length for an evidence span. |
| `max_span_length` | `int` | `500` | >= 50 | Maximum character length for an evidence span. |
| `relevance_threshold` | `float` | `0.3` | 0.0--1.0 | Minimum relevance score to keep a span. |
| `numeric_content_boost` | `float` | `0.2` | 0.0--1.0 | Relevance boost applied to spans containing numeric content. |
| `relevance_computation_method` | `RelevanceMethod` | `hybrid` | -- | Method for computing relevance: `semantic`, `keyword`, or `hybrid`. |
| `chunk_size` | `int` | `8000` | 1000--20000 | Character size for chunking long sources. |
| `chunk_overlap` | `int` | `1000` | 0--5000 | Character overlap between adjacent chunks. |
| `max_chunks_per_source` | `int` | `5` | 1--10 | Maximum chunks to process per source. |

### InterleavedGenerationConfig

**Stage 2: Interleaved Generation** -- Generate claims with `[N]` citations.

| Field | Type | Default | Constraints | Description |
|-------|------|---------|-------------|-------------|
| `max_claims_per_section` | `int` | `10` | 1--50 | Maximum claims to generate per report section. |
| `min_evidence_similarity` | `float` | `0.5` | 0.0--1.0 | Minimum similarity between a claim and its evidence. |
| `retry_on_entailment_failure` | `bool` | `True` | -- | Retry generation when entailment check fails. |
| `max_retries` | `int` | `3` | 0--10 | Maximum retry attempts on entailment failure. |

### ConfidenceClassificationConfig

**Stage 3: Confidence Classification** -- Route claims by confidence level.

| Field | Type | Default | Constraints | Description |
|-------|------|---------|-------------|-------------|
| `high_threshold` | `float` | `0.70` | 0.0--1.0 | Confidence score above which a claim is classified as high-confidence. |
| `low_threshold` | `float` | `0.40` | 0.0--1.0 | Confidence score below which a claim is classified as low-confidence. |
| `quote_match_bonus` | `float` | `0.4` | 0.0--1.0 | Confidence bonus when the claim contains a direct quote match. |
| `hedging_word_penalty` | `float` | `0.2` | 0.0--1.0 | Confidence penalty when hedging words are detected. |
| `estimation_method` | `ConfidenceEstimationMethod` | `linguistic` | -- | Method: `linguistic`, `embedding_similarity`, or `hybrid`. |

### IsolatedVerificationConfig

**Stage 4: Isolated Verification** -- Produce verdicts for each claim.

| Field | Type | Default | Constraints | Description |
|-------|------|---------|-------------|-------------|
| `enable_nei_verdict` | `bool` | `True` | -- | Allow "Not Enough Information" (NEI) verdicts. |
| `verification_model_tier` | `str` | `bulk_analysis` | -- | Model tier for full verification calls. |
| `quick_verification_tier` | `str` | `bulk_analysis` | -- | Model tier for quick verification calls. |

### CitationCorrectionConfig

**Stage 5: Citation Correction** -- Swap citations from existing pool.

| Field | Type | Default | Constraints | Description |
|-------|------|---------|-------------|-------------|
| `correction_method` | `CorrectionMethod` | `keyword_semantic_hybrid` | -- | Method: `keyword_semantic_hybrid`, `keyword_only`, or `semantic_only`. |
| `lambda_weight` | `float` | `0.8` | 0.0--1.0 | Weight for blending keyword and semantic scores (higher = more keyword). |
| `correction_threshold` | `float` | `0.6` | 0.0--1.0 | Minimum score to accept a corrected citation. |
| `allow_alternate_citations` | `bool` | `True` | -- | Allow substituting an alternative source when the original fails. |

### NumericQAVerificationConfig

**Stage 6: Numeric QA Verification** -- Deep verification of numeric claims.

| Field | Type | Default | Constraints | Description |
|-------|------|---------|-------------|-------------|
| `rounding_tolerance` | `float` | `0.05` | 0.0--0.5 | Tolerance for numeric rounding differences (e.g., 0.05 = 5%). |
| `answer_comparison_method` | `AnswerComparisonMethod` | `f1` | -- | Comparison method: `exact_match`, `f1`, or `lerc`. |
| `require_unit_match` | `bool` | `True` | -- | Require measurement units to match (e.g., "million" vs "billion"). |
| `require_entity_match` | `bool` | `True` | -- | Require the named entity to match (e.g., correct company name). |

### VerificationRetrievalConfig

**Stage 7: ARE Verification Retrieval** -- Atomic fact decomposition + search + revision.

Implements the ARE (Atomic fact decomposition-based Retrieval and Editing) pattern. Related papers: [ARE](https://arxiv.org/abs/2410.16708), [FActScore](https://arxiv.org/abs/2305.14251), [SAFE](https://arxiv.org/abs/2403.18802).

| Field | Type | Default | Constraints | Description |
|-------|------|---------|-------------|-------------|
| `trigger_on_verdicts` | `list[str]` | `["unsupported", "partial"]` | -- | Verdicts that trigger verification retrieval. |
| `max_atomic_facts_per_claim` | `int` | `5` | 1--10 | Maximum atomic facts to extract from a single claim. |
| `max_searches_per_fact` | `int` | `2` | 1--5 | Maximum search attempts per atomic fact. |
| `max_external_urls_per_search` | `int` | `3` | 1--10 | Maximum URLs to crawl per external search. |
| `entailment_threshold` | `float` | `0.6` | 0.0--1.0 | Minimum entailment score to accept evidence. |
| `internal_search_threshold` | `float` | `0.7` | 0.0--1.0 | Similarity threshold for internal pool match. |
| `softening_strategy` | `SofteningStrategy` | `hedge` | -- | Strategy for softening unverified facts: `hedge`, `qualify`, or `parenthetical`. |
| `decomposition_timeout_seconds` | `float` | `10.0` | 1.0--60.0 | Timeout for atomic decomposition step. |
| `search_timeout_seconds` | `float` | `10.0` | 1.0--60.0 | Timeout for search step. |
| `crawl_timeout_seconds` | `float` | `15.0` | 1.0--60.0 | Timeout for URL crawling step. |
| `decomposition_tier` | `str` | `bulk_analysis` | -- | Model tier for decomposition LLM calls. |
| `entailment_tier` | `str` | `bulk_analysis` | -- | Model tier for entailment LLM calls. |
| `reconstruction_tier` | `str` | `analytical` | -- | Model tier for claim reconstruction LLM calls. |
| `softening_tier` | `str` | `fast` | -- | Model tier for softening LLM calls. |

### GroundingValidationConfig

Grounding validation for `<analysis>` and `<free>` blocks. Validates that analysis blocks are logically derived from preceding citations and free blocks contain only structural content.

| Field | Type | Default | Constraints | Description |
|-------|------|---------|-------------|-------------|
| `enabled` | `bool` | `True` | -- | Enable grounding validation for analysis blocks. |
| `max_blocks_to_validate` | `int` | `20` | 1--50 | Maximum blocks to validate per report. |
| `min_analysis_length` | `int` | `30` | 10--100 | Minimum character length for an analysis block to be validated. |
| `allow_topic_sentences` | `bool` | `True` | -- | Allow topic sentences without requiring citation support. |
| `max_preceding_citations` | `int` | `10` | 1--20 | Maximum preceding citations to consider for grounding check. |
| `hedging_prefix` | `str` | `"Based on the evidence presented, "` | -- | Prefix added to ungrounded analysis blocks. |

### PostVerificationConfig

Post-generation verification of structured output. Runs stages 4-6 on claims extracted from structured output (e.g., MeetingPrepOutput) without requiring interleaved generation.

| Field | Type | Default | Constraints | Description |
|-------|------|---------|-------------|-------------|
| `enabled` | `bool` | `True` | -- | Enable post-verification of structured output. |
| `max_claims_to_verify` | `int` | `50` | 1--200 | Maximum claims to verify from structured output. |
| `include_stage4_isolation` | `bool` | `True` | -- | Run Stage 4 (Isolated Verification) on extracted claims. |
| `include_stage5_correction` | `bool` | `True` | -- | Run Stage 5 (Citation Correction) on extracted claims. |
| `include_stage6_numeric` | `bool` | `True` | -- | Run Stage 6 (Numeric QA) on extracted claims. |
| `confidence_threshold` | `float` | `0.6` | 0.0--1.0 | Minimum confidence to accept a claim without further verification. |
| `skip_low_priority_claims` | `bool` | `True` | -- | Skip verification for low-priority claims. |

### ReactSynthesisConfig

Configuration for ReAct-based synthesis mode (used when `synthesis_mode = react`).

| Field | Type | Default | Constraints | Description |
|-------|------|---------|-------------|-------------|
| `max_tool_calls` | `int` | `40` | 5--250 | Maximum total tool calls during synthesis. |
| `tool_budget_per_section` | `int` | `10` | 3--50 | Maximum tool calls per report section. |
| `retrieval_window_size` | `int` | `3` | 1--10 | Number of evidence spans to retrieve per tool call. |
| `grounding_threshold` | `float` | `0.6` | 0.0--1.0 | Minimum grounding score to accept a claim. |
| `embedding_high_threshold` | `float` | `0.7` | 0.0--1.0 | Embedding score above which grounding is accepted without LLM judge. |
| `embedding_low_threshold` | `float` | `0.4` | 0.0--1.0 | Embedding score below which grounding is rejected without LLM judge. |
| `use_llm_judge_for_borderline` | `bool` | `True` | -- | Use LLM judge for scores between low and high thresholds. |
| `enable_post_processing` | `bool` | `False` | -- | Enable post-processing of synthesized output. |
| `use_sectioned_synthesis` | `bool` | `False` | -- | Generate report section-by-section instead of all at once. |

---

## Preset Configurations

### Fast (Stages 1-3 only)

Best for low-latency applications where speed matters more than verification depth.

```python
from databricks_deep_research.citation.config import CitationConfig

fast = CitationConfig(
    enable_evidence_preselection=True,
    enable_interleaved_generation=True,
    enable_confidence_classification=True,
    enable_citation_correction=False,
    enable_numeric_qa_verification=False,
    enable_verification_retrieval=False,
)
```

### Balanced (Stages 1-5)

Good default for most use cases. Adds correction but skips expensive numeric QA and ARE retrieval.

```python
balanced = CitationConfig(
    enable_evidence_preselection=True,
    enable_interleaved_generation=True,
    enable_confidence_classification=True,
    enable_citation_correction=True,
    enable_numeric_qa_verification=False,
    enable_verification_retrieval=False,
)
```

### Full (All 7 stages)

Maximum verification depth. Enables ARE retrieval for unsupported claims.

```python
full = CitationConfig(
    enable_evidence_preselection=True,
    enable_interleaved_generation=True,
    enable_confidence_classification=True,
    enable_citation_correction=True,
    enable_numeric_qa_verification=True,
    enable_verification_retrieval=True,
)
```

---

## See Also

- [Citation Pipeline](../concepts/citation-pipeline.md)
- [Citation Verification Guide](../guides/citation-verification.md)
