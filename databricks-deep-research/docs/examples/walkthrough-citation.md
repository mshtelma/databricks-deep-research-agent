# Walkthrough: Citation Pipeline

> Add 7-stage citation verification to a research workflow.

## Overview

This walkthrough shows how to enable and configure the citation verification pipeline in a research workflow, ensuring every claim is backed by real evidence. The example YAML lives at `examples/citation_pipeline.yaml` and produces a verified research report where every factual sentence carries `[N]` citation markers that map back to specific evidence spans from the collected sources.

The pipeline follows the same four-stage research pattern as the [simple research walkthrough](walkthrough-simple-research.md) -- coordinator, background, plan-and-execute, synthesizer -- but adds citation verification as a post-synthesis step inside the synthesizer node.

---

## How Citation Verification Works

The citation pipeline runs 7 stages sequentially after the synthesizer generates a draft report. Each stage adds a layer of verification. Stages can be individually enabled or disabled (see [Citation Pipeline](../concepts/citation-pipeline.md) for the full conceptual treatment).

1. **Evidence Pre-selection** -- Extract and rank relevant text spans from collected sources. Long sources are chunked; snippets are scored by relevance to the query.
2. **Interleaved Generation** -- Generate claims constrained by the pre-selected evidence. Each claim is produced alongside `[N]` citation markers (the ReClaim pattern).
3. **Confidence Classification** -- Route claims into high/medium/low confidence buckets. High-confidence claims skip deep verification; low-confidence claims go through the full pipeline.
4. **Isolated Verification** -- Run NLI-style entailment checks on each claim against its cited evidence. Produces a verdict: supported, partial, unsupported, or contradicted.
5. **Citation Correction** -- Swap incorrect citations with better-matching evidence from the pool. Can keep, replace, remove, or add alternate citations.
6. **Numeric QA** -- Deep verification for claims containing numbers, percentages, or statistics. Uses QA-based extraction to compare claimed values against source values.
7. **ARE Verification** -- Atomic fact decomposition followed by external search for unsupported or partial claims. Decomposes complex claims into atomic facts, searches for corroborating evidence, and softens unverifiable facts with hedging language.

---

## The Complete YAML

```yaml
id: citation_pipeline
name: Research with Citation Pipeline
description: Research pipeline with reduced iterations, designed for citation verification testing
version: 1
required_inputs: [query]
output_keys: [report]

pools:
  - name: sources
    dedup_key: url
    max_items: 100
  - name: observations
    dedup_content_hash: true
    max_items: 50

root:
  id: main
  type: sequence
  label: Citation Research Pipeline
  children:
    # Step 1: Classify the query
    - id: coordinator
      type: agent
      label: Query Classifier
      config:
        subtype: coordinator
        model_tier: simple
        output_key: coordination

    # Step 2: Quick background search
    - id: background
      type: agent
      label: Background Investigator
      config:
        subtype: background
        model_tier: simple
        output_key: background
        tools: [web_search]
        max_tool_calls: 3

    # Step 3: Research cycle (reduced iterations for faster citation testing)
    - id: research_cycle
      type: plan_and_execute
      label: Research Cycle
      config:
        planner:
          subtype: planner
          model_tier: analytical
          output_key: plan
        items_path: steps
        item_state_key: current_step
        body:
          id: researcher
          type: agent
          label: Researcher
          config:
            subtype: researcher
            model_tier: analytical
            output_key: findings
            pool_inject:
              - pool: observations
                threshold: 0
                max_items: 10
                max_item_chars: 500
            tools: [web_search, web_crawl]
            pool_writes:
              - pool: observations
                extract: findings
              - pool: sources
                extract: sources
            max_tool_calls: 6
        evaluator:
          subtype: reflector
          model_tier: analytical
          output_key: evaluation
        max_iterations: 3
        min_iterations: 1
        max_replan_cycles: 1

    # Step 4: Synthesize report (source for citation verification)
    - id: synthesizer
      type: agent
      label: Report Synthesizer
      config:
        subtype: synthesizer
        model_tier: analytical
        output_key: report
        pool_inject:
          - pool: observations
            threshold: 0
          - pool: sources
            threshold: 0
        pool_tools:
          - observations
          - sources
        max_tool_calls: 5
```

---

## Key Configuration

### Enabling Citation Verification

Citation verification is activated on the **synthesizer** node. There are two ways to turn it on:

**Option A: `grounding_mode` field (preferred)**

```yaml
- id: synthesizer
  type: agent
  label: Report Synthesizer
  config:
    subtype: synthesizer
    grounding_mode: reclaim    # Enables the full 7-stage pipeline
    output_key: report
```

**Option B: `output_schema` flags (for finer control)**

```yaml
- id: synthesizer
  type: agent
  label: Report Synthesizer
  config:
    subtype: synthesizer
    output_key: report
    output_schema:
      synthesis_mode: reclaim
      enable_citation_verification: true
      generation_mode: strict       # strict | natural | classical
      target_word_count: 600
      max_tokens: 2000
      enable_are_retrieval: false   # Toggle Stage 7
```

### Grounding Modes

| Mode | Behavior |
|------|----------|
| `none` | No citation verification. The synthesizer produces free-form prose. |
| `classical_lite` | Lightweight post-hoc citation key mapping. Applies only Stage 1 (evidence pre-selection) and basic key assignment. |
| `reclaim` | Full 7-stage pipeline. Interleaved claim generation with evidence, NLI verification, citation correction, numeric QA, and optional ARE retrieval. |

### Generation Modes

The `generation_mode` inside `output_schema` controls how aggressively the synthesizer cites:

| Mode | Behavior |
|------|----------|
| `classical` | Free-form prose with `[Title](url)` links. Best text quality. Skips verification stages 3--6. |
| `natural` | Light-touch `[N]` citations. Runs full verification. |
| `strict` | Heavy `[N]` constraints. Maximum citations per claim. Default for reclaim mode. |

### Evidence Pool for Verification

The citation pipeline operates on evidence collected in the `sources` and `observations` pools. The synthesizer must have access to both via `pool_inject`:

```yaml
pool_inject:
  - pool: observations
    threshold: 0          # Include all observations
  - pool: sources
    threshold: 0          # Include all sources
```

The pre-selection stage (Stage 1) ranks evidence spans from the sources pool by relevance to the query. Without sources in the pool, the pipeline has no evidence to verify against.

---

## Section-by-Section Annotation

### Metadata

```yaml
id: citation_pipeline
name: Research with Citation Pipeline
description: Research pipeline with reduced iterations, designed for citation verification testing
version: 1
required_inputs: [query]
output_keys: [report]
```

Standard workflow metadata. The `description` notes that iteration counts are intentionally low -- this workflow is tuned for testing citation quality, not exhaustive research.

### Pools

```yaml
pools:
  - name: sources
    dedup_key: url
    max_items: 100
  - name: observations
    dedup_content_hash: true
    max_items: 50
```

**`sources`** -- Every URL discovered by the researcher. Deduplication by URL prevents storing the same page twice. Capped at 100 items, which is more than enough for a 3-iteration research cycle.

**`observations`** -- Distilled findings from each research step. Deduplication by content hash collapses identical facts discovered via different searches. Capped at 50 items to keep the synthesizer context manageable.

Both pools feed into the citation pipeline: the evidence pre-selector (Stage 1) ranks spans from the `sources` pool, and the interleaved generator (Stage 2) uses `observations` as additional context.

### Steps 1--3: Coordinator, Background, Research Cycle

These three steps are identical in structure to the [simple research walkthrough](walkthrough-simple-research.md). The key differences are in the iteration limits:

```yaml
max_iterations: 3       # Reduced from 10
min_iterations: 1       # Reduced from 2
max_replan_cycles: 1    # Reduced from 3
```

Fewer iterations means less research but faster time-to-citation. For testing the citation pipeline, 3 iterations with 6 tool calls each is sufficient to gather enough sources for verification.

The researcher also injects existing observations to avoid duplicate research:

```yaml
pool_inject:
  - pool: observations
    threshold: 0
    max_items: 10
    max_item_chars: 500
```

### Step 4: Synthesizer

```yaml
- id: synthesizer
  type: agent
  label: Report Synthesizer
  config:
    subtype: synthesizer
    model_tier: analytical
    output_key: report
    pool_inject:
      - pool: observations
        threshold: 0
      - pool: sources
        threshold: 0
    pool_tools:
      - observations
      - sources
    max_tool_calls: 5
```

The synthesizer consumes all pools and generates the report. When `grounding_mode` is set to `reclaim` (or `output_schema.synthesis_mode` is `reclaim`), the synthesizer delegates to the citation pipeline after generating its initial draft.

**`pool_tools`** lets the synthesizer search pools on demand via tool calls. With `max_tool_calls: 5`, it can pull specific observations or sources when the injected context is too large for the model window.

---

## Expected Events

A citation-enabled run emits all the standard research events plus citation-specific events during the synthesis phase. Below is the sequence you will see after the synthesizer starts:

```
node_started (synthesizer)
synthesis_started {total_observations: 12, total_sources: 8}

# --- Citation pipeline events ---
claim_generated {claim_index: 0, citation_keys: ["ArXiv"], claim_role: "fact"}
claim_generated {claim_index: 1, citation_keys: ["Nature"], claim_role: "fact"}
claim_generated {claim_index: 2, citation_keys: [], claim_role: "analysis"}
...

claim_verified {claim_index: 0, verdict: "supported", confidence: 0.92}
claim_verified {claim_index: 1, verdict: "partial", confidence: 0.65}
claim_verified {claim_index: 2, verdict: "supported", confidence: 0.88, claim_role: "analysis"}
...

citation_corrected {claim_index: 1, action: "replace", original_key: "Nature", corrected_key: "IEEE"}
...

numeric_claim_detected {claim_index: 5, numeric_value: "37.2%", verification_status: "pending"}
...

verification_summary {
    total_claims: 15,
    verified_claims: 13,
    corrected_citations: 2,
    removed_claims: 0,
    softened_claims: 1,
    overall_confidence: 0.87
}
# --- End citation events ---

agent_output {output_key: "report"}
node_completed (synthesizer)
workflow_completed
```

### Citation Event Types

| Event | When | Key Fields |
|-------|------|------------|
| `ClaimGeneratedEvent` | Stage 2: a claim is produced with evidence | `claim_text`, `claim_index`, `citation_keys`, `claim_role` |
| `ClaimVerifiedEvent` | Stage 4: a claim is verified against evidence | `claim_index`, `verdict`, `confidence`, `verification_method` |
| `CitationCorrectedEvent` | Stage 5: a citation is corrected | `claim_index`, `action` (`keep`/`replace`/`remove`/`add_alternate`), `original_key`, `corrected_key` |
| `NumericClaimDetectedEvent` | Stage 6: a numeric claim is queued for QA | `claim_index`, `numeric_value`, `verification_status` |
| `VerificationSummaryEvent` | After all stages: aggregate statistics | `total_claims`, `verified_claims`, `corrected_citations`, `overall_confidence` |

---

## Performance Tuning

### Disable Stages 6--7 for Faster Execution

Numeric QA (Stage 6) and ARE verification (Stage 7) are the most expensive stages. For most use cases, stages 1--5 provide sufficient verification. Disable the expensive stages via `output_schema`:

```yaml
- id: synthesizer
  type: agent
  config:
    subtype: synthesizer
    grounding_mode: reclaim
    output_schema:
      enable_are_retrieval: false    # Disables Stage 7 (ARE)
```

Stage 6 (Numeric QA) runs automatically only on claims that contain numeric content. If your research domain is non-numeric, it adds negligible overhead.

### Adjust Confidence Thresholds

The confidence classifier (Stage 3) routes claims by confidence level. Raising the `high_threshold` sends more claims through deep verification (more accurate, slower). Lowering it lets more claims skip verification (faster, less thorough).

Default thresholds in `ConfidenceClassificationConfig`:

| Threshold | Default | Effect of raising |
|-----------|---------|-------------------|
| `high_threshold` | 0.70 | Fewer claims skip verification |
| `low_threshold` | 0.40 | Fewer claims get deep (Stage 4) verification |

### Limit Max Spans per Source

Reducing `max_spans_per_source` in the evidence pre-selection config shrinks the evidence pool, making downstream stages faster:

```python
from databricks_deep_research.citation import CitationConfig, EvidencePreselectionConfig

config = CitationConfig(
    evidence_preselection=EvidencePreselectionConfig(
        max_spans_per_source=5,   # Default: 10
        relevance_threshold=0.4,  # Default: 0.3 -- higher = fewer spans
    ),
)
```

### Reduce Research Iterations

The citation pipeline verifies whatever the research cycle produces. Fewer research iterations means fewer sources but faster end-to-end execution. The example YAML already uses reduced limits (`max_iterations: 3`). For very fast testing, drop to `max_iterations: 1`.

---

## Running It

```python
from databricks_deep_research import WorkflowRunner, load_workflow
from openai import AsyncOpenAI

workflow = load_workflow("examples/citation_pipeline.yaml")
runner = WorkflowRunner(workflow, AsyncOpenAI(), model_mapping={
    "simple": "gpt-4o-mini",
    "analytical": "gpt-4o",
    "complex": "gpt-4o",
})

# Run to completion
result = await runner.run(query="What are the economic impacts of carbon taxes?")

# The report contains [N] citation markers
print(result.output)
```

For streaming, iterate over events to track citation progress in real time:

```python
async for event in runner.stream(
    query="What are the economic impacts of carbon taxes?"
):
    if event.event_type == "claim_verified":
        print(f"  Claim {event.claim_index}: {event.verdict} "
              f"(confidence: {event.confidence:.2f})")
    elif event.event_type == "verification_summary":
        print(f"Verification complete: {event.verified_claims}/{event.total_claims} "
              f"claims verified, {event.corrected_citations} citations corrected")
    elif event.event_type == "agent_stream_chunk":
        print(event.chunk, end="", flush=True)

# Final report with citations
print(runner.last_result.output)
```

---

## See Also

- [Citation Pipeline](../concepts/citation-pipeline.md) -- conceptual overview of all 7 stages
- [Citation Verification Guide](../guides/citation-verification.md) -- stage-by-stage configuration and tuning
- [Citation Config Reference](../reference/citation-config-reference.md) -- field-by-field reference for `CitationConfig`
