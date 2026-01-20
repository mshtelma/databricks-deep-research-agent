# 7-Stage Citation Verification Pipeline

## Overview

The citation verification pipeline ensures every factual claim is traced to specific evidence with verification verdicts. It implements patterns from peer-reviewed research: ReClaim, FActScore, SAFE, ARE, CoVe, CiteFix, and QAFactEval.

## Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    7-STAGE CITATION VERIFICATION PIPELINE                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Sources with content/snippets                                               │
│           ↓                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ STAGE 1: EVIDENCE PRE-SELECTION                                         ││
│  │ • Chunk documents (8000 chars, 1000 overlap)                            ││
│  │ • Hybrid keyword + semantic scoring                                     ││
│  │ • Numeric content boost (+0.2)                                          ││
│  │ • Snippet fallback for failed fetches (0.5 relevance)                   ││
│  │ Output: RankedEvidence pool                                             ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│           ↓                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ STAGE 2: INTERLEAVED GENERATION (ReClaim Pattern)                       ││
│  │ • Three modes: strict, natural, classical                               ││
│  │ • LLM generates with evidence pool constraints                          ││
│  │ • Citation markers: [0], [1] → [Arxiv], [Zhipu]                         ││
│  │ • Parse claims from human-readable key version                          ││
│  │ Output: Content + InterleavedClaim[]                                    ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│           ↓                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ STAGE 3: CONFIDENCE CLASSIFICATION (HaluGate-style)                     ││
│  │ • High-confidence indicators: "according to", "states that"             ││
│  │ • Low-confidence indicators: "may", "might", "reportedly"               ││
│  │ • Quote match bonus                                                     ││
│  │ • Routes to quick vs full verification                                  ││
│  │ Output: Confidence score (0.0-1.0)                                      ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│           ↓                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ STAGE 4: ISOLATED VERIFICATION (CoVe Pattern)                           ││
│  │ • Claim verified WITHOUT generation context                             ││
│  │ • Prevents "I generated this so it's true" bias                         ││
│  │ • Verdicts: SUPPORTED / PARTIAL / UNSUPPORTED / CONTRADICTED            ││
│  │ • Quick verification for high-confidence claims                         ││
│  │ Output: VerificationResult with verdict + reasoning                     ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│           ↓                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ STAGE 5: CITATION CORRECTION (CiteFix Pattern)                          ││
│  │ • Hybrid keyword + semantic matching                                    ││
│  │ • Actions: REPLACE / REMOVE / ADD_ALTERNATE                             ││
│  │ • Finds better evidence for unsupported claims                          ││
│  │ Output: Corrected citations or demotion to UNSUPPORTED                  ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│           ↓                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ STAGE 6: NUMERIC QA VERIFICATION (QAFactEval Pattern)                   ││
│  │ • Detect: currency ($3.2B), percentages (25%), counts (1M users)        ││
│  │ • Handle multipliers: T/B/M/K                                           ││
│  │ • QA-based: answer from claim vs answer from evidence                   ││
│  │ • Tolerance for rounding, unit conversions                              ││
│  │ Output: NumericVerification with match/mismatch                         ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│           ↓                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ STAGE 7: ARE-STYLE VERIFICATION RETRIEVAL                               ││
│  │                                                                         ││
│  │  7a. ATOMIC DECOMPOSITION (FActScore)                                   ││
│  │      "OpenAI released GPT-4 in March 2023, achieving 90% on bar exam."  ││
│  │      → "OpenAI released GPT-4."                                         ││
│  │      → "GPT-4 was released in March 2023."                              ││
│  │      → "GPT-4 achieved 90% on the bar exam."                            ││
│  │                                                                         ││
│  │  7b. PER-FACT VERIFICATION                                              ││
│  │      • Internal pool search (BM25 + keyword)                            ││
│  │      • External Brave Search if internal fails                          ││
│  │      • Entailment check (NLI-style scoring 0.0-1.0)                     ││
│  │                                                                         ││
│  │  7c. CLAIM RECONSTRUCTION                                               ││
│  │      • Verified facts: keep with citations                              ││
│  │      • Unverified facts: apply softening strategies                     ││
│  │        - Hedge: "reportedly", "allegedly"                               ││
│  │        - Qualify: "Some evidence suggests..."                           ││
│  │        - Parenthetical: "(unverified)", "(disputed)"                    ││
│  │                                                                         ││
│  │ Output: Revised claims with softened unverified facts                   ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│           ↓                                                                  │
│  Final Report with verified claims, citations, verdicts                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Stage Details

### Stage 1: Evidence Pre-Selection

**File**: `src/services/citation/evidence_selector.py`

**Purpose**: Extract relevant evidence spans from sources BEFORE generation.

**Process**:
1. **Document Chunking**: Split long documents (8000 char chunks, 1000 char overlap)
2. **Content Segmentation**: Parse paragraphs and sentences
3. **Relevance Scoring**: Hybrid keyword + semantic matching
4. **Numeric Boost**: +0.2 relevance for content with statistics
5. **Deduplication**: Merge overlapping spans

**Key Data Structure**:
```python
@dataclass
class RankedEvidence:
    source_url: str
    source_title: str
    quote: str                    # The evidence text
    relevance_score: float        # 0.0-1.0
    section_heading: str | None
    has_numeric_content: bool
    is_snippet_based: bool        # True for Brave snippets (lower confidence)
```

**Configuration**:
```yaml
evidence_preselection:
  max_spans_per_source: 10
  min_span_length: 30
  max_span_length: 500
  numeric_content_boost: 0.2
```

### Stage 2: Interleaved Generation

**File**: `src/services/citation/claim_generator.py`

**Purpose**: Generate synthesis with inline citations constrained by evidence pool.

**Generation Modes**:

| Mode | Description | Citation Style | Verification |
|------|-------------|----------------|--------------|
| **Strict** | Heavy constraints, max citations | `[Arxiv]`, `[Zhipu]` | Full pipeline |
| **Natural** | Balanced quality + citations | `[Arxiv]`, `[Zhipu]` | Full pipeline |
| **Classical** | Free-form prose | `[Title](url)` | None |

**ReClaim Pattern**:
1. LLM receives evidence pool with indices `[0]`, `[1]`, `[2]`
2. LLM generates content with inline markers
3. Markers replaced with human-readable keys: `[0]` → `[Arxiv]`
4. Claims parsed from final content (position tracking)

**Citation Key Generation**:
- Domain-based: `arxiv.org` → `[Arxiv]`
- Title-based: "GLM-4.7 Technical Report" → `[GLM47]`
- Collision handling: `[Arxiv]`, `[Arxiv-2]`, `[Arxiv-3]`

**Output**:
```python
@dataclass
class InterleavedClaim:
    claim_text: str
    position_start: int
    position_end: int
    evidence_index: int
    citation_key: str
    confidence: float
    claim_type: str              # "general" or "numeric"
```

### Stage 3: Confidence Classification

**File**: `src/services/citation/confidence_classifier.py`

**Purpose**: Route claims to appropriate verification depth using linguistic indicators.

**High-Confidence Indicators** (+0.15 each):
- "according to", "states that", "reports that"
- "shows that", "indicates that", "confirms that"
- "demonstrates that", "based on", "the study found"

**Low-Confidence Indicators** (-0.15 each):
- "may", "might", "could", "possibly", "perhaps"
- "likely", "probably", "appears to", "seems to"
- "suggests that", "approximately", "reportedly"

**Routing**:
| Score Range | Verification Type |
|-------------|-------------------|
| ≥ 0.7 | Quick verification (Simple tier) |
| 0.4 - 0.7 | Standard verification (Analytical tier) |
| < 0.4 | Full verification (Analytical tier) |

### Stage 4: Isolated Verification

**File**: `src/services/citation/isolated_verifier.py`

**Purpose**: Verify claims WITHOUT generation context (CoVe pattern).

**Key Innovation**: The verifier receives ONLY:
- The claim text
- The cited evidence

It does NOT receive:
- The full generated report
- Other claims
- Generation context

This prevents "I generated this so it's true" confirmation bias.

**Verification Output**:
```python
@dataclass
class VerificationResult:
    verdict: str                  # SUPPORTED/PARTIAL/UNSUPPORTED/CONTRADICTED
    reasoning: str                # Explanation
    key_match: bool               # Core claim matches evidence
    issues: list[str]             # Specific problems found
```

**Verdicts**:

| Verdict | Meaning |
|---------|---------|
| **SUPPORTED** | Evidence fully entails claim |
| **PARTIAL** | Some aspects supported, missing qualifiers |
| **UNSUPPORTED** | Evidence exists but doesn't support claim |
| **CONTRADICTED** | Evidence refutes claim |

### Stage 5: Citation Correction

**File**: `src/services/citation/citation_corrector.py`

**Purpose**: Fix citations for partially-supported and unsupported claims.

**Correction Actions**:

| Action | Description |
|--------|-------------|
| **REPLACE** | Find better evidence from pool |
| **REMOVE** | No suitable evidence → demote to UNSUPPORTED |
| **ADD_ALTERNATE** | Keep original, add alternate citation |

**Process**:
1. Filter claims where `verdict != SUPPORTED`
2. For each claim, search evidence pool using hybrid matching
3. If better evidence found → REPLACE
4. If no alternative → REMOVE (triggers softening in Stage 7)

### Stage 6: Numeric QA Verification

**File**: `src/services/citation/numeric_verifier.py`

**Purpose**: Verify numeric claims using QA-based evaluation.

**Numeric Detection Patterns**:
- Currency: `$3.2B`, `€1.5 million`, `¥100M`
- Percentages: `25%`, `0.5%`
- Counts: `1M users`, `500K downloads`
- Multipliers: T (trillion), B (billion), M (million), K (thousand)

**QA-Based Verification**:
1. Generate question about the numeric value
2. Answer question from CLAIM
3. Answer question from EVIDENCE
4. Compare answers for semantic match

**Tolerance Handling**:
- Rounding errors (3.2B vs 3.19B)
- Unit conversions ($1,000M vs $1B)
- Approximate language ("about $3B" vs $2.9B)

**Output**:
```python
@dataclass
class NumericVerification:
    claim_value: str
    evidence_value: str
    normalized_claim: float
    normalized_evidence: float
    match: bool
    match_type: str              # "exact", "approximate", "mismatch"
    derivation: str | None       # How value was computed
```

### Stage 7: ARE-Style Verification Retrieval

**Files**:
- `src/services/citation/atomic_decomposer.py`
- `src/services/citation/verification_retriever.py`

**Purpose**: Verify and revise unsupported/partial claims using atomic fact decomposition.

#### Step 7a: Atomic Decomposition

Based on FActScore methodology:

```
Original Claim:
"OpenAI released GPT-4 in March 2023, achieving 90% on the bar exam."

Atomic Facts:
1. "OpenAI released GPT-4."
2. "GPT-4 was released in March 2023."
3. "GPT-4 achieved 90% on the bar exam."
```

**Rules for Atomic Facts**:
1. Each is a single, simple statement
2. Independently verifiable
3. Replace all pronouns with explicit references
4. Make sense without original claim
5. Don't add information not present in claim

**Optimization**: Skip decomposition for short claims (<8 words, likely already atomic)

#### Step 7b: Per-Fact Verification

For each atomic fact:

1. **Internal Pool Search**
   - BM25 ranking over evidence pool
   - Threshold: 0.7 similarity score
   - Returns top 3 matches

2. **External Search** (if internal fails)
   - Generate verification query from atomic fact
   - Brave Search API
   - Web crawl matching results

3. **Entailment Check** (NLI-style)
   - Score 0.0-1.0:
     - 1.0: Direct explicit statement
     - 0.8: Strongly implied
     - 0.6: Partially supports
     - 0.4: Tangentially related
     - 0.2: Same topic, doesn't support
     - 0.0: Contradicts
   - Threshold for verified: ≥ 0.6

#### Step 7c: Claim Reconstruction

**For Verified Facts** (entailment ≥ 0.6):
- Keep as-is
- Add citation if new evidence found

**For Unverified Facts** (entailment < 0.6):
Apply softening strategies:

| Strategy | Example |
|----------|---------|
| **Hedge** | "reportedly", "allegedly", "according to some sources" |
| **Qualify** | "Some evidence suggests that...", "Early reports indicate..." |
| **Parenthetical** | "(unverified)", "(needs citation)", "(disputed)" |

**Reconstruction Logic**:
- ALL verified → claim stands with citations
- ALL unverified → soften entire claim
- MIXED → reconstruct preserving verified, softening unverified

## Generation Modes Comparison

| Aspect | Strict | Natural | Classical |
|--------|--------|---------|-----------|
| Citation density | Maximum | Balanced | Minimal |
| Text quality | Mechanical | Natural | Best |
| Verification | Full 7-stage | Full 7-stage | None |
| Citation style | `[Arxiv]` | `[Arxiv]` | `[Title](url)` |
| Use case | High-stakes research | General use | Quick answers |

## Configuration

```yaml
citation_verification:
  generation_mode: natural      # strict, natural, classical
  enable_evidence_preselection: true
  enable_interleaved_generation: true
  enable_confidence_classification: true
  isolated_verification: true
  enable_citation_correction: true
  enable_numeric_qa_verification: true
  enable_verification_retrieval: true   # Stage 7

  evidence_preselection:
    max_spans_per_source: 10
    min_span_length: 30
    max_span_length: 500
    numeric_content_boost: 0.2

  verification_retrieval:
    enable_verification_retrieval: true
    decomposition_tier: analytical
    softening_strategies:
      - hedge
      - qualify
      - parenthetical
```

## Per-Depth Configuration

Different depths use different verification settings:

| Depth | Generation Mode | Numeric QA | Stage 7 |
|-------|-----------------|------------|---------|
| Light | natural | disabled | disabled |
| Medium | natural | enabled | enabled (1 search) |
| Extended | strict | enabled | enabled (full budget) |

## Post-Verification for Structured Output

When generating **JSON structured output** (e.g., Pydantic models), the full 7-stage pipeline isn't applicable because there are no ReClaim-style citation markers. Instead, post-verification runs stages 4-6 on claims extracted from the structured output.

### When to Use Post-Verification

| Generation Mode | Verification | Output Format | Use Case |
|----------------|--------------|---------------|----------|
| ReClaim | Full 7-stage | Markdown | Standard research reports |
| **Simple** | **Post-verify (4-6)** | **JSON** | **Structured output with verification** |
| Simple | None | JSON | Quick structured output without verification |
| Simple | None | Markdown | Standard synthesis without verification |

### Post-Verification Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. SIMPLE STRUCTURED GENERATION                             │
│    run_structured_synthesizer(state, llm, MySchema)         │
│    → LLM generates JSON with structured_output parameter    │
│    → Pydantic validates schema compliance                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. CLAIM EXTRACTION FROM STRUCTURED OUTPUT                  │
│    StructuredClaimExtractor.extract(output)                 │
│    → Auto-discovers text fields with source_refs            │
│    → Patterns: field + field_source_refs, or source_refs    │
│    → No hardcoded schema knowledge required                 │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. POST-VERIFICATION (Stages 4-6)                           │
│    PostVerifier.verify_claims(claims, evidence_pool)        │
│    → Stage 4: IsolatedVerifier (CoVe pattern)               │
│    → Stage 5: CitationCorrector (CiteFix pattern)           │
│    → Stage 6: NumericVerifier (QAFactEval pattern)          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. APPLY CORRECTIONS TO STRUCTURED OUTPUT                   │
│    → Update source_refs with corrected citations            │
│    → Return verified Pydantic model                         │
└─────────────────────────────────────────────────────────────┘
```

### Claim Extraction Auto-Discovery

The `StructuredClaimExtractor` automatically finds verifiable claims by walking the Pydantic model:

**Pattern 1**: `field` + `field_source_refs`
```python
class MyOutput(BaseModel):
    executive_summary: str
    executive_summary_source_refs: list[str]
```

**Pattern 2**: Object with `source_refs` sibling
```python
class KeyInsight(BaseModel):
    insight: str
    source_refs: list[str]
```

Claims are prioritized by field name (summary, overview = high priority) and text length.

### Configuration

```python
from deep_research.agent.orchestrator import OrchestrationConfig

config = OrchestrationConfig(
    output_format="json",
    output_schema=MyOutputSchema,
    verify_sources=True,                  # Master toggle
    enable_post_verification=True,        # Run stages 4-6
    structured_system_prompt="...",       # Optional custom prompt
    structured_user_prompt="...",         # Optional custom prompt
)
```

YAML configuration:
```yaml
citation_verification:
  post_verification:
    enabled: true
    max_claims_to_verify: 50
    include_stage4_isolation: true   # CoVe pattern
    include_stage5_correction: true  # CiteFix pattern
    include_stage6_numeric: true     # QAFactEval pattern
    confidence_threshold: 0.6
    skip_low_priority_claims: true
```

## Key Files

| File | Lines | Purpose |
|------|-------|---------|
| `pipeline.py` | 2637 | Main orchestration |
| `verification_retriever.py` | 1384 | Stage 7 ARE implementation |
| `atomic_decomposer.py` | 585 | Stage 7 fact decomposition |
| `evidence_selector.py` | 642 | Stage 1 pre-selection |
| `citation_corrector.py` | 527 | Stage 5 correction |
| `claim_generator.py` | 503 | Stage 2 generation |
| `numeric_verifier.py` | 442 | Stage 6 numeric QA |
| `isolated_verifier.py` | 316 | Stage 4 verification |
| `confidence_classifier.py` | 243 | Stage 3 classification |
| `post_verifier.py` | ~300 | Post-verification orchestration |
| `claim_extractor.py` | ~200 | Extract claims from structured output |

## See Also

- [Scientific Foundations](./scientific-foundations.md) - Research papers
- [LLM Interaction](./llm-interaction.md) - Model tier routing
- [Configuration](./configuration.md) - YAML settings
