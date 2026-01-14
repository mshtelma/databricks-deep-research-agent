# Scientific Foundations

## Overview

The Deep Research Agent implements patterns from peer-reviewed research on fact verification, citation generation, and claim decomposition. This document summarizes the key papers and how their methods are applied.

## Implemented Research Patterns

| Pattern | Paper | Use in System |
|---------|-------|---------------|
| **ReClaim** | arXiv:2407.01796 | Stage 2: Interleaved generation with evidence constraints |
| **FActScore** | arXiv:2305.14251 | Stage 7a: Atomic fact decomposition methodology |
| **SAFE** | arXiv:2403.18802 | Multi-step reasoning with search verification |
| **ARE** | arXiv:2410.16708 | Stage 7: Atomic facts > sub-questions for retrieval |
| **CoVe** | arXiv:2309.11495 | Stage 4: Isolated verification without generation context |
| **RARR** | arXiv:2210.08726 | Post-hoc attribution framework |
| **CiteFix** | arXiv:2504.15629 | Stage 5: Hybrid citation correction |
| **QAFactEval** | arXiv:2112.08542 | Stage 6: QA-based numeric verification |

## Paper Summaries

### 1. ReClaim: Reference-Constrained Claim Generation

**Citation**: arXiv:2407.01796

**Key Insight**: Constrain LLM generation with pre-selected evidence to produce verifiable claims.

**Method**:
1. Pre-select relevant evidence spans from sources
2. Provide evidence pool to LLM during generation
3. LLM must cite evidence using index markers `[0]`, `[1]`
4. Every claim is immediately tied to supporting evidence

**Application in System**: Stage 2 (Interleaved Generation)
- Evidence pool created in Stage 1
- LLM generates with inline citations
- Markers replaced with human-readable keys

**Benefit**: Claims are verifiable by construction, not post-hoc.

---

### 2. FActScore: Fine-grained Atomic Evaluation of Factual Precision

**Citation**: arXiv:2305.14251 (EMNLP 2023)

**Key Insight**: Complex claims should be decomposed into atomic facts for accurate verification.

**Method**:
1. Break claims into independent, self-contained atomic facts
2. Each fact is a single, simple statement
3. Replace pronouns with explicit references
4. Verify each fact independently

**Example**:
```
Original: "OpenAI released GPT-4 in March 2023, achieving 90% on the bar exam."

Atomic Facts:
1. "OpenAI released GPT-4."
2. "GPT-4 was released in March 2023."
3. "GPT-4 achieved 90% on the bar exam."
```

**Application in System**: Stage 7a (Atomic Decomposition)
- Complex claims decomposed before verification
- Each atomic fact verified independently
- Partial support is precisely identified

**Benefit**: Granular verification catches partial errors.

---

### 3. SAFE: Multi-Step Reasoning for Factual Accuracy

**Citation**: arXiv:2403.18802 (Google DeepMind)

**Key Insight**: Multi-step reasoning with search verification achieves 72% human agreement.

**Method**:
1. Generate claim and supporting reasoning
2. Search for evidence to verify reasoning steps
3. Evaluate each step independently
4. Aggregate results for final verdict

**Application in System**: Stage 7b (Per-Fact Verification)
- Each atomic fact verified with search
- Internal pool searched first
- External search if internal fails
- Entailment scoring for final verdict

**Benefit**: Structured reasoning improves verification accuracy.

---

### 4. ARE: Atomic fact decomposition-based Retrieval and Editing

**Citation**: arXiv:2410.16708 (October 2024)

**Key Insight**: Atomic facts are better search queries than sub-questions (+8% retrieval accuracy).

**Method**:
1. Decompose claims into atomic facts
2. Use facts directly as search queries (not questions)
3. Retrieve evidence for each fact
4. Edit claims based on verification results

**Comparison with RARR**:
| Approach | Query Type | Retrieval Accuracy |
|----------|------------|-------------------|
| RARR | Sub-questions | Baseline |
| ARE | Atomic facts | +8% improvement |

**Application in System**: Stage 7 (Verification Retrieval)
- Facts used as search queries
- Internal pool search with BM25
- External Brave search if needed
- Claim reconstruction based on results

**Benefit**: More accurate retrieval than question-based approaches.

---

### 5. CoVe: Chain of Verification

**Citation**: arXiv:2309.11495

**Key Insight**: Verify claims in isolation to prevent confirmation bias.

**Problem**: When LLMs verify their own generations, they tend to confirm what they generated ("I generated this so it's true").

**Solution**: Provide verifier with ONLY:
- The claim text
- The cited evidence

Do NOT provide:
- Full generated report
- Generation context
- Other claims

**Application in System**: Stage 4 (Isolated Verification)
- Verifier receives only claim + evidence
- No generation context
- Independent verdict

**Benefit**: Honest verification without generation bias.

---

### 6. RARR: Retrofitting Attribution Using Research and Revision

**Citation**: arXiv:2210.08726 (ACL 2023)

**Key Insight**: Post-hoc attribution can add citations to existing text.

**Method**:
1. Generate sub-questions for each claim
2. Search for evidence to answer questions
3. Attribute claims to found evidence
4. Revise claims if evidence contradicts

**Application in System**: Conceptual foundation for Stage 7
- While ARE replaced sub-questions with atomic facts
- RARR's revision framework informs claim reconstruction
- Post-hoc attribution concept used throughout

**Note**: ARE outperforms RARR because facts are better queries than questions.

---

### 7. CiteFix: Hybrid Citation Correction

**Citation**: arXiv:2504.15629

**Key Insight**: Combine keyword and semantic matching for citation correction.

**Method**:
1. Identify claims with incorrect/missing citations
2. Search evidence pool using:
   - Keyword matching (BM25)
   - Semantic similarity (embeddings)
3. Hybrid scoring for best match
4. Replace, remove, or add citations

**Correction Actions**:
| Action | When Used |
|--------|-----------|
| REPLACE | Better evidence found |
| REMOVE | No suitable evidence |
| ADD_ALTERNATE | Additional support found |

**Application in System**: Stage 5 (Citation Correction)
- Filters claims with `verdict != SUPPORTED`
- Hybrid search for alternatives
- Updates citations or demotes to UNSUPPORTED

**Benefit**: Fixes misattributions without regeneration.

---

### 8. QAFactEval: Question-Answering for Factual Evaluation

**Citation**: arXiv:2112.08542

**Key Insight**: Use QA to verify factual claims, especially numeric ones.

**Method**:
1. Generate question about the claim
2. Answer question from claim text
3. Answer question from evidence text
4. Compare answers for consistency

**Example**:
```
Claim: "Tesla's revenue was $3.2 billion in Q4 2023."
Question: "What was Tesla's Q4 2023 revenue?"

Answer from claim: $3.2 billion
Answer from evidence: $3.19 billion

Result: Match (within tolerance)
```

**Application in System**: Stage 6 (Numeric QA Verification)
- Detect numeric claims (currency, percentages, counts)
- Generate verification questions
- Compare claim and evidence answers
- Handle rounding, unit conversions

**Benefit**: Semantic comparison catches numeric errors.

---

## Key Research Findings Applied

### Finding 1: ARE > RARR for Retrieval

**Paper**: ARE (2024)

**Finding**: Atomic facts are better search queries than sub-questions (+8% retrieval accuracy).

**Why**: Facts are direct statements that match source text. Questions introduce semantic transformation that loses precision.

**Application**: Stage 7 uses atomic facts, not sub-questions, for retrieval.

---

### Finding 2: Isolated Verification Eliminates Bias

**Paper**: CoVe

**Finding**: Verification in generation context produces biased results.

**Why**: LLMs tend to confirm their own generations when they can see the full context.

**Application**: Stage 4 provides only claim + evidence, no generation context.

---

### Finding 3: Numeric Claims Need Special Handling

**Papers**: QAFactEval, FActScore

**Finding**: Simple string matching fails for numeric claims due to:
- Rounding (3.2B vs 3.19B)
- Unit variations ($1B vs $1,000M)
- Approximations ("about $3B")

**Application**: Stage 6 normalizes and semantically compares numeric values.

---

### Finding 4: Evidence-First Prevents Hallucination

**Paper**: ReClaim

**Finding**: Pre-selecting evidence before generation constrains LLM to verifiable claims.

**Why**: LLM can only cite from available evidence pool, preventing unsupported claims.

**Application**: Stage 1 pre-selects evidence, Stage 2 constrains generation.

---

### Finding 5: Granular Softening Preserves Accuracy

**Paper**: ARE

**Finding**: Soften only unverified atomic facts, not entire claims.

**Why**: A claim may be partially correct. Softening everything loses verified information.

**Application**: Stage 7c reconstructs claims preserving verified facts, softening only unverified ones.

---

## Academic Papers Included

The `specs/003-claim-level-citations/pdf/` folder contains 10 academic papers:

| File | Paper | Topic |
|------|-------|-------|
| `factscore_2305.14251.pdf` | FActScore | Atomic fact decomposition |
| `reclaim_2407.01796.pdf` | ReClaim | Reference-constrained generation |
| `rarr_2210.08726.pdf` | RARR | Post-hoc attribution |
| `cove_2309.11495.pdf` | CoVe | Isolated verification |
| `citefix_2504.15629.pdf` | CiteFix | Hybrid citation correction |
| `qafacteval_2112.08542.pdf` | QAFactEval | QA-based evaluation |
| `ragas_2309.15217.pdf` | RAGAS | RAG evaluation metrics |
| `ever_2311.09114.pdf` | EVER | Evidence verification |
| `veriscore_2406.19276.pdf` | VeriScore | Verification scoring |
| `llm_augmenter_2302.12813.pdf` | LLM Augmenter | External knowledge |

## Research Comparison Documents

The specs folder also contains competitive analysis:

| Document | Content |
|----------|---------|
| `scientific_approach.md` | Deep dive into ARE/FActScore/SAFE |
| `sota_research.md` | State-of-the-art survey |
| `sota_eval_datasets.md` | Benchmark datasets |
| `deerflow_comparison.md` | DeerFlow feature comparison |
| `open_webui_comparison.md` | Open WebUI analysis |

## See Also

- [Citation Pipeline](./citation-pipeline.md) - Implementation details
- [LLM Interaction](./llm-interaction.md) - Model patterns
- [specs/003-claim-level-citations/](../specs/003-claim-level-citations/) - Full specification
