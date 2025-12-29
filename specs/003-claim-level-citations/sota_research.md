# Detailed Summaries of Three Papers on LLM Hallucination and Citation Accuracy

## Paper 1: CiteFix — Enhancing RAG Accuracy Through Post-Processing Citation Correction

### Overview and Motivation

CiteFix addresses a critical but often overlooked problem in Retrieval-Augmented Generation (RAG) systems: **incorrect citations**. While RAG systems have revolutionized information retrieval by combining search with LLM generation, the authors discovered through developing a commercial RAG product that LLMs frequently cite the wrong source documents—even when the information itself is correct.

The key insight is striking: in their analysis of "Model C" (an anonymized production LLM), approximately **80% of unverifiable facts were not hallucinations but citation errors**—the model generated correct information but attributed it to the wrong retrieved document. This aligns with industry studies reporting citation accuracy rates of only ~74% for popular generative search engines.

### Problem Definition

The paper distinguishes between two types of errors in RAG responses:
- **Hallucinated facts**: Information fabricated by the LLM not present in any retrieved document
- **Incorrectly cited facts**: Correct information present in retrieved documents but attributed to the wrong source

This distinction is crucial because incorrect citations, while less severe than hallucinations, still undermine user trust and reduce the actionability of responses—users cannot verify claims against their cited sources.

### Proposed Methodology

CiteFix operates as a **streaming-compatible post-processing layer** that runs on LLM responses as they're generated. The framework follows these steps:

**Step 1: Factual Point Segmentation**
- Split the LLM response *A* into distinct "factual points" {x₀, x₁, ..., x_{L-1}}
- A factual point is defined as a text segment the LLM attributes to specific citations
- Uses regular expressions to segment responses delimited by citation markers

**Step 2: Citation Correction**
For each factual point xᵢ with Cᵢ citations, compute similarity scores with all retrieved documents and select the top Cᵢ documents that maximize: sᵢⱼ = f(xᵢ, x̂ⱼ)

The paper proposes **six methods** for the similarity function f:

#### Method 1: Keyword-Based Matching
- Computes intersection between tokens in the factual point and retrieved documents
- Simple but effective baseline
- TF-IDF variants were tested but performed poorly due to domain-specific keyword ambiguity (e.g., "yield" means different things in agriculture vs. finance)

#### Method 2: Keyword + Semantic Context Matching
- Combines keyword matching with query-document semantic similarity:
  ```
  f(xᵢ, x̂ⱼ) = λ·f_keyword(xᵢ, x̂ⱼ) + (1-λ)·r(q, x̂ⱼ)
  ```
- λ = 0.8 found optimal empirically
- Adds mild preference for documents more relevant to the original query

#### Method 3: BERT Score
- Leverages contextual embeddings from LongFormer model
- Computes token-level similarity using bi-directional attention
- For each token in factual point, finds maximum similarity across all tokens in retrieved document, then averages:
  ```
  f(xᵢ, x̂ⱼ) = (1/|xᵢ|) Σ max e(tᵢₗ)ᵀe(t̂ⱼₖ)
  ```
- Handles paraphrasing better than keyword methods

#### Method 4: Fine-tuned BERT Score (ColBERT-inspired)
- Fine-tunes the embedding model specifically for citation attribution
- Training uses triplets: (factual point, positive reference, negative reference)
- Two dataset preparation strategies:
    1. Find similar documents via embedding similarity, prompt LLM to generate facts present in one but not the other
    2. Generate RAG responses, use LLM to verify which facts are grounded in which documents
- Uses cross-entropy loss to maximize score with positive references

#### Method 5: LLM-Based Matching
- Uses a secondary LLM to identify the most relevant reference for each factual point
- Simple prompt requesting only reference number (avoids CoT to minimize latency/cost)
- Captures contextual and semantic nuances beyond rule-based methods

#### Method 6: Attention Map Analysis (Proof of Concept)
- Examines attention patterns of the response-generating LLM to identify which retrieved documents influenced each factual point
- Demonstrated via toy experiment with Qwen 2.5-2B showing attention peaks align with relevant document positions
- Marked as future work

### Evaluation Metrics

The authors developed **Mean Question-Level Accuracy (MQLA)**, a composite metric capturing:
- **Relevancy URL**: Fraction of cited URLs relevant to the question
- **Relevancy Keywords**: Ratio of relevant keywords in response
- **Relevancy Facts**: Ratio of relevant facts in response
- **Correctness**: Ratio of facts supported by cited references
- **Completeness**: Whether all aspects of the question are addressed

MQLA = 1 if all relevancy/correctness metrics ≥ 0.8 AND hallucinated facts ≤ 1, else 0

### Experimental Results

**Dataset**: 50 representative questions, 2.5 days of human auditing by 2 experts per configuration

**Key Findings**:

| Method                | MQLA Improvement | Relevancy URL | % Facts Correctly Cited | p90 Latency |
|-----------------------|------------------|---------------|-------------------------|-------------|
| Keyword               | +12.7%           | -0.9%         | +12%                    | 0.014s      |
| Keyword + Semantic    | +15.5%           | -0.9%         | +13.6%                  | 0.015s      |
| BERT Score            | +2.6%            | -1%           | +3.2%                   | 0.389s      |
| Fine-tuned BERT Score | +15.8%           | +1.5%         | +13.7%                  | 0.389s      |
| LLM-Based             | +1.9%            | +0.9%         | +7%                     | 1.586s      |

**Cross-LLM Evaluation** (critical finding):
- Different LLMs pair optimally with different correction strategies
- Model A: Keyword + Semantic Context works best (+21% MQLA)
- Model B: Fine-tuned BERT Score works best (+21% MQLA)
- Model C: Both approaches yield +15.5-15.8% MQLA
- Qwen 2.5 14B: Keyword + Semantic Context slightly better

**Cost Implications**:
CiteFix enables using smaller, cheaper models while maintaining quality:
- Model C with CiteFix achieves higher MQLA than Model A (12x more expensive, 3x slower) without CiteFix

### Limitations and Future Work
- Current methods are model-specific; no universal best approach
- Attention map method needs further development
- Dataset preparation for fine-tuned models could be improved
- Framework opens possibilities for content-document relationship applications (e.g., ad placement)

---

## Paper 2: Chain-of-Verification (CoVe) — Reducing Hallucination Through Self-Verification

### Overview and Motivation

Chain-of-Verification (CoVe) tackles hallucination by having LLMs **deliberate on and verify their own responses** before finalizing them. The core insight is that LLMs often answer short, specific verification questions more accurately than they answer complex questions requiring long-form generation.

The authors observe that hallucinations become particularly problematic in:
- Long-form generation (exposure bias compounds errors)
- Torso and tail distribution facts (less frequent in training data)
- List-based answers (many opportunities for individual errors)

### The CoVe Framework

CoVe implements a **four-step verification pipeline**:

#### Step 1: Generate Baseline Response
- Standard LLM generation given a query
- This serves both as the starting point for verification AND the baseline for comparison
- Uses few-shot prompting with task-specific examples

#### Step 2: Plan Verifications
- Conditioned on the original query + baseline response
- Generate a series of verification questions targeting factual claims
- Questions are not templated—LLM generates natural language questions
- Example: For "The Mexican–American War was from 1846 to 1848," generate "When did the Mexican American war start and end?"

#### Step 3: Execute Verifications
- Answer each verification question
- **Critical design choice**: How much context to include when answering

The paper explores **four execution variants**:

**Joint Execution**
- Planning and execution in single prompt
- Verification questions and answers generated together
- Problem: Answers can condition on baseline response, risking repetition of hallucinations

**2-Step Execution**
- Separate prompts for planning and execution
- Execution prompt only contains questions (not baseline response)
- Prevents direct copying of hallucinated answers

**Factored Execution**
- Each verification question answered independently
- Separate prompt per question, none containing baseline response
- Eliminates interference between answer contexts
- More computationally expensive but can be parallelized/batched

**Factor+Revise Execution**
- After factored answering, explicitly cross-check each verification answer against baseline
- Additional LLM prompt per question for inconsistency detection
- Most sophisticated variant for longform generation

#### Step 4: Generate Final Verified Response
- Few-shot prompt incorporating all previous reasoning
- Context includes: baseline response + verification Q&A pairs + (for factor+revise) cross-check results
- Generates revised response incorporating verification insights

### Experimental Setup

**Tasks Evaluated**:

1. **Wikidata List Questions**
    - Form: "Who are some [Profession]s born in [City]?"
    - 56 questions, ~600 gold entities each
    - Metric: Precision (micro-averaged)

2. **Wiki-Category List (QUEST dataset)**
    - Harder: varied questions like "Name some Mexican animated horror films"
    - 55 questions, ~8 answers each
    - Metric: Precision

3. **MultiSpanQA (Closed-Book)**
    - Reading comprehension with multiple independent answers
    - 418 factoid questions
    - Metrics: F1, Precision, Recall

4. **Biography Generation (FACTScore)**
    - Generate biographies for selected entities
    - 183 test cases from Min et al. (2023)
    - Metric: FACTSCORE (retrieval-augmented fact-checking)

**Models**: Llama 65B (few-shot), Llama 2 70B Chat (zero-shot, CoT)

### Key Results

**List-Based Tasks (Wikidata)**:

| Method            | Precision | Positives | Negatives (Hallucinations) |
|-------------------|-----------|-----------|----------------------------|
| Few-shot baseline | 0.17      | 0.59      | 2.95                       |
| CoVe (joint)      | 0.29      | 0.41      | 0.98                       |
| CoVe (two-step)   | 0.36      | 0.38      | 0.68                       |
| CoVe (factored)   | 0.32      | 0.38      | 0.79                       |

**Precision more than doubles**, with massive reduction in hallucinated entities (2.95 → 0.68).

**Closed-Book QA (MultiSpanQA)**:

| Method                     | F1       | Precision | Recall |
|----------------------------|----------|-----------|--------|
| Llama 2 70B Chat Zero-shot | 0.20     | 0.13      | 0.40   |
| Llama 65B Few-shot         | 0.39     | 0.40      | 0.38   |
| CoVe (joint)               | 0.46     | 0.50      | 0.42   |
| CoVe (factored)            | **0.48** | 0.50      | 0.46   |

**23% improvement** in F1 over few-shot baseline.

**Biography Generation (FACTSCORE)**:

| Method                   | FACTSCORE | Avg. # Facts |
|--------------------------|-----------|--------------|
| InstructGPT Zero-shot    | 41.1      | 26.3         |
| ChatGPT Zero-shot        | 58.7      | 34.7         |
| PerplexityAI (Retrieval) | 61.6      | 40.8         |
| Llama 65B Few-shot       | 55.9      | 16.6         |
| CoVe (joint)             | 60.8      | 12.8         |
| CoVe (factored)          | 63.7      | 11.7         |
| CoVe (factor+revise)     | **71.4**  | 12.3         |

**Key observations**:
- CoVe (factor+revise) achieves **71.4 FACTSCORE**, outperforming even PerplexityAI despite not using retrieval
- Improvement from 55.9 → 71.4 represents **28% gain**
- Works especially well for more frequent facts; PerplexityAI still better for very rare facts (where retrieval is essential)

### Critical Findings

**1. Instruction-tuning and CoT don't help hallucination**
- Llama 2 Chat (instruction-tuned) performs worse than Llama 65B (few-shot)
- Chain-of-Thought prompting alone provides no improvement for hallucination reduction

**2. Shortform questions are more accurate than longform**
- Only ~17% of entities in list-based answers are correct
- But ~70% of individual verification questions answered correctly
- This accuracy gap is what makes verification effective

**3. Factored approaches consistently outperform joint**
- Preventing verification answers from attending to baseline response is crucial
- Avoids repetition of hallucinated content

**4. LLM-generated questions outperform heuristics**
- Compared to templated yes/no questions like "Does X answer the question?"
- Model-generated verification questions yield higher precision

**5. Open questions outperform yes/no questions**
- Yes/no format tends to make models agree with however the question is phrased
- Open-ended verification questions produce more reliable answers

### Limitations
- Does not eliminate hallucinations completely
- Only addresses directly stated factual inaccuracies (not reasoning errors or opinions)
- Increased computational cost from additional generation
- Upper bound limited by model's inherent knowledge

---

## Paper 3: EVER — Real-Time Verification and Rectification for LLM Hallucination

### Overview and Motivation

EVER (REal-Time VErification and Rectification) addresses a fundamental limitation of existing hallucination mitigation approaches: **the snowballing problem**. When hallucinations occur early in generation, they can propagate and compound through subsequent sentences, making post-hoc correction increasingly difficult.

The paper distinguishes between:
- **Intrinsic hallucinations**: Generated content contradicts the source/evidence
- **Extrinsic hallucinations**: Content cannot be verified (neither supported nor refuted by evidence)

EVER's key innovation is performing verification and rectification **sentence-by-sentence during generation**, rather than waiting until completion.

### The EVER Framework

#### Stage 1: Generation
Two modes supported:
- **Non-Retrieval Generation (NRG)**: LLM generates based solely on internal knowledge
- **Retrieval-Augmented Generation (RAG)**: External context included in prompt

After generating each sentence, EVER immediately transitions to validation rather than continuing generation.

#### Stage 2: Concept-Level Validation
A three-step process for each generated sentence:

**Step 2a: Key Concepts Identification**
- Extract factual concepts (dates, numbers, jobs, locations, names, etc.)
- Uses in-context learning with examples
- Example: "Shin Jea-hwan is an artistic gymnast, born on November 2, 1998, and raised by traveling circus performers"
    - Extracts: "artistic gymnast", "November 2, 1998", "traveling circus performers"

**Step 2b: Validation Question Generation**
- Convert each concept to a Yes/No verification question
- Example: "Is Shin Jea-hwan an artistic gymnast?"
- Uses few-shot prompting for smaller models, zero-shot for GPT-3.5

**Step 2c: Support Checking**
- For each validation question, determine: **True**, **False**, or **NEI (Not Enough Information)**
- Two evidence gathering modes:
    - **Self-Query (SQ)**: LLM directly answers from internal knowledge
    - **Evidence Retrieval (ER)**: Retrieves external evidence (e.g., Google Search)
- Uses Chain-of-Thought prompting: "PROVIDE REASONING FIRST BEFORE DECISION"

**Classification Logic**:
- True → Evidence supports the concept
- False → Evidence contradicts (intrinsic hallucination detected)
- NEI → No relevant evidence found (extrinsic hallucination detected)

#### Stage 3: Rectifying Hallucinations

**For Intrinsic Hallucinations (False)**:
- Revise the specific incorrect concept based on retrieved evidence
- Align facts with verifiable truths
- Example: Change "November 2, 1998" to "March 3, 1998" if evidence contradicts

**For Extrinsic Hallucinations (NEI)**:
- Rewrite the entire sentence using:
    - Feedback identifying the problematic concept
    - Retrieved evidence as reference
- Example: Rewrite unverifiable claim about "traveling circus performers" to "made his international debut in 2017"

#### Stage 4: Processing Remaining Extrinsic Hallucinations

After rectification, re-validate the sentence:
- If intrinsic hallucinations persist → Additional rectification rounds (rarely needed)
- If extrinsic hallucinations persist after one round:
    - **Short-form generation**: Abstain from answering
    - **Long-form generation**: Mark with "not sure" warning flag

The warning flag enhances trustworthiness by transparently indicating uncertainty.

### EVER Variants

The paper evaluates three configurations:
1. **EVER (NRG+SQ)**: Non-retrieval generation + Self-query validation
2. **EVER (NRG+ER)**: Non-retrieval generation + Evidence retrieval validation
3. **EVER (RAG+ER)**: Retrieval-augmented generation + Evidence retrieval validation

### Extension: Preference Tuning with EVER Data

Beyond direct hallucination mitigation, EVER enables **creating preference data for DPO (Direct Preference Optimization)**:
- Preferred response (y_w): EVER-rectified output
- Dispreferred response (y_l): Original non-rectified output
- Fine-tune model using DPO loss to internalize factuality preferences

### Experimental Results

#### Biography Generation Task

**Dataset**: 183 entities from Min et al. (2023)
**Metric**: FACTSCORE (ChatGPT + Retrieval fact-checking)

**Non-Retrieval Scenario**:
| Model | Method | FACTSCORE |
|-------|--------|-----------|
| Llama 2 7B Chat | Zero-Shot | 36.8% |
| Llama 2 7B Chat | Dola | 36.8% |
| Llama 2 7B Chat | EVER (NRG+SQ) | **46.7%** |
| Llama 1 65B | CoVe | 71.4% |
| Llama 1 65B | EVER (NRG+SQ) | **72.9%** |
| GPT 3.5 Turbo | Zero-Shot | 71.8% |
| GPT 3.5 Turbo | EVER (NRG+SQ) | **75.2%** |

**Retrieval-Augmented Rectification**:
| Model | Method | FACTSCORE |
|-------|--------|-----------|
| Llama 2 7B Chat | RRAR | 37.8% |
| Llama 2 7B Chat | EVER (NRG+ER) | **76.9%** |
| GPT 3.5 Turbo | RRAR | 74.3% |
| GPT 3.5 Turbo | EVER (NRG+ER) | **94.5%** |

**Retrieval-Augmented Generation + Rectification**:
| Model | Method | FACTSCORE |
|-------|--------|-----------|
| Llama 2 7B Chat | Self-RAG | 81.2% |
| Llama 2 7B Chat | EVER (RAG+ER) | **86.4%** |
| GPT 3.5 Turbo | RAG | 92.7% |
| GPT 3.5 Turbo | EVER (RAG+ER) | **95.8%** |

**Key insight**: EVER (NRG+ER) with Llama 2 7B achieves **76.9% FACTSCORE**—higher than many larger models without retrieval during generation!

#### Rarity Analysis

Unlike RRAR (post-hoc revision), EVER maintains stable factual precision across entity rarity:
- RRAR degrades significantly for rare entities
- EVER's sentence-by-sentence validation prevents error accumulation

#### Multi-Hop Reasoning (HotPotQA)

| Retrieval | Method | EM | F1 |
|-----------|--------|-----|-----|
| N/A | Few-Shot CoT | 32.6% | 46.8% |
| N/A | EVER (NRG+SQ) | **34.7%** | **48.3%** |
| Google | CRITIC | 40.3% | 52.9% |
| Dataset | EVER (NRG+ER) | **42.3%** | **58.1%** |
| Dataset | IRCoT | 48.4% | 57.8% |
| Dataset | EVER (RAG+ER) | **51.4%** | **61.2%** |

EVER outperforms CRITIC because CRITIC corrects reasoning chains as a whole, while EVER addresses errors step-by-step, preventing snowballing.

#### Preference Tuning Results

Fine-tuning Llama-2-7B-chat on biography generation:

| Method | FACTSCORE |
|--------|-----------|
| Vanilla | 36.8% |
| FactTune-FS | 45.4% |
| EVER-PREF (NRG+SQ) | 47.3% |
| RAG-PREF | 50.2% |
| EVER-PREF (NRG+ER) | 52.8% |
| EVER-PREF (RAG+ER) | **53.9%** |

Using EVER-generated data as preference pairs significantly improves factuality compared to alternatives.

### Extrinsic Hallucination Analysis

Human annotation of 300 "Not Enough Info" instances revealed:
- **65%**: Evidence doesn't mention the concept at all
- **15%**: Evidence relevant but requires additional inference
- **9%**: Subjective/opinion-based content (hard to verify objectively)
- **11%**: Misclassified (actually True or False)

The 89% accuracy of NEI classifications demonstrates EVER's reliability and justifies user warnings.

### Efficiency Analysis

| Method | Biography (s) | HotPotQA (s) |
|--------|--------------|--------------|
| RRAR | 210.5 | - |
| IRCoT | - | 67.2 |
| CRITIC | - | 83.8 |
| EVER (NRG+SQ) | 195.7 | 73.6 |
| EVER (NRG+ER) | 141.8 | 86.9 |
| EVER (RAG+ER) | 115.4 | 62.8 |

EVER achieves comparable or better runtime through:
- Simplified prompts (shorter, few-shot/zero-shot)
- Parallel validation of extracted concepts

### Limitations
- Focuses on text attribution rather than comprehensive fact-checking
- External evidence may itself contain errors
- One round of rectification may not resolve all issues (though empirically sufficient in most cases)

---

## Synthesis: How These Papers Relate

These three papers represent **complementary approaches** to the hallucination and attribution problem:

| Paper | Focus | Approach | When Applied |
|-------|-------|----------|--------------|
| **CiteFix** | Citation accuracy (correct sources) | Post-processing similarity matching | After complete generation |
| **CoVe** | Factual accuracy (correct facts) | Self-verification through decomposition | After draft, before final |
| **EVER** | Both intrinsic & extrinsic hallucination | Real-time sentence-level verification | During generation |

**Key shared insights**:
1. Breaking down verification into smaller questions improves accuracy
2. Retrieval augmentation significantly helps factuality
3. Different models may benefit from different approaches
4. Post-hoc correction alone cannot fully address snowballing errors (EVER's motivation)

**Potential combinations**:
- EVER's real-time verification + CiteFix's citation correction as final pass
- CoVe's verification question planning + EVER's sentence-level timing
- All three could be combined: EVER during generation → CoVe for final verification → CiteFix for citation cleanup
# Detailed Summaries of Three Research Papers on Factual Consistency Evaluation

## Paper 1: FACTSCORE - Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation

### Overview and Motivation

FACTSCORE addresses a critical challenge in evaluating large language models: how to measure the factual accuracy of long-form text generation. The authors identify two fundamental problems with existing evaluation approaches:

1. **Granularity Problem**: Generated text contains a mixture of true and false information, making binary (correct/incorrect) judgments inadequate. Even a single sentence averages 4.4 pieces of information in ChatGPT outputs, with 40% containing a mixture of supported and unsupported facts.

2. **Cost Problem**: Human evaluation of factual accuracy is extremely time-consuming and expensive.

### Core Methodology

**Definition of FACTSCORE**: The metric represents the percentage of "atomic facts" in a generation that are supported by a reliable knowledge source. An atomic fact is defined as a short sentence conveying exactly one piece of information.

**Two Key Ideas**:

1. **Atomic Fact as Unit**: Rather than evaluating sentences (which contain multiple facts), FACTSCORE breaks text into atomic facts. This avoids the subjective "partial support" label and provides finer-grained evaluation. For example, "Bridget Moynahan is an American actress, model and producer" becomes multiple atomic facts: "Bridget Moynahan is American," "Bridget Moynahan is an actress," etc.

2. **Knowledge Source Dependency**: Factual precision is evaluated relative to a specific knowledge source (e.g., Wikipedia) rather than attempting to determine absolute truth. This acknowledges that different sources may conflict.

**Formal Definition**:
- For a language model M, prompts X, and knowledge source C
- For response y = M(x), extract atomic facts Ay
- FACTSCORE = average percentage of atomic facts in Ay supported by C

### Experimental Setup

**Domain**: People biographies were chosen because:
- Statements are objective and verifiable (not subjective)
- Broad scope covering diverse nationalities, professions, and rarity levels
- Wikipedia provides comprehensive coverage

**Models Evaluated**:
- InstructGPT (text-davinci-003)
- ChatGPT
- PerplexityAI (search-augmented)

**Data Collection Pipeline**:
1. **Entity Sampling**: 183 people entities sampled from Wikidata, stratified across 20 categories (5 frequency levels × 4 nationality regions)
2. **Generation**: Models prompted with "Tell me a bio of <entity>"
3. **Atomic Fact Generation**: Human annotators break generations into atomic facts (aided by InstructGPT-generated initial breakdowns)
4. **Labeling**: Each atomic fact labeled as Supported, Not-supported, or Irrelevant

**Annotation Quality**: Inter-annotator agreement rates of 96%, 90%, and 88% for InstructGPT, ChatGPT, and PerplexityAI respectively. Cost was approximately $4 per generation.

### Human Evaluation Results

| Model | FACTSCORE | % Responding | Avg Facts/Response |
|-------|-----------|--------------|-------------------|
| InstructGPT | 42.5% | 99.5% | 26.3 |
| ChatGPT | 58.3% | 85.8% | 34.7 |
| PerplexityAI | 71.5% | 90.7% | 40.8 |

**Key Findings**:

1. **All models struggle significantly**: Even the best-performing PerplexityAI (with search) only achieves 71.5% factual precision.

2. **Rarity matters dramatically**: ChatGPT's FACTSCORE drops from 80% for very frequent entities to just 16% for very rare entities. Surprisingly, even PerplexityAI shows a 50% relative drop for rare entities despite having search access.

3. **Position effects**: Facts mentioned later in generations have significantly worse precision, likely due to:
    - Earlier information being more common in training data
    - Error propagation affecting later content

4. **PerplexityAI Error Analysis** (30 Not-supported samples categorized):
    - 33.3% Single-sentence contradiction (word-level: dates, numbers, entities)
    - 23.3% Page-level contradiction (information missing from Wikipedia)
    - 16.7% Subjective statements copied from Wikipedia
    - 10.0% Single-sentence contradiction (beyond word level)
    - 10.0% Annotation errors
    - 3.3% Irrelevant facts due to search errors
    - 3.3% Wikipedia internal inconsistencies

### Automated FACTSCORE Estimation

Since human evaluation costs $4 per generation, the authors developed automated estimators.

**Estimator Components**:
1. **Atomic Fact Generation**: InstructGPT breaks sentences into atomic facts (found effective and close to human)
2. **Fact Validation**: Various approaches tested

**Validation Approaches**:

| Method | Description |
|--------|-------------|
| No-context LM | Prompt: "<atomic-fact> True or False?" |
| Retrieve→LM | Retrieve k passages, concatenate with fact, prompt LM |
| Nonparametric Probability (NP) | Mask tokens, compute nonparametric likelihood |
| Retrieve→LM + NP | Ensemble of both methods |

**Evaluation Metrics**:
- **Error Rate (ER)**: Difference between ground truth and estimated FACTSCORE
- **Ranking Preservation**: Whether estimated scores preserve correct ordering of models

**Results**:
- Retrieval significantly helps (No-context LM performs poorly)
- Retrieve→LM alone tends to overestimate FACTSCORE
- Ensembling with NP reduces error rates substantially
- Best estimators achieve <2% error rate
- LLAMA+NP best for InstructGPT/ChatGPT; ChatGPT best for PerplexityAI

### Large-Scale Evaluation of 13 LMs

Using the automated estimator, the authors evaluated 6,500 generations from 13 models (would have cost $26K with humans):

**Key Rankings** (by estimated FACTSCORE):
1. Human (~92%)
2. GPT-4 (~73%)
3. ChatGPT (~67%)
4. Alpaca 65B (~63%)
5. InstructGPT (~52%)
6. Alpaca/Vicuna 13B (~47%)
7. Vicuna/Alpaca 7B (~40%)
8. MPT-Chat 7B (~30%)
9. Dolly 12B/Oasst-pythia 12B (~24%)
10. StableLM 7B (~17%)

**Insights**:
- All LMs substantially less factual than humans
- GPT-4 comparable to ChatGPT in precision but generates more facts (61 vs 37) and abstains less
- Clear correlation between model size and factual precision within model families
- Large variance among public models of similar size (7B: Alpaca/Vicuna ~40% vs StableLM ~17%)

### Limitations

1. **Scope**: Focused on biographies and Wikipedia; generalization to other domains (news, scientific literature) needs validation
2. **Assumptions**: May not work when facts are nuanced, debatable, or when knowledge sources conflict
3. **Precision vs Recall**: FACTSCORE only measures precision, not whether important information is missing
4. **Abstention Trade-off**: Models that abstain more or generate fewer facts may score higher

---

## Paper 2: LLM-AUGMENTER - Check Your Facts and Try Again: Improving Large Language Models with External Knowledge and Automated Feedback

### Overview and Motivation

LLM-AUGMENTER addresses the practical challenge of deploying LLMs in mission-critical applications where hallucination is unacceptable. The key insight is that rather than retraining expensive LLMs, we can augment them with plug-and-play (PnP) modules that:
1. Ground responses in external knowledge
2. Iteratively improve responses using automated feedback

### System Architecture

LLM-AUGMENTER formulates human-system conversation as a Markov Decision Process (MDP) with five components:

**State Space (S)**: Dialog states encoded in Working Memory as a six-tuple:
- q: current user query
- e: evidence consolidated from external knowledge
- o: set of LLM-generated candidate responses
- u: utility score for each response
- f: verbalized feedback
- hq: dialog history

**Action Space (A)**:
- Call Knowledge Consolidator to gather evidence
- Call Prompt Engine to query LLM for responses
- Send response to user

**Transition Probability P(s'|s,a)**: State transitions after actions

**Reward R(s,a)**: External reward from environment (users/simulators)

**Discount Factor γ**: For future reward weighting

### Module Descriptions

#### 1. Working Memory
Tracks all essential conversation information including query, evidence, candidate responses, utility scores, feedback, and history.

#### 2. Policy Module
Selects optimal actions to maximize expected reward. Can be:
- **Rule-based**: Domain experts encode IF-THEN rules
- **Trainable**: Neural network (e.g., T5) optimized via REINFORCE

**Three-stage Policy Learning**:
1. Bootstrap from rule-based policy
2. Learn with user simulators (self-improvement)
3. Refine with real human interactions

#### 3. Action Executor

**Knowledge Consolidator** (following Ma et al., 2022):
- **Retriever**: Generates search queries, calls APIs (Bing Search, REST APIs for databases)
- **Entity Linker**: Enriches raw evidence by linking entities to Wikipedia descriptions, forming evidence graphs
- **Evidence Chainer**: Prunes irrelevant evidence, forms shortlists of evidence chains relevant to queries

**Prompt Engine**:
Constructs prompts containing:
- Task instruction
- User query and dialog history
- Evidence from Knowledge Consolidator
- Feedback from Utility module

#### 4. Utility Module
Generates utility scores and feedback for candidate responses.

**Two Types of Utility Functions**:
1. **Model-based**: Learned scorers for fluency, informativeness, factuality
2. **Rule-based**: Heuristics for compliance with specific requirements

**Feedback Generation**: A seq2seq model Q generates textual feedback:
f = Q(q, e, o, hq)

This feedback is used to revise prompts and elicit better responses from the LLM.

### Experiments: Information Seeking Dialog

#### Datasets

**News Chat (DSTC7 Track 2 repurposed)**:
- Goal: Generate informative, knowledge-grounded responses
- Data: Reddit threads with news URLs (2021-2022)
- 1,370 examples with ROUGE-F1 filtered oracle passages

**Customer Service (DSTC11 Track 5)**:
- Users seek information from FAQs and customer reviews
- Built on MultiWOZ 2.1
- 14,768 training sessions, validation set used for evaluation

#### Experimental Setup

- **LLM**: ChatGPT (black-box)
- **Knowledge Consolidator**: BM25 retriever over web documents/FAQs/reviews
- **Utility**: Knowledge F1 (KF1) measuring overlap between response and evidence
- **Feedback**: Template-based NLG or ChatGPT self-criticism
- **Policy**: Rule-based (always use KC, evaluate with ChatGPT, provide feedback)

#### Metrics
- Knowledge F1 (KF1), BLEU-4, ROUGE-1, METEOR
- BLEURT, BERTScore, chrF, BARTScore
- Human evaluation: Usefulness and Humanness

#### Results

**News Chat**:
| Setting | KF1 | BLEU |
|---------|-----|------|
| ChatGPT (no knowledge) | 26.71 | 1.01 |
| + BM25 knowledge | 34.96 | 6.71 |
| + BM25 + feedback | 36.41 | 7.63 |
| + gold knowledge | 57.44 | 19.24 |
| + gold + feedback | 60.76 | 21.49 |

**Customer Service**:
| Setting | KF1 | BLEU |
|---------|-----|------|
| ChatGPT (no knowledge) | 31.33 | 4.70 |
| + BM25 knowledge | 34.07 | 4.78 |
| + BM25 + feedback | 37.41 | 3.86 |
| + gold knowledge | 45.63 | 6.54 |
| + gold + feedback | 52.83 | 5.63 |

**Key Findings**:

1. **External knowledge dramatically helps**: Even with zero-shot ChatGPT performing reasonably, task-specific knowledge provides ~10 point KF1 improvement

2. **Automated feedback further improves**: +3.3 KF1 on News Chat, +7.2 on Customer Service with gold knowledge

3. **Trainable policy works**: RL-trained policy (T5-Base) surpasses random policy after 600 interactions, reaching ~37.5 KF1

4. **Human evaluation confirms**: LLM-AUGMENTER improves over ChatGPT by 32.3% in Usefulness and 12.9% in Humanness

#### Ablation Studies

**Policy Variants**:
- No-knowledge consolidator: 31.5 KF1
- Self-ask (use KC when LLM suggests): 33.0 KF1
- Always-use KC: 34.0 KF1

ChatGPT self-asks for external knowledge in 24% of examples.

**Feedback Types**:
- No feedback: 34.07 KF1
- Rule-based feedback: 37.41 KF1
- Self-criticism feedback: 37.10 KF1

Both feedback types help comparably, though self-criticism provides more detailed suggestions.

**Utility + Feedback Interaction**:
Best performance comes from combining utility functions with feedback-augmented prompting. Simply prompting twice and re-ranking is insufficient.

### Experiments: Wiki Question Answering (OTT-QA)

#### Dataset
- Open-domain QA requiring multi-hop reasoning over tables and passages
- ~40K instances, 400K tables, 6M passages from Wikipedia
- 13% single-hop, 57% two-hop, 30% multi-hop questions

#### Setup
- **Knowledge Consolidator**: DPR retriever + CORE (linker + chainer for evidence chains)
- **Utility**: Recall (token overlap with evidence)
- **Evaluation**: Token-level Precision, Recall, F1

#### Results

| Setting | P | R | F1 |
|---------|---|---|-----|
| ChatGPT (closed-book) | 0.48 | 1.52 | 0.59 |
| + DPR | 2.08 | 4.31 | 2.38 |
| + CORE | 7.06 | 14.77 | 8.08 |
| + CORE + feedback | 8.93 | 33.87 | 11.80 |

**Key Findings**:
- Closed-book ChatGPT performs very poorly (hallucinates, abstains 17%)
- Consolidated evidence (CORE) far outperforms raw retrieval (DPR)
- Feedback provides substantial recall boost (+19 points)
- Gap remains vs fine-tuned SOTA, indicating need for better prompting strategies

### Limitations and Future Directions

1. **Latency**: Interactive feedback with ChatGPT slows response time (queried twice per response)
2. **Policy training**: Main results use manual policy; RL experiments limited by ChatGPT bandwidth
3. **Utility function design**: Current focus on KF1; safety-oriented utilities needed for broader applications

---

## Paper 3: QAFactEval - Improved QA-Based Factual Consistency Evaluation for Summarization

### Overview and Motivation

QAFactEval addresses the fragmented state of factual consistency evaluation for summarization. Two main paradigms exist:

1. **Entailment-based metrics**: Determine if summary content is entailed by source document
2. **QA-based metrics**: Generate questions from summary, answer using source, compare answers

Prior work reaches conflicting conclusions about which paradigm is superior, largely due to inconsistent experimental setups. This paper:
1. Systematically analyzes QA-based metric components
2. Proposes optimized QAFACTEVAL metric
3. Shows QA and entailment metrics can be combined for further gains

### QA-Based Metric Pipeline

The paper decomposes QA-based metrics into four components:

#### 1. Answer Selection
Extract "information units" (answers) from the summary to ask questions about.

**Methods Tested**:
- **NER**: Named entities only (~3 answers/summary)
- **NP Chunks**: Noun phrase chunks (~10+ answers/summary)
- **Max NP**: Maximally sized noun phrases via dependency parsing
- **All**: Combination of above methods

**Finding**: NP Chunks performs best, balancing coverage and precision.

#### 2. Question Generation (QG)
Generate questions conditioned on selected answers using summary as context.

**Models Tested**:
- BART-large trained on QA2D (declarative sentence dataset)
- BART-large trained on SQuAD
- T5-base trained on SQuAD
- MixQG-base/large (trained on 9 diverse QA datasets)

**Key Finding**: BART-large (QA2D) works best despite producing longer (~17 tokens vs ~10), more extractive questions (~20% novel unigrams vs ~47% for T5-SQuAD). More abstractive models hallucinate, producing questions QA models struggle to answer.

#### 3. Question Answering (QA)
Answer generated questions using source document as context.

**Models Tested**:
- Electra-large/base (extractive, SQuAD 2.0)
- MADE (multi-dataset adapters on RoBERTa)
- T5-base (abstractive, SQuAD)
- UnifiedQA-base (abstractive, 8 diverse datasets)

**Surprising Finding**: QA model choice has relatively small impact on performance, suggesting QA ability is not the bottleneck.

#### 4. Answer Overlap Evaluation
Compare QA model output with originally selected answer.

**Methods Tested**:
- Exact Match (EM)
- Word F1
- LERC (learned 1-5 score from MOCHA dataset)
    - LERC (orig): BERT-base
    - LERC (RoBERTa): RoBERTa-large
    - LERC (QuIP): Question-infused pretrained RoBERTa
- IsAnsweredInput: Binary (0/1) answerability score

**Finding**: Learned LERC metrics substantially outperform EM/F1. QuIP initialization provides slight additional gain.

#### 5. Question Filtering (IsAnsweredSumm Filter)
Filter out noisy questions that are unanswerable even using the summary (from which they were generated).

**Implementation**: Electra-large determines if question is answerable using summary as context.

#### 6. Answerability Penalty
For questions deemed unanswerable using the source document, set overlap score to 0 rather than using most probable answer.

**Rationale**: Selected answer may appear in both summary and source in different contexts (intrinsic error). QA model might extract it as most probable answer despite factual inconsistency.

### Ablation Study Results

| Component | Best Choice | Benchmark Score |
|-----------|-------------|-----------------|
| **Full QAFACTEVAL** | - | **77.5** |
| Answer Selection | NP Chunks | - |
| " | Max NP | 75.7 |
| " | NER | 66.4 |
| " | ALL | 75.7 |
| Question Generation | BART-large (QA2D) | - |
| " | BART-large (SQuAD) | 74.3 |
| " | T5-base (SQuAD) | 67.0 |
| " | MixQG-base | 75.1 |
| " | MixQG-large | 74.9 |
| Question Answering | Electra-large | - |
| " | Electra-base | 77.0 |
| " | MADE | 77.4 |
| " | T5-base | 76.1 |
| " | UnifiedQA-base | 75.7 |
| Answer Overlap | LERC (QuIP) | - |
| " | EM | 68.4 |
| " | F1 | 71.7 |
| " | IsAnsweredInput | 73.3 |
| " | LERC (orig) | 71.8 |
| " | LERC (RoBERTa) | 77.3 |
| Filtering/Answerability | Both | - |
| " | No IsAnsweredSumm Filter | 73.8 |
| " | No Answerability Penalty | 72.1 |
| " | Neither | 67.4 |

**Critical Insights**:
1. Question filtering and answerability penalty together provide ~10 point improvement
2. Answer overlap metric choice matters greatly (EM vs LERC: ~9 points)
3. QG model matters substantially; QA model less so
4. NER-only answer selection insufficient due to low coverage

### Entailment-Based Metrics Included

| Metric | Description |
|--------|-------------|
| MNLI | RoBERTa-large on MNLI, max entailment over source sentences |
| ANLI | Same approach with adversarial ANLI training |
| SCZeroShot | MNLI + Vitamin-C (contrastive examples) training |
| BertScore-FFCI | BERTScore with RoBERTa-MNLI backbone |
| DAE | Dependency arc entailment with synthetic training |
| FactCC | Document-level RoBERTa-base on synthetic data |
| DocNLI | Document-level entailment model |

### Learned Metrics

**SCConv** (from Laban et al., 2021):
- Creates M×N entailment matrix (M source sentences × N summary sentences)
- Bins into H×N histogram
- 1D convolution to produce per-summary-sentence scores
- Trained on 50K synthetic FactCC examples

**QAFACTEVAL-NLI** (proposed):
- Extracts K answers from summary
- Produces length-K score array
- Converts to histogram, applies 1D convolution for single QA score
- Concatenates with SCConv NLI score
- Linear layer produces final combined score
- Can be trained on synthetic or supervised (validation set) data

### SummaC Benchmark Results

The benchmark consists of 6 datasets: CGS, XSF (XSum), Polytope, FactCC, SummEval, FRANK

| Model Type | Model | Benchmark Avg |
|------------|-------|---------------|
| Misc | BARTScore | 68.9 |
| | BLANC | 61.8 |
| | FactCC | 70.4 |
| Entailment | BertScore-FFCI | 65.4 |
| | DAE | 72.7 |
| | ANLI | 74.4 |
| | MNLI | 75.7 |
| | DocNLI | 68.5 |
| | SCZeroShot | 72.8 |
| QA | QuestEval | 68.2 |
| | **QAFACTEVAL** | **77.8** |
| Learned | SCConv (synthetic) | 74.3 |
| | QAFACTEVAL-NLI (synthetic) | 78.3 |
| | **QAFACTEVAL-NLI (supervised)** | **79.5*** |

*Statistically significant improvement (p<0.01)

**Key Results**:
1. QAFACTEVAL outperforms prior QA metric (QuestEval) by **14%** absolute
2. QAFACTEVAL outperforms all entailment-based metrics
3. Combining QA and NLI signals (QAFACTEVAL-NLI) provides further improvement
4. Supervised fine-tuning on validation data helps most datasets

### Correlation Analysis

Instance-level Pearson correlations with human judgments:

| Model | XSF | SummEval | FRANK-CNN | FRANK-XSum | QAGs-CNN | QAGs-XSum |
|-------|-----|----------|-----------|------------|----------|-----------|
| QuestEval | 0.45 | 0.41 | 0.52 | 0.24 | 0.51 | 0.23 |
| QAFACTEVAL | 0.29 | **0.61** | **0.66** | **0.32** | **0.68** | **0.44** |
| SCConv | 0.12 | 0.50 | 0.59 | 0.30 | 0.03 | 0.06 |
| QAFACTEVAL-NLI | 0.19 | **0.61** | **0.66** | 0.25 | 0.65 | **0.48** |

**Observations**:
- QAFACTEVAL shows strong correlation across most datasets
- BertScore-FFCI correlates best with XSF (word-level annotations align with token-level metric)
- SCConv struggles on QAGs datasets
- QAFACTEVAL-NLI generally performs well except on XSF

### Dataset-Specific Insights

Different metrics excel on different datasets due to annotation characteristics:
- **XSF**: Word-level factuality annotations favor token-level metrics
- **FactCC**: Synthetic training data similarity benefits models trained on same distribution
- **Polytope**: Extractive summaries with specific error types

### Illustrative Examples

**Example 1** (Entity-centric error):
- Document: "Paul Merson has restarted his row with Andros Townsend..."
- Summary: "Paul Merson is not happy with Andros Townsend's call-up to the England squad last week"
- BART-QA2D question: "What is Paul Merson not happy with to the England squad last week?" (less fluent but more extractive)
- MixQG question: "What is Paul Merson not happy with?" (fluent but vague)
- BART-QA2D enables correct QA extraction; entailment metric fails due to novel unigrams

**Example 2** (General question challenge):
- Document about Twisted Sister's final tour
- Summary: "The band will perform two shows"
- Question: "Who will perform two shows?"
- QA labels as unanswerable despite answer being extractable
- Demonstrates limitation with generic questions

### Contributions Summary

1. **Comprehensive component analysis**: First systematic study of all QA-based metric components
2. **14% improvement**: Over prior QA metrics on standard benchmark
3. **Complementary signals**: Demonstrated QA and NLI metrics combine effectively
4. **State-of-the-art**: QAFACTEVAL-NLI achieves best performance
5. **Extensive benchmarking**: 10 additional metrics, both classification and correlation analysis

### Limitations

1. **English-only**: All underlying models trained on English data
2. **Bias propagation**: Training data biases may affect metric judgments
3. **Imperfect detection**: Metrics don't catch all factual inconsistencies
4. **Environmental cost**: Multiple large models required for inference

---

## Cross-Paper Synthesis

These three papers represent complementary approaches to the factual consistency problem:

| Paper | Focus | Key Innovation | Evaluation Paradigm |
|-------|-------|----------------|---------------------|
| FACTSCORE | Long-form generation evaluation | Atomic fact decomposition | Human annotation + automated estimation |
| LLM-AUGMENTER | Improving LLM outputs | External knowledge + feedback loops | Task performance (KF1, human eval) |
| QAFactEval | Summarization evaluation | Optimized QA pipeline + NLI combination | Classification + correlation |

**Common Themes**:
1. **Decomposition helps**: Breaking text into smaller units (atomic facts, questions) improves evaluation
2. **External knowledge is crucial**: Retrieval-augmented approaches outperform closed-book methods
3. **Component optimization matters**: Careful tuning of pipeline components yields significant gains
4. **Hybrid approaches work best**: Combining different signals (QA + NLI, retrieval + NP) improves over single methods

# Detailed Summaries of Three RAG Evaluation & Attribution Papers

## Paper 1: RAGAS - Automated Evaluation of Retrieval Augmented Generation

### Overview and Motivation

RAGAS (Retrieval Augmented Generation Assessment) addresses a critical gap in evaluating RAG systems: the lack of reference-free, automated metrics that can assess multiple quality dimensions without requiring ground truth annotations. The authors argue that while RAG systems are increasingly adopted to reduce hallucinations and provide LLMs with external knowledge, evaluating these systems remains challenging because:

1. **Multiple components interact**: RAG systems combine retrieval and generation modules, each affecting overall quality
2. **Traditional metrics are insufficient**: Perplexity doesn't predict downstream performance; extractive QA datasets don't represent real-world usage
3. **Ground truth is often unavailable**: In production settings, human-annotated reference answers are rarely available
4. **API-based models limit evaluation options**: Token probabilities aren't accessible for closed models like GPT-4

### The Three Core Metrics

#### 1. Faithfulness (F)

**Definition**: Measures whether claims in the generated answer can be inferred from the retrieved context.

**Computation Process**:
- **Step 1 - Statement Extraction**: Use an LLM to decompose the answer into atomic statements/claims
- **Step 2 - Verification**: For each statement, use an LLM to determine if it can be inferred from the context
- **Final Score**: F = |V| / |S|, where |V| is verified statements and |S| is total statements

**Prompt for Statement Extraction**:
```
Given a question and answer, create one or more statements from each sentence in the given answer.
```

**Prompt for Verification**:
```
Consider the given context and following statements, then determine whether they are supported by the information present in the context. Provide a brief explanation for each statement before arriving at the verdict (Yes/No).
```

#### 2. Answer Relevance (AR)

**Definition**: Measures whether the answer directly addresses the question appropriately, penalizing incomplete or redundant information (independent of factuality).

**Computation Process**:
- Generate n potential questions from the answer using an LLM
- Compute embeddings for all generated questions and the original question
- Calculate cosine similarity between original and generated questions
- **Final Score**: AR = (1/n) × Σ sim(q, qᵢ)

**Key Insight**: If an answer is highly relevant, questions generated from it should be semantically similar to the original question.

#### 3. Context Relevance (CR)

**Definition**: Measures whether the retrieved context is focused and contains minimal irrelevant information.

**Rationale**:
- Long contexts increase computational costs
- LLMs struggle with information in the middle of long passages ("lost in the middle" phenomenon)
- Focused contexts improve answer quality

**Computation Process**:
- Use LLM to extract sentences from context crucial for answering the question
- **Final Score**: CR = (extracted sentences) / (total sentences in context)

**Prompt**:
```
Please extract relevant sentences from the provided context that can potentially help answer the following question. If no relevant sentences are found, or if you believe the question cannot be answered from the given context, return the phrase "Insufficient Information".
```

### WikiEval Dataset

To validate the metrics, the authors created WikiEval:

**Construction**:
- Selected 50 Wikipedia pages covering events since 2022 (ensuring recency beyond model training cutoff)
- Used ChatGPT to generate questions from introductory sections
- Generated answers using ChatGPT with context

**Annotation Process**:
- Two fluent English annotators rated each dimension
- 95% agreement on faithfulness and context relevance
- 90% agreement on answer relevance
- Disagreements resolved through discussion

**Creating Contrastive Examples**:
- **Low Faithfulness**: Generated answers without context access
- **Low Answer Relevance**: Prompted for deliberately incomplete answers
- **Low Context Relevance**: Added related but less relevant sentences from Wikipedia backlinks

### Experimental Results

| Metric | Faithfulness | Answer Relevance | Context Relevance |
|--------|--------------|------------------|-------------------|
| RAGAS | **0.95** | **0.78** | **0.70** |
| GPT Score | 0.72 | 0.52 | 0.63 |
| GPT Ranking | 0.54 | 0.40 | 0.52 |

**Key Findings**:
- RAGAS significantly outperforms direct scoring (asking LLM to rate 0-10) and ranking baselines
- Faithfulness achieves highest alignment with human judgments
- Context relevance is hardest to evaluate—ChatGPT struggles with sentence selection for longer contexts

### Technical Implementation

- **LLM Used**: gpt-3.5-turbo-16k via OpenAI API
- **Embedding Model**: text-embedding-ada-002 for answer relevance computation
- **Integration**: Compatible with LlamaIndex and LangChain frameworks
- **Open Source**: Available at github.com/explodinggradients/ragas

---

## Paper 2: RARR - Researching and Revising What Language Models Say, Using Language Models

### Core Problem and Innovation

RARR (Retrofit Attribution using Research and Revision) tackles a fundamental problem: LLMs generate fluent but often unsupported or misleading content, and most lack built-in attribution mechanisms. Even retrieval-augmented models don't guarantee attribution—they may include information outside retrieved documents, ignore retrievals, or contradict them.

**Key Innovation**: Instead of constraining LLMs during generation, RARR retrofits attribution post-hoc through a research-and-revise workflow:
1. Generate text with any LLM
2. Research: Retrieve evidence for claims
3. Revise: Edit text to be consistent with evidence while preserving original properties

### Task Formulation: Editing for Attribution

**Input**: Text passage x produced by any generation model
**Output**:
- Revised passage y
- Attribution report A containing evidence snippets e₁, ..., eₘ

**Two Evaluation Dimensions**:

#### Attribution Metrics

**Sentence-level AIS (Attributable to Identified Sources)**:
```
AttrAIS(y, A) = avg_{s∈y} AIS(s, A)
```
- Binary score for each sentence
- Requires annotator to affirm "According to A, s" is appropriate
- Maximum M=5 evidence snippets (found sufficient through manual inspection)

**Automated Attribution (Attrauto)**:
- Uses NLI model from Honovich et al. (2022)
- For each sentence s and evidence e: compute entailment probability NLI(e, s)
- Decontextualize sentences before scoring
```
Attrauto(y, A) = avg_{s∈y} max_{e∈A} NLI(e, s)
```

#### Preservation Metrics

**Intent Preservation (Presintent)**:
- Human judgment: Does revision completely preserve original intent?
- Binary: 1.0 if completely preserved, 0.0 otherwise

**Levenshtein Preservation (PresLev)**:
```
PresLev(x, y) = max(1 - Lev(x, y)/length(x), 0)
```
- 1.0 if identical, 0.0 if completely rewritten
- Penalizes any unnecessary changes

**Combined Metric**:
```
Prescomb(x, y) = Presintent(x, y) · PresLev(x, y)
```

**F1AP**: Harmonic mean of AttrAIS and Prescomb

### RARR Architecture (Figure 2 in paper)

#### Research Stage

**1. Query Generation (CQGen - Comprehensive Question Generation)**:
- Few-shot prompting with PaLM to generate questions covering all verifiable aspects
- Sample 3 times and union results for diversity
- Example prompt structure shows text → multiple verification queries

**2. Evidence Retrieval**:
- Google Search for K=5 web pages per query
- Extract candidate snippets via 4-sentence sliding window
- Rank by query-document relevance using T5-large model (fine-tuned on MS MARCO)
- Keep top J=1 evidence per query

#### Revision Stage

**1. Agreement Model**:
- For each (query, evidence) pair, check if evidence agrees with current text
- Uses chain-of-thought prompting
- Explicitly states implied answers from both text and evidence before judgment
- If agreement detected → skip editing

**2. Edit Model**:
- Only runs if disagreement detected
- Chain-of-thought: identify specific span to edit, then generate revision
- Reject edits with >50 characters or >0.5× original text length change

**3. Attribution Report Selection**:
- Select ≤M=5 snippets maximizing coverage of all queries
- Coverage score: Cover(A, q₁:N) = Σᵢ max_{e∈A} Sᵣₑₗₑᵥₐₙ꜀ₑ(qᵢ, e)
- Exhaustive search for optimal subset

### Few-Shot Prompts (Detailed Examples)

**Query Generation Example**:
```
You said: Your nose switches back and forth between nostrils. When you sleep, you switch about every 45 minutes...
To verify it,
a) I googled: Does your nose switch between nostrils?
b) I googled: How often does your nostrils switch?
c) I googled: Why does your nostril switch?
d) I googled: What is nasal cycle?
```

**Agreement Model Example**:
```
You said: [text about 45 minutes]
I checked: How often do your nostrils switch?
I found this article: [evidence about 2 hours]
Your nose's switching time is about every 2 hours, not 45 minutes.
This disagrees with what you said.
```

**Edit Model Example**:
```
This suggests 45 minutes switch time in your statement is wrong.
My fix: Your nose switches back and forth between nostrils. When you sleep, you switch about every 2 hours...
```

### Experimental Setup

**Evaluation Benchmarks** (150 dev + 150 test passages each):

| Dataset | Model | Description |
|---------|-------|-------------|
| NQ | PaLM/GPT-3 | Factoid statements (knowledge-intensive) |
| SQA | PaLM/GPT-3 | Reasoning chains (multi-step) |
| QReCC | LaMDA/GPT-3 | Dialog responses (context-dependent) |

**Baselines**:
- **EFEC**: T5-based editor fine-tuned on FEVER, conditions on multiple evidence at once
- **LaMDA Research**: Dialog system's built-in research-and-revise workflow

### Main Results (Table 1)

| Setting | Model | Attribution (AIS before→after) | Preservation (Prescomb) | F1AP |
|---------|-------|-------------------------------|------------------------|------|
| NQ | EFEC | 35.4→48.3 | 10.4 | 17.1 |
| NQ | LaMDA | 18.3→30.4 | 21.1 | 24.9 |
| NQ | **RARR** | 35.4→**43.4** | **83.1** | **57.0** |
| SQA | EFEC | 24.5→51.7 | 3.8 | 7.1 |
| SQA | LaMDA | 15.8→27.0 | 33.7 | 30.0 |
| SQA | **RARR** | 24.5→**31.5** | **84.6** | **45.9** |
| QReCC | EFEC | 13.2→48.7 | 23.7 | 31.9 |
| QReCC | LaMDA | 16.0→27.1 | 12.0 | 16.6 |
| QReCC | **RARR** | 13.2→**28.3** | **78.1** | **41.5** |

**Key Findings**:
- RARR preserves original intent >90% of the time; EFEC and LaMDA only 6-40%
- RARR increases attribution by up to 13% absolute while changing only 10-20% of text
- RARR is the only method with robust F1AP across all three diverse datasets

### Ablation Studies

**Query Generation Alternatives**:
| Method | NQ F1AP | NQ F1AP (no wiki) |
|--------|---------|-------------------|
| Full RARR (CQGen) | 68.1 | 64.3 |
| Sentences as queries | 67.8 | 60.3 |
| Entire input as query | 63.8 | - |

- CQGen more robust to corpus shifts
- Sentences-as-queries susceptible to confirmation bias (retrieves echoing content)

**Agreement Model Necessity**:
- Without agreement model: preservation drops (82.6 vs 89.6 on NQ)
- Leads to spurious edits when evidence doesn't explicitly disagree

**Downstream Task Impact**:
- NQ: RARR actually improves short answer accuracy by ~5%
- SQA: Slight drop (~2.6%) due to noisy retrievals and unchanged subsequent reasoning

### Qualitative Analysis

**Successful Revision Types**:
- Entity corrections (Figure 7a): "Henry Roth" → "Jules Shear"
- Number corrections (Figure 7b): "1745" → "1825"
- Larger necessary revisions (Figure 7c): Multiple author names corrected

**Failure Cases**:
- Misleading evidence (Figure 7d): MeTV confused with original NBC airing
- Incomplete reasoning updates (Figure 7e): "Homer has 4 fingers" corrected but "needs one hand to count to 5" not updated

**Human Oracle**: Manual editing achieved 88% preservation with 100% attribution → 93.6 F1AP, indicating significant headroom

### Limitations Acknowledged

1. **Task-specific preservation needs**: Poetry requires rhyme preservation; long documents need discourse coherence
2. **Attribution scope ambiguity**: Self-evident statements, subjective claims, varying directness levels
3. **Conflicting evidence**: Current implementation is "permissive"—any supporting source counts
4. **Computational cost**: Requires large LLM prompting; suggests distillation to smaller models

---

## Paper 3: RECLAIM - Ground Every Sentence: Improving Retrieval-Augmented LLMs with Interleaved Reference-Claim Generation

### Problem Statement

Prior attributed text generation (ATG) methods provide coarse-grained citations (passage-level or paragraph-level), which:
- Contain significant irrelevant information
- Increase fact-checking time
- Reduce verifiability

**RECLAIM's Goal**: Enable sentence-level fine-grained attributions in long-form question answering through an interleaved generation approach.

### Task Formulation

**Input**: Query q + Reference passages D from RAG system

**Output**: O = {r₁, c₁, r₂, c₂, ..., rₙ, cₙ} where:
- rᵢ = fine-grained reference (sentence-level citation text from passages)
- cᵢ = claim (portion of answer grounded in rᵢ)

**Key Difference from Prior Work**: References are explicit text excerpts, not just source numbers/URLs.

### Method Architecture

#### Training Dataset Construction (Section 3.2)

**Base Datasets**: WebGLM-QA and ELI5

**Step 1 - Reference Passage Retrieval**:
- WebGLM-QA: Already has aligned passages; add irrelevant noise via:
    - BM25 top-100 retrieval
    - Reranker scoring (BGE-M3)
    - Select high-BM25-rank but low-rerank-score passages as noise
- ELI5: BM25 top-100 with reranking; includes natural noise

**Step 2 - Model Answer Generation**:
- WebGLM-QA: Use original model responses
- ELI5: Generate with Llama-3.1-405B-Instruct from top-5 passages (due to distribution mismatch with human answers)

**Step 3 - Multi-Stage Citation Search**:
1. Use Llama-3.1-405B-Instruct to segment answers into clauses
2. For each clause, identify minimal citation set from passages
3. **NLI Filtering**: Discard pairs where entailment probability < 0.8

**Dataset Statistics (Table 1)**:
| Dataset | Samples | Avg Answer Len | Avg Citation Len | Avg Passage Len |
|---------|---------|----------------|------------------|-----------------|
| WebGLM-QA | 4,582 | 98.53 | 154.62 | 326.57 |
| WebGLM-QA Extended | 2,605 | 83.85 | 114.10 | 396.16 |
| ELI5 Default | 3,383 | 91.33 | 132.34 | 545.01 |
| ELI5 Rerank | 2,653 | 107.54 | 158.52 | 542.02 |

#### Generation Methods

**Method 1: RECLAIM Unified (Section 3.3)**
- Single fine-tuned LLM
- One-step generation: Query + Passages → Full attributed answer
- Simple but less effective for attribution quality

**Method 2: RECLAIM w/IG (Interleaving Generation) (Section 3.4)**

Two separately trained models alternating:

**ReferModel** (Reference Generation):
```
ReferGen = {rᵢ₊₁ | Prompt, {r₁, c₁, ..., rᵢ, cᵢ}}
```
- Receives full context (instruction, query, passages, prior output)
- Selects next sentence-level citation

**ClaimModel** (Claim Generation):
```
ClaimGen = {cᵢ₊₁ | {r₁, c₁, ..., rᵢ, cᵢ, rᵢ₊₁}}
```
- Receives only prior output (NOT full context)
- Key insight: Citation already contains answer information; full context unnecessary and potentially harmful

**Training Data Transformation for ClaimModel**:
| Input | Output |
|-------|--------|
| `According to <ref>citation text</ref>` | `We can know that: <claim>answer clause</claim>` |

#### Constrained Decoding (Section 3.5)

Ensures generated references exactly match passage sentences:

1. **Sentence Segmentation**: Split passages into individual sentences
2. **Encoding**: Tokenize sentences into vectors
3. **Prefix Tree Construction**: Organize encoded sentences into trie structure
4. **Constrained Inference**: At each step, select highest-probability token satisfying current trie path; continue until leaf node

**Benefit**: Guarantees 100% consistency between citations and source passages.

### Evaluation Framework

#### Datasets

| Dataset | Samples | Question Type | Passages |
|---------|---------|---------------|----------|
| ASQA | 948 | Ambiguous factoid | 5 oracle Wikipedia |
| ELI5 | 1,000 | Why/How/What | 5 oracle Sphere |
| EXPERTQA | 1,000 | Expert domain | 5 BM25+reranked |

#### Metrics

**Answer Quality**:
- **EM Rec.** (ASQA): Exact match recall of gold short answers
- **Claim Rec.** (ELI5, EXPERTQA): % key claims in answer (via NLI)
- **MAUVE**: Neural-human text similarity for fluency

**Citation Quality**:
- **CAS** (Correct Attribution Score): % sentences fully supported by citations
- **CRS** (Citation Redundancy Score): % non-redundant citation sentences

**Verifiability**:
- **Citation Length**: Shorter = easier fact-checking
- **AR** (Attribution Ratio): % attributed sentences
- **CR** (Consistency Ratio): String match between citations and passages

### Main Results (Table 2)

**ASQA Dataset**:
| Method | MAUVE | EM Rec. | CAS | CRS |
|--------|-------|---------|-----|-----|
| ALCE (ChatGPT) | 64.4 | 48.9 | 74.5 | 72.7 |
| ALCE (Llama3-8B) | 79.2 | 55.2 | 54.7 | 54.6 |
| Self-RAG | 70.6 | 38.7 | 53.3 | 66.2 |
| RS+RL | 84.4 | 47.7 | 75.5 | 69.4 |
| FRONT | 72.5 | 56.5 | 72.2 | 66.0 |
| RECLAIM 3-shot | 90.1 | 50.7 | 77.7 | 62.1 |
| RECLAIM Unified | 89.8 | 53.3 | 68.2 | 58.9 |
| **RECLAIM w/IG** | 88.1 | 53.5 | **92.1** | **86.1** |

**ELI5 Dataset**:
| Method | MAUVE | Claim Rec. | CAS | CRS |
|--------|-------|------------|-----|-----|
| ALCE (ChatGPT) | 59.4 | 21.3 | 57.8 | 56.0 |
| **RECLAIM w/IG** | 71.6 | 17.8 | **89.9** | **67.5** |

**Cross-Dataset Performance**:
- RECLAIM w/IG achieves ~90% CAS across all three datasets
- Compared to ALCE+ChatGPT: +31.3% CAS, +16.7% CRS, +25.7% MAUVE (avg), -6.0% accuracy

### Verifiability Analysis (Table 3)

| Method | Citation Len | Claim Len | CR | AR |
|--------|-------------|-----------|-----|-----|
| ALCE | 536.3 | 85.5 | 100.0 | 91.3 |
| RS+RL | 327.0 | 39.9 | 100.0 | 94.5 |
| RECLAIM 3-shot | 106.8 | 59.8 | 75.5 | 100.0 |
| RECLAIM Unified | 77.9 | 52.9 | 100.0 | 100.0 |
| **RECLAIM w/IG** | **82.8** | 68.9 | **100.0** | **100.0** |

**Key Finding**: RECLAIM's citation length is ~22% of ALCE, dramatically reducing fact-checking time.

### Ablation Studies (Table 4)

| Method | ASQA CAS | ELI5 CAS | EXPERTQA CAS |
|--------|----------|----------|--------------|
| ReferModel-Only w/Extend | 70.5 | 59.9 | 61.7 |
| ReferModel-Only w/Sum | **94.5** | **96.2** | **97.7** |
| ClaimModel-Only | 89.6 | 84.4 | 85.6 |
| **RECLAIM w/IG (both)** | 92.1 | 89.9 | 90.1 |

**Insights**:
- ReferModel-Only w/Sum achieves highest CAS but sacrifices fluency (MAUVE: 29.2 vs 88.1)
- ClaimModel-Only has higher accuracy but lower fluency, produces excessively long answers
- Full RECLAIM w/IG achieves best balance

### NLI Filtering Necessity (Figure 3)

Training on filtered vs. unfiltered data:
- **After filtering**: Higher scores on all metrics (Correct, Fluency, CAS, CRS)
- Validates importance of NLI threshold (θ=0.8) for training data quality

### Faithfulness Analysis (Figure 4)

Plots faithfulness (to full reference passages) vs. accuracy:
- **Inverse relationship observed**: Higher faithfulness correlates with lower accuracy
- RECLAIM w/IG achieves highest faithfulness (~0.95)
- Interpretation: Stricter grounding in citations limits information sources but improves credibility

### Limitations Acknowledged

1. **Training data specificity**: Only LFQA tasks; reduced generalization
2. **Output length**: Explicit citations double output length
3. **Multi-source synthesis**: Not specifically enhanced for combining multiple references
4. **Template rigidity**: Not all sentences need citations; current format forces it
5. **Multi-hop reasoning**: Limited capability for synthesizing conclusions across sources

---

## Cross-Paper Synthesis

### Complementary Contributions

| Paper | Focus | Key Innovation | Primary Metric |
|-------|-------|----------------|----------------|
| RAGAS | Evaluation | Reference-free metrics for RAG quality dimensions | Faithfulness, AR, CR |
| RARR | Post-hoc correction | Research-and-revise to add attribution after generation | Attribution + Preservation |
| RECLAIM | Generation method | Interleaved reference-claim generation for sentence-level attribution | CAS (90%+) |

### Evolution of Attribution Granularity

1. **RAGAS (2023)**: Evaluates existing systems; doesn't modify generation
2. **RARR (2022)**: Retrofits attribution post-hoc; preserves original text style
3. **RECLAIM (2024)**: Generates with attribution built-in; sentence-level citations

### Shared Technical Approaches

- **NLI Models**: All three use NLI for entailment verification
- **LLM Prompting**: Few-shot prompting central to RAGAS and RARR
- **Statement Decomposition**: RAGAS extracts statements; RECLAIM generates claims

### Trade-offs Identified

| Dimension | Higher Attribution | Higher Answer Quality |
|-----------|-------------------|----------------------|
| RARR | Lower preservation | Better preservation |
| RECLAIM | Lower accuracy (6%) | Higher accuracy |
| General | More constraints | More flexibility |

This tension between attribution quality and answer quality/flexibility represents a fundamental challenge in the field that all three papers acknowledge.


# Detailed Summary: VERISCORE - Evaluating the Factuality of Verifiable Claims in Long-Form Text Generation

## Paper Overview

**Authors:** Yixiao Song, Yekyung Kim, Mohit Iyyer (UMass Amherst)
**Published:** arXiv:2406.19276v1, June 27, 2024
**Repository:** https://github.com/Yixiao-Song/VeriScore

---

## 1. Problem Statement and Motivation

### The Core Challenge
Existing factuality metrics like FACTSCORE (Min et al., 2023) and SAFE (Wei et al., 2024) decompose text into "atomic claims" and verify each against knowledge bases like Wikipedia. However, these approaches have a critical flaw: **they assume all extracted content is verifiable**, which doesn't hold for most real-world generation tasks.

### Why Existing Metrics Fail

**Issue 1: Over-decomposition**
Many long-form outputs contain complex assertions that cannot be made "atomic" without losing critical context. For example:
> "The impeachment of Andrew Johnson set a precedent that impeachment should be reserved for clear cases of serious misconduct rather than political disagreements."

This cannot be meaningfully split into smaller claims without losing the causal relationship.

**Issue 2: Extraction of Unverifiable Content**
FACTSCORE and SAFE extract everything from text, including:
- Metaphors (e.g., "Betacyanin is like a superhero cape")
- Subjective opinions ("I am 1000% better")
- Personal experiences ("My grandpa assembled a TV")
- Advice and instructions
- Hypotheticals

This unfairly penalizes models during aggregation because unverifiable claims cannot be supported by evidence.

**Issue 3: Domain Limitation**
FACTSCORE was optimized for biography generation, a fact-dense and formulaic domain. It doesn't generalize well to:
- Long-form question answering (LFQA)
- Creative writing
- Multi-domain instruction following

---

## 2. VERISCORE: The Proposed Solution

### Key Innovations

**Innovation 1: Verifiable Claims Only**
VERISCORE introduces the concept of **verifiable claims** based on linguistic frameworks of events and states (Maienborn, 2003, 2019):

> **Verifiable claims** describe a single event or state with all necessary modifiers (spatial, temporal, or relative clauses) that help denote entities or events in the real world.

**Innovation 2: Inter-Sentence Context**
VERISCORE considers context from surrounding sentences when extracting claims, eliminating the need for expensive claim revision steps present in SAFE.

**Innovation 3: Dual Implementation**
- **Closed-model version:** Uses GPT-4/GPT-4o with few-shot prompting
- **Open-weight version:** Fine-tuned Mistral-7B and Llama3-8B models for cost efficiency

---

## 3. The VERISCORE Pipeline

### Stage 1: Claim Extraction

**Sliding Window Approach**
The extraction uses a sliding window format:
```
(context1: 0-3 sentences) <SOS>focused sentence<EOS> (context2: 0-1 sentence)
```

The LLM extracts verifiable claims from the focused sentence while using context to:
- Resolve pronouns
- Complete definite phrases
- Maintain temporal/spatial references

**For QA Tasks:** The question is always prepended to context1
**For Non-QA Tasks:** The first sentence of a paragraph is prepended if the paragraph exceeds 5 sentences

**What Gets Excluded:**
- Stories and personal experiences
- Hypotheticals (subjunctive mood)
- Subjective opinions
- Advice and instructions
- Suggestions

### Stage 2: Evidence Retrieval

- Uses Google Search via the Serper API
- Each claim c serves as the search query
- Retrieves top n ≤ 10 search results
- Collects title, snippet, and link for each result
- Combines results into an evidence list

**Key Finding:** Top 5 search results show higher utility (>30% informative), with the first result at 35.6% usefulness. Results 6-9 range from 27-29.2%, and the last result drops to only 13.3%.

### Stage 3: Claim Verification

**Three Possible Outcomes:**

| Scenario | Definition |
|----------|------------|
| **Supported** | All parts of the claim are supported by evidence; no evidence contradicts any part |
| **Contradicted** | At least one part is contradicted by evidence |
| **Inconclusive (a)** | At least one part is neither supported nor contradicted |
| **Inconclusive (b)** | At least one part is both supported and contradicted by different evidence |

**Practical Simplification:** Since direct contradictions are rare (<3% in human studies), VERISCORE combines contradicted and inconclusive into a single **unsupported** category, making verification a binary classification task.

### Stage 4: Score Calculation

VERISCORE adopts the **F1@K metric** from SAFE:

**K** = minimum number of factual claims a response must contain for perfect recall (set as median across all model responses per domain)

**Formulas:**
- S(r) = number of supported claims in response r
- P(r) = S(r) / |C| (precision)
- R(r) = min(S(r)/K, 1) (recall)
- F1@K(r) = 2P(r)R(r) / (P(r) + R(r)) if S(r) > 0, else 0
- **VERISCORE** = average F1@K across all responses

---

## 4. Validation Studies

### Human Evaluation on Claim Extraction

**Setup:**
- 15 randomly sampled texts from 8 diverse datasets
- Pairwise comparison: VERISCORE vs. SAFE
- 3 human annotators
- 360 total data points
- Half extracted by GPT-4, half by Claude 3

**Results:**
- **Fleiss κ = 0.7662** (substantial agreement)
- Full agreement on 99/120 triple-annotated items
- **SAFE preferred only 26 times** out of 360, with 19 being marginal preferences
- **VERISCORE preferred 93%** of the time, even on biography generation

**Annotator-Identified Issues with SAFE:**

1. **Indiscriminate extraction:** Extracts subjective content and personal experiences
2. **Over-decomposition causing meaning overlap:**
   ```
   "Longwood House is a place."
   "Longwood House is a Napoleonic Museum."
   "Longwood House is one of the best Napoleonic Museums."
   "Longwood House is one of the best Napoleonic Museums in the world."
   ```
3. **Trivial or vague claims:** "3.2 is a number" or "All My Sons has key themes"

### Verifiable-to-Sentence Ratio Analysis

The ratio of verifiable claims to sentences varies significantly by domain:

| Domain | VerRatio |
|--------|----------|
| WritingPrompts (creative) | 0.03 |
| ShareGPT | 0.92 |
| FreshQA | 1.00 |
| ELI5 | 1.71 |
| AskHistorians | 1.90 |
| Biography | 2.08 |
| LongFact | 2.24 |
| FreshBooks | 2.31 |

This demonstrates VERISCORE effectively discriminates between verifiable and unverifiable content.

### Human Evaluation on Claim Verification

**Setup:**
- 320 GPT-4-extracted claims
- 3 annotators
- Two-level assessment: evidence level and claim level

**Results:**
- **82% complete agreement** on 50 triple-annotated items
- **Fleiss κ = 0.7316** (substantial agreement)
- **Only 55% of claims are supported** by Google Search results
- **~42% are inconclusive**
- **<3% are directly contradicted**

**Reasons for Inconclusive:**
1. Claim too general (e.g., "A systematic review on sex differences... was published in 2019" without specifying which review)
2. No direct mention of claim parts or their connections in evidence

### Automatic Verifier Selection

Tested Mixtral-8×22B, Claude 3 Opus, GPT-4, and GPT-4o on human-annotated data:

**Winner: GPT-4o** with ternary labels achieved the most balanced performance:
- Overall F1: 0.841
- Supported F1: 0.841
- Unsupported F1: 0.731

---

## 5. Open-Weight Implementation

### Fine-Tuned Claim Extractor

**Base Model:** Mistral-7B-Instruct-v0.2
**Training Data:** GPT-4-generated extractions from:
- 100 samples from Scruples
- 200 samples from other datasets
- Model responses from 12 different LLMs

**Data Processing:**
- 99,592 input-output pairs total
- 80% removal of short marked sentences (<10 chars)
- 50% removal of "No verifiable claim" outputs
- Split: 95% train, 4% validation, 1% test

**Quality Metrics:**
- Exact match: 43.7%
- RougeL: 0.801
- Cohen's κ between GPT-4 and Mistral: 0.4320 (moderate agreement)

### Fine-Tuned Claim Verifier

**Base Model:** Llama3-8B-Instruct
**Training Data:** 13,403 GPT-4o-generated verification labels
- 9,996 supported
- 3,407 unsupported (triplicated during training to address imbalance)

**Performance:** F1 = 0.841 on human-annotated data

**Cost Savings:** The fine-tuned pipeline eliminates ~$1,038 USD cost for evaluating 400 GPT-4o generations.

---

## 6. Benchmarking Results

### Datasets Used

| Dataset | Description | VerRatio |
|---------|-------------|----------|
| LongFact | 38 topics, object & concept prompts | 2.24 |
| Biography | From PerplexityAI, InstructGPT, ChatGPT | 2.08 |
| ELI5 | r/explainlikeimfive Q&A | 1.71 |
| AskHistorians | r/AskHistorians Q&A | 1.90 |
| FreshBooks | Continuations of 20 non-fiction books (2023-2024) | 2.31 |
| ShareGPT | User-shared ChatGPT conversations | 0.92 |
| FreshQA | Dynamic QA with changing answers | 1.00 |
| WritingPrompts | r/WritingPrompts creative stories | 0.03 |

### Model Performance Rankings (Average VERISCORE)

| Rank | Model | Avg. Score |
|------|-------|------------|
| 1 | **GPT-4o** | **66.5** |
| 2 | GPT-4-0125-preview | 65.5 |
| 3 | Claude-3-Opus | 61.2 |
| 4 | **Mixtral-8×22B-Instruct** (best open) | **58.5** |
| 5 | DBRX-Instruct | 57.2 |
| 6 | Mixtral-8×7B-Instruct | 56.9 |
| 7 | GPT-3.5-turbo-0613 | 56.0 |
| 8 | Claude-3-Sonnet | 54.4 |
| 9 | Claude-3-Haiku | 52.2 |
| 10 | Mistral-7B-Instruct-v0.2 | 51.5 |
| 11 | OLMo-7B-Instruct | 49.6 |
| 12 | Qwen1.5-1.8B-Chat | 48.2 |
| 13 | GPT-3.5-turbo-1106 | 43.5 |
| 14 | Vicuna-7B-v1.5 | 43.3 |
| 15 | Mistral-7B-Instruct-v0.1 | 39.6 |
| 16 | Gemma-2B-it | 27.4 |

### Key Findings

**Finding 1: Closed models are more factual**
GPT models generally outperform Claude 3 models, though DBRX and Mixtral approach Claude 3 performance.

**Finding 2: Model size correlates with VERISCORE**
Larger models consistently achieve higher scores across domains.

**Finding 3: Cross-domain performance doesn't correlate well**
Kendall's τ correlations between domains show that a model's factuality on one task doesn't predict performance on others:

| Domain Pair | Kendall's τ |
|-------------|-------------|
| LF - Bio | 0.65 |
| LF - ELI5 | 0.71 |
| ELI5 - AskH | 0.82 |
| Bio - S.GPT | 0.39 |
| FBs - S.GPT | 0.63 |

**Implication:** Multiple tasks are needed for comprehensive factuality evaluation.

**Finding 4: F1@K favors longer outputs**
Models generating shorter, to-the-point responses are penalized even if they have higher precision. This is debatable since longer responses provide more details but tend to contain less accurate later content.

---

## 7. Qualitative Analysis and Limitations

### Challenge 1: Claim Complexity Outside Entity-Centric Tasks

**Problem:** Claims from LFQA and other domains are inherently longer than biography claims:
- FACTSCORE biography claims: average 7 words, max 18
- ELI5 claims: average 12 words, max 25

**Example of irreducible complexity:**
> "Travelers and crusaders during the medieval period depended on established infrastructure to secure clean and consistent sources of water."

Cannot be meaningfully split without losing causal relationships.

**Example of partial verification insufficiency:**
> "[Chuck Norris's victory in the 1968 World Full-Contact Karate Championships] solidified [his reputation as one of the best martial artists in the world]."

Verifying the bracketed parts separately doesn't verify the connection "solidified."

### Challenge 2: Google Search Limitations

**Supported claims resemble encyclopedic writing:**
Verification succeeds via semantic or string match.

**Unsupported claims often lack direct contradicting evidence:**
Example:
> "Japanese people encountered tigers in the form of stuffed animals before the Meiji era."

No search results mention "Meiji era" and "stuffed tigers" together. Extensive search found no supporting or contradicting evidence.

**Some claims require extensive inference:**
> "Marshall's leadership and strategic acumen ensured the maneuver was carried out flawlessly during a field maneuver in the Philippines."

Such claims encapsulate achievements or historical movements that require inferring from large document bodies, beyond search snippet capabilities.

### Formal Limitations Acknowledged

1. **Definition Challenge:** "Verifiable claim" remains a working definition rather than formal. Determining whether a sentence describes one event or multiple states is philosophically complex.

2. **Processing Speed:** Sliding window approach is slow (~4 hours for 400 GPT-4o responses on one RTX8000 GPU without parallelization; each response averages 40 sentences).

3. **Occasional Extraction of Unverifiable Claims:** For extremely non-factual content (e.g., WritingPrompts), small amounts of unverifiable claims may still be extracted.

4. **Hyperparameter Optimization:** Authors did not exhaustively search for optimal fine-tuning hyperparameters due to resource constraints.

---

## 8. Issues with SAFE (Detailed Analysis)

The paper dedicates significant attention to SAFE's shortcomings:

1. **Minimal prompt modification:** SAFE only adds a brief task description to FACTSCORE's biography-optimized prompt.

2. **Multi-step overhead:** SAFE's three-step pipeline (extraction → revision → relevance check) adds 35 minutes processing time per 100 claims and ~$1.7 cost just for prompt templates.

3. **Counterproductive relevance check:** The relevance check removes 11% of claims, of which 58% are actually relevant. It removes claims like "Castello Maniance in Siracusa, Italy was built from 1232 to 1239" as irrelevant when asked about beautiful castle interiors.

4. **Limited evaluation:** SAFE was only validated on FACTSCORE's biography data despite claiming to target broader domains.

---

## 9. Technical Details

### Claim Extraction Prompt Design

**For Non-QA Inputs:**
- 13 few-shot examples
- Instructions to focus on named entities and numbers
- Explicit exclusions for stories, experiences, hypotheticals, subjective content
- Output format: bullet list of claims or "No verifiable claim."

**For QA Inputs:**
- 10 few-shot examples
- Question always prepended to context
- Additional instruction not to extract claims from the question itself

### Verification Prompt Design

**Binary Classification Template:**
```
Supported: A claim is supported if everything in the claim is supported 
and nothing is contradicted by search results.

Unsupported: If a claim is not supported, mark it as unsupported.
```

**Note for Claude 3:** Required rearranging element order (search results → claim → task description → decision) to ensure consistent output formatting.

### FreshBooks Dataset Construction

20 non-fiction books published 2023-2024, including:
- "It's OK to Be Angry About Capitalism" (Bernie Sanders)
- "Takeover: Hitler's Final Rise to Power" (Timothy Ryback)
- "The Making of a Leader: George C. Marshall" (Josiah Bunting III)
- Various academic handbooks on cryptography, green finance, language studies

10 paragraphs from each book, all from chapter/section beginnings.

---

## 10. Related Work Context

### Factual Error Detection
- Prior work targets individual sentences (FEVER, FEVEROUS, AVERITEC)
- Li et al. (2023) test paragraph-level detection but don't locate errors

### Long-Form Factuality Evaluation
- Decomposition approach common in: Kamoi et al., Gao et al., Wang et al., Chern et al., Wanner et al., Wei et al., Guan et al., Chen et al.
- Retrieval commonly used for up-to-date knowledge
- Pipeline helps generate post-hoc citations and iterative improvement

---

## 11. Key Contributions Summary

1. **New metric (VERISCORE)** that exclusively focuses on verifiable claims, with formal definition grounded in linguistic event/state frameworks

2. **Comprehensive validation** through human studies confirming:
    - Superior claim extraction (93% preference over SAFE)
    - High inter-annotator agreement (κ > 0.73)
    - Reliable verification

3. **Dual implementation:**
    - High-quality closed-model version (GPT-4/GPT-4o)
    - Cost-effective open-weight version (fine-tuned Mistral/Llama3)

4. **Large-scale benchmarking** of 16 LLMs across 8 diverse domains with 50 prompts each

5. **Key insight:** Cross-task factuality correlation is low, necessitating multi-task evaluation for comprehensive assessment

6. **Identification of fundamental limitations** in decompositional factuality evaluation for complex, non-entity-centric claims

---

## 12. Future Directions Suggested

1. **Improve verification of complex claims** beyond semantic/string matching
2. **Develop faster claim extraction** without sliding windows
3. **Formal definition refinement** for verifiable claims
4. **Address the F1@K length bias** for fair cross-model comparison
5. **Sophisticated search strategies** for claims requiring extensive background reasoning