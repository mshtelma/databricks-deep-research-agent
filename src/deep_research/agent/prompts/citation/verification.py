"""Stage 4: Isolated Verification prompts.

This module contains prompts for CoVe-style verification where claims
are checked against evidence IN ISOLATION (no generation context).
"""

ISOLATED_VERIFICATION_PROMPT = """You are a Fact Checker verifying whether a claim is supported by evidence.

## CRITICAL: Isolated Verification
- You are checking this claim IN ISOLATION
- You have NO context about how this claim was generated
- Base your judgment ONLY on the evidence provided below

## Claim to Verify
"{claim_text}"

## Supporting Evidence
Source: {source_title}
URL: {source_url}
Quote: "{evidence_quote}"

## Verdict Categories

### SUPPORTED
The claim is FULLY entailed by the evidence:
- All facts in the claim are present in the evidence
- Numbers match exactly (or are correctly rounded)
- No extrapolation beyond what the evidence states

### PARTIAL
The claim is PARTIALLY supported:
- Some aspects are supported by the evidence
- Other aspects are not mentioned (neither confirmed nor denied)
- May involve reasonable inference from the evidence

### UNSUPPORTED
The claim has NO evidence basis:
- The evidence does not address this claim
- Cannot determine if the claim is true or false
- Different from CONTRADICTED - this is "we don't know"

### CONTRADICTED
The evidence DIRECTLY opposes the claim:
- The evidence states the opposite
- Numbers are clearly different (not a rounding difference)
- Factual disagreement between claim and evidence

## Response Format
```json
{{
  "verdict": "SUPPORTED" | "PARTIAL" | "UNSUPPORTED" | "CONTRADICTED",
  "reasoning": "Detailed explanation of why this verdict was chosen",
  "key_match": "Quote the specific part of evidence that supports/contradicts",
  "issues": ["List any specific issues found"]
}}
```

Verify the claim against the evidence:"""


QUICK_VERIFICATION_PROMPT = """Quickly verify if this claim matches the evidence.

## Claim
"{claim_text}"

## Evidence
"{evidence_quote}"

## Quick Check
1. Is the core fact in the claim present in the evidence? (Y/N)
2. Do any numbers match exactly? (Y/N/NA)
3. Is there any contradiction? (Y/N)

Based on these checks:
- If 1=Y and 3=N: SUPPORTED
- If 1=Partial and 3=N: PARTIAL
- If 1=N and 3=N: UNSUPPORTED
- If 3=Y: CONTRADICTED

Respond with just the verdict (SUPPORTED/PARTIAL/UNSUPPORTED/CONTRADICTED):"""


NEI_DETECTION_PROMPT = """Determine if there is enough evidence to verify this claim.

## Claim
"{claim_text}"

## Available Evidence
{evidence_text}

## Assessment
1. Does the evidence ADDRESS the topic of the claim?
2. Is there SUFFICIENT information to make a judgment?
3. Would a human expert need additional sources?

## Response
If evidence is insufficient, respond: NEI (Not Enough Information)
If evidence is sufficient, respond: SUFFICIENT

Your response:"""


CONTRADICTION_ANALYSIS_PROMPT = """Analyze if there is a direct contradiction between claim and evidence.

## Claim
"{claim_text}"

## Evidence
"{evidence_quote}"

## Contradiction Types
1. **Numeric Contradiction**: Numbers are significantly different
   - "$3.2B revenue" vs "$2.8B revenue" = CONTRADICTION
   - "$3.2B" vs "$3.2 billion" = NOT a contradiction (same value)

2. **Factual Contradiction**: Opposite facts stated
   - "Company was founded in 2010" vs "Established in 2015" = CONTRADICTION

3. **Logical Contradiction**: Mutually exclusive statements
   - "Revenue increased" vs "Revenue declined" = CONTRADICTION

## Response
Is there a direct contradiction?
- YES: [Explain the specific contradiction]
- NO: [The claim may be unsupported but not contradicted]

Your analysis:"""


NUMERIC_QA_PROMPT = """Generate QA pairs to verify a numeric claim.

## Claim with Numeric Value
"{claim_text}"

## Evidence Quote
"{evidence_quote}"

## Numeric Value Details
- Raw value: {raw_value}
- Unit: {unit}
- Entity: {entity}

## Task
Generate 2-3 fact-checking questions about this numeric value.
For each question:
1. Ask a specific question about the numeric value
2. Answer from the CLAIM ONLY
3. Answer from the EVIDENCE ONLY
4. Compare answers to check if they match

## Response Format (JSON array)
```json
[
  {{
    "question": "What was the revenue figure mentioned?",
    "claim_answer": "The claim states $3.2 billion",
    "evidence_answer": "The evidence mentions $3.2B revenue"
  }},
  {{
    "question": "What entity does this value refer to?",
    "claim_answer": "Company X's Q4 2024 revenue",
    "evidence_answer": "Company X fourth quarter revenue"
  }}
]
```

Generate QA pairs for verification:"""


CITATION_CORRECTION_PROMPT = """You are a Citation Corrector. A claim has been flagged as potentially misattributed.
Your task is to decide whether to KEEP, REPLACE, or REMOVE the citation.

## Claim to Verify
"{claim}"

## Current Evidence
{current_evidence}

## Available Evidence Options
{evidence_options}

## Decision Criteria

### KEEP
- The current evidence adequately supports the claim
- Key entities (numbers, names, dates) match between claim and evidence
- The claim can be reasonably inferred from the evidence

### REPLACE
- One of the evidence options better supports the claim
- The current evidence is too tangential or weak
- Choose the option with highest keyword/semantic overlap

### REMOVE
- None of the evidence options support this claim
- The claim appears to be unsupported by any available source
- This is a last resort when no suitable evidence exists

## Response Format (JSON)
```json
{{
  "action": "keep" | "replace" | "remove",
  "evidence_index": <1-5 if replacing, null otherwise>,
  "reasoning": "Brief explanation of your decision"
}}
```

Make your correction decision:"""
