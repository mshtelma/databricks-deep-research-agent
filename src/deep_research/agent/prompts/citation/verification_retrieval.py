"""Stage 7: ARE-style Verification Retrieval prompts.

This module contains prompts for the ARE (Atomic fact decomposition-based
Retrieval and Editing) approach to verifying and revising unsupported claims.

Scientific basis:
- ARE: https://arxiv.org/abs/2410.16708
- FActScore: https://arxiv.org/abs/2305.14251
- SAFE: https://arxiv.org/abs/2403.18802
"""

# =============================================================================
# Atomic Fact Decomposition (FActScore-style)
# =============================================================================

ATOMIC_DECOMPOSITION_PROMPT = """You are decomposing a claim into atomic facts.

## Claim to Decompose
"{claim_text}"

## Instructions
Break this claim into independent, self-contained atomic facts.

Rules for atomic facts:
1. Each fact should be a single, simple statement
2. Each fact should be independently verifiable
3. Replace all pronouns with explicit references
4. Each fact should make sense without the original claim
5. Do NOT add information not present in the original claim
6. Do NOT duplicate facts - each should be unique

## Examples

Input: "OpenAI released GPT-4 in March 2023, which scored 90% on the bar exam."
Output:
1. "OpenAI released GPT-4."
2. "GPT-4 was released in March 2023."
3. "GPT-4 scored 90% on the bar exam."

Input: "Tesla sold 500,000 vehicles in Q3 2024 and became the most valuable automaker."
Output:
1. "Tesla sold 500,000 vehicles in Q3 2024."
2. "Tesla became the most valuable automaker."

Input: "The study found that 75% of participants improved."
Output:
1. "A study was conducted."
2. "75% of participants in the study improved."

## Response Format (JSON)
```json
{{
  "atomic_facts": ["fact 1", "fact 2", ...],
  "reasoning": "Brief explanation of how you decomposed the claim"
}}
```

Decompose the claim into atomic facts:"""


# =============================================================================
# Entailment Check (NLI-style)
# =============================================================================

ENTAILMENT_CHECK_PROMPT = """You are checking if evidence supports a factual claim.

## Fact to Verify
"{fact_text}"

## Evidence
Source: {source_url}
Quote: "{evidence_quote}"

## Task
Determine if the evidence ENTAILS (supports) the fact.

## Entailment Scoring
- 1.0: Evidence directly and explicitly states the fact
- 0.8: Evidence strongly implies the fact with minimal inference
- 0.6: Evidence partially supports the fact (some aspects covered)
- 0.4: Evidence is tangentially related but doesn't confirm
- 0.2: Evidence is about the same topic but doesn't support the fact
- 0.0: Evidence contradicts the fact

## Key Considerations
- Numbers must match exactly (or be close enough to be rounding)
- Entity names must match
- Time periods must align
- Causal claims need explicit support

## Response Format (JSON)
```json
{{
  "entails": true/false,
  "score": 0.0-1.0,
  "reasoning": "Brief explanation of the assessment",
  "key_match": "Quote the specific part that matches or conflicts"
}}
```

Assess whether the evidence entails the fact:"""


# =============================================================================
# Evidence Extraction from Crawled Content
# =============================================================================

EVIDENCE_EXTRACTION_PROMPT = """You are extracting evidence to verify a fact from web content.

## Fact to Verify
"{fact_text}"

## Source Content
URL: {source_url}
Title: {source_title}

Content:
{source_content}

## Task
Find the MOST RELEVANT quote (1-3 sentences) that could verify or refute the fact.

## Guidelines
1. Look for explicit statements about the fact
2. Prioritize quotes with specific numbers, dates, or names
3. Include enough context for the quote to make sense
4. If multiple relevant quotes exist, choose the most authoritative
5. If no relevant content exists, indicate this clearly

## Response Format (JSON)
```json
{{
  "quote_text": "Exact quote from source (or null if none found)",
  "relevance_score": 0.0-1.0,
  "has_numeric_content": true/false,
  "section_heading": "Section name if available (or null)",
  "reasoning": "Why this quote is (or is not) relevant"
}}
```

Extract the most relevant evidence:"""


# =============================================================================
# Claim Reconstruction with Verified/Softened Facts
# =============================================================================

CLAIM_RECONSTRUCTION_PROMPT = """You are reconstructing a claim based on verification results.

## Original Claim
"{original_claim}"

## Atomic Facts with Verification Status
{facts_with_status}

## Instructions
Reconstruct the claim following these rules:

### For VERIFIED facts (entailment_score >= 0.6):
- Keep the fact as-is
- Add [Citation] marker if new evidence was found
- Example: "Tesla sold 500K vehicles [Reuters]"

### For UNVERIFIED facts (entailment_score < 0.6):
- Add hedging language to indicate uncertainty
- Options:
  - "reportedly" - "Tesla reportedly became the most valuable..."
  - "according to some sources" - "According to some sources, Tesla..."
  - "it is claimed that" - "It is claimed that Tesla..."
  - "allegedly" - "Tesla allegedly became..."
- Do NOT remove the fact entirely - keep the information but mark uncertainty

### General Guidelines:
- Maintain natural sentence flow and readability
- Preserve the original claim's structure where possible
- Combine multiple atomic facts back into coherent sentences
- If ALL facts are unverified, soften the entire claim
- If ALL facts are verified, the claim can stand as-is with citations

## Example
Original: "Tesla sold 500,000 vehicles in Q3 2024 and became the most valuable automaker."

Facts:
1. "Tesla sold 500,000 vehicles in Q3 2024" - VERIFIED [Reuters]
2. "Tesla became the most valuable automaker" - UNVERIFIED

Output: "Tesla sold 500,000 vehicles in Q3 2024 [Reuters], though its claim to being the world's most valuable automaker remains disputed."

## Response Format
Return ONLY the reconstructed claim text (no JSON, no explanation).

Reconstruct the claim:"""


# =============================================================================
# Claim Reconstruction - Alternative Softening Strategies
# =============================================================================

CLAIM_SOFTENING_HEDGE_PROMPT = """You are softening a claim that lacks supporting evidence.

## Original Claim
"{claim_text}"

## Task
Rewrite this claim using HEDGING language to indicate uncertainty.

## Hedging Techniques
- "reportedly" - indicates unconfirmed reports
- "allegedly" - indicates unverified allegations
- "according to some sources" - indicates disputed information
- "it is believed that" - indicates unconfirmed belief
- "may have" / "might have" - indicates possibility
- "appears to" - indicates uncertain observation

## Guidelines
- Maintain the claim's informational value
- Make clear this is not definitively established
- Do NOT make up false sources or citations
- Keep similar length to original
- Preserve the core meaning while adding uncertainty

## Response
Return ONLY the softened claim text:"""


CLAIM_SOFTENING_QUALIFY_PROMPT = """You are softening a claim that lacks supporting evidence.

## Original Claim
"{claim_text}"

## Task
Rewrite this claim using QUALIFYING phrases to indicate uncertainty.

## Qualifying Techniques
- "Some evidence suggests that..." - partial support
- "It is believed that..." - unconfirmed belief
- "There are indications that..." - tentative evidence
- "Early reports indicate that..." - unconfirmed reports
- "Preliminary findings suggest..." - initial evidence

## Guidelines
- Start with a qualifying phrase
- Maintain the claim's informational value
- Keep similar length to original
- Preserve the core meaning

## Response
Return ONLY the qualified claim text:"""


CLAIM_SOFTENING_PARENTHETICAL_PROMPT = """You are softening a claim that lacks supporting evidence.

## Original Claim
"{claim_text}"

## Task
Add PARENTHETICAL markers to indicate the claim needs verification.

## Parenthetical Options
- "(unverified)" - at the end of claim
- "(needs citation)" - for missing source
- "(disputed)" - for contested claims
- "(approximate)" - for uncertain numbers

## Guidelines
- Keep the original claim intact
- Add one or more parenthetical markers
- Use sparingly - don't overload with markers
- Preserve readability

## Response
Return ONLY the claim with parenthetical markers:"""


# =============================================================================
# Query Generation for Additional Search
# =============================================================================

VERIFICATION_QUERY_PROMPT = """You are generating a search query to verify a factual claim.

## Fact to Verify
"{fact_text}"

## Original Research Query (for context)
"{research_query}"

## Previous Query Attempts (if any)
{previous_queries}

## Task
Generate a specific search query to find authoritative evidence that would:
1. Directly support OR refute this fact
2. Come from reliable sources (news, academic, official)
3. Contain the specific details mentioned in the fact

## Guidelines for Good Queries
- Focus on the CORE FACT being claimed
- Include key entities (names, organizations, dates)
- Include specific numbers or metrics if present
- Avoid using the exact wording (find independent sources)
- If this is a retry, try different phrasings or synonyms

{reformulation_guidance}

## Response Format (JSON)
```json
{{
  "query": "your search query here",
  "reasoning": "why this query will find relevant evidence",
  "search_strategy": "what type of source you expect to find"
}}
```

Generate the search query:"""


REFORMULATION_GUIDANCE = """
## REFORMULATION REQUIRED
Previous queries did not find supporting evidence. Try:
- Different synonyms or alternative phrasings
- Broader scope (e.g., "Tesla vehicle sales" instead of "Tesla Q3 2024 sales")
- Narrower scope (more specific entity or time period)
- Alternative source types (official reports, press releases, news articles)
- Different language or terminology used in the industry
"""
