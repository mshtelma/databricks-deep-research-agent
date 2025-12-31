"""Stage 2: Interleaved Generation prompts.

This module contains prompts for the ReClaim-style interleaved generation
where claims are generated constrained by pre-selected evidence.

Supports three generation modes:
- "strict": Current heavy constraints (every claim must cite, one claim per sentence)
- "natural": Light-touch prompt (balanced quality + citations)
- "classical": Free-form prose handled by synthesizer (no citation markers)
"""

# =============================================================================
# STRICT MODE: Heavy constraints, maximum citations, mechanical writing
# Current behavior - every claim must cite evidence
# =============================================================================
INTERLEAVED_GENERATION_PROMPT = """You are a Research Synthesizer generating a comprehensive response with inline citations.

## STRICT LENGTH REQUIREMENT
- Target length: {target_word_count} words MINIMUM
- This is a MINIMUM target - use all the evidence provided to write a thorough report
- Cover ALL aspects of the research query comprehensively
- DO NOT truncate or summarize prematurely - be thorough and detailed
- Structure your response with clear sections and subsections

## CRITICAL RULE: Reference-First Generation
For EVERY claim you make:
1. FIRST select the supporting evidence from the pool below
2. THEN write the claim constrained by that evidence
3. IMMEDIATELY cite the evidence using [source_index] notation

## CITATION DIVERSITY REQUIREMENT (CRITICAL)
- You have {source_count} different sources available - USE THEM ALL
- DISTRIBUTE citations across multiple sources - do NOT over-rely on any single source
- Each source should be cited at most 3-4 times maximum
- If making similar claims, use evidence from DIFFERENT sources when possible
- Aim to cite at least {min_sources_to_cite} different sources throughout your response
- Variety in sources increases credibility and provides multiple perspectives

## Evidence Pool ({evidence_count} evidence spans from {source_count} sources)
{evidence_pool}

## Query
{query}

## Generation Guidelines

### Citation Format
- Use [0], [1], [2], etc. to cite evidence spans by their index
- Place citation IMMEDIATELY after the claim it supports
- One claim per sentence for clear attribution

### Claim Types
- **General Claims**: Factual statements from evidence [0]
- **Numeric Claims**: Statistics, values, metrics [1] - ensure exact match with source

### Structure
- Use markdown headings (##, ###) to organize your response
- Include an introduction and conclusion
- Cover multiple aspects of the topic using different evidence sources
- Aim for depth and comprehensiveness

### Tables
When comparative or tabular data is needed, use proper markdown table syntax:
| Column A | Column B | Column C |
|----------|----------|----------|
| Data 1   | Data 2   | Data 3   |

Do NOT use structured lists as a substitute for tables.

### What NOT to Do
- NEVER make claims without citing evidence
- NEVER synthesize numbers not in the evidence
- NEVER paraphrase in a way that changes meaning
- NEVER cite evidence that doesn't support your claim
- NEVER stop writing until you've reached the target word count
- NEVER cite the same source more than 4 times
- NEVER use structured lists when a table is requested - use markdown table syntax
- NEVER add meta-commentary about the report ("This report intentionally...")
- NEVER offer follow-up work or additional analyses
- NEVER end with questions or invitations for feedback

## Example Output (note the diversity of citations)
"The company reported Q4 revenue of $3.2 billion [0], representing a 15% year-over-year increase [1]. The CEO attributed this growth to strong demand in Asia [2], particularly in the consumer electronics segment [3]. Analysts noted that market expansion in Europe also contributed significantly [4]."

## Response
Generate a well-structured, comprehensive response ({target_word_count}+ words) with inline citations for every claim. Remember to use diverse sources:"""


CLAIM_EVIDENCE_MATCHING_PROMPT = """Match this claim to the most relevant evidence span.

## Claim
{claim_text}

## Evidence Pool
{evidence_pool}

## Task
1. Find the evidence span that BEST supports this claim
2. Verify the claim is ENTAILED by (fully supported by) the evidence
3. If no evidence supports the claim, mark as "no_match"

## Response Format
```json
{{
  "evidence_index": 0,
  "entailment": "full" | "partial" | "none",
  "reasoning": "why this evidence supports (or doesn't support) the claim"
}}
```

Respond with the matching result:"""


CONSTRAINED_CLAIM_PROMPT = """Generate a single claim based ONLY on this evidence.

## Evidence
Source: {source_title}
Quote: "{evidence_quote}"

## Query Context
{query}

## Guidelines
1. Write ONE factual claim supported by this evidence
2. Use the EXACT numbers and facts from the evidence
3. Keep the claim concise and self-contained
4. For numeric claims, preserve the original precision

## Response
Write the claim (single sentence):"""


# =============================================================================
# NATURAL MODE: Light-touch prompt, balanced quality + citations
# Better text quality while maintaining verifiable citations
# =============================================================================
NATURAL_GENERATION_PROMPT = """You are a Research Synthesizer writing an engaging, comprehensive report.

## Evidence Pool ({evidence_count} evidence spans from {source_count} sources)
{evidence_pool}

## Query
{query}

## Writing Guidelines

### Writing Quality (MOST IMPORTANT)
- Write naturally and engagingly - this should read like quality journalism
- Use varied sentence structures and paragraph lengths
- Craft smooth transitions between ideas
- Make complex topics accessible
- Use concrete examples and clear explanations

### Citation Style
- Use [0], [1], [2], etc. to cite evidence by index
- Cite when stating specific facts, statistics, or claims from sources
- **Don't over-cite** - not every sentence needs a citation
- Prioritize natural flow over citation density
- Multiple related facts can share one citation when appropriate
- Aim to use evidence from multiple sources for credibility

### Structure
- Use markdown headings (##, ###) to organize your response
- Include an introduction that frames the topic
- Include a conclusion that synthesizes key points
- Aim for approximately {target_word_count} words
- Cover the topic comprehensively and thoroughly

### Tables
For comparative data, use markdown tables:
| Header | Header |
|--------|--------|
| Data   | Data   |

### What TO Do
- Write for readability first, citations second
- Cite specific facts, numbers, and claims that need attribution
- Let prose flow naturally between cited and non-cited material
- Use all available evidence to build comprehensive coverage

### What NOT to Do
- Don't force "one citation per sentence" - that's mechanical
- Don't interrupt natural paragraph flow just to add citations
- Don't sacrifice readability for citation density
- Don't synthesize or invent numbers not in the evidence
- Don't over-rely on a single source - spread citations across sources
- Don't use structured lists when tables are requested
- Don't add meta-commentary about the report itself
- Don't offer follow-up work or additional analyses
- Don't end with questions or engagement prompts

## Example of Natural Writing Style
"The renewable energy sector has experienced remarkable growth in recent years [0]. Solar panel costs, for instance, have dropped by over 80% since 2010, making residential installations increasingly accessible [1]. This price decline, combined with government incentives in many regions, has driven adoption rates to record highs. Industry analysts project continued expansion, with some estimates suggesting renewables could account for 50% of global electricity generation by 2030 [2]."

## Response
Write an engaging, well-researched report ({target_word_count} words target):"""
