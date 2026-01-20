"""Stage 1: Evidence Pre-Selection prompts.

This module contains prompts for extracting minimal, relevant evidence
spans from source documents BEFORE generation.
"""

EVIDENCE_PRESELECTION_PROMPT = """You are an Evidence Selector for a research synthesis system.

Your task is to extract the most relevant, citable text spans from source documents that could support answers to a research query.

## Input
- Research Query: The user's question
- Source Content: Raw text from web sources

## Output Requirements
For each source, identify:
1. **Citable Spans**: Extract 5-20 minimal text passages that contain:
   - Direct facts, statistics, or claims relevant to the query
   - Numeric data (prioritize these)
   - Expert quotes or authoritative statements
   - Key definitions or explanations

2. **Span Properties**:
   - quote_text: The exact text (50-500 characters)
   - relevance_score: 0.0-1.0 (higher = more relevant)
   - has_numeric: true if contains numbers/statistics
   - section: Section heading if identifiable

## Guidelines
- Extract MINIMAL spans - just the supporting fact, not surrounding context
- Prefer spans with numeric data (boost relevance by 0.2)
- Avoid spans that are:
  - Navigation text, boilerplate, or disclaimers
  - Incomplete sentences that lack context
  - Redundant (same information as another span)
- Keep spans self-contained - they should make sense in isolation

## Query
{query}

## Source: {source_title}
URL: {source_url}

Content:
{source_content}

## Response Format
Respond with a JSON array of evidence spans:
```json
{{
  "spans": [
    {{
      "quote_text": "exact quote from source",
      "relevance_score": 0.85,
      "has_numeric": true,
      "section": "Financial Results"
    }}
  ]
}}
```

Extract the most relevant evidence spans from this source:"""


RELEVANCE_SCORING_PROMPT = """Rate the relevance of this evidence span to the research query.

## Query
{query}

## Evidence Span
{quote_text}

## Scoring Criteria
- 0.9-1.0: Directly answers the query or provides critical supporting data
- 0.7-0.89: Highly relevant context or related facts
- 0.5-0.69: Moderately relevant, provides background
- 0.3-0.49: Tangentially related
- 0.0-0.29: Not relevant to the query

Consider:
1. Does the span contain information that directly addresses the query?
2. Does it contain numeric data relevant to the query?
3. Is it from an authoritative source on this topic?

Respond with just a number between 0.0 and 1.0:"""
