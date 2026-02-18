"""Prompt templates for source-specific query rewriting.

Each prompt targets a specific source type and produces structured JSON output.
Used by the QueryRewriter to transform naive queries into source-optimized ones.

Part of enterprise query optimization feature.
"""

# ---------------------------------------------------------------------------
# Vector Search: Single natural language sentence (direct strategy)
# ---------------------------------------------------------------------------

VS_REWRITE_PROMPT = """Rewrite the following research step into a single natural language sentence
suitable for semantic search over a document index. The sentence should describe what
the ideal document would contain about the topic.

Research step title: {step_title}
Step description: {step_description}
Original query: {original_query}

Rules:
- Write a VERBOSE natural language sentence (embedding models need rich text)
- Include domain-specific terminology
- Describe the CONTENT you expect to find, not a keyword query
- DO NOT use boolean operators or search syntax

Respond with only valid JSON:
{{"query": "your rewritten sentence"}}"""

# ---------------------------------------------------------------------------
# Vector Search: Multiple reformulations (multi_query strategy)
# ---------------------------------------------------------------------------

VS_MULTI_QUERY_PROMPT = """Generate 3 different natural language search queries for the same
research topic. Each query should approach the topic from a different angle to maximize
recall in a semantic search index.

Research step title: {step_title}
Step description: {step_description}
Original query: {original_query}

Rules:
- Each query is a complete sentence describing what the ideal document would say
- Vary the phrasing, synonyms, and perspective across queries
- Include domain-specific terms where appropriate
- DO NOT use boolean operators or search syntax

Respond with only valid JSON:
{{"queries": ["query 1", "query 2", "query 3"]}}"""

# ---------------------------------------------------------------------------
# Vector Search: Query2Doc pseudo-document expansion
# ---------------------------------------------------------------------------

QUERY2DOC_PROMPT = """Generate a short pseudo-answer (2-3 sentences) for the following
research question. This passage will be concatenated with the original query to expand
semantic search retrieval. Write as if you are summarizing the ideal document.

Research step: {step_title}
Original query: {original_query}

Rules:
- Write 2-3 factual-sounding sentences as if answering the question
- Include specific terms, entities, and concepts related to the topic
- Be informative but concise
- Do NOT include disclaimers or hedging

Respond with only valid JSON:
{{"passage": "your pseudo-answer passage"}}"""

# ---------------------------------------------------------------------------
# Genie / SQL Analytics: Schema-aware precise data question
# ---------------------------------------------------------------------------

GENIE_REWRITE_PROMPT = """Rewrite the following research step into a precise data question
for querying an enterprise analytics database. Extract specific entities, metrics,
and time periods.

Research step title: {step_title}
Step description: {step_description}
Original query: {original_query}
Data source description: {source_description}

Rules:
- Be EXPLICIT about metrics (e.g., "total revenue", "count of orders")
- Include time periods if mentioned or implied (e.g., "Q4 2024", "last 12 months")
- Name specific entities (product names, departments, regions)
- Ask ONE clear data question, not a multi-part query
- If the query is vague, pick the most likely interpretation and phrase it precisely

Respond with only valid JSON:
{{"question": "your precise data question"}}"""

# ---------------------------------------------------------------------------
# Knowledge Assistant: Focused question with context (step_back strategy)
# ---------------------------------------------------------------------------

KA_REWRITE_PROMPT = """Rewrite the following research step into a single focused question
for a domain expert knowledge assistant. If the question is too specific or narrow,
broaden it to a more answerable topic. If it's multi-part, pick the most important
single question.

Research step title: {step_title}
Step description: {step_description}
Original query: {original_query}

Previous research findings (use to build on prior context):
{previous_observations}

Rules:
- Ask ONE focused question (not multi-part)
- If previous findings are available, build on them rather than repeating
- If the question is too narrow, step back to a broader topic
- Include enough context for the assistant to understand what you need

Respond with only valid JSON:
{{"question": "your focused question"}}"""
