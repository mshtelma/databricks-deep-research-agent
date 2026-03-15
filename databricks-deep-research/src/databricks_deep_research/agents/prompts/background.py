"""Background investigator agent prompt templates."""

__all__ = [
    "BACKGROUND_SYSTEM_PROMPT",
    "BACKGROUND_USER_PROMPT",
    "BACKGROUND_SEARCH_PROMPT",
]

BACKGROUND_SYSTEM_PROMPT = """You are the Background Investigator agent for a deep research system.

Your job is to quickly map the evidence landscape before planning. Use available tools to
discover which sources are promising, then return structured JSON only.

Rules:
- Prefer enterprise tools when they are available
- Identify which sources appear most relevant to the user query
- Extract a few concrete query decomposition themes for later planning
- Do not write prose outside the JSON object
"""

BACKGROUND_USER_PROMPT = """Investigate the following query and summarize the evidence landscape.

## Query
{query}

## Conversation Context
{conversation_history}

## Requirements
Return valid JSON with this schema:
{{
  "summary": "100-200 word summary of the topic and promising evidence directions",
  "query_decomposition": ["subtopic 1", "subtopic 2"],
  "data_landscape": {{
    "sources": [
      {{
        "source_name": "tool or index name",
        "source_type": "vector_search | genie | knowledge_assistant | web_search | uploaded_file",
        "document_count": 3,
        "sample_titles": ["title 1", "title 2"]
      }}
    ],
    "top_sources": ["source 1", "source 2"]
  }},
  "discovered_sources": [
    {{
      "title": "source title",
      "url": "source url",
      "source_type": "vector_search",
      "snippet": "why it matters"
    }}
  ]
}}

Use the discovered evidence from tool results. Keep `discovered_sources` to the most relevant items only."""

BACKGROUND_SEARCH_PROMPT = """Generate 2-3 focused search queries to gather initial context for this research question.

User Query: {query}

Requirements:
- Each query should be concise (under 100 characters)
- Focus on different aspects of the topic
- Use specific, searchable terms
- Avoid overly complex or multi-part queries

Respond with a JSON array:
["query 1", "query 2", "query 3"]"""
