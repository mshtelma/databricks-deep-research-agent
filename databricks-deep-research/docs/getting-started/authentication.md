# Authentication

> Configure LLM and tool credentials for the framework.

## Overview
The framework needs credentials for two things:
1. **LLM calls** — via AsyncOpenAI client (Databricks Model Serving or OpenAI directly)
2. **Tool execution** — Brave API key for web search, Databricks workspace for enterprise tools

## Databricks Model Serving (Recommended)
```python
from openai import AsyncOpenAI

# Option 1: Databricks config profile
import os
os.environ["DATABRICKS_CONFIG_PROFILE"] = "my-profile"

client = AsyncOpenAI(
    base_url="https://your-workspace.databricks.com/serving-endpoints",
    api_key=os.environ["DATABRICKS_TOKEN"],
)

# Option 2: Direct token
client = AsyncOpenAI(
    base_url="https://your-workspace.databricks.com/serving-endpoints",
    api_key="dapi...",
)
```

## Model Mapping
```python
from databricks_deep_research import WorkflowRunner, ModelTier

runner = WorkflowRunner(workflow, client, model_mapping={
    "simple": "databricks-meta-llama-3-1-70b-instruct",
    "analytical": "databricks-meta-llama-3-1-70b-instruct",
    "complex": "databricks-meta-llama-3-1-405b-instruct",
})
```

The framework uses three model tiers to balance cost, latency, and capability:

| Tier | Purpose | Typical Use |
|------|---------|-------------|
| **simple** | Fast, cheap responses | Query rewriting, classification, short extractions |
| **analytical** | Balanced cost and quality | Research steps, reflection, evidence evaluation |
| **complex** | Reasoning-heavy tasks | Planning, synthesis, multi-hop analysis |

You can map multiple tiers to the same model — the distinction is purely semantic and lets you swap models per tier without changing agent logic.

## Advanced: ModelTierConfig
```python
from databricks_deep_research import FrameworkLLMClient

llm = FrameworkLLMClient(
    client=client,
    model_mapping={
        "simple": ModelTierConfig(
            endpoints=["llama-70b", "mixtral-8x7b"],
            fallback_on_429=True,
            rotation_strategy="round_robin",
            tokens_per_minute=200000,
        ),
        "analytical": "llama-70b",
        "complex": "llama-405b",
    },
)
```

## OpenAI Direct (Standalone)
```python
client = AsyncOpenAI(api_key="sk-...")
runner = WorkflowRunner(workflow, client, model_mapping={
    "simple": "gpt-4o-mini",
    "analytical": "gpt-4o",
    "complex": "gpt-4o",
})
```

## On-Behalf-Of (OBO) Token Flow
For Databricks Apps, the user's OBO token enables per-user access to enterprise data sources:
```python
runner = WorkflowRunner(workflow, client, model_mapping=mapping)
result = await runner.run(
    query="...",
    user_token=request.state.obo_token,  # From Databricks Apps auth
)
```
The user_token propagates through WorkflowState to enterprise tools (vector_search, genie, knowledge_assistant).

## Tool Credentials
| Tool | Credential | Environment Variable |
|------|-----------|---------------------|
| web_search | Brave API key | `BRAVE_API_KEY` |
| web_crawl | None (public URLs) | — |
| vector_search | Databricks token / OBO | `DATABRICKS_TOKEN` |
| genie | Databricks token / OBO | `DATABRICKS_TOKEN` |
| knowledge_assistant | Databricks token / OBO | `DATABRICKS_TOKEN` |

## See Also
- [Quick Start](quickstart.md)
- [LLM Client](../concepts/llm-client.md)
- [Enterprise Data Sources](../guides/enterprise-data-sources.md)
