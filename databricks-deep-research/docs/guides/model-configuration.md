# Model Configuration

> From single-model hello-world to production multi-endpoint failover in four levels.

## Overview

The framework routes LLM calls through **model tiers** — named slots like `simple`, `analytical`, `complex` (or any custom name). Each tier resolves to one or more Databricks serving endpoints at runtime. This guide shows four progressively richer ways to configure tiers.

---

## Level 1: Single Model

One model for everything. Good for prototyping.

```python
from databricks_deep_research import WorkflowRunner

runner = WorkflowRunner.from_databricks(model="databricks-claude-haiku-4-5")
result = await runner.run("my_workflow.yaml", query="What is AI?")
```

All tiers (`simple`, `analytical`, `complex`) map to the same endpoint.

---

## Level 2: Per-Tier Mapping

Different models for different tasks. The planner gets a balanced model; the synthesizer gets the strongest.

```python
runner = WorkflowRunner.from_databricks(model_mapping={
    "simple": "databricks-claude-haiku-4-5",
    "analytical": "databricks-claude-sonnet-4-5",
    "complex": "databricks-claude-opus-4-5",
})
```

Agent nodes in your YAML reference tiers by name:

```yaml
config:
  subtype: coordinator
  model_tier: simple       # → haiku
```

You can also define custom tier names:

```python
runner = WorkflowRunner.from_databricks(model_mapping={
    "simple": "databricks-claude-haiku-4-5",
    "analytical": "databricks-claude-sonnet-4-5",
    "complex": "databricks-claude-opus-4-5",
    "bulk_analysis": "databricks-gpt-5-2",  # custom tier
})
```

---

## Level 3: Multi-Endpoint with Failover

For production. Multiple endpoints per tier with automatic rate-limit failover and health tracking.

```python
from databricks_deep_research import WorkflowRunner, ModelTierConfig

runner = WorkflowRunner.from_databricks(model_mapping={
    "analytical": ModelTierConfig(
        endpoints=["databricks-claude-haiku-4-5", "databricks-gpt-5-mini"],
        fallback_on_429=True,
        tokens_per_minute=200000,
        rotation_strategy="PRIORITY",
    ),
    "complex": ModelTierConfig(
        endpoints=["databricks-claude-opus-4-5", "databricks-claude-sonnet-4-5"],
        fallback_on_429=True,
        rotation_strategy="PRIORITY",
    ),
    "simple": "databricks-claude-haiku-4-5",  # simple string still works
})
```

### ModelTierConfig Fields

| Field | Default | Description |
|-------|---------|-------------|
| `endpoints` | (required) | List your models in preference order. The first healthy one is used. |
| `fallback_on_429` | `True` | On rate limit, the client immediately tries the next endpoint instead of waiting. |
| `tokens_per_minute` | `0` | Proactive throttling. When an endpoint approaches this limit within a 60-second window, the client prefers a different endpoint. `0` means no limit. |
| `rotation_strategy` | `"PRIORITY"` | `PRIORITY`: always try the first endpoint. `ROUND_ROBIN`: distribute requests evenly across healthy endpoints. Use ROUND_ROBIN when endpoints have similar capabilities. |

### Health Tracking

The framework tracks per-endpoint health automatically:

- **3 consecutive errors** → endpoint marked unhealthy (skipped during selection)
- **429 (rate limit)** → 60-second cooldown window
- **Any success** → resets error counter, marks healthy
- **TPM budget** → when `tokens_per_minute` is set, the client tracks usage per 60-second window

---

## Level 4: YAML-Defined Models

Make your workflow self-contained — define model tiers directly in the YAML file.

```yaml
models:
  simple:
    endpoints:
      - databricks-claude-haiku-4-5
      - databricks-gemini-3-flash
    tokens_per_minute: 200000
    fallback_on_429: true
    rotation_strategy: priority

  analytical:
    endpoints:
      - databricks-claude-haiku-4-5
      - databricks-gpt-5-mini
    fallback_on_429: true

  complex: databricks-claude-opus-4-5   # simple string works in YAML too
```

Python — no `model_mapping` needed:

```python
runner = WorkflowRunner.from_databricks()
result = await runner.run("my_workflow.yaml", query="...")
```

The runner reads the `models:` section from the YAML and applies it automatically. YAML models override any Python-supplied `model_mapping` when present.

### Custom Tiers in YAML

```yaml
models:
  bulk_analysis:
    endpoints:
      - databricks-gpt-5-2
      - databricks-gemini-3-flash
    tokens_per_minute: 500000
    rotation_strategy: round_robin
```

Agent nodes reference the custom tier:

```yaml
config:
  subtype: researcher
  model_tier: bulk_analysis
```

---

## Programmatic Parsing

Use `parse_model_config()` to convert raw dicts (e.g., from your own config files) into the framework's model mapping format:

```python
from databricks_deep_research import parse_model_config, FrameworkLLMClient

raw = {
    "simple": "fast-model",
    "complex": {"endpoints": ["big-a", "big-b"], "fallback_on_429": True},
}
mapping = parse_model_config(raw)

client = FrameworkLLMClient.from_databricks(model_mapping=mapping)
```

---

## Feature Comparison

| Feature | Simple String | ModelTierConfig | YAML models: |
|---------|---------------|-----------------|--------------|
| Single endpoint | Yes | Yes | Yes |
| Multi-endpoint failover | No | Yes | Yes |
| Health tracking | No | Yes | Yes |
| TPM budget | No | Yes | Yes |
| Rotation strategy | No | Yes | Yes |
| Custom tier names | Yes | Yes | Yes |

---

## Precedence Rules

When both Python `model_mapping` and YAML `models:` are present, **YAML wins**. This is the simplest correct mental model:

- **YAML `models:`** = source of truth for the workflow
- **Python `model_mapping`** = source of truth when no YAML `models:` present
- Want Python to override? Don't put `models:` in your YAML

---

## Limitations

- **No `${ENV_VAR}` expansion** in model names. The framework's YAML loader is intentionally simple. For env-var-based config, use `parse_model_config()` with your own expansion logic in Python.
- **Health state is per-run.** Each `run()` or `stream()` call starts with fresh health tracking. This prevents stale rate-limit windows from previous runs from incorrectly blocking healthy endpoints.

---

## See Also

- [LLM Client](../concepts/llm-client.md) — full client API reference
- [YAML Workflow Authoring](yaml-workflow-authoring.md) — complete YAML authoring guide
- [Workflow Definition Schema](../reference/workflow-definition-schema.md) — schema reference
