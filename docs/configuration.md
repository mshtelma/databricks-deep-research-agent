# Configuration System

## Overview

The Deep Research Agent uses a central YAML configuration file (`config/app.yaml`) for all settings. The configuration supports environment variable interpolation, per-depth profiles, and Pydantic validation.

## Configuration Files

| File | Purpose |
|------|---------|
| `config/app.yaml` | Central configuration (production) |
| `config/app.test.yaml` | Test configuration (faster models, fewer iterations) |
| `config/app.example.yaml` | Documented example configuration |

## Configuration Structure

```yaml
# Default model role (simple, analytical, complex)
default_role: analytical

# Model endpoints with rate limits and capabilities
endpoints:
  databricks-gemini-flash:
    endpoint_identifier: google-gemini-flash
    max_context_window: 1000000
    tokens_per_minute: 500000
    supports_structured_output: true

  databricks-claude-sonnet:
    endpoint_identifier: anthropic-claude-sonnet
    max_context_window: 200000
    tokens_per_minute: 100000
    supports_structured_output: true

  databricks-claude-sonnet-er:
    endpoint_identifier: anthropic-claude-sonnet-er
    max_context_window: 200000
    tokens_per_minute: 50000
    supports_structured_output: true

# Model roles (tiers) with priority-ordered endpoints
models:
  simple:
    endpoints:
      - databricks-gemini-flash
    temperature: 0.3
    max_tokens: 4000

  analytical:
    endpoints:
      - databricks-claude-sonnet
      - databricks-gemini-flash  # fallback
    temperature: 0.7
    max_tokens: 8000
    fallback_on_429: true

  complex:
    endpoints:
      - databricks-claude-sonnet-er
      - databricks-claude-sonnet  # fallback
    temperature: 0.7
    max_tokens: 16000
    reasoning_effort: medium

# Agent configuration
agents:
  coordinator:
    enable_clarification: true
    max_clarifying_questions: 3

  planner:
    max_plan_iterations: 3
    preserve_completed_steps: true

  researcher:
    max_search_queries: 2
    max_urls_to_crawl: 3

  reflector:
    enforce_min_steps: true

  background:
    max_queries: 3
    enable_background: true

# Search configuration
search:
  brave:
    requests_per_second: 1.0
    default_result_count: 10
    timeout_seconds: 30

# Research type profiles (per-depth)
research_types:
  light:
    steps:
      min: 1
      max: 3
      prompt_guidance: "Quick overview with 1-3 focused steps."
    report_limits:
      min_words: 800
      max_words: 1200
      max_tokens: 2000
    researcher:
      mode: classic
      max_search_queries: 2
      max_urls_to_crawl: 3
      max_tool_calls: 8
    citation_verification:
      generation_mode: natural
      enable_numeric_qa_verification: false
      enable_verification_retrieval: false

  medium:
    steps:
      min: 3
      max: 6
      prompt_guidance: "Balanced research with 3-6 steps."
    report_limits:
      min_words: 1200
      max_words: 2000
      max_tokens: 4000
    researcher:
      mode: react
      max_tool_calls: 12
    citation_verification:
      enable_numeric_qa_verification: true
      enable_verification_retrieval: true

  extended:
    steps:
      min: 5
      max: 10
      prompt_guidance: "Comprehensive research with 5-10 steps."
    report_limits:
      min_words: 1500
      max_words: 3200
      max_tokens: 8000
    researcher:
      mode: react
      max_tool_calls: 20
    citation_verification:
      generation_mode: strict
      enable_numeric_qa_verification: true
      enable_verification_retrieval: true

# Citation verification settings (global defaults)
citation_verification:
  generation_mode: natural
  enable_evidence_preselection: true
  enable_interleaved_generation: true
  enable_confidence_classification: true
  isolated_verification: true
  enable_citation_correction: true
  enable_numeric_qa_verification: true
  enable_verification_retrieval: true

  evidence_preselection:
    max_spans_per_source: 10
    min_span_length: 30
    max_span_length: 500
    numeric_content_boost: 0.2

  verification_retrieval:
    decomposition_tier: analytical
    softening_strategies:
      - hedge
      - qualify
      - parenthetical

# Rate limiting
rate_limiting:
  max_retries: 3
  initial_delay_seconds: 1.0
  max_delay_seconds: 60.0
  backoff_strategy: exponential
  jitter: true
```

## Environment Variable Interpolation

The configuration supports environment variable substitution:

```yaml
# Required variable (fails if not set)
endpoint_identifier: ${MODEL_ENDPOINT}

# Optional with default value
endpoint_identifier: ${MODEL_ENDPOINT:-databricks-llama-70b}
database_name: ${LAKEBASE_DATABASE:-deep_research}
```

**Syntax**:
- `${VAR}` - Required variable (error if unset)
- `${VAR:-default}` - Optional with default value

## Configuration Loading

**File**: `src/core/app_config.py`

```python
from src.core.app_config import get_app_config

# Load configuration (cached)
app_config = get_app_config()

# Access settings
default_role = app_config.default_role
endpoints = app_config.endpoints
models = app_config.models
```

### Custom Config Path

Set `APP_CONFIG_PATH` environment variable to use a different config file:

```bash
# Use test configuration
APP_CONFIG_PATH=config/app.test.yaml make test-integration

# Use custom configuration
APP_CONFIG_PATH=/path/to/custom.yaml python -m src.main
```

## Configuration Accessors

**File**: `src/agent/config.py`

### Global Config

```python
from src.core.app_config import get_app_config

app_config = get_app_config()
default_role = app_config.default_role
```

### Agent-Specific Config

```python
from src.agent.config import (
    get_coordinator_config,
    get_planner_config,
    get_researcher_config,
    get_reflector_config,
    get_background_config,
)

planner_config = get_planner_config()
max_iterations = planner_config.max_plan_iterations
```

### Per-Depth Config

```python
from src.agent.config import (
    get_research_type_config,
    get_step_limits,
    get_report_limits,
    get_researcher_config_for_depth,
    get_citation_config_for_depth,
)

# Full config for a depth
research_config = get_research_type_config("medium")

# Step limits
step_limits = get_step_limits("medium")
# StepLimits(min=3, max=6, prompt_guidance="...")

# Report limits
report_limits = get_report_limits("extended")
# ReportLimitConfig(min_words=1500, max_words=3200, max_tokens=8000)

# Researcher mode and limits
researcher_config = get_researcher_config_for_depth("extended")
# ResearcherTypeConfig(mode=REACT, max_tool_calls=20, ...)

# Citation config (merged with global)
citation_config = get_citation_config_for_depth("extended")
# CitationVerificationConfig with per-depth overrides
```

### Citation Config Merging

Per-depth citation config is **merged** with global config:

```python
def get_citation_config_for_depth(depth: str) -> CitationVerificationConfig:
    """
    Merge per-depth config with global config.

    - Fields explicitly set in per-depth override global
    - Unset fields inherit from global
    """
    global_config = get_app_config().citation_verification
    per_depth = get_research_type_config(depth).citation_verification

    # Deep merge
    merged = global_config.model_copy()
    if per_depth:
        for field, value in per_depth.model_dump(exclude_unset=True).items():
            setattr(merged, field, value)

    return merged
```

## Pydantic Models

### Endpoint Configuration

```python
class EndpointConfig(BaseModel):
    endpoint_identifier: str
    max_context_window: int = 128000
    tokens_per_minute: int = 100000
    supports_structured_output: bool = True
```

### Model Role Configuration

```python
class ModelRoleConfig(BaseModel):
    endpoints: list[str]
    temperature: float = 0.7
    max_tokens: int = 8000
    reasoning_effort: str | None = None
    fallback_on_429: bool = False
```

### Research Type Configuration

```python
class StepLimits(BaseModel):
    min: int
    max: int
    prompt_guidance: str | None = None

class ReportLimitConfig(BaseModel):
    min_words: int
    max_words: int
    max_tokens: int

class ResearcherTypeConfig(BaseModel):
    mode: ResearcherMode  # "classic" or "react"
    max_search_queries: int = 2
    max_urls_to_crawl: int = 3
    max_tool_calls: int = 8

class ResearchTypeConfig(BaseModel):
    steps: StepLimits
    report_limits: ReportLimitConfig
    researcher: ResearcherTypeConfig
    citation_verification: CitationVerificationConfig | None = None
```

### Citation Configuration

```python
class CitationVerificationConfig(BaseModel):
    generation_mode: str = "natural"  # strict, natural, classical
    enable_evidence_preselection: bool = True
    enable_interleaved_generation: bool = True
    enable_confidence_classification: bool = True
    isolated_verification: bool = True
    enable_citation_correction: bool = True
    enable_numeric_qa_verification: bool = True
    enable_verification_retrieval: bool = True

    evidence_preselection: EvidencePreselectionConfig | None = None
    verification_retrieval: VerificationRetrievalConfig | None = None
```

## Validation

Configuration is validated at startup:

```python
# src/core/app_config.py

def get_app_config() -> AppConfig:
    """
    Load and validate configuration.

    Raises:
    - FileNotFoundError: Config file not found
    - ValidationError: Invalid configuration
    """
    config_path = os.getenv("APP_CONFIG_PATH", "config/app.yaml")

    with open(config_path) as f:
        raw_config = yaml.safe_load(f)

    # Interpolate environment variables
    interpolated = interpolate_env_vars(raw_config)

    # Validate with Pydantic
    return AppConfig.model_validate(interpolated)
```

### Validation Errors

If configuration is invalid, startup fails with clear error:

```
pydantic.error_wrappers.ValidationError: 1 validation error for AppConfig
endpoints -> databricks-claude-sonnet -> max_context_window
  field required (type=value_error.missing)
```

## Test Configuration

`config/app.test.yaml` uses faster settings for tests:

```yaml
# Minimal iterations for speed
agents:
  planner:
    max_plan_iterations: 1

# Light depth only
research_types:
  light:
    steps:
      min: 1
      max: 2

# Disable expensive features
citation_verification:
  enable_verification_retrieval: false
  enable_numeric_qa_verification: false
```

Usage:

```bash
# Integration tests use test config
APP_CONFIG_PATH=config/app.test.yaml pytest -m integration
```

## Configuration Best Practices

### 1. Use Environment Variables for Secrets

```yaml
# Never hardcode secrets
brave_api_key: ${BRAVE_API_KEY}
databricks_token: ${DATABRICKS_TOKEN}
```

### 2. Use Defaults for Optional Settings

```yaml
# Provide sensible defaults
max_retries: ${MAX_RETRIES:-3}
timeout_seconds: ${TIMEOUT:-30}
```

### 3. Separate Test and Production Config

```bash
# Production
make prod  # Uses config/app.yaml

# Tests
make test-integration  # Uses config/app.test.yaml
```

### 4. Validate Early

Configuration is validated at startup, not at first use. This catches errors immediately rather than during runtime.

## See Also

- [Architecture](./architecture.md) - System overview
- [Deployment](./deployment.md) - Environment setup
- [LLM Interaction](./llm-interaction.md) - Model tier routing
