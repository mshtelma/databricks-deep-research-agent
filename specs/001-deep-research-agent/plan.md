# Implementation Plan: Deep Research Agent - Central YAML Configuration

**Branch**: `001-deep-research-agent` | **Date**: 2025-12-24 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification FR-081 to FR-090 (Central Configuration)

## Summary

Implement a central YAML-based configuration system (`config/app.yaml`) that consolidates all model endpoints, model roles (micro/simple/analytical/complex), agent settings, and search configuration. The system supports environment variable interpolation (`${VAR:-default}`), validates at startup, and falls back to sensible defaults when the config file is absent.

## Technical Context

**Language/Version**: Python 3.11+
**Primary Dependencies**: PyYAML, Pydantic v2, FastAPI
**Storage**: YAML file (`config/app.yaml`), Pydantic models for validation
**Testing**: pytest with pytest-asyncio
**Target Platform**: Linux server (Databricks workspace)
**Project Type**: Web application (Python backend + React frontend)
**Performance Goals**: Config load <100ms, validation errors reported within 5s of startup
**Constraints**: Must work without config file (defaults), environment variable interpolation for secrets
**Scale/Scope**: Team-scale (10-100 concurrent users)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Clients and Workspace Integration | ✅ PASS | Config system doesn't affect LLM/Workspace client patterns |
| II. Typing-First Python | ✅ PASS | Config models use Pydantic with full type annotations |
| III. Avoid Runtime Introspection | ✅ PASS | Pydantic handles validation at boundaries, no hasattr/isinstance |
| IV. Linting and Static Type Enforcement | ✅ PASS | All config types will have strict annotations, mypy-compatible |

**Gate Status**: PASSED - No constitution violations

## Project Structure

### Documentation (this feature)

```text
specs/001-deep-research-agent/
├── plan.md              # This file
├── research.md          # Phase 0 output - YAML config decisions
├── data-model.md        # Updated with AppConfig entity
├── quickstart.md        # Configuration quick start guide
├── contracts/           # No API changes for config
└── tasks.md             # Phase 2 output
```

### Source Code (repository root)

```text
src/
├── core/
│   ├── config.py            # Existing Settings class - UPDATE to load YAML
│   └── app_config.py        # NEW: Central YAML config loader
├── services/
│   └── llm/
│       ├── config.py        # UPDATE: Load from central config
│       └── types.py         # Existing types - unchanged
├── agent/
│   └── config.py            # NEW: Agent configuration from YAML
└── ...

config/
├── app.yaml                 # NEW: Central configuration file
└── app.example.yaml         # NEW: Example configuration with comments

tests/
├── unit/
│   └── core/
│       ├── test_app_config.py   # NEW: Config loader tests
│       └── test_config.py       # UPDATE: Integration with YAML config
└── ...
```

**Structure Decision**: Single project structure with new `config/` directory at repository root for YAML configuration files.

## Phase 0: Research & Decisions

### Research Topics

1. **YAML Configuration Patterns**: Best practices for YAML config in Python applications
2. **Environment Variable Interpolation**: Approaches for `${VAR:-default}` syntax in YAML
3. **Pydantic YAML Integration**: Loading YAML into Pydantic models
4. **Hot Reload Patterns**: File watching for configuration changes (optional FR-090)

### Research Findings

See [research.md](./research.md) for detailed findings.

**Key Decisions**:

| Decision | Choice | Rationale |
|----------|--------|-----------|
| YAML Library | PyYAML (already in deps) | Standard, well-tested, no new dependencies |
| Env Var Interpolation | Custom preprocessor with regex | Simple, no external deps, `${VAR:-default}` syntax |
| Validation | Pydantic v2 models | Already in use, strong typing, good error messages |
| Hot Reload | Optional (file watcher) | Complexity vs value tradeoff - implement as enhancement |
| Default Fallback | In-code defaults | Allows running without config file in development |

## Phase 1: Design & Contracts

### 1.1 YAML Configuration Schema

```yaml
# config/app.yaml - Full schema
# Secrets use environment variables: ${DATABRICKS_TOKEN}

# Default role used when no specific role is requested
default_role: analytical

# Model endpoint definitions
endpoints:
  databricks-gpt-oss-20b:
    endpoint_identifier: databricks-gpt-oss-20b
    max_context_window: 32000
    tokens_per_minute: 200000
    supports_structured_output: true

  databricks-gpt-oss-120b:
    endpoint_identifier: databricks-gpt-oss-120b
    max_context_window: 128000
    tokens_per_minute: 200000
    supports_structured_output: true

  databricks-llama-4-maverick:
    endpoint_identifier: databricks-meta-llama-4-maverick
    max_context_window: 128000
    tokens_per_minute: 200000

  databricks-claude-3-7-sonnet:
    endpoint_identifier: databricks-claude-3-7-sonnet
    max_context_window: 200000
    tokens_per_minute: 50000
    temperature: 0.5  # Override role default

  databricks-claude-sonnet-4:
    endpoint_identifier: databricks-claude-sonnet-4
    max_context_window: 200000
    tokens_per_minute: 50000

  databricks-gemma-3-12b:
    endpoint_identifier: databricks-gemma-3-12b
    max_context_window: 32000
    tokens_per_minute: 300000

# Model roles (tiers) with priority-ordered endpoints
models:
  # TIER 1: MICRO - Ultra-lightweight (pattern matching, entity extraction)
  micro:
    endpoints:
      - databricks-gpt-oss-20b
      - databricks-gemma-3-12b
    temperature: 0.5
    max_tokens: 4000
    tokens_per_minute: 200000
    rotation_strategy: priority
    fallback_on_429: true

  # TIER 2: SIMPLE - Lightweight (query gen, claim extraction, validation)
  simple:
    endpoints:
      - databricks-gpt-oss-20b
      - databricks-llama-4-maverick
    temperature: 0.5
    max_tokens: 8000
    reasoning_effort: low
    tokens_per_minute: 200000
    rotation_strategy: priority
    fallback_on_429: true

  # TIER 3: ANALYTICAL - Medium (research synthesis, fact checking, planning)
  analytical:
    endpoints:
      - databricks-gpt-oss-120b
      - databricks-gpt-oss-20b
    temperature: 0.7
    max_tokens: 12000
    reasoning_effort: medium
    tokens_per_minute: 200000
    rotation_strategy: priority
    fallback_on_429: true

  # TIER 4: COMPLEX - Heavy (report generation, complex synthesis)
  complex:
    endpoints:
      - databricks-gpt-oss-120b
      - databricks-claude-3-7-sonnet
      - databricks-claude-sonnet-4
    temperature: 0.7
    max_tokens: 25000
    reasoning_effort: high
    reasoning_budget: 8000
    tokens_per_minute: 50000
    rotation_strategy: priority
    fallback_on_429: true

# Agent configuration
agents:
  researcher:
    max_search_queries: 2
    max_search_results: 10
    max_urls_to_crawl: 3
    content_preview_length: 3000
    content_storage_length: 10000
    max_previous_observations: 3
    page_contents_limit: 8000
    max_generated_queries: 3

  planner:
    max_plan_iterations: 3

  coordinator:
    max_clarification_rounds: 3
    enable_clarification: true

  synthesizer:
    max_report_length: 50000

# Search configuration
search:
  brave:
    requests_per_second: 1.0
    default_result_count: 10
    freshness: "month"  # pd, pw, pm, py (day, week, month, year)

# Truncation limits (for consistency across codebase)
truncation:
  log_preview: 200
  error_message: 500
  query_display: 100
  source_snippet: 300
```

### 1.2 Pydantic Configuration Models

```python
# src/core/app_config.py

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class ReasoningEffort(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SelectionStrategy(str, Enum):
    PRIORITY = "priority"
    ROUND_ROBIN = "round_robin"


class EndpointConfig(BaseModel):
    """Configuration for a single model endpoint."""

    endpoint_identifier: str
    max_context_window: int = Field(gt=0)
    tokens_per_minute: int = Field(gt=0)

    # Optional overrides
    temperature: float | None = Field(None, ge=0, le=2)
    max_tokens: int | None = Field(None, gt=0)
    reasoning_effort: ReasoningEffort | None = None
    reasoning_budget: int | None = Field(None, gt=0)
    supports_structured_output: bool = False

    class Config:
        frozen = True


class ModelRoleConfig(BaseModel):
    """Configuration for a model role (tier)."""

    endpoints: list[str] = Field(min_length=1)
    temperature: float = Field(0.7, ge=0, le=2)
    max_tokens: int = Field(8000, gt=0)
    reasoning_effort: ReasoningEffort = ReasoningEffort.LOW
    reasoning_budget: int | None = Field(None, gt=0)
    tokens_per_minute: int = Field(100000, gt=0)
    rotation_strategy: SelectionStrategy = SelectionStrategy.PRIORITY
    fallback_on_429: bool = True

    class Config:
        frozen = True


class ResearcherConfig(BaseModel):
    """Configuration for the Researcher agent."""

    max_search_queries: int = Field(2, ge=1, le=10)
    max_search_results: int = Field(10, ge=1, le=50)
    max_urls_to_crawl: int = Field(3, ge=1, le=20)
    content_preview_length: int = Field(3000, ge=100)
    content_storage_length: int = Field(10000, ge=1000)
    max_previous_observations: int = Field(3, ge=1, le=10)
    page_contents_limit: int = Field(8000, ge=1000)
    max_generated_queries: int = Field(3, ge=1, le=10)

    class Config:
        frozen = True


class PlannerConfig(BaseModel):
    """Configuration for the Planner agent."""

    max_plan_iterations: int = Field(3, ge=1, le=10)

    class Config:
        frozen = True


class CoordinatorConfig(BaseModel):
    """Configuration for the Coordinator agent."""

    max_clarification_rounds: int = Field(3, ge=0, le=5)
    enable_clarification: bool = True

    class Config:
        frozen = True


class SynthesizerConfig(BaseModel):
    """Configuration for the Synthesizer agent."""

    max_report_length: int = Field(50000, ge=1000)

    class Config:
        frozen = True


class AgentsConfig(BaseModel):
    """Configuration for all agents."""

    researcher: ResearcherConfig = Field(default_factory=ResearcherConfig)
    planner: PlannerConfig = Field(default_factory=PlannerConfig)
    coordinator: CoordinatorConfig = Field(default_factory=CoordinatorConfig)
    synthesizer: SynthesizerConfig = Field(default_factory=SynthesizerConfig)

    class Config:
        frozen = True


class BraveSearchConfig(BaseModel):
    """Configuration for Brave Search API."""

    requests_per_second: float = Field(1.0, gt=0, le=10)
    default_result_count: int = Field(10, ge=1, le=50)
    freshness: str = Field("month", pattern=r"^(pd|pw|pm|py)$")

    class Config:
        frozen = True


class SearchConfig(BaseModel):
    """Configuration for search services."""

    brave: BraveSearchConfig = Field(default_factory=BraveSearchConfig)

    class Config:
        frozen = True


class TruncationConfig(BaseModel):
    """Configuration for text truncation limits."""

    log_preview: int = Field(200, ge=10)
    error_message: int = Field(500, ge=50)
    query_display: int = Field(100, ge=10)
    source_snippet: int = Field(300, ge=50)

    class Config:
        frozen = True


class AppConfig(BaseModel):
    """Central application configuration loaded from YAML."""

    default_role: str = "analytical"
    endpoints: dict[str, EndpointConfig] = Field(default_factory=dict)
    models: dict[str, ModelRoleConfig] = Field(default_factory=dict)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    truncation: TruncationConfig = Field(default_factory=TruncationConfig)

    @model_validator(mode="after")
    def validate_endpoint_references(self) -> "AppConfig":
        """Ensure all role endpoints exist in endpoints dict."""
        for role_name, role_config in self.models.items():
            for endpoint_id in role_config.endpoints:
                if endpoint_id not in self.endpoints:
                    raise ValueError(
                        f"Role '{role_name}' references undefined endpoint: {endpoint_id}"
                    )

        if self.default_role and self.default_role not in self.models:
            raise ValueError(f"default_role '{self.default_role}' not found in models")

        return self

    class Config:
        frozen = True
```

### 1.3 YAML Loader with Environment Variable Interpolation

```python
# src/core/yaml_loader.py

import os
import re
from pathlib import Path
from typing import Any

import yaml


ENV_VAR_PATTERN = re.compile(r"\$\{([^}:]+)(?::-([^}]*))?\}")


def interpolate_env_vars(value: Any) -> Any:
    """Recursively interpolate environment variables in YAML values.

    Supports:
    - ${VAR} - Required variable, raises if not set
    - ${VAR:-default} - Optional variable with default value
    """
    if isinstance(value, str):
        def replace_var(match: re.Match) -> str:
            var_name = match.group(1)
            default = match.group(2)

            env_value = os.environ.get(var_name)
            if env_value is not None:
                return env_value
            if default is not None:
                return default
            raise ValueError(f"Environment variable '{var_name}' is not set and no default provided")

        return ENV_VAR_PATTERN.sub(replace_var, value)

    elif isinstance(value, dict):
        return {k: interpolate_env_vars(v) for k, v in value.items()}

    elif isinstance(value, list):
        return [interpolate_env_vars(item) for item in value]

    return value


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    """Load YAML configuration with environment variable interpolation.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Parsed and interpolated configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If environment variable interpolation fails
        yaml.YAMLError: If YAML parsing fails
    """
    with open(config_path) as f:
        raw_config = yaml.safe_load(f)

    if raw_config is None:
        return {}

    return interpolate_env_vars(raw_config)
```

### 1.4 Config Loading and Caching

```python
# src/core/app_config.py (continued)

import logging
from functools import lru_cache
from pathlib import Path

from .yaml_loader import load_yaml_config

logger = logging.getLogger(__name__)

# Default config paths
DEFAULT_CONFIG_PATH = Path("config/app.yaml")
FALLBACK_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "app.yaml"


def get_default_config() -> AppConfig:
    """Create AppConfig with sensible defaults (no YAML file needed)."""
    return AppConfig(
        default_role="analytical",
        endpoints={
            "databricks-llama-70b": EndpointConfig(
                endpoint_identifier="databricks-meta-llama-3-1-70b-instruct",
                max_context_window=128000,
                tokens_per_minute=200000,
            ),
            "databricks-llama-8b": EndpointConfig(
                endpoint_identifier="databricks-meta-llama-3-1-8b-instruct",
                max_context_window=128000,
                tokens_per_minute=300000,
            ),
        },
        models={
            "simple": ModelRoleConfig(
                endpoints=["databricks-llama-8b", "databricks-llama-70b"],
                temperature=0.3,
                max_tokens=4000,
                reasoning_effort=ReasoningEffort.LOW,
            ),
            "analytical": ModelRoleConfig(
                endpoints=["databricks-llama-70b"],
                temperature=0.7,
                max_tokens=8000,
                reasoning_effort=ReasoningEffort.MEDIUM,
            ),
            "complex": ModelRoleConfig(
                endpoints=["databricks-llama-70b"],
                temperature=0.7,
                max_tokens=16000,
                reasoning_effort=ReasoningEffort.HIGH,
                reasoning_budget=8000,
            ),
        },
    )


@lru_cache(maxsize=1)
def load_app_config(config_path: Path | None = None) -> AppConfig:
    """Load application configuration from YAML file.

    Args:
        config_path: Path to YAML config file. If None, searches default locations.

    Returns:
        Validated AppConfig instance

    Note:
        Falls back to default configuration if no config file is found.
        This allows running without explicit configuration in development.
    """
    # Determine config file path
    if config_path is None:
        if DEFAULT_CONFIG_PATH.exists():
            config_path = DEFAULT_CONFIG_PATH
        elif FALLBACK_CONFIG_PATH.exists():
            config_path = FALLBACK_CONFIG_PATH
        else:
            logger.info("No config file found, using default configuration")
            return get_default_config()

    if not config_path.exists():
        logger.info(f"Config file not found at {config_path}, using defaults")
        return get_default_config()

    try:
        raw_config = load_yaml_config(config_path)
        config = AppConfig.model_validate(raw_config)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}")
        raise


def get_app_config() -> AppConfig:
    """Get the cached application configuration.

    This is the primary entry point for accessing configuration.
    """
    return load_app_config()


def clear_config_cache() -> None:
    """Clear the configuration cache (useful for testing and hot reload)."""
    load_app_config.cache_clear()
```

### 1.5 Integration with Existing LLM Config

```python
# src/services/llm/config.py (UPDATED)

from src.core.app_config import get_app_config
from src.services.llm.types import ModelEndpoint, ModelRole, ReasoningEffort, SelectionStrategy


class ModelConfig:
    """Model configuration manager - now loads from central YAML config."""

    def __init__(self) -> None:
        self._app_config = get_app_config()
        self._endpoints: dict[str, ModelEndpoint] = {}
        self._roles: dict[str, ModelRole] = {}
        self._load_from_app_config()

    def _load_from_app_config(self) -> None:
        """Load configuration from central AppConfig."""
        # Convert endpoint configs to ModelEndpoint
        for endpoint_id, ep_config in self._app_config.endpoints.items():
            self._endpoints[endpoint_id] = ModelEndpoint(
                id=endpoint_id,
                endpoint_identifier=ep_config.endpoint_identifier,
                max_context_window=ep_config.max_context_window,
                tokens_per_minute=ep_config.tokens_per_minute,
                temperature=ep_config.temperature,
                max_tokens=ep_config.max_tokens,
                reasoning_effort=ReasoningEffort(ep_config.reasoning_effort.value)
                if ep_config.reasoning_effort else None,
                reasoning_budget=ep_config.reasoning_budget,
                supports_structured_output=ep_config.supports_structured_output,
            )

        # Convert role configs to ModelRole
        for role_name, role_config in self._app_config.models.items():
            self._roles[role_name] = ModelRole(
                name=role_name,
                endpoints=role_config.endpoints,
                temperature=role_config.temperature,
                max_tokens=role_config.max_tokens,
                reasoning_effort=ReasoningEffort(role_config.reasoning_effort.value),
                reasoning_budget=role_config.reasoning_budget,
                rotation_strategy=SelectionStrategy(role_config.rotation_strategy.value),
                fallback_on_429=role_config.fallback_on_429,
            )

    def get_role(self, role_name: str) -> ModelRole:
        """Get model role configuration."""
        if role_name not in self._roles:
            raise ValueError(f"Unknown model role: {role_name}")
        return self._roles[role_name]

    def get_endpoint(self, endpoint_id: str) -> ModelEndpoint:
        """Get endpoint configuration."""
        if endpoint_id not in self._endpoints:
            raise ValueError(f"Unknown endpoint: {endpoint_id}")
        return self._endpoints[endpoint_id]

    def get_endpoints_for_role(self, role_name: str) -> list[ModelEndpoint]:
        """Get all endpoints for a role in priority order."""
        role = self.get_role(role_name)
        return [
            self.get_endpoint(ep_id)
            for ep_id in role.endpoints
            if ep_id in self._endpoints
        ]

    def get_default_role(self) -> str:
        """Get the default role name."""
        return self._app_config.default_role
```

### 1.6 Agent Configuration Access

```python
# src/agent/config.py (NEW)

from src.core.app_config import get_app_config, ResearcherConfig, PlannerConfig, CoordinatorConfig, SynthesizerConfig


def get_researcher_config() -> ResearcherConfig:
    """Get Researcher agent configuration."""
    return get_app_config().agents.researcher


def get_planner_config() -> PlannerConfig:
    """Get Planner agent configuration."""
    return get_app_config().agents.planner


def get_coordinator_config() -> CoordinatorConfig:
    """Get Coordinator agent configuration."""
    return get_app_config().agents.coordinator


def get_synthesizer_config() -> SynthesizerConfig:
    """Get Synthesizer agent configuration."""
    return get_app_config().agents.synthesizer


def get_truncation_limit(limit_name: str) -> int:
    """Get a truncation limit by name.

    Args:
        limit_name: One of 'log_preview', 'error_message', 'query_display', 'source_snippet'

    Returns:
        The configured truncation limit
    """
    config = get_app_config().truncation
    return getattr(config, limit_name)
```

## Implementation Tasks

### Task 1: Create Core Configuration Infrastructure

**Files to create**:
- `src/core/yaml_loader.py` - YAML loading with env var interpolation
- `src/core/app_config.py` - Pydantic models and loader

**Files to update**:
- `src/core/__init__.py` - Export new config functions

**Acceptance Criteria**:
- [ ] Environment variable interpolation works for `${VAR}` and `${VAR:-default}`
- [ ] Validation errors are clear and actionable
- [ ] Falls back to defaults when no config file exists
- [ ] All config models are frozen (immutable)

### Task 2: Create Example Configuration Files

**Files to create**:
- `config/app.yaml` - Default configuration with common endpoints
- `config/app.example.yaml` - Documented example with all options

**Acceptance Criteria**:
- [ ] Example file shows all configuration options with comments
- [ ] Default file works out of the box with Databricks endpoints

### Task 3: Integrate with LLM Service

**Files to update**:
- `src/services/llm/config.py` - Load from central config
- `src/services/llm/client.py` - Use centralized config

**Acceptance Criteria**:
- [ ] ModelConfig loads from AppConfig
- [ ] Existing tests continue to pass
- [ ] No hardcoded values remain in LLM service

### Task 4: Integrate with Agent Nodes

**Files to create**:
- `src/agent/config.py` - Agent configuration accessors

**Files to update**:
- `src/agent/nodes/researcher.py` - Use centralized config
- `src/agent/nodes/planner.py` - Use centralized config
- `src/agent/nodes/coordinator.py` - Use centralized config
- `src/agent/nodes/synthesizer.py` - Use centralized config

**Acceptance Criteria**:
- [ ] All hardcoded limits moved to config
- [ ] Agent nodes use config accessors
- [ ] Existing tests continue to pass

### Task 5: Integrate with Search Service

**Files to update**:
- `src/services/search/brave.py` - Use centralized config for rate limiting

**Acceptance Criteria**:
- [ ] Brave rate limit from config
- [ ] Freshness setting from config
- [ ] Existing tests continue to pass

### Task 6: Add Startup Validation

**Files to update**:
- `src/main.py` - Validate config at startup
- Add startup event to load and validate configuration

**Acceptance Criteria**:
- [ ] Config validated before accepting requests
- [ ] Clear error message on invalid config
- [ ] Application fails fast on config errors

### Task 7: Write Unit Tests

**Files to create**:
- `tests/unit/core/test_yaml_loader.py` - Env var interpolation tests
- `tests/unit/core/test_app_config.py` - Config loading and validation tests
- `tests/unit/agent/test_config.py` - Agent config accessor tests

**Acceptance Criteria**:
- [ ] 90%+ coverage for config modules
- [ ] Tests for missing env vars
- [ ] Tests for invalid config values
- [ ] Tests for endpoint reference validation

### Task 8: Update Documentation

**Files to update**:
- `CLAUDE.md` - Add configuration section
- `README.md` - Add configuration quick start
- `specs/001-deep-research-agent/data-model.md` - Add AppConfig entity

**Acceptance Criteria**:
- [ ] Configuration section in CLAUDE.md
- [ ] Quick start guide for configuration
- [ ] Data model updated with AppConfig

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Breaking existing defaults | Medium | High | Comprehensive test coverage before refactor |
| Config file not found in prod | Low | Medium | Explicit fallback to defaults with logging |
| Invalid env var interpolation | Low | Medium | Clear error messages, example config |
| Performance impact | Low | Low | Config is cached, loaded once at startup |

## Complexity Tracking

No constitution violations requiring justification.

## Dependencies

- PyYAML (already in dependencies)
- Pydantic v2 (already in dependencies)
- No new external dependencies required

## Success Metrics

- [ ] SC-029: System starts successfully with default configuration when `config/app.yaml` is not present
- [ ] SC-030: Configuration validation errors are reported with clear, actionable messages within 5 seconds of application startup
- [ ] All existing unit tests pass (112+)
- [ ] All E2E tests pass
- [ ] mypy strict mode passes
