"""Central application configuration loaded from YAML."""

import logging
from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from src.core.yaml_loader import load_yaml_config

logger = logging.getLogger(__name__)

# Default config paths
_this_file = Path(__file__).resolve()
_src_root = _this_file.parent.parent  # app_config.py -> core -> src
_project_root = _src_root.parent  # src -> project root
DEFAULT_CONFIG_PATH = _project_root / "config" / "app.yaml"


class ReasoningEffort(str, Enum):
    """Reasoning effort levels for LLM calls."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SelectionStrategy(str, Enum):
    """Endpoint selection strategy."""

    PRIORITY = "priority"
    ROUND_ROBIN = "round_robin"


class BackoffStrategy(str, Enum):
    """Backoff strategy for rate limit retries."""

    EXPONENTIAL = "exponential"  # delay = base * (2 ** attempt)
    LINEAR = "linear"  # delay = base * (attempt + 1)


class EndpointConfig(BaseModel):
    """Configuration for a single model endpoint."""

    endpoint_identifier: str
    max_context_window: int = Field(gt=0)
    tokens_per_minute: int = Field(gt=0)

    # Optional overrides (inherit from role if not set)
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, gt=0)
    reasoning_effort: ReasoningEffort | None = None
    reasoning_budget: int | None = Field(default=None, gt=0)
    supports_structured_output: bool = False

    model_config = {"frozen": True}


class ModelRoleConfig(BaseModel):
    """Configuration for a model role (tier)."""

    endpoints: list[str] = Field(min_length=1)
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=8000, gt=0)
    reasoning_effort: ReasoningEffort = ReasoningEffort.LOW
    reasoning_budget: int | None = Field(default=None, gt=0)
    tokens_per_minute: int = Field(default=100000, gt=0)
    rotation_strategy: SelectionStrategy = SelectionStrategy.PRIORITY
    fallback_on_429: bool = True

    model_config = {"frozen": True}


class ResearcherConfig(BaseModel):
    """Configuration for the Researcher agent."""

    max_search_queries: int = Field(default=2, ge=1, le=10)
    max_search_results: int = Field(default=10, ge=1, le=50)
    max_urls_to_crawl: int = Field(default=3, ge=1, le=20)
    content_preview_length: int = Field(default=3000, ge=100)
    content_storage_length: int = Field(default=10000, ge=1000)
    max_previous_observations: int = Field(default=3, ge=1, le=10)
    page_contents_limit: int = Field(default=8000, ge=1000)
    max_generated_queries: int = Field(default=3, ge=1, le=10)

    model_config = {"frozen": True}


class PlannerConfig(BaseModel):
    """Configuration for the Planner agent."""

    max_plan_iterations: int = Field(default=3, ge=1, le=10)

    model_config = {"frozen": True}


class CoordinatorConfig(BaseModel):
    """Configuration for the Coordinator agent."""

    max_clarification_rounds: int = Field(default=3, ge=0, le=5)
    enable_clarification: bool = True

    model_config = {"frozen": True}


class SynthesizerConfig(BaseModel):
    """Configuration for the Synthesizer agent."""

    max_report_length: int = Field(default=50000, ge=1000)

    model_config = {"frozen": True}


class BackgroundConfig(BaseModel):
    """Configuration for the Background Investigator agent."""

    max_search_queries: int = Field(default=2, ge=1, le=5)
    max_results_per_query: int = Field(default=3, ge=1, le=10)
    max_total_results: int = Field(default=5, ge=1, le=20)

    model_config = {"frozen": True}


class AgentsConfig(BaseModel):
    """Configuration for all agents."""

    researcher: ResearcherConfig = Field(default_factory=ResearcherConfig)
    planner: PlannerConfig = Field(default_factory=PlannerConfig)
    coordinator: CoordinatorConfig = Field(default_factory=CoordinatorConfig)
    synthesizer: SynthesizerConfig = Field(default_factory=SynthesizerConfig)
    background: BackgroundConfig = Field(default_factory=BackgroundConfig)

    model_config = {"frozen": True}


class BraveSearchConfig(BaseModel):
    """Configuration for Brave Search API."""

    requests_per_second: float = Field(default=1.0, gt=0, le=10)
    default_result_count: int = Field(default=10, ge=1, le=50)
    freshness: str = Field(default="pm", pattern=r"^(pd|pw|pm|py)$")

    model_config = {"frozen": True}


class SearchConfig(BaseModel):
    """Configuration for search services."""

    brave: BraveSearchConfig = Field(default_factory=BraveSearchConfig)

    model_config = {"frozen": True}


class TruncationConfig(BaseModel):
    """Configuration for text truncation limits."""

    log_preview: int = Field(default=200, ge=10)
    error_message: int = Field(default=500, ge=50)
    query_display: int = Field(default=100, ge=10)
    source_snippet: int = Field(default=300, ge=50)

    model_config = {"frozen": True}


class RateLimitingConfig(BaseModel):
    """Configuration for rate limit retry behavior."""

    max_retries: int = Field(default=3, ge=0, le=10)
    base_delay_seconds: float = Field(default=2.0, gt=0, le=30)
    max_delay_seconds: float = Field(default=60.0, gt=0, le=300)
    backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL
    jitter: bool = True

    model_config = {"frozen": True}

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number (0-indexed).

        Args:
            attempt: Current attempt number (0 = first retry)

        Returns:
            Delay in seconds (capped at max_delay_seconds)
        """
        if self.backoff_strategy == BackoffStrategy.EXPONENTIAL:
            delay: float = self.base_delay_seconds * (2**attempt)
        else:  # LINEAR
            delay = self.base_delay_seconds * (attempt + 1)

        return min(delay, self.max_delay_seconds)


class AppConfig(BaseModel):
    """Central application configuration loaded from YAML."""

    default_role: str = "analytical"
    endpoints: dict[str, EndpointConfig] = Field(default_factory=dict)
    models: dict[str, ModelRoleConfig] = Field(default_factory=dict)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    truncation: TruncationConfig = Field(default_factory=TruncationConfig)
    rate_limiting: RateLimitingConfig = Field(default_factory=RateLimitingConfig)

    @model_validator(mode="after")
    def validate_endpoint_references(self) -> "AppConfig":
        """Ensure all role endpoints exist in endpoints dict."""
        errors: list[str] = []

        for role_name, role_config in self.models.items():
            for endpoint_id in role_config.endpoints:
                if endpoint_id not in self.endpoints:
                    errors.append(
                        f"Role '{role_name}' references undefined endpoint: '{endpoint_id}'"
                    )

        if self.default_role and self.models and self.default_role not in self.models:
            errors.append(f"default_role '{self.default_role}' not found in models")

        if errors:
            raise ValueError("\n".join(errors))

        return self

    model_config = {"frozen": True}


def get_default_config() -> AppConfig:
    """Create AppConfig with sensible defaults (no YAML file needed).

    Returns:
        AppConfig with default endpoints and roles for development.
    """
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
        config_path = DEFAULT_CONFIG_PATH

    if not config_path.exists():
        logger.info(f"Config file not found at {config_path}, using default configuration")
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
