"""LLM configuration loader - uses central AppConfig."""

import logging

from src.core.app_config import AppConfig, get_app_config
from src.services.llm.types import (
    ModelEndpoint,
    ModelRole,
    ReasoningEffort,
    SelectionStrategy,
)

logger = logging.getLogger(__name__)


class ModelConfig:
    """Model configuration manager - loads from central YAML config."""

    def __init__(self) -> None:
        """Initialize model configuration from central AppConfig."""
        self._app_config = get_app_config()
        self._endpoints: dict[str, ModelEndpoint] = {}
        self._roles: dict[str, ModelRole] = {}
        self._load_from_app_config()

    def _load_from_app_config(self) -> None:
        """Load configuration from central AppConfig."""
        # Convert endpoint configs to ModelEndpoint
        for endpoint_id, ep_config in self._app_config.endpoints.items():
            reasoning_effort: ReasoningEffort | None = None
            if ep_config.reasoning_effort is not None:
                reasoning_effort = ReasoningEffort(ep_config.reasoning_effort.value)

            self._endpoints[endpoint_id] = ModelEndpoint(
                id=endpoint_id,
                endpoint_identifier=ep_config.endpoint_identifier,
                max_context_window=ep_config.max_context_window,
                tokens_per_minute=ep_config.tokens_per_minute,
                temperature=ep_config.temperature,
                max_tokens=ep_config.max_tokens,
                reasoning_effort=reasoning_effort,
                reasoning_budget=ep_config.reasoning_budget,
                supports_structured_output=ep_config.supports_structured_output,
                supports_temperature=ep_config.supports_temperature,
                supports_prompt_caching=ep_config.supports_prompt_caching,
            )

        # Convert role configs to ModelRole
        for role_name, role_config in self._app_config.models.items():
            self._roles[role_name] = ModelRole(
                name=role_name,
                endpoints=list(role_config.endpoints),
                temperature=role_config.temperature,
                max_tokens=role_config.max_tokens,
                reasoning_effort=ReasoningEffort(role_config.reasoning_effort.value),
                reasoning_budget=role_config.reasoning_budget,
                rotation_strategy=SelectionStrategy(role_config.rotation_strategy.value),
                fallback_on_429=role_config.fallback_on_429,
            )

        logger.debug(
            f"Loaded {len(self._endpoints)} endpoints and {len(self._roles)} roles from config"
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

    @property
    def roles(self) -> dict[str, ModelRole]:
        """Get all roles."""
        return self._roles

    @property
    def endpoints(self) -> dict[str, ModelEndpoint]:
        """Get all endpoints."""
        return self._endpoints

    @property
    def app_config(self) -> "AppConfig":
        """Get the underlying AppConfig for accessing global settings like prompt_caching."""
        return self._app_config
