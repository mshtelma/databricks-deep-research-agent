"""Configuration catalog schemas for the config API.

Provides response schemas for endpoint and model catalog information,
used by the agent editor to populate model override dropdowns.

Part of 009-custom-agent-config (T018).
"""

from pydantic import Field

from deep_research.schemas.common import BaseSchema


class EndpointInfo(BaseSchema):
    """Information about a single model endpoint."""

    name: str
    """Endpoint identifier name."""

    endpoint_identifier: str
    """Model serving endpoint identifier."""

    max_context_window: int
    """Maximum context window in tokens."""

    supports_structured_output: bool
    """Whether this endpoint supports structured JSON output."""


class ModelCategoryInfo(BaseSchema):
    """Information about a model tier/category."""

    name: str
    """Category/tier name."""

    default_endpoints: list[str]
    """Default endpoint IDs for this tier (in priority order)."""

    temperature: float
    """Default temperature for this tier."""

    max_tokens: int
    """Default max tokens for this tier."""


class EndpointCatalogResponse(BaseSchema):
    """Response schema for the endpoint catalog API."""

    categories: dict[str, ModelCategoryInfo] = Field(default_factory=dict)
    """Model tier categories with their default configuration."""

    endpoints: dict[str, EndpointInfo] = Field(default_factory=dict)
    """All available endpoints by ID."""


class ServingEndpointSummary(BaseSchema):
    """Summary of a workspace serving endpoint for autocomplete."""

    name: str
    """Serving endpoint name (used as the model identifier)."""

    endpoint_type: str = "CUSTOM"
    """Endpoint type (FOUNDATION_MODEL_API, CUSTOM, EXTERNAL_MODEL, etc.)."""

    state: str = "UNKNOWN"
    """Endpoint state (READY, NOT_READY, etc.)."""


class ServingEndpointsResponse(BaseSchema):
    """Response listing workspace serving endpoints."""

    endpoints: list[ServingEndpointSummary] = Field(default_factory=list)
    """All serving endpoints visible to the user."""

    config_endpoint_names: list[str] = Field(default_factory=list)
    """Endpoint identifiers that are also configured in YAML (for dedup)."""


class DeploymentDefaultsResponse(BaseSchema):
    """Default values for the Agent Designer deployment wizards.

    Currently exposes only the framework Git ref used by Mode-2 (shell-app)
    deploys. Resolved server-side so generated shell apps share the same
    default pin.
    """

    framework_git_tag: str = Field(
        ...,
        description=(
            "Default Git ref (for example, 'main' or 'v0.2.0') for the "
            "shell-app pyproject pin."
        ),
    )
