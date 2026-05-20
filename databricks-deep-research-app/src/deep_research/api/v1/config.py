"""Configuration catalog API endpoints.

Provides read-only access to model endpoint catalog for agent editor UI.

Part of 009-custom-agent-config (T019).
"""

import logging
import time

from fastapi import APIRouter, Request

from deep_research.core.app_config import get_app_config
from deep_research.middleware.auth import CurrentUser
from deep_research.schemas.config import (
    DeploymentDefaultsResponse,
    EndpointCatalogResponse,
    EndpointInfo,
    ModelCategoryInfo,
    ServingEndpointsResponse,
    ServingEndpointSummary,
)
from deep_research.services.deployment.framework_version import framework_git_tag

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/config", tags=["Config"])


@router.get("/model-catalog", response_model=EndpointCatalogResponse)
async def get_model_catalog(_user: CurrentUser) -> EndpointCatalogResponse:
    """Get the model endpoint catalog.

    Returns all available model tiers (categories) and their endpoints.
    Used by the agent editor to populate model override dropdowns.
    """
    app_config = get_app_config()

    # Build categories from model roles
    categories: dict[str, ModelCategoryInfo] = {}
    for role_name, role_config in app_config.models.items():
        categories[role_name] = ModelCategoryInfo(
            name=role_name,
            default_endpoints=role_config.endpoints,
            temperature=role_config.temperature,
            max_tokens=role_config.max_tokens,
        )

    # Build endpoints from endpoint configs
    endpoints: dict[str, EndpointInfo] = {}
    for ep_name, ep_config in app_config.endpoints.items():
        endpoints[ep_name] = EndpointInfo(
            name=ep_name,
            endpoint_identifier=ep_config.endpoint_identifier,
            max_context_window=ep_config.max_context_window,
            supports_structured_output=ep_config.supports_structured_output,
        )

    return EndpointCatalogResponse(
        categories=categories,
        endpoints=endpoints,
    )


# Simple module-level cache for workspace endpoints
_serving_cache: ServingEndpointsResponse | None = None
_serving_cache_time: float = 0.0
_SERVING_CACHE_TTL = 120.0  # 2 minutes


@router.get("/serving-endpoints", response_model=ServingEndpointsResponse)
async def get_serving_endpoints(
    request: Request,
    _user: CurrentUser,
) -> ServingEndpointsResponse:
    """List workspace serving endpoints for model override autocomplete.

    Returns all serving endpoints from the Databricks workspace.
    Results are cached for 2 minutes to avoid excessive API calls.
    Requires authentication (uses workspace client or OBO token).
    """
    global _serving_cache, _serving_cache_time

    if _serving_cache is not None and (time.monotonic() - _serving_cache_time) < _SERVING_CACHE_TTL:
        return _serving_cache

    from deep_research.services.discovery_service import DiscoveryService

    app_config = get_app_config()
    obo_token: str | None = getattr(request.state, "obo_token", None)
    discovery = DiscoveryService()

    try:
        sources, _ = await discovery.discover_serving_endpoints(
            user_token=obo_token, include_all=True,
        )
    except Exception:
        logger.warning("SERVING_ENDPOINTS_DISCOVERY_FAILED", exc_info=True)
        return ServingEndpointsResponse()

    # Collect YAML endpoint identifiers for dedup
    config_identifiers = [
        ep.endpoint_identifier for ep in app_config.endpoints.values()
    ]

    endpoints = [
        ServingEndpointSummary(
            name=src.endpoint_name,
            endpoint_type=(
                src.metadata.get("endpoint_type", "CUSTOM")
                if isinstance(src.metadata, dict) else "CUSTOM"
            ),
            state=(
                src.metadata.get("state", "UNKNOWN")
                if isinstance(src.metadata, dict) else "UNKNOWN"
            ),
        )
        for src in sources
        if src.endpoint_name
    ]

    result = ServingEndpointsResponse(
        endpoints=endpoints,
        config_endpoint_names=config_identifiers,
    )
    _serving_cache = result
    _serving_cache_time = time.monotonic()
    return result


@router.get("/deployment-defaults", response_model=DeploymentDefaultsResponse)
async def get_deployment_defaults(
    _user: CurrentUser,
) -> DeploymentDefaultsResponse:
    """Default values for the Agent Designer deployment wizards.

    Mode-2 (shell-app) and Mode-3 (MLflow agent) both pin
    ``databricks-deep-research`` from a Git ref. We default that ref via the
    shared ``framework_git_tag`` helper so the wizard and deploy artifact use
    the same pin.
    """
    return DeploymentDefaultsResponse(framework_git_tag=framework_git_tag())
