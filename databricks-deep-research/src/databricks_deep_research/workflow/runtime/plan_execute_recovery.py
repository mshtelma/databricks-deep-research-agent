from __future__ import annotations

from typing import Any

from databricks_deep_research.pools.pool_state import PoolState
from databricks_deep_research.workflow.runtime_core.selectors import (
    select_discovered_sources,
    select_prior_sources,
)
from databricks_deep_research.workflow.state import WorkflowState


def coerce_discovered_sources(state: WorkflowState) -> list[dict[str, Any]]:
    raw = select_discovered_sources(state)
    return [item for item in raw if isinstance(item, dict)]


def coerce_prior_sources(state: WorkflowState) -> list[dict[str, Any]]:
    """Dict-filtered prior-turn sources staged under ``prior_sources_for_seed``."""
    raw = select_prior_sources(state)
    return [item for item in raw if isinstance(item, dict)]


def hydrate_pools_from_discovered_sources(
    pools: dict[str, PoolState],
    discovered_sources: list[dict[str, Any]],
) -> bool:
    if not discovered_sources:
        return False
    hydrated = False
    sources_pool = pools.get("sources")
    if sources_pool is not None:
        for item in discovered_sources:
            if sources_pool.add(item):
                hydrated = True
    observations_pool = pools.get("observations")
    if observations_pool is not None:
        for item in discovered_sources:
            summary = str(item.get("summary") or item.get("snippet") or "").strip()
            if not summary:
                continue
            observation = {"text": summary, "source": "discovered", "url": str(item.get("url") or "")}
            if observations_pool.add(observation):
                hydrated = True
    discovery_pool = pools.get("discovery_sources")
    if discovery_pool is not None:
        for item in discovered_sources:
            if discovery_pool.add(item):
                hydrated = True
    return hydrated
