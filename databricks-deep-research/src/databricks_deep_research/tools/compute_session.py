"""Run-scoped compute-tool lookup helpers.

The run's Python scratchpad (the ``compute`` tool's persistent namespace) is
resolver-cache-scoped: :class:`ToolResolver` shares its instance cache with
factories via ``extras["_resolver_cache"]`` so sibling closures
(``compute_namespace``, ``table_load``) can find the run's compute singleton.
This module formalizes that idiom for consumers outside the factory layer
(e.g. the workflow executor's tool node). Isolation semantics come for free:
``isolate`` subworkflows drop ``_resolver_cache`` from the copied extras, so
lookups there see the child's fresh compute tool, never the parent's.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from databricks_deep_research.tools.builtins.compute import PythonComputeTool


def get_compute_tool(
    extras: Mapping[str, Any], name: str = "compute"
) -> PythonComputeTool | None:
    """Return the run's cached compute singleton from the resolver cache.

    Looks up ``extras["_resolver_cache"]`` first by ``name`` (the conventional
    declaration name), then by scanning for any cached compute tool, since
    workflows may declare their compute tool under a different name. Returns
    ``None`` when the run has no resolver cache or no compute tool.
    """
    cache = extras.get("_resolver_cache")
    if not isinstance(cache, Mapping):
        return None
    candidate = cache.get(name)
    if isinstance(candidate, PythonComputeTool):
        return candidate
    for value in cache.values():
        if isinstance(value, PythonComputeTool):
            return value
    return None
