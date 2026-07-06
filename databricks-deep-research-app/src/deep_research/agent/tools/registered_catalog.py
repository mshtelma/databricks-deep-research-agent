"""Process-level catalog of operator-registered Python tools.

Built lazily from ``app.yaml``'s ``tools.registered_tools`` entries
("module:attr", imported ONCE at startup — startup config is operator-trusted,
unlike stored workflow rows). Workflows reference catalog entries by key via
``kind: registered`` (dict lookup only). Plugin-provided tools keep flowing
through their existing enterprise-override path and are NOT duplicated here.
"""

from __future__ import annotations

import importlib
import logging
import threading
from collections.abc import Mapping
from typing import Any

from databricks_deep_research.tools.api import tool as framework_tool
from databricks_deep_research.tools.protocol import ResearchTool

from deep_research.core.app_config import get_app_config

logger = logging.getLogger(__name__)

_CATALOG: dict[str, ResearchTool] | None = None
_LOCK = threading.Lock()


def _load_entry(entry: str) -> tuple[str, ResearchTool] | None:
    module_path, _, attr = entry.partition(":")
    if not module_path or not attr:
        logger.warning(
            "REGISTERED_TOOL_BAD_ENTRY entry=%r expected 'module:attr'", entry
        )
        return None
    try:
        module = importlib.import_module(module_path)
        value: Any = getattr(module, attr)
    except (ImportError, AttributeError) as exc:
        logger.warning("REGISTERED_TOOL_LOAD_FAILED entry=%r err=%s", entry, exc)
        return None
    candidate: Any = value
    if not hasattr(candidate, "definition") and callable(candidate):
        candidate = framework_tool(candidate)
    definition = getattr(candidate, "definition", None)
    name = getattr(definition, "name", None)
    if not isinstance(name, str) or not name:
        logger.warning(
            "REGISTERED_TOOL_INVALID entry=%r (no ResearchTool definition)", entry
        )
        return None
    key = f"{module_path}.{attr}"
    return key, candidate


def get_registered_tool_catalog() -> Mapping[str, ResearchTool]:
    """Return the (lazily built, cached) registered-tool catalog."""
    global _CATALOG
    if _CATALOG is None:
        with _LOCK:
            if _CATALOG is None:
                catalog: dict[str, ResearchTool] = {}
                for entry in get_app_config().tools.registered_tools:
                    loaded = _load_entry(entry)
                    if loaded is None:
                        continue
                    key, loaded_tool = loaded
                    if key in catalog:
                        logger.warning("REGISTERED_TOOL_DUPLICATE key=%s", key)
                        continue
                    catalog[key] = loaded_tool
                logger.info(
                    "REGISTERED_TOOL_CATALOG_READY entries=%d keys=%s",
                    len(catalog),
                    sorted(catalog),
                )
                _CATALOG = catalog
    return _CATALOG


def registered_tool_keys() -> list[str]:
    """Catalog keys for Designer surfacing / validation."""
    return sorted(get_registered_tool_catalog())


def reset_registered_tool_catalog() -> None:
    """Test seam: drop the cached catalog."""
    global _CATALOG
    _CATALOG = None
