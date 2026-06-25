"""Tool factories for creating ResearchTool instances from YAML declarations."""

from databricks_deep_research.tools.factories.builtin import BuiltinToolFactory
from databricks_deep_research.tools.factories.databricks import DatabricksToolFactory
from databricks_deep_research.tools.factories.decorated import DecoratedToolFactory

# ---------------------------------------------------------------------------
# BUILTIN_FACTORIES allow-list
# ---------------------------------------------------------------------------
# SECURITY: This dict is the ONLY authoritative mapping from factory_ref
# strings to factory callables.  Only entries explicitly added here can be
# referenced by custom_tool_defs rows.
#
# NEVER:
#   - add a key whose value is derived from user input
#   - call importlib.import_module to resolve a factory_ref
#   - use getattr / eval to look up a factory by name at runtime
#
# The dict maps human-readable stable identifiers to the factory class that
# knows how to build that tool kind.  The same factory class may appear under
# multiple keys when the class handles several related kinds.
BUILTIN_FACTORIES: dict[str, type] = {
    # Builtin tool kinds (web search, crawl, file, compute, text_table)
    "web_search_v1": BuiltinToolFactory,
    "web_crawl_v1": BuiltinToolFactory,
    "web_research_v1": BuiltinToolFactory,
    "file_search_v1": BuiltinToolFactory,
    "compute_v1": BuiltinToolFactory,
    "compute_namespace_v1": BuiltinToolFactory,
    "read_skill_v1": BuiltinToolFactory,
    "table_discovery_v1": BuiltinToolFactory,
    "table_search_v1": BuiltinToolFactory,
    "table_read_v1": BuiltinToolFactory,
    "table_neighbors_v1": BuiltinToolFactory,
    "table_load_v1": BuiltinToolFactory,
    "table_aggregate_v1": BuiltinToolFactory,
    # Databricks-hosted tool kinds (vector search, Genie, KA)
    "vector_search_v1": DatabricksToolFactory,
    "genie_v1": DatabricksToolFactory,
    "knowledge_assistant_v1": DatabricksToolFactory,
}


def resolve_factory(factory_ref: str) -> type:
    """Resolve a factory_ref string to its factory class.

    SECURITY: This function performs a dict lookup ONLY against the static
    BUILTIN_FACTORIES allow-list.  It MUST NOT call importlib.import_module,
    getattr, eval, or any other dynamic import mechanism — doing so would
    allow arbitrary code execution via user-supplied factory_ref values.

    Args:
        factory_ref: A key that must exist in BUILTIN_FACTORIES.

    Returns:
        The factory class registered under factory_ref.

    Raises:
        ValueError: If factory_ref is not in the allow-list.
    """
    if factory_ref not in BUILTIN_FACTORIES:
        raise ValueError(f"factory_ref_not_in_allowlist: {factory_ref!r}")
    return BUILTIN_FACTORIES[factory_ref]


__all__ = [
    "BuiltinToolFactory",
    "DatabricksToolFactory",
    "DecoratedToolFactory",
    "BUILTIN_FACTORIES",
    "resolve_factory",
]
