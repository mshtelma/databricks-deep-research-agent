"""
Plugin Discovery
================

Discovers plugins via Python entry points mechanism and provides
functions to query plugin-provided resources (data sources, templates, agents).

Plugins register themselves in their pyproject.toml under:

[project.entry-points."deep_research.plugins"]
my_plugin = "mypackage.plugin:MyPlugin"

Enterprise Data Source Integration (007-enterprise-data-sources):
- T061: get_plugin_data_sources(), apply_plugin_source_constraints()
- T062: get_plugin_templates()
- T063: get_plugin_agents()
- T064: emit_data_source_event(), emit_template_applied_event(), emit_agent_selected_event()
"""

from __future__ import annotations

import importlib.metadata
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from deep_research.agent.tools.base import ResearchContext
from deep_research.plugins.base import (
    CustomAgentDefinition,
    CustomAgentProvider,
    DataSourceDefinition,
    DataSourceProvider,
    PromptTemplateDefinition,
    TemplateProvider,
)

if TYPE_CHECKING:
    from deep_research.plugins.manager import PluginManager

logger = logging.getLogger(__name__)

PLUGIN_ENTRY_POINT_GROUP = "deep_research.plugins"


# =============================================================================
# Entry Point Discovery (Existing)
# =============================================================================


def discover_plugins() -> list[type[Any]]:
    """
    Discover all plugins registered via entry points.

    Looks for entry points in the 'deep_research.plugins' group.
    Each entry point should point to a class implementing ResearchPlugin.

    Returns:
        List of plugin classes (not instances).
        Failed loads are logged and skipped.

    Example pyproject.toml for a plugin:
        [project.entry-points."deep_research.plugins"]
        my_plugin = "mypackage.plugin:MyPlugin"
    """
    plugins: list[type[Any]] = []

    try:
        entry_points = importlib.metadata.entry_points(group=PLUGIN_ENTRY_POINT_GROUP)
    except Exception as e:
        logger.warning("Failed to get entry points: %s", e)
        return plugins

    for ep in entry_points:
        try:
            logger.debug("Loading plugin from entry point: %s", ep.name)
            plugin_cls = ep.load()
            plugins.append(plugin_cls)
            logger.info("Discovered plugin: %s from %s", ep.name, ep.value)
        except ImportError as e:
            logger.warning(
                "Failed to import plugin '%s' from '%s': %s",
                ep.name,
                ep.value,
                e,
            )
        except Exception as e:
            logger.error(
                "Error loading plugin '%s': %s",
                ep.name,
                e,
            )

    return plugins


def discover_tools() -> list[type[Any]]:
    """
    Discover all tools registered via entry points.

    Looks for entry points in the 'deep_research.tools' group.
    This is separate from plugins - core tools register here.

    Returns:
        List of tool classes (not instances).
    """
    tools: list[type[Any]] = []

    try:
        entry_points = importlib.metadata.entry_points(group="deep_research.tools")
    except Exception as e:
        logger.warning("Failed to get tool entry points: %s", e)
        return tools

    for ep in entry_points:
        try:
            logger.debug("Loading tool from entry point: %s", ep.name)
            tool_cls = ep.load()
            tools.append(tool_cls)
            logger.info("Discovered tool: %s from %s", ep.name, ep.value)
        except ImportError as e:
            logger.warning(
                "Failed to import tool '%s' from '%s': %s",
                ep.name,
                ep.value,
                e,
            )
        except Exception as e:
            logger.error(
                "Error loading tool '%s': %s",
                ep.name,
                e,
            )

    return tools


def get_plugin_count() -> int:
    """Get count of discovered plugins without loading them."""
    try:
        entry_points = importlib.metadata.entry_points(group=PLUGIN_ENTRY_POINT_GROUP)
        return len(list(entry_points))
    except Exception:
        return 0


# =============================================================================
# T061: Plugin Data Source Integration
# =============================================================================


@dataclass
class PluginDataSource:
    """Data source with plugin attribution.

    Wraps DataSourceDefinition with the plugin name that provided it.
    """

    definition: DataSourceDefinition
    plugin_name: str


def get_plugin_data_sources(
    plugin_manager: PluginManager,
    context: ResearchContext,
) -> list[PluginDataSource]:
    """
    Query DataSourceProvider.get_data_sources() from all registered plugins.

    T061: Integrates plugin-provided data sources with the source browser.
    Each data source is attributed to its providing plugin.

    Args:
        plugin_manager: Initialized PluginManager instance.
        context: Research context for filtering sources.

    Returns:
        List of PluginDataSource objects with plugin attribution.

    Example:
        sources = get_plugin_data_sources(plugin_manager, context)
        for source in sources:
            print(f"{source.definition.name} from {source.plugin_name}")
    """
    results: list[PluginDataSource] = []

    for plugin in plugin_manager.get_plugins():
        if isinstance(plugin, DataSourceProvider):
            try:
                data_sources = plugin.get_data_sources(context)
                for ds in data_sources:
                    results.append(PluginDataSource(
                        definition=ds,
                        plugin_name=plugin.name,
                    ))
                logger.debug(
                    "Retrieved %d data sources from plugin '%s'",
                    len(data_sources),
                    plugin.name,
                )
            except Exception as e:
                logger.warning(
                    "Failed to get data sources from plugin '%s': %s",
                    plugin.name,
                    str(e)[:200],
                )

    logger.info(
        "PLUGIN_DATA_SOURCES_RETRIEVED total=%d plugins=%d",
        len(results),
        len([p for p in plugin_manager.get_plugins() if isinstance(p, DataSourceProvider)]),
    )

    return results


@dataclass
class MergedSourceConstraints:
    """Merged source constraints from multiple plugins.

    When multiple plugins provide constraints, they are merged as follows:
    - allowed_types: Intersection (must satisfy all plugins)
    - allowed_sources: Intersection (must satisfy all plugins)
    - required_sources: Union (must query all required sources)
    - excluded_sources: Union (exclude from any plugin)
    """

    allowed_types: set[str] | None = None
    allowed_sources: set[str] | None = None
    required_sources: set[str] = field(default_factory=set)
    excluded_sources: set[str] = field(default_factory=set)
    contributing_plugins: list[str] = field(default_factory=list)


def apply_plugin_source_constraints(
    plugin_manager: PluginManager,
    context: ResearchContext,
) -> MergedSourceConstraints:
    """
    Apply get_source_constraints() from all active plugins.

    T061: Merges source constraints from all DataSourceProvider plugins.
    Constraints are merged conservatively:
    - Intersect allowed types/sources (most restrictive)
    - Union required/excluded sources (include all requirements)

    Args:
        plugin_manager: Initialized PluginManager instance.
        context: Research context for constraint evaluation.

    Returns:
        MergedSourceConstraints with combined constraints from all plugins.

    Example:
        constraints = apply_plugin_source_constraints(plugin_manager, context)
        if constraints.required_sources:
            print(f"Must query: {constraints.required_sources}")
    """
    merged = MergedSourceConstraints()

    for plugin in plugin_manager.get_plugins():
        if isinstance(plugin, DataSourceProvider):
            try:
                constraints = plugin.get_source_constraints(context)
                if constraints is None:
                    continue

                merged.contributing_plugins.append(plugin.name)

                # Merge allowed_types (intersection)
                if constraints.allowed_types is not None:
                    if merged.allowed_types is None:
                        merged.allowed_types = set(constraints.allowed_types)
                    else:
                        merged.allowed_types &= constraints.allowed_types

                # Merge allowed_sources (intersection)
                if constraints.allowed_sources is not None:
                    allowed_set = set(constraints.allowed_sources)
                    if merged.allowed_sources is None:
                        merged.allowed_sources = allowed_set
                    else:
                        merged.allowed_sources &= allowed_set

                # Merge required_sources (union)
                merged.required_sources.update(constraints.required_sources)

                # Merge excluded_sources (union)
                merged.excluded_sources.update(constraints.excluded_sources)

                logger.debug(
                    "Applied source constraints from plugin '%s': "
                    "allowed_types=%s required=%d excluded=%d",
                    plugin.name,
                    constraints.allowed_types,
                    len(constraints.required_sources),
                    len(constraints.excluded_sources),
                )

            except Exception as e:
                logger.warning(
                    "Failed to get source constraints from plugin '%s': %s",
                    plugin.name,
                    str(e)[:200],
                )

    if merged.contributing_plugins:
        logger.info(
            "PLUGIN_SOURCE_CONSTRAINTS_APPLIED plugins=%s required=%d excluded=%d",
            merged.contributing_plugins,
            len(merged.required_sources),
            len(merged.excluded_sources),
        )

    return merged


def filter_sources_by_constraints(
    sources: list[DataSourceDefinition],
    constraints: MergedSourceConstraints,
) -> list[DataSourceDefinition]:
    """
    Filter a list of sources by merged constraints.

    Utility function to apply MergedSourceConstraints to a source list.

    Args:
        sources: List of data source definitions.
        constraints: Merged constraints from plugins.

    Returns:
        Filtered list of sources that satisfy all constraints.
    """
    filtered: list[DataSourceDefinition] = []

    for source in sources:
        # Check allowed_types
        if (
            constraints.allowed_types is not None
            and source.type not in constraints.allowed_types
        ):
            continue

        # Check allowed_sources
        if (
            constraints.allowed_sources is not None
            and source.name not in constraints.allowed_sources
        ):
            continue

        # Check excluded_sources
        if source.name in constraints.excluded_sources:
            continue

        filtered.append(source)

    return filtered


# =============================================================================
# T062: Plugin Template Integration
# =============================================================================


@dataclass
class PluginTemplate:
    """Template with plugin attribution.

    Wraps PromptTemplateDefinition with the plugin name that provided it.
    """

    definition: PromptTemplateDefinition
    plugin_name: str


def get_plugin_templates(
    plugin_manager: PluginManager,
    context: ResearchContext,
) -> list[PluginTemplate]:
    """
    Query TemplateProvider.get_templates() from all registered plugins.

    T062: Integrates plugin-provided templates with the template library.
    Each template is attributed to its providing plugin.

    Args:
        plugin_manager: Initialized PluginManager instance.
        context: Research context for filtering templates.

    Returns:
        List of PluginTemplate objects with plugin attribution.

    Example:
        templates = get_plugin_templates(plugin_manager, context)
        for template in templates:
            print(f"{template.definition.name} from {template.plugin_name}")
    """
    results: list[PluginTemplate] = []

    for plugin in plugin_manager.get_plugins():
        if isinstance(plugin, TemplateProvider):
            try:
                templates = plugin.get_templates(context)
                for tmpl in templates:
                    results.append(PluginTemplate(
                        definition=tmpl,
                        plugin_name=plugin.name,
                    ))
                logger.debug(
                    "Retrieved %d templates from plugin '%s'",
                    len(templates),
                    plugin.name,
                )
            except Exception as e:
                logger.warning(
                    "Failed to get templates from plugin '%s': %s",
                    plugin.name,
                    str(e)[:200],
                )

    logger.info(
        "PLUGIN_TEMPLATES_RETRIEVED total=%d plugins=%d",
        len(results),
        len([p for p in plugin_manager.get_plugins() if isinstance(p, TemplateProvider)]),
    )

    return results


def resolve_template_variable(
    plugin_manager: PluginManager,
    variable_name: str,
    context: ResearchContext,
    plugin_name: str | None = None,
) -> str | None:
    """
    Resolve a template variable using plugin-provided resolvers.

    Queries TemplateProvider.resolve_variable() from plugins.
    If plugin_name is provided, only that plugin is queried.
    Otherwise, all TemplateProvider plugins are tried in order.

    Args:
        plugin_manager: Initialized PluginManager instance.
        variable_name: Name of the variable to resolve.
        context: Research context for resolution.
        plugin_name: Optional specific plugin to query.

    Returns:
        Resolved value or None if no plugin can resolve it.
    """
    for plugin in plugin_manager.get_plugins():
        if not isinstance(plugin, TemplateProvider):
            continue

        # If specific plugin requested, skip others
        if plugin_name is not None and plugin.name != plugin_name:
            continue

        try:
            value = plugin.resolve_variable(variable_name, context)
            if value is not None:
                logger.debug(
                    "Resolved variable '%s' via plugin '%s'",
                    variable_name,
                    plugin.name,
                )
                return value
        except Exception as e:
            logger.warning(
                "Failed to resolve variable '%s' via plugin '%s': %s",
                variable_name,
                plugin.name,
                str(e)[:200],
            )

    return None


# =============================================================================
# T063: Plugin Agent Integration
# =============================================================================


@dataclass
class PluginAgent:
    """Custom agent with plugin attribution.

    Wraps CustomAgentDefinition with the plugin name that provided it.
    """

    definition: CustomAgentDefinition
    plugin_name: str


def get_plugin_agents(
    plugin_manager: PluginManager,
    context: ResearchContext,
) -> list[PluginAgent]:
    """
    Query CustomAgentProvider.get_custom_agents() from all registered plugins.

    T063: Integrates plugin-provided agents with the agent selector.
    Each agent is attributed to its providing plugin.

    Args:
        plugin_manager: Initialized PluginManager instance.
        context: Research context for filtering agents.

    Returns:
        List of PluginAgent objects with plugin attribution.

    Example:
        agents = get_plugin_agents(plugin_manager, context)
        for agent in agents:
            print(f"{agent.definition.name} from {agent.plugin_name}")
    """
    results: list[PluginAgent] = []

    for plugin in plugin_manager.get_plugins():
        if isinstance(plugin, CustomAgentProvider):
            try:
                agents = plugin.get_custom_agents(context)
                for agent in agents:
                    results.append(PluginAgent(
                        definition=agent,
                        plugin_name=plugin.name,
                    ))
                logger.debug(
                    "Retrieved %d custom agents from plugin '%s'",
                    len(agents),
                    plugin.name,
                )
            except Exception as e:
                logger.warning(
                    "Failed to get custom agents from plugin '%s': %s",
                    plugin.name,
                    str(e)[:200],
                )

    logger.info(
        "PLUGIN_AGENTS_RETRIEVED total=%d plugins=%d",
        len(results),
        len([p for p in plugin_manager.get_plugins() if isinstance(p, CustomAgentProvider)]),
    )

    return results


# =============================================================================
# T064: Lifecycle Event Emission Helpers
# =============================================================================


async def emit_data_source_query_event(
    plugin_manager: PluginManager,
    job_id: str,
    source_type: str,
    source_name: str,
    query: str,
    result_count: int,
    duration_ms: float,
    filters: dict[str, Any] | None = None,
    success: bool = True,
    error_message: str | None = None,
) -> None:
    """
    Emit DataSourceQueryEvent after a source is queried.

    T064: Called by tool execution when querying enterprise data sources.
    Provides observability for Vector Search, Genie, Knowledge Assistants.

    Args:
        plugin_manager: Initialized PluginManager instance.
        job_id: Current job ID (string form).
        source_type: Type of data source (e.g., 'vector_search', 'genie').
        source_name: Name of the specific source.
        query: The query that was executed.
        result_count: Number of results returned.
        duration_ms: Query execution time in milliseconds.
        filters: Optional filters applied to query.
        success: Whether the query succeeded.
        error_message: Error message if query failed.

    Example:
        await emit_data_source_query_event(
            plugin_manager,
            job_id="123",
            source_type="vector_search",
            source_name="product_docs",
            query="How to configure...",
            result_count=5,
            duration_ms=150.0,
        )
    """
    from deep_research.plugins.lifecycle.events import DataSourceQueryEvent

    event = DataSourceQueryEvent(
        job_id=job_id,
        source_type=source_type,
        source_name=source_name,
        query=query,
        filters=filters,
        result_count=result_count,
        duration_ms=duration_ms,
        success=success,
        error_message=error_message,
        timestamp=datetime.now(UTC),
    )

    try:
        await plugin_manager.emit_hook("on_data_source_query", event)
        logger.debug(
            "EMITTED_DATA_SOURCE_QUERY source=%s results=%d success=%s",
            source_name,
            result_count,
            success,
        )
    except Exception as e:
        logger.warning(
            "Failed to emit DataSourceQueryEvent: %s",
            str(e)[:200],
        )


async def emit_template_applied_event(
    plugin_manager: PluginManager,
    job_id: str,
    template_id: str,
    template_type: str,
    template_source: str,
    variables: dict[str, Any],
) -> None:
    """
    Emit TemplateAppliedEvent when a template is used.

    T064: Called when a template is applied during research.
    Tracks template usage for analytics and debugging.

    Args:
        plugin_manager: Initialized PluginManager instance.
        job_id: Current job ID (string form).
        template_id: ID of the applied template.
        template_type: Template type ('system', 'step', 'synthesis', 'query').
        template_source: Source of template ('system', 'plugin', 'user').
        variables: Variables used in template rendering.

    Example:
        await emit_template_applied_event(
            plugin_manager,
            job_id="123",
            template_id="sales_report",
            template_type="synthesis",
            template_source="plugin",
            variables={"company_name": "Acme Inc"},
        )
    """
    from deep_research.plugins.lifecycle.events import TemplateAppliedEvent

    event = TemplateAppliedEvent(
        job_id=job_id,
        template_id=template_id,
        template_type=template_type,
        template_source=template_source,
        variables=variables,
        timestamp=datetime.now(UTC),
    )

    try:
        await plugin_manager.emit_hook("on_template_applied", event)
        logger.debug(
            "EMITTED_TEMPLATE_APPLIED template=%s type=%s source=%s",
            template_id,
            template_type,
            template_source,
        )
    except Exception as e:
        logger.warning(
            "Failed to emit TemplateAppliedEvent: %s",
            str(e)[:200],
        )


async def emit_custom_agent_selected_event(
    plugin_manager: PluginManager,
    job_id: str,
    agent_id: str,
    agent_name: str,
    agent_source: str,
) -> None:
    """
    Emit CustomAgentSelectedEvent when a custom agent is selected.

    T064: Called when a custom agent is selected for research.
    Tracks custom agent usage for analytics.

    Args:
        plugin_manager: Initialized PluginManager instance.
        job_id: Current job ID (string form).
        agent_id: ID of the selected agent.
        agent_name: Name of the selected agent.
        agent_source: Source of agent ('plugin', 'user').

    Example:
        await emit_custom_agent_selected_event(
            plugin_manager,
            job_id="123",
            agent_id="sales_researcher",
            agent_name="Sales Research Specialist",
            agent_source="plugin",
        )
    """
    from deep_research.plugins.lifecycle.events import CustomAgentSelectedEvent

    event = CustomAgentSelectedEvent(
        job_id=job_id,
        agent_id=agent_id,
        agent_name=agent_name,
        agent_source=agent_source,
        timestamp=datetime.now(UTC),
    )

    try:
        await plugin_manager.emit_hook("on_custom_agent_selected", event)
        logger.debug(
            "EMITTED_CUSTOM_AGENT_SELECTED agent=%s source=%s",
            agent_name,
            agent_source,
        )
    except Exception as e:
        logger.warning(
            "Failed to emit CustomAgentSelectedEvent: %s",
            str(e)[:200],
        )


async def emit_data_landscape_built_event(
    plugin_manager: PluginManager,
    job_id: str,
    sources_queried: int,
    sources_with_results: int,
    top_source: str | None,
    top_source_relevance: float | None,
    total_duration_ms: float,
) -> None:
    """
    Emit DataLandscapeBuiltEvent when discovery builds the data landscape.

    T064: Called after discovery phase completes source relevance assessment.
    Tracks discovery performance and source effectiveness.

    Args:
        plugin_manager: Initialized PluginManager instance.
        job_id: Current job ID (string form).
        sources_queried: Number of sources queried during discovery.
        sources_with_results: Number of sources that returned results.
        top_source: Name of the most relevant source (if any).
        top_source_relevance: Relevance score of top source.
        total_duration_ms: Total discovery duration in milliseconds.

    Example:
        await emit_data_landscape_built_event(
            plugin_manager,
            job_id="123",
            sources_queried=5,
            sources_with_results=3,
            top_source="product_docs",
            top_source_relevance=0.85,
            total_duration_ms=2500.0,
        )
    """
    from deep_research.plugins.lifecycle.events import DataLandscapeBuiltEvent

    event = DataLandscapeBuiltEvent(
        job_id=job_id,
        sources_queried=sources_queried,
        sources_with_results=sources_with_results,
        top_source=top_source,
        top_source_relevance=top_source_relevance,
        total_duration_ms=total_duration_ms,
        timestamp=datetime.now(UTC),
    )

    try:
        await plugin_manager.emit_hook("on_data_landscape_built", event)
        logger.debug(
            "EMITTED_DATA_LANDSCAPE_BUILT sources_queried=%d with_results=%d top=%s",
            sources_queried,
            sources_with_results,
            top_source,
        )
    except Exception as e:
        logger.warning(
            "Failed to emit DataLandscapeBuiltEvent: %s",
            str(e)[:200],
        )


# =============================================================================
# Convenience: Combined Discovery Result
# =============================================================================


@dataclass
class PluginDiscoveryResult:
    """Combined result from all plugin discovery functions.

    Convenience dataclass for getting all plugin-provided resources at once.
    """

    data_sources: list[PluginDataSource]
    templates: list[PluginTemplate]
    agents: list[PluginAgent]
    source_constraints: MergedSourceConstraints


def discover_all_plugin_resources(
    plugin_manager: PluginManager,
    context: ResearchContext,
) -> PluginDiscoveryResult:
    """
    Discover all resources from registered plugins in one call.

    Convenience function that calls all discovery functions and returns
    a combined result.

    Args:
        plugin_manager: Initialized PluginManager instance.
        context: Research context for filtering.

    Returns:
        PluginDiscoveryResult with all discovered resources.
    """
    return PluginDiscoveryResult(
        data_sources=get_plugin_data_sources(plugin_manager, context),
        templates=get_plugin_templates(plugin_manager, context),
        agents=get_plugin_agents(plugin_manager, context),
        source_constraints=apply_plugin_source_constraints(plugin_manager, context),
    )
