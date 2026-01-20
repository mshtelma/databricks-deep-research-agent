"""
Deep Research Agent - A pip-installable research agent framework for Databricks.

This package provides the core functionality for building research agents with
multi-agent architecture, citation tracking, and extensibility via plugins.

Quick Start
-----------
Run a research query:

    from deep_research import run_research

    result = await run_research(
        query="What are the latest developments in AI?",
        user_id="user@example.com",
    )
    print(result.final_report)

Create a custom plugin:

    from deep_research.plugins import ResearchPlugin, ToolProvider
    from deep_research.agent.tools import ResearchTool, ResearchContext, ToolResult

    class MyPlugin(ResearchPlugin, ToolProvider):
        @property
        def name(self) -> str:
            return "my-plugin"

        @property
        def version(self) -> str:
            return "1.0.0"

        def initialize(self, app_config):
            pass

        def shutdown(self):
            pass

        def get_tools(self, context: ResearchContext) -> list[ResearchTool]:
            return [MyCustomTool()]

Public API Exports
------------------

Core:
    run_research: Async function to run a complete research session
    stream_research: Async generator for streaming research events
    ResearchState: State container for research session

Plugins:
    ResearchPlugin: Base protocol for all plugins
    ToolProvider: Protocol for plugins that provide tools
    PromptProvider: Protocol for plugins that customize prompts
    PluginManager: Manages plugin discovery and lifecycle
    discover_plugins: Function to discover plugins via entry points

Tools:
    ResearchTool: Protocol for research tools
    ToolDefinition: Tool metadata (name, description, parameters)
    ToolResult: Result returned by tool execution
    ResearchContext: Context passed to tool execution
    ToolRegistry: Registry for managing tools

Configuration:
    AppConfig: Central application configuration
    get_app_config: Get the current app configuration

LLM:
    LLMClient: Client for LLM interactions

Pipeline:
    PipelineConfig: Declarative pipeline configuration
    AgentConfig: Configuration for individual agents
    PipelineExecutor: Executes research pipelines
    PipelineCustomization: Customization for pipelines

Output:
    OutputTypeProvider: Protocol for custom output types
    OutputTypeRegistry: Registry for output types
    SynthesizerConfig: Configuration for synthesis

Conversation:
    ConversationProvider: Protocol for conversation handling
    IntentClassifier: Protocol for intent classification
    ConversationHandler: Protocol for conversation handling
"""

__version__ = "0.1.0"


def __getattr__(name: str):
    """Lazy loading of exports to avoid circular imports."""
    # Core orchestrator
    if name == "run_research":
        from deep_research.agent.orchestrator import run_research

        return run_research
    if name == "stream_research":
        from deep_research.agent.orchestrator import stream_research

        return stream_research
    if name == "ResearchState":
        from deep_research.agent.state import ResearchState

        return ResearchState

    # Plugin protocols
    if name == "ResearchPlugin":
        from deep_research.plugins.base import ResearchPlugin

        return ResearchPlugin
    if name == "ToolProvider":
        from deep_research.plugins.base import ToolProvider

        return ToolProvider
    if name == "PromptProvider":
        from deep_research.plugins.base import PromptProvider

        return PromptProvider
    if name == "PluginManager":
        from deep_research.plugins.manager import PluginManager

        return PluginManager
    if name == "discover_plugins":
        from deep_research.plugins.discovery import discover_plugins

        return discover_plugins

    # Tool infrastructure
    if name == "ResearchTool":
        from deep_research.agent.tools.base import ResearchTool

        return ResearchTool
    if name == "ToolDefinition":
        from deep_research.agent.tools.base import ToolDefinition

        return ToolDefinition
    if name == "ToolResult":
        from deep_research.agent.tools.base import ToolResult

        return ToolResult
    if name == "ResearchContext":
        from deep_research.agent.tools.base import ResearchContext

        return ResearchContext
    if name == "ToolRegistry":
        from deep_research.agent.tools.registry import ToolRegistry

        return ToolRegistry

    # Configuration
    if name == "AppConfig":
        from deep_research.core.app_config import AppConfig

        return AppConfig
    if name == "get_app_config":
        from deep_research.core.app_config import get_app_config

        return get_app_config

    # LLM Client
    if name == "LLMClient":
        from deep_research.services.llm.client import LLMClient

        return LLMClient

    # Pipeline
    if name == "PipelineConfig":
        from deep_research.agent.pipeline.config import PipelineConfig

        return PipelineConfig
    if name == "AgentConfig":
        from deep_research.agent.pipeline.config import AgentConfig

        return AgentConfig
    if name == "PipelineExecutor":
        from deep_research.agent.pipeline.executor import PipelineExecutor

        return PipelineExecutor
    if name == "PipelineCustomization":
        from deep_research.agent.pipeline.protocols import PipelineCustomization

        return PipelineCustomization

    # Output
    if name == "OutputTypeProvider":
        from deep_research.output.protocol import OutputTypeProvider

        return OutputTypeProvider
    if name == "OutputTypeRegistry":
        from deep_research.output.registry import OutputTypeRegistry

        return OutputTypeRegistry
    if name == "SynthesizerConfig":
        from deep_research.output.base import SynthesizerConfig

        return SynthesizerConfig

    # Conversation
    if name == "ConversationProvider":
        from deep_research.conversation.protocol import ConversationProvider

        return ConversationProvider
    if name == "IntentClassifier":
        from deep_research.conversation.intent import IntentClassifier

        return IntentClassifier
    if name == "ConversationHandler":
        from deep_research.conversation.handler import ConversationHandler

        return ConversationHandler

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Public API
__all__ = [
    # Version
    "__version__",
    # Core Orchestrator
    "run_research",
    "stream_research",
    "ResearchState",
    # Plugin Protocols
    "ResearchPlugin",
    "ToolProvider",
    "PromptProvider",
    "PluginManager",
    "discover_plugins",
    # Tool Infrastructure
    "ResearchTool",
    "ToolDefinition",
    "ToolResult",
    "ResearchContext",
    "ToolRegistry",
    # Configuration
    "AppConfig",
    "get_app_config",
    # LLM
    "LLMClient",
    # Pipeline
    "PipelineConfig",
    "AgentConfig",
    "PipelineExecutor",
    "PipelineCustomization",
    # Output
    "OutputTypeProvider",
    "OutputTypeRegistry",
    "SynthesizerConfig",
    # Conversation
    "ConversationProvider",
    "IntentClassifier",
    "ConversationHandler",
]
