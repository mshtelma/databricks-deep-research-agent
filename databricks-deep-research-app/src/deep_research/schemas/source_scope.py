"""Source Scope schemas for controlling data source selection.

These models allow users to control which categories of data sources
are available for research (enterprise only, web only, or all).

Part of 007-enterprise-data-sources feature (T028).
"""

from collections.abc import Callable
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class SourceScope(StrEnum):
    """Scope of data sources to use during research.

    Users can restrict research to specific source categories:
    - ENTERPRISE_ONLY: Only use Databricks enterprise sources (VS, Genie, KA)
    - WEB_ONLY: Only use web search (Brave)
    - ALL: Use all available sources (default)
    """

    ENTERPRISE_ONLY = "enterprise_only"
    WEB_ONLY = "web_only"
    ALL = "all"


class SourceScopeConfig(BaseModel):
    """Configuration for source scope filtering.

    Provides methods to filter available sources based on scope.
    """

    scope: SourceScope = SourceScope.ALL
    """Which category of sources to use."""

    enabled_sources: list[str] | None = None
    """If set, only these specific sources (by name) are enabled.
    Takes precedence over scope for fine-grained control."""

    disabled_sources: list[str] = Field(default_factory=list)
    """Sources to explicitly disable (by name)."""

    # Type-level toggles
    enable_vector_search: bool = True
    """Enable Vector Search sources."""

    enable_genie: bool = True
    """Enable Genie sources."""

    enable_knowledge_assistant: bool = True
    """Enable Knowledge Assistant sources."""

    enable_web_search: bool = True
    """Enable web search (Brave)."""

    enable_uploaded_files: bool = True
    """Enable uploaded file search."""

    class Config:
        """Pydantic configuration."""

        use_enum_values = True

    def is_type_enabled(self, source_type: str) -> bool:
        """Check if a source type is enabled by scope and toggles.

        Args:
            source_type: Type of source (vector_search, genie, etc.)

        Returns:
            True if the type is enabled.
        """
        # Check scope first
        if self.scope == SourceScope.ENTERPRISE_ONLY and source_type in ("web_search",):
            return False
        if self.scope == SourceScope.WEB_ONLY and source_type in (
            "vector_search", "genie", "knowledge_assistant"
        ):
            return False

        # Check type-level toggles
        type_toggle_map = {
            "vector_search": self.enable_vector_search,
            "genie": self.enable_genie,
            "knowledge_assistant": self.enable_knowledge_assistant,
            "web_search": self.enable_web_search,
            "uploaded_file": self.enable_uploaded_files,
        }

        return type_toggle_map.get(source_type, True)

    def is_source_enabled(self, source_name: str, source_type: str) -> bool:
        """Check if a specific source is enabled.

        Args:
            source_name: Name of the source.
            source_type: Type of the source.

        Returns:
            True if the source is enabled.
        """
        # Check if type is enabled
        if not self.is_type_enabled(source_type):
            return False

        # Check explicit disabled list
        if source_name in self.disabled_sources:
            return False

        # Check explicit enabled list (if set, only those are allowed)
        if self.enabled_sources is not None:
            return source_name in self.enabled_sources

        return True

    def filter_sources(
        self,
        sources: list[Any],
        name_getter: Callable[[Any], str] = lambda x: x.name,
        type_getter: Callable[[Any], str] = lambda x: x.type,
    ) -> list[Any]:
        """Filter a list of sources based on scope configuration.

        Args:
            sources: List of source objects.
            name_getter: Function to get source name from object.
            type_getter: Function to get source type from object.

        Returns:
            Filtered list of sources.
        """
        return [
            s for s in sources
            if self.is_source_enabled(name_getter(s), type_getter(s))
        ]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "scope": self.scope,
            "enabled_sources": self.enabled_sources,
            "disabled_sources": self.disabled_sources,
            "enable_vector_search": self.enable_vector_search,
            "enable_genie": self.enable_genie,
            "enable_knowledge_assistant": self.enable_knowledge_assistant,
            "enable_web_search": self.enable_web_search,
            "enable_uploaded_files": self.enable_uploaded_files,
        }


# Default scope configurations for common use cases
DEFAULT_SCOPE_CONFIGS = {
    "enterprise_only": SourceScopeConfig(
        scope=SourceScope.ENTERPRISE_ONLY,
        enable_web_search=False,
    ),
    "web_only": SourceScopeConfig(
        scope=SourceScope.WEB_ONLY,
        enable_vector_search=False,
        enable_genie=False,
        enable_knowledge_assistant=False,
    ),
    "all": SourceScopeConfig(
        scope=SourceScope.ALL,
    ),
}
