"""
Unified configuration management with simplified precedence and validation.

This module replaces the complex configuration logic with a cleaner approach that supports
dot notation access, clear precedence rules, and proper validation.
"""

import os
import yaml
from typing import Any, Dict, List, Optional, Union, TypeVar, Generic
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import logging

from deep_research_agent.core.error_handler import retry, safe_call

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ConfigSource(Enum):
    """Configuration source types for tracking precedence."""
    OVERRIDE = "override"
    ENVIRONMENT = "environment" 
    YAML = "yaml"
    DEFAULT = "default"


@dataclass
class ConfigValue:
    """Configuration value with source tracking."""
    
    value: Any
    source: ConfigSource
    key_path: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConfigSchema:
    """Base configuration schema with validation."""
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        return []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


class AgentConfigSchema(ConfigSchema):
    """Agent configuration schema with validation."""
    
    def __init__(
        self,
        llm_endpoint: str = "databricks-claude-3-7-sonnet",
        max_research_loops: int = 2,
        initial_query_count: int = 3,
        max_concurrent_searches: int = 2,
        batch_delay_seconds: float = 1.0,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        timeout_seconds: int = 30,
        max_retries: int = 3,
        enable_streaming: bool = True,
        enable_citations: bool = True,
        search_provider: str = "tavily",
        **kwargs
    ):
        self.llm_endpoint = llm_endpoint
        self.max_research_loops = max_research_loops
        self.initial_query_count = initial_query_count
        self.max_concurrent_searches = max_concurrent_searches
        self.batch_delay_seconds = batch_delay_seconds
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.enable_streaming = enable_streaming
        self.enable_citations = enable_citations
        self.search_provider = search_provider
        super().__init__(**kwargs)
    
    def validate(self) -> List[str]:
        """Validate agent configuration."""
        errors = []
        
        if self.max_research_loops < 1:
            errors.append("max_research_loops must be at least 1")
        
        if self.initial_query_count < 1:
            errors.append("initial_query_count must be at least 1")
        
        if self.max_concurrent_searches < 1:
            errors.append("max_concurrent_searches must be at least 1")
        
        if self.batch_delay_seconds < 0:
            errors.append("batch_delay_seconds must be non-negative")
        
        if not 0 <= self.temperature <= 2:
            errors.append("temperature must be between 0 and 2")
        
        if self.max_tokens < 100:
            errors.append("max_tokens must be at least 100")
        
        if self.timeout_seconds < 1:
            errors.append("timeout_seconds must be at least 1")
        
        if self.max_retries < 0:
            errors.append("max_retries must be non-negative")
        
        if self.search_provider not in ["tavily", "brave", "vector", "hybrid"]:
            errors.append(f"search_provider must be one of: tavily, brave, vector, hybrid")
        
        return errors


class ToolConfigSchema(ConfigSchema):
    """Tool configuration schema."""
    
    def __init__(
        self,
        enabled: bool = True,
        timeout_seconds: Optional[int] = None,
        max_retries: Optional[int] = None,
        **tool_specific_config
    ):
        self.enabled = enabled
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.config = tool_specific_config
        super().__init__()


class UnifiedConfigManager:
    """
    Unified configuration manager with simplified precedence and dot notation support.
    
    Precedence order (highest to lowest):
    1. Override dictionary (programmatic overrides)
    2. Environment variables
    3. YAML configuration file
    4. Default values
    """
    
    def __init__(
        self,
        yaml_path: Optional[str] = None,
        override_config: Optional[Dict[str, Any]] = None,
        environment: Optional[str] = None
    ):
        self.yaml_path = yaml_path
        self.override_config = override_config or {}
        self.environment = environment or os.getenv("ENVIRONMENT", "default")
        
        self._yaml_config: Optional[Dict[str, Any]] = None
        self._config_cache: Dict[str, ConfigValue] = {}
        
        # Secret resolution function
        self._secret_resolver = self._get_secret_resolver()
    
    @retry("load_yaml_config", "io_operation")
    def load_yaml_config(self, force_reload: bool = False) -> Dict[str, Any]:
        """Load YAML configuration with caching."""
        if self._yaml_config is not None and not force_reload:
            return self._yaml_config
        
        yaml_path = self._resolve_yaml_path()
        if not yaml_path:
            logger.warning("No YAML configuration file found, using defaults")
            self._yaml_config = {}
            return self._yaml_config
        
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
                self._yaml_config = config
                logger.info(f"Loaded YAML config from {yaml_path}")
                return config
        
        except Exception as e:
            logger.error(f"Failed to load YAML config from {yaml_path}: {e}")
            self._yaml_config = {}
            return {}
    
    def _resolve_yaml_path(self) -> Optional[str]:
        """Resolve YAML configuration file path."""
        if self.yaml_path and Path(self.yaml_path).exists():
            return self.yaml_path
        
        # Search common locations
        search_paths = [
            Path.cwd() / "agent_config.yaml",
            Path.cwd() / "deep_research_agent" / "agent_config.yaml",
            Path(__file__).parent.parent / "agent_config.yaml",
            Path.cwd() / "config" / "agent_config.yaml"
        ]
        
        for path in search_paths:
            if path.exists():
                return str(path)
        
        return None
    
    def get(self, key_path: str, default: Any = None, value_type: type = None) -> Any:
        """
        Get configuration value using dot notation with precedence rules.
        
        Args:
            key_path: Dot-separated key path (e.g., "models.default.temperature")
            default: Default value if not found
            value_type: Optional type to cast the value to
            
        Returns:
            Configuration value
        """
        # Check cache first
        cache_key = f"{key_path}:{self.environment}"
        if cache_key in self._config_cache:
            cached_value = self._config_cache[cache_key]
            return self._cast_value(cached_value.value, value_type)
        
        # Resolve value using precedence
        config_value = self._resolve_config_value(key_path, default)
        
        # Cache the result
        self._config_cache[cache_key] = config_value
        
        # Cast and return
        final_value = self._cast_value(config_value.value, value_type)
        return final_value
    
    def _resolve_config_value(self, key_path: str, default: Any) -> ConfigValue:
        """Resolve configuration value using precedence rules."""
        
        # 1. Check override config
        override_value = self._get_nested_value(self.override_config, key_path)
        if override_value is not None:
            resolved_value = self._resolve_secret(override_value)
            return ConfigValue(
                value=resolved_value,
                source=ConfigSource.OVERRIDE,
                key_path=key_path
            )
        
        # 2. Check environment variables
        env_var = self._key_path_to_env_var(key_path)
        env_value = os.getenv(env_var)
        if env_value is not None:
            resolved_value = self._resolve_secret(env_value)
            return ConfigValue(
                value=resolved_value,
                source=ConfigSource.ENVIRONMENT,
                key_path=key_path,
                metadata={"env_var": env_var}
            )
        
        # 3. Check YAML config (with environment overrides)
        yaml_config = self.load_yaml_config()
        
        # Check environment-specific override first
        if self.environment != "default":
            env_override_path = f"environments.{self.environment}.{key_path}"
            env_value = self._get_nested_value(yaml_config, env_override_path)
            if env_value is not None:
                resolved_value = self._resolve_secret(env_value)
                return ConfigValue(
                    value=resolved_value,
                    source=ConfigSource.YAML,
                    key_path=key_path,
                    metadata={"environment": self.environment}
                )
        
        # Check base YAML config
        yaml_value = self._get_nested_value(yaml_config, key_path)
        if yaml_value is not None:
            resolved_value = self._resolve_secret(yaml_value)
            return ConfigValue(
                value=resolved_value,
                source=ConfigSource.YAML,
                key_path=key_path
            )
        
        # 4. Return default
        resolved_value = self._resolve_secret(default)
        return ConfigValue(
            value=resolved_value,
            source=ConfigSource.DEFAULT,
            key_path=key_path
        )
    
    def _get_nested_value(self, config: Dict[str, Any], key_path: str) -> Any:
        """Get nested value from dictionary using dot notation."""
        if not config or not key_path:
            return None
        
        keys = key_path.split('.')
        value = config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        
        return value
    
    def _key_path_to_env_var(self, key_path: str) -> str:
        """Convert dot notation key path to environment variable name."""
        return key_path.upper().replace('.', '_')
    
    def _cast_value(self, value: Any, value_type: Optional[type]) -> Any:
        """Cast value to specified type with error handling."""
        if value_type is None or value is None:
            return value
        
        try:
            if value_type is bool:
                if isinstance(value, bool):
                    return value
                if isinstance(value, str):
                    return value.lower() in ("true", "1", "yes", "on", "enabled")
                return bool(value)
            
            return value_type(value)
        
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to cast {value} to {value_type}: {e}")
            return value
    
    def _resolve_secret(self, value: Any) -> Any:
        """Resolve MLflow secret references."""
        if not isinstance(value, str):
            return value
        
        if value.startswith("{{secrets/") and value.endswith("}}"):
            try:
                return self._secret_resolver(value)
            except Exception as e:
                logger.warning(f"Failed to resolve secret {value}: {e}")
                return value
        
        return value
    
    def _get_secret_resolver(self):
        """Get secret resolver function."""
        try:
            from deep_research_agent.core.utils import resolve_secret
            return resolve_secret
        except ImportError:
            logger.warning("Secret resolver not available, using raw values")
            return lambda x: x
    
    def get_agent_config(self) -> AgentConfigSchema:
        """Get validated agent configuration."""
        return AgentConfigSchema(
            llm_endpoint=self.get("models.default.endpoint", "databricks-claude-3-7-sonnet", str),
            max_research_loops=self.get("research.max_research_loops", 2, int),
            initial_query_count=self.get("research.initial_query_count", 3, int),
            max_concurrent_searches=self.get("rate_limiting.max_concurrent_searches", 2, int),
            batch_delay_seconds=self.get("rate_limiting.batch_delay_seconds", 1.0, float),
            temperature=self.get("models.default.temperature", 0.7, float),
            max_tokens=self.get("models.default.max_tokens", 4000, int),
            timeout_seconds=self.get("research.timeout_seconds", 30, int),
            max_retries=self.get("research.max_retries", 3, int),
            enable_streaming=self.get("research.enable_streaming", True, bool),
            enable_citations=self.get("research.enable_citations", True, bool),
            search_provider=self.get("research.search_provider", "tavily", str)
        )
    
    def get_tool_config(self, tool_name: str) -> ToolConfigSchema:
        """Get configuration for a specific tool."""
        base_path = f"tools.{tool_name}"
        
        return ToolConfigSchema(
            enabled=self.get(f"{base_path}.enabled", True, bool),
            timeout_seconds=self.get(f"{base_path}.timeout_seconds", None, int),
            max_retries=self.get(f"{base_path}.max_retries", None, int),
            **self._get_tool_specific_config(tool_name, base_path)
        )
    
    def _get_tool_specific_config(self, tool_name: str, base_path: str) -> Dict[str, Any]:
        """Get tool-specific configuration."""
        config = {}
        
        if tool_name == "tavily_search":
            config.update({
                "api_key": self.get(f"{base_path}.api_key", "{{secrets/msh/TAVILY_API_KEY}}", str),
                "base_url": self.get(f"{base_path}.base_url", "https://api.tavily.com", str),
                "search_depth": self.get(f"{base_path}.search_depth", "basic", str),
                "max_results": self.get(f"{base_path}.max_results", 5, int)
            })
        
        elif tool_name == "brave_search":
            config.update({
                "api_key": self.get(f"{base_path}.api_key", "{{secrets/msh/BRAVE_API_KEY}}", str),
                "base_url": self.get(f"{base_path}.base_url", "https://api.search.brave.com/res/v1", str),
                "max_results": self.get(f"{base_path}.max_results", 5, int)
            })
        
        elif tool_name == "vector_search":
            config.update({
                "index_name": self.get(f"{base_path}.index_name", None, str),
                "text_column": self.get(f"{base_path}.text_column", "content", str),
                "columns": self.get(f"{base_path}.columns", ["source", "title", "url"], list),
                "k": self.get(f"{base_path}.k", 5, int)
            })
        
        return config
    
    def get_model_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get model configurations for different research phases."""
        phases = ["default", "query_generation", "web_research", "reflection", "synthesis", "embedding"]
        configs = {}
        
        for phase in phases:
            phase_config = {
                "endpoint": self.get(f"models.{phase}.endpoint", "databricks-claude-3-7-sonnet", str),
                "temperature": self.get(f"models.{phase}.temperature", 0.7, float),
                "max_tokens": self.get(f"models.{phase}.max_tokens", 4000, int)
            }
            
            # Remove generation parameters for embedding models
            if phase == "embedding":
                phase_config = {
                    "endpoint": phase_config["endpoint"]
                }
            
            configs[phase] = phase_config
        
        return configs
    
    @safe_call("validate_config", fallback=[])
    def validate(self) -> List[str]:
        """Validate entire configuration and return list of errors."""
        errors = []
        
        # Validate agent config
        agent_config = self.get_agent_config()
        errors.extend(agent_config.validate())
        
        # Validate required secrets for enabled tools
        if self.get("tools.tavily_search.enabled", True, bool):
            tavily_key = self.get("tools.tavily_search.api_key", None, str)
            if not tavily_key or tavily_key.startswith("{{secrets/"):
                errors.append("Tavily search is enabled but API key is not configured")
        
        if self.get("tools.brave_search.enabled", True, bool):
            brave_key = self.get("tools.brave_search.api_key", None, str)
            if not brave_key or brave_key.startswith("{{secrets/"):
                errors.append("Brave search is enabled but API key is not configured")
        
        return errors
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary with source information."""
        summary = {
            "yaml_path": self.yaml_path,
            "environment": self.environment,
            "config_sources": {},
            "validation_errors": self.validate()
        }
        
        for cache_key, config_value in self._config_cache.items():
            key_path = config_value.key_path
            summary["config_sources"][key_path] = {
                "source": config_value.source.value,
                "metadata": config_value.metadata
            }
        
        return summary
    
    def clear_cache(self):
        """Clear configuration cache."""
        self._config_cache.clear()
        self._yaml_config = None


# Global configuration manager instance
_global_config_manager: Optional[UnifiedConfigManager] = None


def get_config_manager(
    yaml_path: Optional[str] = None,
    override_config: Optional[Dict[str, Any]] = None,
    environment: Optional[str] = None
) -> UnifiedConfigManager:
    """Get or create global configuration manager."""
    global _global_config_manager
    
    if _global_config_manager is None or any([yaml_path, override_config, environment]):
        _global_config_manager = UnifiedConfigManager(
            yaml_path=yaml_path,
            override_config=override_config,
            environment=environment
        )
    
    return _global_config_manager


# Convenience functions
def get_agent_config() -> AgentConfigSchema:
    """Get agent configuration using global manager."""
    return get_config_manager().get_agent_config()


def get_tool_config(tool_name: str) -> ToolConfigSchema:
    """Get tool configuration using global manager."""
    return get_config_manager().get_tool_config(tool_name)


def get_config_value(key_path: str, default: Any = None, value_type: type = None) -> Any:
    """Get configuration value using global manager."""
    return get_config_manager().get(key_path, default, value_type)


def validate_configuration() -> List[str]:
    """Validate configuration using global manager."""
    return get_config_manager().validate()