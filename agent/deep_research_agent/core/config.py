"""
Configuration management for the research agent.

This module provides centralized configuration loading, validation,
and management with support for environment variables and defaults.
"""

import os
import yaml
import re
from typing import Optional, Dict, Any, List
from pathlib import Path

from .types import AgentConfiguration, ToolConfiguration, ToolType
from .exceptions import ConfigurationError
from .logging import get_logger

logger = get_logger(__name__)


class ConfigManager:
    """Manages configuration for the research agent."""
    
    def __init__(self, config_override: Optional[Dict[str, Any]] = None, yaml_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_override: Optional configuration override values
            yaml_path: Path to YAML configuration file (optional)
        """
        self._config_override = config_override or {}
        self._yaml_path = yaml_path
        self._yaml_config: Optional[Dict[str, Any]] = None
        self._agent_config: Optional[AgentConfiguration] = None
        self._tool_configs: Dict[ToolType, ToolConfiguration] = {}
        
        # Log initialization
        print(f"[ConfigManager] Initializing with yaml_path: {yaml_path}")
        logger.info(f"ConfigManager initialized with yaml_path: {yaml_path}, override keys: {list(config_override.keys()) if config_override else []}")
        
    def get_agent_config(self) -> AgentConfiguration:
        """Get validated agent configuration."""
        if self._agent_config is None:
            self._agent_config = self._load_agent_config()
        return self._agent_config
    
    def get_tool_config(self, tool_type: ToolType) -> ToolConfiguration:
        """Get configuration for a specific tool."""
        if tool_type not in self._tool_configs:
            self._tool_configs[tool_type] = self._load_tool_config(tool_type)
        return self._tool_configs[tool_type]
    
    def _load_agent_config(self) -> AgentConfiguration:
        """Load and validate agent configuration."""
        config_dict = {
            "llm_endpoint": self._get_config_value(
                "llm_endpoint", 
                "LLM_ENDPOINT", 
                "databricks-claude-3-7-sonnet",
                "models.default.endpoint"
            ),
            "max_research_loops": int(self._get_config_value(
                "max_research_loops", 
                "MAX_RESEARCH_LOOPS", 
                2,
                "research.max_research_loops"
            )),
            "initial_query_count": int(self._get_config_value(
                "initial_query_count", 
                "INITIAL_QUERY_COUNT", 
                3,
                "research.initial_query_count"
            )),
            "max_concurrent_searches": int(self._get_config_value(
                "max_concurrent_searches", 
                "MAX_CONCURRENT_SEARCHES", 
                2,
                "rate_limiting.max_concurrent_searches"
            )),
            "batch_delay_seconds": float(self._get_config_value(
                "batch_delay_seconds", 
                "BATCH_DELAY_SECONDS", 
                1.0,
                "rate_limiting.batch_delay_seconds"
            )),
            "temperature": float(self._get_config_value(
                "temperature", 
                "TEMPERATURE", 
                0.7,
                "models.default.temperature"
            )),
            "max_tokens": int(self._get_config_value(
                "max_tokens", 
                "MAX_TOKENS", 
                4000,
                "models.default.max_tokens"
            )),
            "tavily_api_key": self._get_config_value(
                "tavily_api_key", 
                "TAVILY_API_KEY", 
                None,
                "tools.tavily_search.api_key"
            ),
            "vector_search_index": self._get_config_value(
                "vector_search_index", 
                "VECTOR_SEARCH_INDEX", 
                None,
                "tools.vector_search.index_name"
            ),
            "timeout_seconds": int(self._get_config_value(
                "timeout_seconds", 
                "TIMEOUT_SECONDS", 
                30,
                "research.timeout_seconds"
            )),
            "max_retries": int(self._get_config_value(
                "max_retries", 
                "MAX_RETRIES", 
                3,
                "research.max_retries"
            )),
            "enable_streaming": self._get_bool_config_value(
                "enable_streaming", 
                "ENABLE_STREAMING", 
                True,
                "research.enable_streaming"
            ),
            "enable_citations": self._get_bool_config_value(
                "enable_citations", 
                "ENABLE_CITATIONS", 
                True,
                "research.enable_citations"
            ),
            # Intermediate events configuration
            "emit_intermediate_events": self._get_bool_config_value(
                "emit_intermediate_events",
                "EMIT_INTERMEDIATE_EVENTS",
                True,
                "intermediate_events.emit_intermediate_events"
            ),
            "reasoning_visibility": self._get_config_value(
                "reasoning_visibility",
                "REASONING_VISIBILITY", 
                "summarized",
                "intermediate_events.reasoning_visibility"
            ),
            "thought_snapshot_interval_tokens": int(self._get_config_value(
                "thought_snapshot_interval_tokens",
                "THOUGHT_SNAPSHOT_INTERVAL_TOKENS",
                40,
                "intermediate_events.thought_snapshot_interval_tokens"  
            )),
            "thought_snapshot_interval_ms": int(self._get_config_value(
                "thought_snapshot_interval_ms",
                "THOUGHT_SNAPSHOT_INTERVAL_MS",
                800,
                "intermediate_events.thought_snapshot_interval_ms"
            )),
            "max_thought_chars_per_step": int(self._get_config_value(
                "max_thought_chars_per_step",
                "MAX_THOUGHT_CHARS_PER_STEP", 
                1000,
                "intermediate_events.max_thought_chars_per_step"
            )),
        }
        
        try:
            config = AgentConfiguration(**config_dict)
            config.validate()
            return config
        except (ValueError, TypeError) as e:
            raise ConfigurationError(f"Invalid agent configuration: {e}")
    
    def _load_tool_config(self, tool_type: ToolType) -> ToolConfiguration:
        """Load configuration for a specific tool."""
        tool_prefix = tool_type.value.upper()
        
        config_dict = {
            "tool_type": tool_type,
            "enabled": self._get_bool_config_value(
                f"{tool_type.value}_enabled",
                f"{tool_prefix}_ENABLED",
                True,
                f"tools.{tool_type.value}.enabled"
            ),
            "timeout_seconds": self._get_optional_int_config_value(
                f"{tool_type.value}_timeout_seconds",
                f"{tool_prefix}_TIMEOUT_SECONDS"
            ),
            "max_retries": self._get_optional_int_config_value(
                f"{tool_type.value}_max_retries",
                f"{tool_prefix}_MAX_RETRIES"
            ),
            "config": self._get_tool_specific_config(tool_type)
        }
        
        return ToolConfiguration(**config_dict)
    
    def _get_tool_specific_config(self, tool_type: ToolType) -> Dict[str, Any]:
        """Get tool-specific configuration."""
        if tool_type == ToolType.TAVILY_SEARCH:
            return {
                "api_key": self._get_config_value(
                    "tavily_api_key", 
                    "TAVILY_API_KEY", 
                    None,
                    "tools.tavily_search.api_key"
                ),
                "base_url": self._get_config_value(
                    "tavily_base_url", 
                    "TAVILY_BASE_URL", 
                    "https://api.tavily.com",
                    "tools.tavily_search.base_url"
                ),
                "search_depth": self._get_config_value(
                    "tavily_search_depth", 
                    "TAVILY_SEARCH_DEPTH", 
                    "basic",
                    "tools.tavily_search.search_depth"
                ),
                "max_results": int(self._get_config_value(
                    "tavily_max_results", 
                    "TAVILY_MAX_RESULTS", 
                    5,
                    "tools.tavily_search.max_results"
                ))
            }
        elif tool_type == ToolType.BRAVE_SEARCH:
            return {
                "api_key": self._get_config_value(
                    "brave_api_key", 
                    "BRAVE_API_KEY", 
                    None,
                    "tools.brave_search.api_key"
                ),
                "base_url": self._get_config_value(
                    "brave_base_url", 
                    "BRAVE_BASE_URL", 
                    "https://api.search.brave.com/res/v1",
                    "tools.brave_search.base_url"
                ),
                "max_results": int(self._get_config_value(
                    "brave_max_results", 
                    "BRAVE_MAX_RESULTS", 
                    5,
                    "tools.brave_search.max_results"
                ))
            }
        elif tool_type == ToolType.VECTOR_SEARCH:
            return {
                "index_name": self._get_config_value(
                    "vector_search_index", 
                    "VECTOR_SEARCH_INDEX", 
                    None
                ),
                "text_column": self._get_config_value(
                    "vector_search_text_column", 
                    "VECTOR_SEARCH_TEXT_COLUMN", 
                    "content"
                ),
                "columns": self._get_config_value(
                    "vector_search_columns", 
                    "VECTOR_SEARCH_COLUMNS", 
                    ["source", "title", "url"]
                ),
                "k": int(self._get_config_value(
                    "vector_search_k", 
                    "VECTOR_SEARCH_K", 
                    5
                ))
            }
        elif tool_type == ToolType.PYTHON_EXEC:
            return {
                "function_names": self._get_config_value(
                    "python_exec_functions", 
                    "PYTHON_EXEC_FUNCTIONS", 
                    ["system.ai.python_exec"]
                )
            }
        else:
            return {}
    
    def _resolve_secret(self, value: Any) -> Any:
        """
        Resolve MLflow secret syntax to actual values using the centralized SecretResolver.
        
        Args:
            value: Value to resolve (may be a secret reference or regular value)
            
        Returns:
            Resolved value or original if not a secret reference
        """
        from .utils import resolve_secret
        return resolve_secret(value)
    
    def _get_config_value(self, key: str, env_var: str, default: Any, yaml_key_path: Optional[str] = None) -> Any:
        """Get configuration value from override, YAML, environment, or default."""
        # Check override first
        if key in self._config_override:
            value = self._config_override[key]
            return self._resolve_secret(value)
        
        # Check YAML configuration
        if yaml_key_path:
            environment = os.getenv("ENVIRONMENT")
            yaml_value = self.get_yaml_value(yaml_key_path, environment)
            if yaml_value is not None:
                return self._resolve_secret(yaml_value)
        
        # Check environment variable
        env_value = os.getenv(env_var)
        if env_value is not None:
            return self._resolve_secret(env_value)
        
        # Return default (also resolve in case it contains secrets)
        return self._resolve_secret(default)
    
    def _get_bool_config_value(self, key: str, env_var: str, default: bool, yaml_key_path: Optional[str] = None) -> bool:
        """Get boolean configuration value."""
        value = self._get_config_value(key, env_var, default, yaml_key_path)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes", "on")
        return bool(value)
    
    def _get_optional_int_config_value(self, key: str, env_var: str) -> Optional[int]:
        """Get optional integer configuration value."""
        value = self._get_config_value(key, env_var, None)
        if value is None:
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            return None
    
    def get_secrets_config(self) -> Dict[str, str]:
        """Get configuration for secrets and sensitive values."""
        return {
            "tavily_api_key": self._get_config_value(
                "tavily_api_key", 
                "TAVILY_API_KEY", 
                "{{secrets/msh/TAVILY_API_KEY}}"
            ),
            "brave_api_key": self._get_config_value(
                "brave_api_key", 
                "BRAVE_API_KEY", 
                "{{secrets/msh/BRAVE_API_KEY}}"
            ),
            "langsmith_api_key": self._get_config_value(
                "langsmith_api_key", 
                "LANGSMITH_API_KEY", 
                "{{secrets/msh/LANGSMITH_API_KEY}}"
            )
        }
    
    def validate_required_secrets(self) -> List[str]:
        """Validate that required secrets are available."""
        missing_secrets = []
        
        # Check Tavily API key if enabled
        tavily_config = self.get_tool_config(ToolType.TAVILY_SEARCH)
        if tavily_config.enabled:
            tavily_key = tavily_config.config.get("api_key")
            if not tavily_key or tavily_key.startswith("{{secrets/"):
                missing_secrets.append("TAVILY_API_KEY")
        
        # Check Brave API key if enabled
        brave_config = self.get_tool_config(ToolType.BRAVE_SEARCH)
        if brave_config.enabled:
            brave_key = brave_config.config.get("api_key")
            if not brave_key or brave_key.startswith("{{secrets/"):
                missing_secrets.append("BRAVE_API_KEY")
        
        return missing_secrets
    
    def get_databricks_config(self) -> Dict[str, Any]:
        """Get Databricks-specific configuration."""
        return {
            "workspace_url": self._get_config_value(
                "databricks_workspace_url", 
                "DATABRICKS_WORKSPACE_URL", 
                None,
                "databricks.workspace_url"
            ),
            "token": self._get_config_value(
                "databricks_token",
                "DATABRICKS_TOKEN",
                None,
                "databricks.token"
            ),
            "workspace_profile": self._get_config_value(
                "databricks_profile", 
                "DATABRICKS_PROFILE", 
                None,
                "databricks.workspace_profile"
            ),
            "llm_endpoint": self.get_agent_config().llm_endpoint,
        }
    
    def get_search_provider(self) -> str:
        """Get the configured search provider (tavily or brave)."""
        provider = self._get_config_value(
            "search_provider",
            "SEARCH_PROVIDER",
            "tavily",
            "research.search_provider"
        )
        print(f"[ConfigManager.get_search_provider] Final search provider: {provider}")
        logger.info(f"Search provider resolved to: {provider}")
        return provider
    
    def get_model_config(self) -> Dict[str, Dict[str, Any]]:
        """Get model configuration for different research phases."""
        yaml_config = self.load_yaml_config()
        models_config = yaml_config.get("models", {})
        
        # Default model configurations
        default_config = {
            "endpoint": "databricks-claude-3-7-sonnet",
            "temperature": 0.7,
            "max_tokens": 4000
        }
        
        # Model defaults for different phases  
        phase_defaults = {
            "default": default_config,
            "query_generation": {
                "endpoint": "databricks-claude-3-7-sonnet",
                "temperature": 0.5,
                "max_tokens": 2000
            },
            "web_research": default_config.copy(),
            "reflection": {
                "endpoint": "databricks-claude-3-7-sonnet", 
                "temperature": 0.8,
                "max_tokens": 3000
            },
            "synthesis": {
                "endpoint": "databricks-claude-3-7-sonnet",
                "temperature": 0.7,
                "max_tokens": 6000
            },
            "embedding": {
                "endpoint": "databricks-bge-large-en"
                # No generation parameters for embedding
            }
        }
        
        # Merge YAML config with defaults
        result = {}
        for phase, defaults in phase_defaults.items():
            phase_config = models_config.get(phase, {})
            
            if isinstance(phase_config, str):
                # Support old format where model was just a string
                result[phase] = {"endpoint": phase_config, **defaults}
            else:
                # New format: merge with defaults
                merged_config = defaults.copy()
                merged_config.update(phase_config)
                result[phase] = merged_config
        
        return result
    
    def _find_yaml_config_path(self) -> Optional[str]:
        """Find agent_config.yaml in various locations, including MLflow contexts."""
        current_dir = Path.cwd()
        
        # Get the directory where this config.py file is located
        config_module_dir = Path(__file__).parent
        deep_research_agent_dir = config_module_dir.parent  # Should be deep_research_agent/
        
        # Define possible locations for the YAML config
        possible_paths = [
            # Standard development locations
            current_dir / "agent_config.yaml",
            current_dir / "agent_authoring" / "agent_config.yaml",
            deep_research_agent_dir / "agent_config.yaml",
            
            # MLflow model serving contexts - the YAML may be in various locations
            # When MLflow unpacks models, it creates various directory structures
            current_dir / "deep_research_agent" / "agent_config.yaml",
            current_dir / "model" / "deep_research_agent" / "agent_config.yaml",
            current_dir / "model" / "agent_config.yaml",
            
            # MLflow artifact locations (when logged separately)
            current_dir / "config" / "agent_config.yaml",
            current_dir / "model" / "config" / "agent_config.yaml",
            
            # Parent directory checks (in case we're in a subdirectory)
            current_dir.parent / "agent_config.yaml",
            current_dir.parent / "deep_research_agent" / "agent_config.yaml",
            
            # Check relative to where the agent module is located
            config_module_dir / ".." / "agent_config.yaml",
            config_module_dir / ".." / ".." / "agent_config.yaml",
        ]
        
        print(f"[ConfigManager._find_yaml_config_path] Searching in: {len(possible_paths)} locations")
        for i, path in enumerate(possible_paths):
            resolved_path = path.resolve()
            print(f"[ConfigManager._find_yaml_config_path] Checking {i+1}: {resolved_path}")
            if resolved_path.exists():
                yaml_path = str(resolved_path)
                print(f"[ConfigManager._find_yaml_config_path] Found config at: {yaml_path}")
                logger.info(f"Found YAML config at: {yaml_path}")
                return yaml_path
        
        print(f"[ConfigManager._find_yaml_config_path] No YAML config found in any location")
        return None

    def load_yaml_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if self._yaml_config is not None:
            return self._yaml_config
            
        yaml_path = self._yaml_path
        print(f"[ConfigManager.load_yaml_config] Starting YAML load, yaml_path: {yaml_path}")
        
        if not yaml_path:
            # Try to find agent_config.yaml in various locations, including MLflow contexts
            yaml_path = self._find_yaml_config_path()
            
        # If we have a yaml_path but the file doesn't exist, try to find it
        if yaml_path and not Path(yaml_path).exists():
            print(f"[ConfigManager.load_yaml_config] Specified path {yaml_path} doesn't exist, searching for alternatives")
            yaml_path = self._find_yaml_config_path()
        
        if not yaml_path or not Path(yaml_path).exists():
            print(f"[ConfigManager.load_yaml_config] ERROR: No YAML config found!")
            logger.error(f"No YAML configuration file found at {yaml_path}")
            raise ConfigurationError(f"Required YAML configuration file not found at {yaml_path}. The agent cannot run without proper configuration.")
        
        try:
            print(f"[ConfigManager.load_yaml_config] Loading YAML from: {yaml_path}")
            with open(yaml_path, 'r') as f:
                self._yaml_config = yaml.safe_load(f) or {}
                print(f"[ConfigManager.load_yaml_config] Loaded config with keys: {list(self._yaml_config.keys())}")
                
                # Log search provider from YAML
                if 'research' in self._yaml_config:
                    search_provider = self._yaml_config.get('research', {}).get('search_provider')
                    print(f"[ConfigManager.load_yaml_config] Search provider from YAML: {search_provider}")
                    logger.info(f"Search provider from YAML config: {search_provider}")
                
                # Log tool configs from YAML
                if 'tools' in self._yaml_config:
                    tools = self._yaml_config.get('tools', {})
                    for tool_name, tool_config in tools.items():
                        enabled = tool_config.get('enabled', 'not specified')
                        print(f"[ConfigManager.load_yaml_config] Tool {tool_name}: enabled={enabled}")
                        logger.info(f"Tool {tool_name} config: enabled={enabled}")
                
                return self._yaml_config
        except Exception as e:
            print(f"[ConfigManager.load_yaml_config] ERROR loading YAML: {e}")
            logger.error(f"Failed to load YAML configuration from {yaml_path}: {e}")
            raise ConfigurationError(f"Failed to load YAML configuration from {yaml_path}: {e}")
    
    def get_yaml_value(self, key_path: str, environment: Optional[str] = None) -> Any:
        """
        Get value from YAML configuration using dot notation key path.
        
        Args:
            key_path: Dot-separated path to the configuration value (e.g., "llm.temperature")
            environment: Optional environment to check for overrides (e.g., "dev", "prod")
            
        Returns:
            Configuration value or None if not found
        """
        yaml_config = self.load_yaml_config()
        if not yaml_config:
            return None
        
        # Check environment-specific overrides first
        if environment and "environments" in yaml_config and environment in yaml_config["environments"]:
            env_config = yaml_config["environments"][environment]
            env_value = self._get_nested_value(env_config, key_path)
            if env_value is not None:
                return env_value
        
        # Fall back to base configuration
        return self._get_nested_value(yaml_config, key_path)
    
    def _get_nested_value(self, config: Dict[str, Any], key_path: str) -> Any:
        """Get nested value from dictionary using dot notation."""
        keys = key_path.split('.')
        value = config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        
        return value


def get_default_config() -> AgentConfiguration:
    """Get default agent configuration."""
    return ConfigManager().get_agent_config()


def get_config_from_dict(config_dict: Dict[str, Any]) -> AgentConfiguration:
    """Create configuration from dictionary."""
    return ConfigManager(config_dict).get_agent_config()


def get_config_from_yaml(yaml_path: str) -> AgentConfiguration:
    """Create configuration from YAML file."""
    return ConfigManager(yaml_path=yaml_path).get_agent_config()


def validate_configuration(config: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    Validate configuration and return list of issues.
    
    Args:
        config: Optional configuration dictionary to validate
        
    Returns:
        List of validation error messages
    """
    issues = []
    
    try:
        manager = ConfigManager(config)
        agent_config = manager.get_agent_config()
        
        # Check for missing required secrets
        missing_secrets = manager.validate_required_secrets()
        for secret in missing_secrets:
            issues.append(f"Missing required secret: {secret}")
        
    except ConfigurationError as e:
        issues.append(str(e))
    except Exception as e:
        issues.append(f"Configuration validation error: {e}")
    
    return issues