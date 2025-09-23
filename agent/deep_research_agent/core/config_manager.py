"""
Compatibility wrapper for ConfigManager to support legacy tests.

This module provides a ConfigManager class that wraps the unified_config
functionality for backward compatibility with existing tests.
"""

from typing import Any, Dict, Optional, Union
from pathlib import Path
import logging

from deep_research_agent.core.unified_config import UnifiedConfigManager

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Legacy ConfigManager interface for backward compatibility.
    
    This wraps UnifiedConfigManager to provide the expected interface
    for existing tests while using the new unified configuration system.
    """
    
    def __init__(self, config: Optional[Union[Dict, str, Path]] = None):
        """
        Initialize ConfigManager with configuration.
        
        Args:
            config: Configuration dict, path to YAML file, or None for defaults
        """
        if isinstance(config, dict):
            # Direct configuration dictionary
            self._manager = UnifiedConfigManager(override_config=config)
        elif isinstance(config, (str, Path)):
            # Path to configuration file
            self._manager = UnifiedConfigManager(yaml_path=str(config))
        else:
            # Use defaults
            self._manager = UnifiedConfigManager()
        
        # Create config dict for direct access compatibility
        self._config = self._build_config_dict()
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        return self._manager.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        # Note: UnifiedConfigManager may not support set - just update our dict
        if '.' in key:
            keys = key.split('.')
            current = self._config
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value
        else:
            self._config[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Get full configuration as dictionary."""
        return self._config.copy()
    
    @property
    def config(self) -> Dict[str, Any]:
        """Direct access to configuration dictionary."""
        return self._config
    
    def get_model_config(self, model_type: str = "default") -> Dict[str, Any]:
        """
        Get model configuration for specific type.
        
        Args:
            model_type: Type of model configuration to get
            
        Returns:
            Model configuration dictionary
        """
        models = self.get("models", {})
        return models.get(model_type, models.get("default", {}))
    
    def get_search_config(self) -> Dict[str, Any]:
        """Get search configuration."""
        return self.get("tools.search", {})
    
    def get_rate_limits(self) -> Dict[str, Any]:
        """Get rate limiting configuration."""
        return self.get("rate_limiting", {})
    
    def validate(self) -> bool:
        """
        Validate configuration.
        
        Returns:
            True if configuration is valid
        """
        errors = self._manager.validate()
        return len(errors) == 0
    
    def _build_config_dict(self) -> Dict[str, Any]:
        """Build a configuration dictionary for compatibility."""
        # Build a basic config dict using common keys
        config = {}
        
        # Common configuration keys that tests might access
        common_keys = [
            "llm", "search", "research", "agents", "workflow", "events", 
            "timeouts", "test_settings", "models", "multi_agent", "planning",
            "background_investigation", "grounding", "report", "citations",
            "reflexion", "streaming", "tools", "rate_limiting", "quality_metrics"
        ]
        
        for key in common_keys:
            try:
                value = self._manager.get(key, None)
                if value is not None:
                    config[key] = value
            except:
                # Skip if key doesn't exist
                pass
        
        return config