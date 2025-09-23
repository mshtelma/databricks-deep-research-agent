"""
Unified configuration loader with override support.

This module provides a clean interface for loading both agent and deployment
configurations with proper override and environment variable support.
"""

from typing import Optional, Dict, Any, Union, List
from pathlib import Path
import yaml
import os
import logging

from deep_research_agent.config_schema import ResearchConfig
from deep_research_agent.deploy_config_schema import DeploymentConfig, GlobalDeploymentConfig

logger = logging.getLogger(__name__)

class ConfigLoader:
    """Unified configuration loader with override support"""
    
    @staticmethod
    def load_agent(
        config_path: Optional[Path] = None,
        override_path: Optional[Path] = None,
        env: Optional[str] = None
    ) -> ResearchConfig:
        """
        Load agent configuration with override support
        
        Args:
            config_path: Base configuration file (default: conf/base.yaml)
            override_path: Override configuration file (e.g., conf/test.yaml)
            env: Environment name (auto-detects test mode)
            
        Returns:
            Validated ResearchConfig instance
        """
        # Determine environment from explicit argument or environment variable
        if env is None:
            env = os.getenv("ENVIRONMENT", "prod")
        
        logger.info(f"Loading agent configuration for environment: {env}")
        
        # Find base directory properly - relative to this module
        base_dir = Path(__file__).parent.parent  # Go up to agent/
        conf_dir = base_dir / "conf"
        
        # Load base config with proper path resolution
        base_path = config_path or conf_dir / "base.yaml"
        config_dict = {}
        
        if base_path.exists():
            logger.debug(f"Loading base config from: {base_path}")
            with open(base_path) as f:
                config_dict = yaml.safe_load(f) or {}
        else:
            logger.warning(f"Base config file not found: {base_path}")
        
        # Apply overrides based on environment with proper paths
        if env == "test" and not override_path:
            override_path = conf_dir / "test.yaml"
        elif env == "integration" and not override_path:
            override_path = conf_dir / "integration_test.yaml"
        elif override_path is None and env and env != "prod":
            override_path = conf_dir / f"{env}.yaml"
            
        if override_path and override_path.exists():
            logger.info(f"Applying config overrides from: {override_path}")
            with open(override_path) as f:
                overrides = yaml.safe_load(f) or {}
                config_dict = ConfigLoader._deep_merge(config_dict, overrides)
        elif override_path:
            logger.debug(f"Override file not found (skipping): {override_path}")
        
        # Substitute environment variables
        config_dict = ConfigLoader._substitute_env_vars(config_dict)
        
        # Create and validate config
        try:
            config = ResearchConfig(**config_dict)
            logger.info("Agent configuration loaded and validated successfully")
            return config
        except Exception as e:
            logger.error(f"Failed to validate agent configuration: {e}")
            raise
    
    @staticmethod
    def load_deployment(
        env: str = "dev",
        config_path: Optional[Path] = None
    ) -> DeploymentConfig:
        """
        Load deployment configuration for specific environment
        
        Args:
            env: Environment name (dev, staging, prod, etc.)
            config_path: Custom config path (default: auto-detect)
            
        Returns:
            Validated DeploymentConfig instance
        """
        logger.info(f"Loading deployment configuration for environment: {env}")
        
        # Find base directory properly - relative to this module
        base_dir = Path(__file__).parent.parent  # Go up to agent/
        deploy_dir = base_dir / "conf" / "deploy"
        
        # Try global config approach first
        global_config_path = deploy_dir / "base.yaml"
        if global_config_path.exists():
            return ConfigLoader._load_deployment_from_global(env, global_config_path)
        
        # Fall back to per-environment files
        env_path = config_path or deploy_dir / f"{env}.yaml"
        if not env_path.exists():
            raise FileNotFoundError(f"Deployment config not found: {env_path}")
            
        logger.debug(f"Loading deployment config from: {env_path}")
        with open(env_path) as f:
            config_dict = yaml.safe_load(f) or {}
            
        # Substitute environment variables
        config_dict = ConfigLoader._substitute_env_vars(config_dict)
        
        try:
            config = DeploymentConfig(**config_dict)
            logger.info("Deployment configuration loaded and validated successfully")
            return config
        except Exception as e:
            logger.error(f"Failed to validate deployment configuration: {e}")
            raise
    
    @staticmethod
    def _load_deployment_from_global(env: str, global_config_path: Path) -> DeploymentConfig:
        """Load deployment config from global config with environment selection"""
        logger.debug(f"Loading global deployment config from: {global_config_path}")
        
        with open(global_config_path) as f:
            global_dict = yaml.safe_load(f) or {}
            
        # Substitute environment variables
        global_dict = ConfigLoader._substitute_env_vars(global_dict)
        
        # Load global config and extract environment
        global_config = GlobalDeploymentConfig(**global_dict)
        return global_config.get_environment_config(env)
    
    @staticmethod
    def _deep_merge(base: Dict, override: Dict) -> Dict:
        """
        Deep merge override dictionary into base dictionary
        
        Args:
            base: Base configuration dictionary
            override: Override configuration dictionary
            
        Returns:
            Merged configuration dictionary
        """
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigLoader._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    @staticmethod
    def _substitute_env_vars(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Substitute environment variables in configuration
        
        Supports patterns like:
        - ${VAR_NAME}
        - ${VAR_NAME:-default_value}
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            Configuration with environment variables substituted
        """
        if isinstance(config_dict, dict):
            return {k: ConfigLoader._substitute_env_vars(v) for k, v in config_dict.items()}
        elif isinstance(config_dict, list):
            return [ConfigLoader._substitute_env_vars(item) for item in config_dict]
        elif isinstance(config_dict, str) and config_dict.startswith("${") and config_dict.endswith("}"):
            # Extract variable name and default value
            var_expr = config_dict[2:-1]  # Remove ${ and }
            if ":-" in var_expr:
                var_name, default_value = var_expr.split(":-", 1)
                return os.getenv(var_name, default_value)
            else:
                var_name = var_expr
                return os.getenv(var_name, config_dict)  # Return original if not found
        else:
            return config_dict
    
    @staticmethod
    def list_available_configs() -> Dict[str, List[str]]:
        """
        List all available configuration files
        
        Returns:
            Dictionary with agent and deployment config files
        """
        agent_configs = []
        deploy_configs = []
        
        conf_dir = Path("conf")
        if conf_dir.exists():
            agent_configs = [f.name for f in conf_dir.glob("*.yaml")]
            
            deploy_dir = conf_dir / "deploy"
            if deploy_dir.exists():
                deploy_configs = [f.name for f in deploy_dir.glob("*.yaml")]
        
        return {
            "agent": sorted(agent_configs),
            "deployment": sorted(deploy_configs)
        }


# Convenience functions for easy imports
def load_agent_config(env: Optional[str] = None) -> ResearchConfig:
    """Convenience function to load agent configuration"""
    return ConfigLoader.load_agent(env=env)

def load_deployment_config(env: str = "dev") -> DeploymentConfig:
    """Convenience function to load deployment configuration"""
    return ConfigLoader.load_deployment(env=env)
