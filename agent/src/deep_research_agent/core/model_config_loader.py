"""
Model configuration loader that reads from base.yaml
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from .model_manager import ModelConfig, NodeModelConfiguration
from .logging import get_logger

logger = get_logger(__name__)


def load_model_config_from_yaml(config_dict: Dict[str, Any]) -> NodeModelConfiguration:
    """
    Load NodeModelConfiguration from parsed YAML config dictionary.

    Args:
        config_dict: Dictionary loaded from base.yaml

    Returns:
        NodeModelConfiguration with all model configurations
    """
    models_config = config_dict.get('models', {})

    # Create ModelConfig instances for each defined model
    model_configs = {}
    for model_name, model_dict in models_config.items():
        if isinstance(model_dict, dict):
            # Handle both 'endpoint' (single) and 'endpoints' (list) formats
            endpoint = model_dict.get('endpoint')
            if not endpoint:
                endpoints = model_dict.get('endpoints', [])
                # Use first endpoint if multiple are defined (priority-based)
                endpoint = endpoints[0] if endpoints else 'databricks-gpt-oss-20b'

                if endpoints and len(endpoints) > 1:
                    logger.info(
                        f"📌 {model_name}: Using primary endpoint '{endpoint}' "
                        f"(fallbacks: {endpoints[1:]})"
                    )

            # Extract all fields including reasoning-specific ones
            model_config = ModelConfig(
                endpoint=endpoint,
                temperature=model_dict.get('temperature', 0.7),
                max_tokens=model_dict.get('max_tokens', 4000),
                timeout_seconds=model_dict.get('timeout_seconds', 30),
                max_retries=model_dict.get('max_retries', 3),
                reasoning_effort=model_dict.get('reasoning_effort'),
                reasoning_budget=model_dict.get('reasoning_budget')
            )
            model_configs[model_name] = model_config

            # Log if reasoning config is found
            if model_config.reasoning_effort or model_config.reasoning_budget:
                logger.info(
                    f"Loaded reasoning config for {model_name}: "
                    f"effort={model_config.reasoning_effort}, "
                    f"budget={model_config.reasoning_budget}"
                )

    # Create NodeModelConfiguration with the loaded configs
    # Map complexity-based models to workflow roles
    node_config = NodeModelConfiguration(
        default_model=model_configs.get('default', ModelConfig(endpoint='databricks-gpt-oss-20b')),
        # Use simple model for query generation if defined, otherwise use dedicated or web_research
        query_generation_model=model_configs.get('query_generation') or model_configs.get('simple') or model_configs.get('web_research'),
        # Use analytical model for web research if defined, otherwise use dedicated
        web_research_model=model_configs.get('web_research') or model_configs.get('analytical'),
        # Use analytical model for reflection if defined
        reflection_model=model_configs.get('reflection') or model_configs.get('analytical'),
        # Use complex model for synthesis if defined, otherwise use dedicated
        synthesis_model=model_configs.get('synthesis') or model_configs.get('complex'),
        embedding_model=model_configs.get('embedding', ModelConfig(endpoint='databricks-bge-large-en'))
    )

    # Store all model configs for dynamic access by new roles
    node_config._all_models = model_configs

    return node_config


def create_model_manager_from_config(
    config_path: Optional[Path] = None,
    model_selector: Optional['ModelSelector'] = None
):
    """
    Create a ModelManager instance with configuration from base.yaml.

    Args:
        config_path: Path to configuration file (defaults to conf/base.yaml)
        model_selector: Optional ModelSelector for rate limiting

    Returns:
        ModelManager instance configured from YAML (with or without rate limiting)
    """
    from .model_manager import ModelManager

    if config_path is None:
        # Try to find base.yaml
        possible_paths = [
            Path(__file__).parent.parent.parent / "conf" / "base.yaml",
            Path.cwd() / "conf" / "base.yaml",
        ]

        for path in possible_paths:
            if path.exists():
                config_path = path
                logger.info(f"Found config at: {config_path}")
                break
        else:
            logger.warning("No base.yaml found, using defaults")
            return ModelManager(model_selector=model_selector)

    # Load YAML config
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Convert to NodeModelConfiguration
        node_config = load_model_config_from_yaml(config_dict)

        # Create ModelManager WITH model_selector for rate limiting
        logger.info(
            f"Creating ModelManager | "
            f"Rate limiting: {'ENABLED' if model_selector else 'DISABLED'}"
        )

        return ModelManager(
            config=node_config,
            model_selector=model_selector  # PASS IT HERE
        )

    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        logger.info("Falling back to default configuration")
        return ModelManager(model_selector=model_selector)