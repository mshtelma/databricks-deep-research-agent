"""YAML configuration loader with environment variable interpolation."""

import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Pattern matches ${VAR} and ${VAR:-default}
ENV_VAR_PATTERN = re.compile(r"\$\{([^}:]+)(?::-([^}]*))?\}")


def interpolate_env_vars(value: Any) -> Any:
    """Recursively interpolate environment variables in YAML values.

    Supports:
    - ${VAR} - Required variable, raises ValueError if not set
    - ${VAR:-default} - Optional variable with default value

    Args:
        value: Any value from parsed YAML (str, dict, list, or primitive)

    Returns:
        The value with environment variables interpolated

    Raises:
        ValueError: If a required environment variable is not set
    """
    if isinstance(value, str):

        def replace_var(match: re.Match[str]) -> str:
            var_name = match.group(1)
            default = match.group(2)

            env_value = os.environ.get(var_name)
            if env_value is not None:
                return env_value
            if default is not None:
                return default
            raise ValueError(
                f"Environment variable '{var_name}' is not set and no default provided"
            )

        return ENV_VAR_PATTERN.sub(replace_var, value)

    elif isinstance(value, dict):
        return {k: interpolate_env_vars(v) for k, v in value.items()}

    elif isinstance(value, list):
        return [interpolate_env_vars(item) for item in value]

    return value


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    """Load YAML configuration with environment variable interpolation.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Parsed and interpolated configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If environment variable interpolation fails
        yaml.YAMLError: If YAML parsing fails
    """
    logger.debug(f"Loading configuration from {config_path}")

    with open(config_path) as f:
        raw_config = yaml.safe_load(f)

    if raw_config is None:
        logger.warning(f"Configuration file {config_path} is empty")
        return {}

    try:
        interpolated = interpolate_env_vars(raw_config)
        logger.info(f"Successfully loaded configuration from {config_path}")
        if not isinstance(interpolated, dict):
            raise ValueError(f"Expected dict from YAML config, got {type(interpolated).__name__}")
        return interpolated
    except ValueError as e:
        logger.error(f"Environment variable interpolation failed: {e}")
        raise
