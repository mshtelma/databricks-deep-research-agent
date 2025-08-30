"""Simple config loading utility for deployment scripts."""

import yaml
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: Path, environment: str) -> Dict[str, Any]:
    """Load configuration from YAML file for specified environment."""
    
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # Get environment-specific config
    env_config = config_data.get('environments', {}).get(environment, {})
    
    if not env_config:
        raise ValueError(f"Environment '{environment}' not found in config")
    
    # Build unified config with environment variables
    config = {
        'ENVIRONMENT': environment,
        'UC_MODEL_NAME': env_config['model']['catalog'] + '.' + env_config['model']['schema'] + '.' + env_config['model']['name'],
        'endpoint': env_config['endpoint'],
        'command_execution': env_config['command_execution'],
        'profile': env_config['profile'],
        'workspace_path': env_config['workspace_path']
    }
    
    return config