"""Helper utilities for connecting to Databricks workspace and getting OpenAI client."""

import os
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from databricks.sdk import WorkspaceClient
from openai import OpenAI


def load_deploy_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load deployment configuration from YAML file."""
    if config_path is None:
        # Look for deploy_config.yaml in parent directory or current directory
        possible_paths = [
            Path(__file__).parent.parent / "deploy_config.yaml",
            Path(__file__).parent / "deploy_config.yaml", 
            Path("deploy_config.yaml"),
            Path("../deploy_config.yaml")
        ]
        
        for path in possible_paths:
            if path.exists():
                config_path = str(path)
                break
        
        if config_path is None:
            raise FileNotFoundError("Could not find deploy_config.yaml")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_workspace_client(env: str = "dev", profile: Optional[str] = None) -> WorkspaceClient:
    """
    Get WorkspaceClient configured for the specified environment.
    
    Args:
        env: Environment name from deploy_config.yaml (default: "dev")
        profile: Override Databricks CLI profile name (optional)
        
    Returns:
        Configured WorkspaceClient
    """
    if profile is None:
        # Get profile from deploy config
        config = load_deploy_config()
        if env not in config.get("environments", {}):
            available_envs = list(config.get("environments", {}).keys())
            raise ValueError(f"Environment '{env}' not found. Available: {available_envs}")
        
        profile = config["environments"][env]["profile"]
    
    return WorkspaceClient(profile=profile)


def get_databricks_openai_client(workspace_client: WorkspaceClient) -> OpenAI:
    """
    Get OpenAI client configured for Databricks serving endpoints.
    
    Args:
        workspace_client: Databricks WorkspaceClient instance
        
    Returns:
        OpenAI client configured for Databricks endpoints
    """
    return workspace_client.serving_endpoints.get_open_ai_client()


def get_databricks_llm_endpoint(env: str = "dev") -> str:
    """
    Get the configured LLM endpoint for the environment.
    
    Args:
        env: Environment name from deploy_config.yaml
        
    Returns:
        LLM endpoint name (e.g., "databricks-claude-3-7-sonnet")
    """
    config = load_deploy_config()
    
    # Get from model_defaults or environment-specific config
    llm_endpoint = config.get("model_defaults", {}).get("llm_endpoint")
    
    env_config = config.get("environments", {}).get(env, {})
    if "model" in env_config and "llm_endpoint" in env_config["model"]:
        llm_endpoint = env_config["model"]["llm_endpoint"]
    
    if not llm_endpoint:
        # Default to Claude Sonnet if not specified
        llm_endpoint = "databricks-claude-3-7-sonnet"
    
    return llm_endpoint


def check_endpoint_availability(workspace_client: WorkspaceClient, endpoint_name: str) -> bool:
    """
    Check if a serving endpoint is available and ready.
    
    Args:
        workspace_client: Databricks WorkspaceClient
        endpoint_name: Name of the serving endpoint
        
    Returns:
        True if endpoint is available, False otherwise
    """
    try:
        endpoint = workspace_client.serving_endpoints.get(endpoint_name)
        return endpoint.state.ready == "READY"
    except Exception:
        return False


def list_available_endpoints(workspace_client: WorkspaceClient) -> list[str]:
    """
    List all available serving endpoints in the workspace.
    
    Args:
        workspace_client: Databricks WorkspaceClient
        
    Returns:
        List of endpoint names
    """
    try:
        endpoints = workspace_client.serving_endpoints.list()
        return [endpoint.name for endpoint in endpoints if endpoint.state.ready == "READY"]
    except Exception as e:
        print(f"Error listing endpoints: {e}")
        return []


def create_test_client_config(env: str = "dev") -> Dict[str, Any]:
    """
    Create configuration dict for testing with Databricks workspace.
    
    Args:
        env: Environment name
        
    Returns:
        Configuration dict for testing
    """
    config = load_deploy_config()
    env_config = config.get("environments", {}).get(env, {})
    
    return {
        "profile": env_config.get("profile"),
        "llm_endpoint": get_databricks_llm_endpoint(env),
        "workspace_path": env_config.get("workspace_path"),
        "model_defaults": config.get("model_defaults", {}),
        "environment": env
    }


# Example usage functions
def create_test_workspace_setup(env: str = "dev") -> Dict[str, Any]:
    """
    Create a complete workspace setup for testing.
    
    Returns:
        Dict containing workspace_client, openai_client, and config
    """
    try:
        workspace_client = get_workspace_client(env)
        openai_client = get_databricks_openai_client(workspace_client)
        config = create_test_client_config(env)
        llm_endpoint = get_databricks_llm_endpoint(env)
        
        # Check endpoint availability
        endpoint_available = check_endpoint_availability(workspace_client, llm_endpoint)
        
        return {
            "workspace_client": workspace_client,
            "openai_client": openai_client,
            "config": config,
            "llm_endpoint": llm_endpoint,
            "endpoint_available": endpoint_available,
            "available_endpoints": list_available_endpoints(workspace_client)
        }
    except Exception as e:
        print(f"Failed to create workspace setup: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    # Test the helper functions
    print("Testing Databricks workspace connection...")
    
    try:
        setup = create_test_workspace_setup()
        if "error" in setup:
            print(f"Error: {setup['error']}")
        else:
            print(f"✅ Connected to workspace")
            print(f"✅ LLM endpoint: {setup['llm_endpoint']}")
            print(f"✅ Endpoint available: {setup['endpoint_available']}")
            print(f"✅ Available endpoints: {setup['available_endpoints']}")
            print(f"✅ OpenAI client configured: {setup['openai_client'] is not None}")
    except Exception as e:
        print(f"❌ Test failed: {e}")