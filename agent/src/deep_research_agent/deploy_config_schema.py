"""
Pydantic configuration schema for deployment settings.

This module defines deployment-specific configurations like Databricks workspaces,
cluster IDs, MLflow settings, and serving endpoints.
"""

from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings
from typing import Optional, Dict, Any
from enum import Enum

class WorkloadSize(str, Enum):
    """Databricks serving endpoint workload sizes"""
    SMALL = "Small"
    MEDIUM = "Medium"
    LARGE = "Large"

class CommandExecutionConfig(BaseSettings):
    """Databricks Command Execution API configuration"""
    cluster_id: str = Field(..., description="Databricks cluster ID for execution")
    timeout: int = Field(default=300, ge=30, description="Command timeout in seconds")
    auto_start: bool = Field(default=True, description="Auto-start cluster if stopped")
    start_timeout: int = Field(default=600, ge=60, description="Cluster start timeout in seconds")

class MLflowConfig(BaseSettings):
    """MLflow experiment configuration"""
    experiment_path: str = Field(..., description="MLflow experiment path")
    experiment_description: str = Field(default="", description="Experiment description")

class ModelRegistryConfig(BaseModel):
    """Unity Catalog model registry configuration"""
    catalog: str = Field(default="main", description="Unity Catalog catalog name")
    schema_name: str = Field(..., alias="schema", description="Unity Catalog schema name")
    name: str = Field(..., description="Model name in registry")

class EndpointConfig(BaseSettings):
    """Serving endpoint configuration"""
    name: str = Field(..., description="Serving endpoint name")
    workload_size: WorkloadSize = Field(default=WorkloadSize.SMALL, description="Endpoint workload size")
    ready_timeout: int = Field(default=900, ge=60, description="Endpoint ready timeout in seconds")
    check_interval: int = Field(default=30, ge=5, description="Health check interval in seconds")
    validation_retry_delay: int = Field(default=60, ge=10, description="Validation retry delay in seconds")
    validation_retries: int = Field(default=2, ge=0, description="Number of validation retries")

class DeploymentEnvironment(BaseSettings):
    """Environment-specific deployment configuration"""
    profile: str = Field(..., description="Databricks CLI profile name")
    workspace_path: str = Field(..., description="Workspace path for deployment")
    command_execution: CommandExecutionConfig
    mlflow: MLflowConfig
    model: ModelRegistryConfig
    endpoint: EndpointConfig

class DeploymentConfig(BaseSettings):
    """Main deployment configuration"""
    
    class Config:
        env_prefix = "DEPLOY_"
        env_nested_delimiter = "__"
        extra = "forbid"
        
    project_name: str = Field(default="deep-research-agent", description="Project name")
    environment: DeploymentEnvironment
    
    @validator("environment")
    def validate_environment(cls, v):
        """Validate environment configuration"""
        if not v.workspace_path.startswith("/"):
            raise ValueError("workspace_path must be an absolute path starting with '/'")
        return v

class GlobalDeploymentConfig(BaseSettings):
    """Global deployment settings and environment definitions"""
    
    class Config:
        extra = "allow"  # Allow additional environments
        
    project_name: str = Field(default="deep-research-agent")
    
    # Default configurations that can be inherited
    default_command_execution: Dict[str, Any] = Field(default_factory=lambda: {
        "timeout": 300,
        "auto_start": True,
        "start_timeout": 600
    })
    
    default_endpoint: Dict[str, Any] = Field(default_factory=lambda: {
        "workload_size": "Small",
        "ready_timeout": 900,
        "check_interval": 30,
        "validation_retry_delay": 60,
        "validation_retries": 2
    })
    
    # Environment definitions
    environments: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    
    def get_environment_config(self, env_name: str) -> DeploymentConfig:
        """Get deployment config for specific environment"""
        if env_name not in self.environments:
            raise ValueError(f"Environment '{env_name}' not found in configuration")
            
        env_config = self.environments[env_name].copy()
        
        # Merge defaults for command_execution if not fully specified
        if "command_execution" in env_config:
            command_exec = {**self.default_command_execution, **env_config["command_execution"]}
            env_config["command_execution"] = command_exec
            
        # Merge defaults for endpoint if not fully specified  
        if "endpoint" in env_config:
            endpoint = {**self.default_endpoint, **env_config["endpoint"]}
            env_config["endpoint"] = endpoint
            
        return DeploymentConfig(
            project_name=self.project_name,
            environment=DeploymentEnvironment(**env_config)
        )