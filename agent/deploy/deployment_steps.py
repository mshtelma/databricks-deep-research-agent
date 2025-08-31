"""
Core deployment steps extracted from log_and_deploy.py notebook.
These functions are designed to be executed on Databricks via Command Execution API.
"""

import time
import json
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

# Note: These imports will be available when executed on Databricks
# import mlflow
# from databricks import agents
# from databricks.sdk import WorkspaceClient
# from deep_research_agent.databricks_compatible_agent import DatabricksCompatibleAgent

logger = logging.getLogger(__name__)


@dataclass
class StepResult:
    """Result of a deployment step."""
    success: bool
    data: Dict[str, Any]
    error: Optional[str] = None
    duration: float = 0.0


class DeploymentSteps:
    """Container for all deployment step functions."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with deployment configuration."""
        self.config = config
    
    @staticmethod
    def install_dependencies() -> Dict[str, Any]:
        """Install required packages for agent deployment."""
        try:
            import subprocess
            import sys
            
            # Core dependencies with version pinning for stability
            packages = [
                "backoff",
                "databricks-langchain", 
                "langgraph==0.6.6",
                "uv",
                "databricks-agents",
                "mlflow-skinny[databricks]",
                "unitycatalog-ai[databricks]",
                "unitycatalog-langchain[databricks]",
                "tavily-python>=0.3.0",
                "requests>=2.31.0", 
                "pydantic>=2.0.0"
            ]
            
            # Install packages
            for package in packages:
                result = subprocess.run([
                    sys.executable, '-m', 'pip', 'install', '-U', package
                ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    return {
                        "success": False,
                        "error": f"Failed to install {package}: {result.stderr}",
                        "package": package
                    }
            
            # Restart Python to ensure imports work
            print("Restarting Python kernel...")
            if 'dbutils' in globals():
                dbutils.library.restartPython()
            
            return {
                "success": True,
                "packages_installed": len(packages),
                "packages": packages
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "stage": "package_installation"
            }
    
    @staticmethod
    def setup_and_test_agent(config: Dict[str, Any]) -> Dict[str, Any]:
        """Setup environment and test agent functionality."""
        try:
            import os
            import sys
            from typing import List, Dict, Any
            
            # Setup environment variables
            os.environ["DATABRICKS_RESPONSE_FORMAT"] = "adaptive"
            os.environ["AGENT_RETURN_FORMAT"] = "dict"
            
            # Setup secrets using centralized resolver
            try:
                from deep_research_agent.core.config import ConfigManager
                from deep_research_agent.core.types import ToolType
                from deep_research_agent.core.utils import resolve_secret
                
                config_manager = ConfigManager(yaml_path="deep_research_agent/agent_config.yaml")
                tavily_config = config_manager.get_tool_config(ToolType.TAVILY_SEARCH)
                brave_config = config_manager.get_tool_config(ToolType.BRAVE_SEARCH)
                
                # Resolve secrets if tools are enabled
                if tavily_config.enabled:
                    tavily_key = resolve_secret("{{secrets/msh/TAVILY_API_KEY}}")
                    if tavily_key != "{{secrets/msh/TAVILY_API_KEY}}":
                        os.environ["TAVILY_API_KEY"] = tavily_key
                
                if brave_config.enabled:
                    brave_key = resolve_secret("{{secrets/msh/BRAVE_API_KEY}}")
                    if brave_key != "{{secrets/msh/BRAVE_API_KEY}}":
                        os.environ["BRAVE_API_KEY"] = brave_key
                        
            except Exception as secret_error:
                print(f"Secret resolution warning: {secret_error}")
            
            # Add workspace path to Python path for imports
            import os
            current_dir = os.getcwd()
            print(f"Current directory: {current_dir}")
            
            # Add both current directory and the workspace path to sys.path
            if current_dir not in sys.path:
                sys.path.append(current_dir)
            if "." not in sys.path:
                sys.path.append(".")
                
            # Print sys.path for debugging
            print(f"Python path: {sys.path}")
            
            # List files in current directory to verify structure
            print("Files in current directory:")
            for item in sorted(os.listdir('.')):
                if os.path.isdir(item):
                    print(f"  📁 {item}/")
                    if item == "deep_research_agent":
                        # Show contents of deep_research_agent directory
                        print("    Contents:")
                        for subitem in sorted(os.listdir(item)):
                            print(f"      {subitem}")
                else:
                    print(f"  📄 {item}")
            
            # Import and create agent with detailed debugging
            print("=" * 60)
            print("ATTEMPTING IMPORT")
            print("=" * 60)
            
            try:
                print("Attempting to import deep_research_agent.databricks_compatible_agent...")
                from deep_research_agent.databricks_compatible_agent import DatabricksCompatibleAgent
                print("✅ Successfully imported DatabricksCompatibleAgent")
            except ImportError as e:
                print(f"❌ Import error for DatabricksCompatibleAgent: {e}")
                print("Trying to import the package step by step...")
                
                try:
                    print("Step 1: Importing deep_research_agent package...")
                    import deep_research_agent
                    print(f"✅ deep_research_agent imported from: {deep_research_agent.__file__}")
                except ImportError as e2:
                    print(f"❌ Cannot import deep_research_agent package: {e2}")
                    return {
                        "success": False,
                        "error": f"Cannot import deep_research_agent package: {e2}",
                        "stage": "package_import"
                    }
                
                try:
                    print("Step 2: Importing databricks_compatible_agent module...")
                    from deep_research_agent import databricks_compatible_agent
                    print(f"✅ databricks_compatible_agent module imported from: {databricks_compatible_agent.__file__}")
                except ImportError as e3:
                    print(f"❌ Cannot import databricks_compatible_agent module: {e3}")
                    return {
                        "success": False,
                        "error": f"Cannot import databricks_compatible_agent module: {e3}",
                        "stage": "module_import"
                    }
                
                try:
                    print("Step 3: Getting DatabricksCompatibleAgent class...")
                    from deep_research_agent.databricks_compatible_agent import DatabricksCompatibleAgent
                    print("✅ DatabricksCompatibleAgent class imported successfully")
                except ImportError as e4:
                    print(f"❌ Cannot import DatabricksCompatibleAgent class: {e4}")
                    return {
                        "success": False,
                        "error": f"Cannot import DatabricksCompatibleAgent class: {e4}",
                        "stage": "class_import"
                    }
            
            try:
                print("Importing MLflow ResponsesAgentRequest...")
                from mlflow.types.responses import ResponsesAgentRequest
                print("✅ Successfully imported ResponsesAgentRequest")
            except ImportError as e:
                print(f"❌ Failed to import ResponsesAgentRequest: {e}")
                return {
                    "success": False,
                    "error": f"Failed to import ResponsesAgentRequest: {e}",
                    "stage": "mlflow_import"
                }
            
            print("All imports successful, proceeding with agent creation...")
            print("=" * 60)
            
            test_agent = DatabricksCompatibleAgent(yaml_path="deep_research_agent/agent_config.yaml")
            
            # Run test queries
            test_queries = [
                {
                    "name": "Simple Calculation",
                    "query": "What is 6*7 in Python?",
                    "type": "simple"
                },
                {
                    "name": "Research Query", 
                    "query": "What are the latest developments in artificial intelligence in 2024?",
                    "type": "research"
                }
            ]
            
            test_results = {}
            successful_tests = 0
            
            for test_query in test_queries:
                try:
                    start_time = time.time()
                    request = ResponsesAgentRequest(input=[{"role": "user", "content": test_query["query"]}])
                    response = test_agent.predict(request)
                    end_time = time.time()
                    
                    response_time = end_time - start_time
                    
                    if response and hasattr(response, 'output') and response.output:
                        test_results[test_query["name"]] = {
                            "success": True,
                            "response_time": response_time,
                            "query_type": test_query["type"]
                        }
                        successful_tests += 1
                    else:
                        test_results[test_query["name"]] = {
                            "success": False,
                            "error": "Empty or invalid response",
                            "query_type": test_query["type"]
                        }
                        
                except Exception as e:
                    test_results[test_query["name"]] = {
                        "success": False,
                        "error": str(e),
                        "query_type": test_query["type"]
                    }
            
            total_tests = len(test_queries)
            
            if successful_tests == 0:
                return {
                    "success": False,
                    "error": f"All {total_tests} tests failed - agent is not functional",
                    "test_results": test_results
                }
            
            return {
                "success": True,
                "tests_passed": successful_tests,
                "tests_total": total_tests,
                "test_results": test_results,
                "agent_functional": True
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "stage": "agent_setup_test",
                "traceback": str(e.__traceback__) if hasattr(e, '__traceback__') else None
            }
    
    @staticmethod
    def log_model_to_mlflow(config: Dict[str, Any]) -> Dict[str, Any]:
        """Log the LangGraph research agent to MLflow."""
        try:
            import mlflow
            import os
            import sys
            from pkg_resources import get_distribution
            
            # Add workspace path to Python path for imports
            current_dir = os.getcwd()
            if current_dir not in sys.path:
                sys.path.append(current_dir)
            if "." not in sys.path:
                sys.path.append(".")
            
            print("=" * 60)
            print("MLFLOW LOGGING - ATTEMPTING IMPORT")
            print("=" * 60)
            print(f"Current directory: {current_dir}")
            print(f"Python path: {sys.path}")
            
            try:
                print("Attempting to import DatabricksCompatibleAgent for MLflow...")
                from deep_research_agent.databricks_compatible_agent import DatabricksCompatibleAgent
                print("✅ Successfully imported DatabricksCompatibleAgent")
            except ImportError as e:
                print(f"❌ Import error for DatabricksCompatibleAgent: {e}")
                return {
                    "success": False,
                    "error": f"Cannot import DatabricksCompatibleAgent for MLflow logging: {e}",
                    "stage": "mlflow_agent_import"
                }
            
            # Set MLflow registry to Unity Catalog
            mlflow.set_registry_uri("databricks-uc")
            
            # Set or create the MLflow experiment
            experiment_path = config.get('mlflow', {}).get('experiment_path', '/Users/michael.shtelma@databricks.com/langgraph-agent-experiments/dev')
            experiment_description = config.get('mlflow', {}).get('experiment_description', 'LangGraph Research Agent Experiments')
            
            print(f"📊 Setting MLflow experiment: {experiment_path}")
            try:
                mlflow.set_experiment(experiment_path)
                print(f"✅ MLflow experiment set successfully")
            except Exception as e:
                print(f"⚠️  Creating new experiment: {e}")
                mlflow.create_experiment(
                    experiment_path,
                    artifact_location=None,  # Use default location
                    tags={"description": str(experiment_description), "environment": str(config.get("ENVIRONMENT", "dev"))}
                )
                mlflow.set_experiment(experiment_path)
                print(f"✅ MLflow experiment created and set")
            
            # Prepare input example
            input_example = {
                "input": [{"role": "user", "content": "What is 6*7 in Python?"}]
            }
            
            # Create agent instance with correct YAML path
            agent_instance = DatabricksCompatibleAgent(yaml_path="deep_research_agent/agent_config.yaml")
            
            with mlflow.start_run(run_name=f"langgraph-agent-{int(time.time())}") as run:
                # Log configuration parameters
                mlflow.log_params({
                    "agent_type": "langgraph_research",
                    "framework": "langgraph",
                    "deployment_type": "responses_agent",
                    "environment": str(config.get("ENVIRONMENT", "dev"))
                })
                
                # Log model using models-from-code approach
                logged_model_info = mlflow.pyfunc.log_model(
                    artifact_path="model",
                    python_model="deep_research_agent/databricks_compatible_agent.py",
                    # agent_config.yaml is inside deep_research_agent folder, so just include the whole folder
                    code_paths=["deep_research_agent"],
                    input_example=input_example,
                    pip_requirements=[
                        "databricks-langchain",
                        f"langgraph=={get_distribution('langgraph').version}",
                        f"backoff=={get_distribution('backoff').version}",
                        "unitycatalog-ai[databricks]",
                        "unitycatalog-langchain[databricks]", 
                        "tavily-python>=0.3.0",
                        "requests>=2.31.0",
                        "pydantic>=2.0.0"
                    ],
                    metadata={
                        "description": "LangGraph multi-step research agent using Databricks Foundation Models",
                        "capabilities": str(["web_search", "vector_search", "multi_step_reasoning", "citation_generation"]),
                        "tools": str(["tavily", "brave", "vector_search", "unity_catalog_functions"]),
                        "author": "Databricks Migration",
                        "environment": str(config.get("ENVIRONMENT", "dev"))
                    }
                )
                
                return {
                    "success": True,
                    "model_uri": logged_model_info.model_uri,
                    "run_id": run.info.run_id,
                    "artifact_path": logged_model_info.artifact_path
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "stage": "mlflow_logging",
                "traceback": str(e.__traceback__) if hasattr(e, '__traceback__') else None
            }
    
    @staticmethod 
    def register_model(model_uri: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Register model in Unity Catalog."""
        try:
            import mlflow
            
            full_model_name = config["UC_MODEL_NAME"]
            
            registered_model = mlflow.register_model(
                model_uri=model_uri,
                name=full_model_name,
                tags={
                    "type": "research_agent",
                    "framework": "langgraph",
                    "deployment": "databricks_agent_framework", 
                    "environment": str(config.get("ENVIRONMENT", "dev"))
                }
            )
            
            return {
                "success": True,
                "model_name": full_model_name,
                "model_version": registered_model.version,
                "model_uri": model_uri
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "stage": "model_registration",
                "model_name": config.get("UC_MODEL_NAME", "unknown")
            }
    
    @staticmethod
    def deploy_agent_endpoint(model_name: str, model_version: str, endpoint_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy agent using Databricks Agent Framework."""
        try:
            from databricks import agents
            import os
            
            deployment = agents.deploy(
                model_name=model_name,
                model_version=model_version,
                name=endpoint_name,
                
                # Enable monitoring and review
                enable_review_ui=True,
                enable_inference_table=True,
                
                # Resource configuration
                workload_size=config.get("WORKLOAD_SIZE", "Small"),
                scale_to_zero_enabled=(config.get("ENVIRONMENT", "dev") != "prod"),
                
                # Environment configuration
                environment_vars={
                    "TAVILY_API_KEY": "{{secrets/msh/TAVILY_API_KEY}}",
                    "BRAVE_API_KEY": "{{secrets/msh/BRAVE_API_KEY}}",
                    "VECTOR_SEARCH_INDEX": os.getenv("VECTOR_SEARCH_INDEX", "main.msh.docs_index"),
                    "ENVIRONMENT": str(config.get("ENVIRONMENT", "dev")),
                    # Response format settings for Databricks schema compliance
                    "DATABRICKS_RESPONSE_FORMAT": "adaptive",
                    "AGENT_RETURN_FORMAT": "dict",
                    # Serving endpoint indicators
                    "DATABRICKS_MODEL_SERVING": "true",
                    "REQUIRE_DATABRICKS_SCHEMA_COMPLIANCE": "true",
                    "DATABRICKS_ENDPOINT_NAME": endpoint_name,
                    "DATABRICKS_MODEL_NAME": model_name,
                    "DEPLOYMENT_CONTEXT": "serving"
                }
            )
            
            return {
                "success": True,
                "endpoint_name": endpoint_name,
                "query_endpoint": deployment.query_endpoint if hasattr(deployment, 'query_endpoint') else None,
                "model_name": model_name,
                "model_version": model_version
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "stage": "agent_deployment",
                "endpoint_name": endpoint_name,
                "model_name": model_name
            }
    
    @staticmethod
    def wait_for_endpoint(endpoint_name: str, timeout: int = 600) -> Dict[str, Any]:
        """Wait for endpoint to be ready and return status."""
        try:
            from databricks.sdk import WorkspaceClient
            import time
            
            workspace_client = WorkspaceClient()
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                try:
                    endpoint = workspace_client.serving_endpoints.get(endpoint_name)
                    
                    if endpoint.state and endpoint.state.ready:
                        return {
                            "success": True,
                            "endpoint_name": endpoint_name,
                            "status": "ready",
                            "wait_time": time.time() - start_time,
                            "endpoint_url": f"/ml/endpoints/{endpoint_name}"
                        }
                    
                    # Still starting, wait a bit more
                    time.sleep(10)
                    
                except Exception as status_error:
                    # Endpoint might not exist yet
                    time.sleep(10)
                    continue
            
            # Timeout
            return {
                "success": False,
                "error": f"Endpoint not ready after {timeout} seconds",
                "endpoint_name": endpoint_name,
                "timeout": timeout
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "stage": "endpoint_wait",
                "endpoint_name": endpoint_name
            }
    
    @staticmethod
    def validate_endpoint(endpoint_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run validation tests against deployed endpoint."""
        try:
            import requests
            from databricks.sdk import WorkspaceClient
            import time
            
            workspace_client = WorkspaceClient()
            
            # Get endpoint URL
            try:
                endpoint = workspace_client.serving_endpoints.get(endpoint_name)
                endpoint_url = f"{workspace_client.config.host}/serving-endpoints/{endpoint_name}/invocations"
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Could not get endpoint URL: {e}",
                    "endpoint_name": endpoint_name
                }
            
            # Test queries
            test_queries = [
                "What is 6*7 in Python?",
                "Explain quantum computing briefly"
            ]
            
            results = []
            successful_tests = 0
            
            for query in test_queries:
                try:
                    payload = {
                        "input": [{"role": "user", "content": query}]
                    }
                    
                    # Use databricks SDK for authenticated requests
                    response = requests.post(
                        endpoint_url,
                        json=payload,
                        headers={
                            "Authorization": f"Bearer {workspace_client.config.token}",
                            "Content-Type": "application/json"
                        },
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        successful_tests += 1
                        results.append({
                            "query": query,
                            "status_code": response.status_code,
                            "success": True,
                            "response_time": response.elapsed.total_seconds()
                        })
                    else:
                        results.append({
                            "query": query,
                            "status_code": response.status_code,
                            "success": False,
                            "error": response.text[:200]
                        })
                        
                except Exception as e:
                    results.append({
                        "query": query,
                        "success": False,
                        "error": str(e)[:200]
                    })
            
            return {
                "success": successful_tests > 0,
                "tests_passed": successful_tests,
                "tests_total": len(test_queries),
                "results": results,
                "endpoint_name": endpoint_name,
                "endpoint_url": endpoint_url
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "stage": "endpoint_validation",
                "endpoint_name": endpoint_name
            }


# Standalone functions for direct execution via Command Execution API

def install_dependencies():
    """Standalone function for package installation."""
    return DeploymentSteps.install_dependencies()

def setup_and_test_agent(config):
    """Standalone function for agent setup and testing."""
    return DeploymentSteps.setup_and_test_agent(config)

def log_model_to_mlflow(config):
    """Standalone function for MLflow model logging."""
    return DeploymentSteps.log_model_to_mlflow(config)

def register_model(model_uri, config):
    """Standalone function for model registration."""
    return DeploymentSteps.register_model(model_uri, config)

def deploy_agent_endpoint(model_name, model_version, endpoint_name, config):
    """Standalone function for agent deployment."""
    return DeploymentSteps.deploy_agent_endpoint(model_name, model_version, endpoint_name, config)

def wait_for_endpoint(endpoint_name, timeout=600):
    """Standalone function for waiting for endpoint."""
    return DeploymentSteps.wait_for_endpoint(endpoint_name, timeout)

def validate_endpoint(endpoint_name, config):
    """Standalone function for endpoint validation."""
    return DeploymentSteps.validate_endpoint(endpoint_name, config)