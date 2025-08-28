# Databricks notebook source
# MAGIC %md
# MAGIC # LangGraph Research Agent - Enhanced MLflow Logging and Deployment
# MAGIC 
# MAGIC This notebook provides a comprehensive workflow for testing, logging, registering, and deploying
# MAGIC LangGraph agents using the Agent Framework with proper testing at each stage.
# MAGIC 
# MAGIC ## Workflow:
# MAGIC 1. **Setup and Configuration** - Install dependencies and configure parameters
# MAGIC 2. **Pre-deployment Testing** - Test agent locally before logging
# MAGIC 3. **MLflow Logging** - Log agent to MLflow with validation
# MAGIC 4. **Post-logging Testing** - Test logged model functionality
# MAGIC 5. **Model Registration** - Register model in Unity Catalog
# MAGIC 6. **Agent Deployment** - Deploy using Databricks Agent Framework
# MAGIC 7. **Post-deployment Testing** - Validate deployed endpoint
# MAGIC 8. **Summary Report** - Final validation and monitoring setup
# MAGIC 
# MAGIC ## Parameters:
# MAGIC - CATALOG: Unity Catalog catalog name
# MAGIC - SCHEMA: Unity Catalog schema name  
# MAGIC - MODEL_NAME: Model name for registration
# MAGIC - ENDPOINT_NAME: Serving endpoint name
# MAGIC - ENVIRONMENT: Environment (dev/staging/prod)
# MAGIC - WORKLOAD_SIZE: Serving workload size

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Install Required Packages

# COMMAND ----------

# Install core dependencies with version pinning for stability
%pip install -U backoff databricks-langchain langgraph==0.6.6 uv databricks-agents mlflow-skinny[databricks]
%pip install unitycatalog-ai[databricks] unitycatalog-langchain[databricks]
# Install search provider dependencies (both Tavily and Brave use requests)
%pip install tavily-python>=0.3.0 requests>=2.31.0 pydantic>=2.0.0

# COMMAND ----------

print("Restarting Python kernel...")
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Setup and Configuration

# COMMAND ----------

import os
import sys
import logging
import time
import traceback
import json
from typing import Dict, Any, Optional, List
from uuid import uuid4

# MLflow and Databricks imports
import mlflow
from databricks import agents
from databricks.sdk import WorkspaceClient
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse
# ChatMessage removed - using dictionary format as per Databricks standards
from pkg_resources import get_distribution

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

print("✅ Dependencies imported successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Configuration and Helper Functions

# COMMAND ----------

def get_deployment_config() -> Dict[str, Any]:
    """Get deployment configuration from widgets or use defaults."""
    try:
        # Try to get parameters from job widgets
        widget_names = [w.name for w in dbutils.widgets.getAll()] if 'dbutils' in globals() else []
        logger.info(f"Available widgets: {widget_names}")
        
        config = {
            "CATALOG": dbutils.widgets.get("CATALOG") if "CATALOG" in widget_names else "main",
            "SCHEMA": dbutils.widgets.get("SCHEMA") if "SCHEMA" in widget_names else "msh", 
            "MODEL_NAME": dbutils.widgets.get("MODEL_NAME") if "MODEL_NAME" in widget_names else "langgraph_research_agent",
            "ENDPOINT_NAME": dbutils.widgets.get("ENDPOINT_NAME") if "ENDPOINT_NAME" in widget_names else "langgraph-research-agent",
            "ENVIRONMENT": dbutils.widgets.get("ENVIRONMENT") if "ENVIRONMENT" in widget_names else "dev",
            "WORKLOAD_SIZE": dbutils.widgets.get("WORKLOAD_SIZE") if "WORKLOAD_SIZE" in widget_names else "Small"
        }
    except Exception as e:
        logger.info(f"Using default parameters (widget error: {e})")
        config = {
            "CATALOG": "main",
            "SCHEMA": "msh", 
            "MODEL_NAME": "langgraph_research_agent",
            "ENDPOINT_NAME": "langgraph-research-agent",
            "ENVIRONMENT": "dev",
            "WORKLOAD_SIZE": "Small"
        }
    
    # Add derived values
    config["UC_MODEL_NAME"] = f"{config['CATALOG']}.{config['SCHEMA']}.{config['MODEL_NAME']}"
    
    return config

def log_step(step_name: str, status: str = "START", details: str = ""):
    """Log workflow steps with consistent formatting."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    separator = "=" * 60
    
    if status == "START":
        print(f"\n{separator}")
        print(f"🚀 [{timestamp}] {step_name}")
        print(f"{separator}")
        if details:
            print(f"Details: {details}")
    elif status == "SUCCESS":
        print(f"✅ [{timestamp}] {step_name} - SUCCESS")
        if details:
            print(f"Result: {details}")
    elif status == "ERROR":
        print(f"❌ [{timestamp}] {step_name} - ERROR")
        if details:
            print(f"Error: {details}")
        print(f"{separator}")

def handle_error(error: Exception, context: str):
    """Enhanced error handling with context."""
    error_msg = str(error)
    error_traceback = traceback.format_exc()
    
    log_step(context, "ERROR", error_msg)
    logger.error(f"{context} failed: {error_msg}")
    logger.error(f"Traceback:\n{error_traceback}")
    
    # Log to MLflow if available
    try:
        mlflow.log_metric(f"{context.lower().replace(' ', '_')}_success", 0.0)
        mlflow.log_text(f"Error: {error_msg}\n\nTraceback:\n{error_traceback}", f"{context.lower().replace(' ', '_')}_error.txt")
    except:
        pass  # MLflow not available or no active run
    
    return error_msg, error_traceback

def validate_environment() -> Dict[str, Any]:
    """Validate environment setup and return status."""
    validation_results = {
        "python_version": sys.version,
        "current_directory": os.getcwd(),
        "refactored_agent_exists": os.path.exists("research_agent_refactored.py"),
        "environment_vars": {},
        "package_versions": {}
    }
    
    # Check environment variables
    env_vars_to_check = [
        "TAVILY_API_KEY", "BRAVE_API_KEY", "VECTOR_SEARCH_INDEX", "LLM_ENDPOINT",
        "MAX_CONCURRENT_SEARCHES", "BATCH_DELAY_SECONDS"
    ]
    
    for key in env_vars_to_check:
        value = os.getenv(key, "NOT SET")
        if (key == "TAVILY_API_KEY" or key == "BRAVE_API_KEY") and value != "NOT SET":
            # Mask the API key for security
            if value.startswith("{{secrets/"):
                # Secret reference not resolved
                validation_results["environment_vars"][key] = f"UNRESOLVED: {value}"
            else:
                # Real API key - mask it
                value = value[:10] + "..." if len(value) > 10 else value
                validation_results["environment_vars"][key] = f"SET: {value}"
        else:
            validation_results["environment_vars"][key] = value
    
    # Check package versions
    for package in ["langgraph", "databricks-langchain", "mlflow-skinny"]:
        try:
            validation_results["package_versions"][package] = get_distribution(package).version
        except:
            validation_results["package_versions"][package] = "NOT INSTALLED"
    
    return validation_results

def get_model_config() -> Dict[str, Any]:
    """Create MLflow ModelConfig with agent configuration from YAML."""
    # Import here to avoid circular imports
    from core.config import ConfigManager
    from core.types import ToolType
    
    # Initialize configuration manager with YAML loading
    config_manager = ConfigManager(yaml_path="agent_config.yaml")
    agent_config = config_manager.get_agent_config()
    
    # Convert AgentConfiguration to dictionary for MLflow
    from dataclasses import asdict
    config_dict = asdict(agent_config)
    
    # Add model serving endpoint alias for compatibility
    config_dict["model_serving_endpoint"] = config_dict["llm_endpoint"]
    
    # Add response format settings for Databricks schema compatibility
    config_dict["databricks_response_format"] = "adaptive"
    config_dict["agent_return_format"] = "dict"
    
    # Detect which search provider is configured
    search_provider = config_manager.get_search_provider()
    config_dict["search_provider"] = search_provider
    
    # Check which search tools are enabled
    tavily_config = config_manager.get_tool_config(ToolType.TAVILY_SEARCH)
    brave_config = config_manager.get_tool_config(ToolType.BRAVE_SEARCH)
    
    config_dict["tavily_enabled"] = tavily_config.enabled
    config_dict["brave_enabled"] = brave_config.enabled
    
    return config_dict

def exit_with_error(test_results: Dict[str, Any], context: str = "Pre-deployment Testing") -> None:
    """
    Exit the notebook gracefully with error information when tests fail.
    
    Args:
        test_results: Dictionary containing test results with success/failure info
        context: Context description for logging
    """
    # Count failures
    failed_tests = [name for name, result in test_results.items() if not result.get("success", False)]
    total_tests = len(test_results)
    
    if not failed_tests:
        return  # No failures, don't exit with error
    
    # Create detailed error information
    error_details = {
        "failed_tests": failed_tests,
        "total_tests": total_tests,
        "failure_rate": len(failed_tests) / total_tests * 100,
        "test_results": test_results,
        "context": context
    }
    
    # Format error message for logging
    error_msg = f"{len(failed_tests)} out of {total_tests} tests failed ({error_details['failure_rate']:.1f}%)"
    
    # Format detailed failure information
    failure_details = []
    for test_name in failed_tests:
        test_result = test_results[test_name]
        failure_info = {
            "test_name": test_name,
            "error": test_result.get("error", "Unknown error"),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        failure_details.append(failure_info)
    
    # Set task values to pass back to calling job
    if 'dbutils' in globals():
        try:
            dbutils.jobs.taskValues.set(key="deployment_status", value="failed")
            dbutils.jobs.taskValues.set(key="error_message", value=error_msg)
            dbutils.jobs.taskValues.set(key="failed_tests", value=json.dumps(failed_tests))
            dbutils.jobs.taskValues.set(key="failure_details", value=json.dumps(failure_details))
            dbutils.jobs.taskValues.set(key="total_tests", value=str(total_tests))
            dbutils.jobs.taskValues.set(key="context", value=context)
            logger.info("Task values set for failure propagation")
        except Exception as e:
            logger.warning(f"Could not set task values: {e}")
    
    # Log the error
    log_step(context, "ERROR", error_msg)
    logger.error(f"{context} failed: {error_msg}")
    
    # Print detailed failure summary
    print(f"\n❌ {context.upper()} FAILED")
    print("=" * 50)
    print(f"Failed: {len(failed_tests)} out of {total_tests} tests")
    print("\nFailed Tests:")
    for failure in failure_details:
        print(f"  • {failure['test_name']}: {failure['error']}")
    print("=" * 50)
    
    # Exit notebook with detailed error message
    if 'dbutils' in globals():
        # Create detailed exit message with specific test failures
        detailed_msg = f"FAILED: {error_msg}\n\nFailed Tests:\n"
        for failure in failure_details:
            detailed_msg += f"• {failure['test_name']}: {failure['error']}\n"
        detailed_msg += f"\nContext: {context}"
        
        dbutils.notebook.exit(detailed_msg)
    else:
        # For local testing
        raise Exception(f"{context} failed: {error_msg}")

def create_test_queries() -> List[Dict[str, Any]]:
    """Create test queries for validation."""
    return [
        {
            "name": "Simple Calculation",
            "query": "What is 6*7 in Python?",
            "type": "simple",
            "expected_keywords": ["python", "42", "*", "multiply"]
        },
        {
            "name": "Research Query",
            "query": "What are the latest developments in artificial intelligence in 2024?",
            "type": "research",
            "expected_keywords": ["ai", "artificial intelligence", "2024", "developments"]
        },
        {
            "name": "Comparison Query", 
            "query": "Compare Python and JavaScript for web development",
            "type": "research",
            "expected_keywords": ["python", "javascript", "web development", "compare"]
        }
    ]

# Get deployment configuration
DEPLOY_CONFIG = get_deployment_config()

print("=== DEPLOYMENT CONFIGURATION ===")
for key, value in DEPLOY_CONFIG.items():
    print(f"{key}: {value}")
print("================================")

def setup_secrets():
    """Setup required secrets by resolving Databricks secrets to environment variables using centralized SecretResolver."""
    try:
        logger.info("Setting up secrets for agent initialization using centralized resolver")
        
        # Import the centralized secret resolution utilities
        from core.config import ConfigManager
        from core.types import ToolType
        from core.utils import resolve_secret, get_secret_cache_status
        
        config_manager = ConfigManager(yaml_path="agent_config.yaml")
        tavily_config = config_manager.get_tool_config(ToolType.TAVILY_SEARCH)
        brave_config = config_manager.get_tool_config(ToolType.BRAVE_SEARCH)
        
        # List of secrets to pre-resolve and set as environment variables
        secrets_to_resolve = []
        
        # Add Tavily secret if enabled
        if tavily_config.enabled:
            secrets_to_resolve.append({
                "secret_ref": "{{secrets/msh/TAVILY_API_KEY}}",
                "env_var": "TAVILY_API_KEY",
                "service": "Tavily"
            })
        
        # Add Brave secret if enabled
        if brave_config.enabled:
            secrets_to_resolve.append({
                "secret_ref": "{{secrets/msh/BRAVE_API_KEY}}",
                "env_var": "BRAVE_API_KEY", 
                "service": "Brave"
            })
        
        logger.info(f"Will attempt to resolve {len(secrets_to_resolve)} secrets using SecretResolver")
        
        resolved_count = 0
        for secret in secrets_to_resolve:
            try:
                # Check if environment variable is already set
                existing_value = os.getenv(secret["env_var"])
                if existing_value and not existing_value.startswith("{{secrets/"):
                    logger.info(f"Environment variable {secret['env_var']} already set (length: {len(existing_value)})")
                    resolved_count += 1
                    continue
                
                # Use centralized secret resolution
                resolved_value = resolve_secret(secret["secret_ref"])
                
                if resolved_value != secret["secret_ref"]:
                    # Secret was successfully resolved
                    os.environ[secret["env_var"]] = resolved_value
                    logger.info(f"Successfully resolved and set {secret['env_var']} for {secret['service']} (length: {len(resolved_value)})")
                    resolved_count += 1
                else:
                    # Secret could not be resolved
                    logger.warning(
                        f"{secret['service']} secret could not be resolved: {secret['secret_ref']}. "
                        f"Please ensure either:\n"
                        f"  1. Databricks secret 'msh/{secret['env_var']}' is available\n"
                        f"  2. Environment variable '{secret['env_var']}' is set\n"
                        f"  {secret['service']} search will be disabled."
                    )
                    
            except Exception as e:
                logger.warning(f"Failed to resolve {secret['service']} secret: {e}")
                # Continue without failing - the agent will handle missing secrets
        
        # Log final resolution summary
        logger.info(f"Secret resolution completed: {resolved_count}/{len(secrets_to_resolve)} secrets resolved")
        
        # Log cache status for debugging
        cache_status = get_secret_cache_status()
        logger.info(f"SecretResolver cache status: {cache_status['cached_secrets']} cached, {cache_status['failed_secrets']} failed")
        
        # Verify final state
        logger.info("Final secret resolution summary:")
        for secret in secrets_to_resolve:
            env_value = os.getenv(secret["env_var"])
            status = f"SET (length: {len(env_value)})" if env_value else "NOT SET"
            logger.info(f"  {secret['env_var']}: {status}")
                
    except Exception as e:
        logger.error(f"Error in setup_secrets: {e}")
        # Don't fail the entire process for secret resolution issues

print("✅ Configuration and helper functions ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Pre-deployment Testing
# MAGIC 
# MAGIC Test the agent locally before logging to MLflow to catch issues early.

# COMMAND ----------

log_step("Pre-deployment Testing", "START")

# Initialize test results dictionary
test_results = {}

# Basic validation - can we create the agent?
try:
    log_step("Agent Initialization", "START")
    
    # Setup environment
    import os
    os.environ["DATABRICKS_RESPONSE_FORMAT"] = "adaptive" 
    os.environ["AGENT_RETURN_FORMAT"] = "dict"
    
    # Setup secrets
    setup_secrets()
    
    # Import and create agent
    sys.path.append(".")
    from deep_research_agent.databricks_compatible_agent import DatabricksCompatibleAgent
    
    test_agent = DatabricksCompatibleAgent(yaml_path="agent_config.yaml")
    log_step("Agent Initialization", "SUCCESS", "Agent created successfully")
    
    # Run comprehensive tests using test queries
    test_queries = create_test_queries()
    
    for test_query in test_queries:
        try:
            log_step(f"Testing - {test_query['name']}", "START")
            
            # Test the agent with the query
            start_time = time.time()
            request = ResponsesAgentRequest(input=[{"role": "user", "content": test_query["query"]}])
            response = test_agent.predict(request)
            end_time = time.time()
            
            response_time = end_time - start_time
            
            # Validate response
            if response and hasattr(response, 'output') and response.output:
                test_results[test_query["name"]] = {
                    "success": True,
                    "response_time": response_time,
                    "query_type": test_query["type"]
                }
                log_step(f"Testing - {test_query['name']}", "SUCCESS", 
                        f"Response time: {response_time:.2f}s")
            else:
                test_results[test_query["name"]] = {
                    "success": False,
                    "error": "Empty or invalid response",
                    "query_type": test_query["type"]
                }
                log_step(f"Testing - {test_query['name']}", "ERROR", "Empty or invalid response")
                
        except Exception as e:
            test_results[test_query["name"]] = {
                "success": False,
                "error": str(e),
                "query_type": test_query["type"]
            }
            log_step(f"Testing - {test_query['name']}", "ERROR", str(e))
    
    # Print test summary
    print("\n📊 PRE-DEPLOYMENT TEST SUMMARY:")
    print("=" * 50)
    successful_tests = 0
    for test_name, result in test_results.items():
        if result["success"]:
            successful_tests += 1
            print(f"✅ {test_name}")
            print(f"   Response Time: {result.get('response_time', 0):.2f}s")
        else:
            print(f"❌ {test_name}: {result['error']}")
    
    total_tests = len(test_results)
    print(f"\nResults: {successful_tests}/{total_tests} tests passed")
    print("=" * 50)
    
    # Check if we should proceed
    if successful_tests == 0:
        raise Exception(f"All {total_tests} tests failed - agent is not functional")
    elif successful_tests < total_tests:
        log_step("Pre-deployment Testing", "WARNING", f"Only {successful_tests}/{total_tests} tests passed")
    else:
        log_step("Pre-deployment Testing", "SUCCESS", f"All {total_tests} tests passed")
        
except Exception as e:
    log_step("Pre-deployment Testing", "ERROR", str(e))
    
    # If we don't have test_results yet, create a basic one for the error
    if not test_results:
        test_results = {
            "Agent Initialization": {
                "success": False,
                "error": str(e),
                "context": "Failed to create or test basic agent functionality"
            }
        }
    
    # Exit with error information
    exit_with_error(test_results, "Pre-deployment Testing")



# COMMAND ----------


# MAGIC %md
# MAGIC ## Step 5: MLflow Logging with Validation

# COMMAND ----------

def log_agent_to_mlflow(config: Dict[str, Any]) -> Any:
    """Log the LangGraph research agent to MLflow with enhanced validation."""
    log_step("MLflow Logging", "START")
    
    # Set MLflow registry to Unity Catalog
    mlflow.set_registry_uri("databricks-uc")
    
    model_config = get_model_config()
    
    # Ensure response format settings are in model config for deployment
    model_config["databricks_response_format"] = "adaptive"
    model_config["agent_return_format"] = "dict"
    
    # Prepare input example
    input_example = {
        "input": [{"role": "user", "content": "What is 6*7 in Python?"}]
    }
    
    # Import and create the agent instance (models-from-code approach) - using wrapper
    from deep_research_agent.databricks_compatible_agent import DatabricksCompatibleAgent
    
    # Create agent with model config using the Databricks-compatible wrapper
    agent_instance = DatabricksCompatibleAgent(model_config)
    
    # Set the model for MLflow (like old/agent.py does)
    mlflow.models.set_model(agent_instance)
    
    with mlflow.start_run(run_name=f"langgraph-agent-{int(time.time())}") as run:
        # Log configuration and test results
        mlflow.log_params({
            "agent_type": "langgraph_research",
            "llm_endpoint": model_config["llm_endpoint"], 
            "max_research_loops": model_config["max_research_loops"],
            "initial_query_count": model_config["initial_query_count"],
            "framework": "langgraph",
            "deployment_type": "responses_agent",
            "environment": config["ENVIRONMENT"]
        })
        
        # Log test results
        for test_name, result in test_results.items():
            mlflow.log_metric(f"pretest_{test_name.lower().replace(' ', '_')}_success", 
                            1.0 if result["success"] else 0.0)
            if result["success"]:
                mlflow.log_metric(f"pretest_{test_name.lower().replace(' ', '_')}_response_time", 
                                result.get("response_time", 0))
        
        # YAML configuration is included in code_paths - no need to log separately
        if os.path.exists("agent_config.yaml"):
            log_step("YAML Configuration", "SUCCESS", "agent_config.yaml will be included with code")
        else:
            log_step("YAML Configuration", "WARNING", "agent_config.yaml not found")
        
        # Log model using models-from-code with python_model parameter - using wrapper
        logged_model_info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model="deep_research_agent/databricks_compatible_agent.py",  # Required - points to the wrapper file with mlflow.models.set_model()
            model_config="agent_config.yaml" if os.path.exists("agent_config.yaml") else None,  # Optional YAML config file
            code_paths=[
                "deep_research_agent"  # Include entire package directory to preserve module structure
            ] + (["agent_config.yaml"] if os.path.exists("agent_config.yaml") else []),  # Dependencies (excluding the main python_model file)
            input_example=input_example,
            pip_requirements=[
                "databricks-langchain",
                f"langgraph=={get_distribution('langgraph').version}",
                f"backoff=={get_distribution('backoff').version}", 
                f"databricks-connect=={get_distribution('databricks-connect').version}",
                "unitycatalog-ai[databricks]",
                "unitycatalog-langchain[databricks]",
                "tavily-python>=0.3.0",
                "requests>=2.31.0",
                "pydantic>=2.0.0"
            ],
            metadata={
                "description": "LangGraph multi-step research agent using Databricks Foundation Models",
                "capabilities": ["web_search", "vector_search", "multi_step_reasoning", 
                              "citation_generation", "python_execution"],
                "tools": ["tavily", "brave", "vector_search", "unity_catalog_functions"],
                "search_providers": ["tavily", "brave"],
                "configured_provider": model_config.get("search_provider", "tavily"),
                "author": "Databricks Migration",
                "environment": config["ENVIRONMENT"],
                "test_results": {name: result["success"] for name, result in test_results.items()}
            }
        )
        
        log_step("MLflow Logging", "SUCCESS", f"Model URI: {logged_model_info.model_uri}")
        return logged_model_info

# Execute MLflow logging
try:
    logged_model_info = log_agent_to_mlflow(DEPLOY_CONFIG)
except Exception as e:
    error_msg, error_traceback = handle_error(e, "MLflow Logging")
    
    # Create a failure result for MLflow logging
    mlflow_failure = {
        "MLflow Logging": {
            "success": False,
            "error": error_msg,
            "traceback": error_traceback
        }
    }
    
    # Exit gracefully with error information
    exit_with_error(mlflow_failure, "MLflow Logging")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Post-logging Validation
# MAGIC 
# MAGIC Test the logged model to ensure it can be loaded and works correctly.

# COMMAND ----------

log_step("Post-logging Validation", "START")

# Load the logged model
try:
    log_step("Loading Logged Model", "START")
    loaded_model = mlflow.pyfunc.load_model(logged_model_info.model_uri)
    log_step("Loading Logged Model", "SUCCESS", "Model loaded successfully")
except Exception as e:
    error_msg, error_traceback = handle_error(e, "Loading Logged Model")
    
    # Create a failure result for model loading
    loading_failure = {
        "Loading Logged Model": {
            "success": False,
            "error": error_msg,
            "traceback": error_traceback
        }
    }
    
    # Exit gracefully with error information
    exit_with_error(loading_failure, "Loading Logged Model")

# Test loaded model with same queries
loaded_test_results = {}

for test_query in create_test_queries():
    try:
        log_step(f"Testing Loaded Model - {test_query['name']}", "START")
        
        # Test with loaded model
        start_time = time.time()
        response = loaded_model.predict({"input": [{"role": "user", "content": test_query["query"]}]})
        end_time = time.time()
        
        response_time = end_time - start_time
        
        # Validate response structure
        if hasattr(response, 'output') and response.output:
            # Extract text content properly from the response structure
            if hasattr(response.output[0], 'content') and response.output[0].content:
                # Content is a list like [{'type': 'output_text', 'text': 'actual text'}]
                content_items = response.output[0].content
                if isinstance(content_items, list) and len(content_items) > 0:
                    first_item = content_items[0]
                    if isinstance(first_item, dict) and 'text' in first_item:
                        response_text = first_item['text']
                    else:
                        response_text = str(first_item)
                else:
                    response_text = str(content_items)
            else:
                response_text = str(response.output[0])
            
            response_length = len(response_text)
            
            loaded_test_results[test_query["name"]] = {
                "success": True,
                "response_time": response_time,
                "response_length": response_length,
                "has_custom_outputs": hasattr(response, 'custom_outputs') and response.custom_outputs is not None
            }
            
            log_step(f"Testing Loaded Model - {test_query['name']}", "SUCCESS",
                   f"Response time: {response_time:.2f}s, Length: {response_length} chars")
        else:
            loaded_test_results[test_query["name"]] = {"success": False, "error": "Invalid response structure"}
            log_step(f"Testing Loaded Model - {test_query['name']}", "ERROR", "Invalid response structure")
            
    except Exception as e:
        error_msg, _ = handle_error(e, f"Testing Loaded Model - {test_query['name']}")
        loaded_test_results[test_query["name"]] = {"success": False, "error": error_msg}

# Print loaded model test summary
print("\n📊 POST-LOGGING TEST SUMMARY:")
print("=" * 50)
for test_name, result in loaded_test_results.items():
    if result["success"]:
        print(f"✅ {test_name}")
        print(f"   Response Time: {result.get('response_time', 0):.2f}s")
        print(f"   Has Custom Outputs: {result.get('has_custom_outputs', False)}")
    else:
        print(f"❌ {test_name}: {result['error']}")

log_step("Post-logging Validation", "SUCCESS", f"Loaded model tested with {len(create_test_queries())} queries")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Model Registration

# COMMAND ----------

def register_model(model_uri: str, config: Dict[str, Any]) -> Any:
    """Register model in Unity Catalog with enhanced metadata."""
    log_step("Model Registration", "START")
    
    full_model_name = config["UC_MODEL_NAME"]
    
    try:
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=full_model_name,
            tags={
                "type": "research_agent",
                "framework": "langgraph", 
                "deployment": "databricks_agent_framework",
                "environment": config["ENVIRONMENT"],
                "test_status": "passed" if all(r["success"] for r in loaded_test_results.values()) else "partial"
            }
        )
        
        log_step("Model Registration", "SUCCESS", 
               f"Model: {full_model_name} v{registered_model.version}")
        return registered_model
        
    except Exception as e:
        handle_error(e, f"Model Registration for {full_model_name}")
        raise

# Execute model registration
try:
    registered_model = register_model(logged_model_info.model_uri, DEPLOY_CONFIG)
except Exception as e:
    error_msg, error_traceback = handle_error(e, "Model Registration")
    
    # Create a failure result for model registration
    registration_failure = {
        "Model Registration": {
            "success": False,
            "error": error_msg,
            "traceback": error_traceback
        }
    }
    
    # Exit gracefully with error information
    exit_with_error(registration_failure, "Model Registration")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Agent Deployment

# COMMAND ----------

def deploy_agent(config: Dict[str, Any], model_version: str) -> Any:
    """Deploy agent using Databricks Agent Framework."""
    log_step("Agent Deployment", "START")
    
    full_model_name = config["UC_MODEL_NAME"]
    endpoint_name = config["ENDPOINT_NAME"]
    
    try:
        deployment = agents.deploy(
            model_name=full_model_name,
            model_version=model_version,
            name=endpoint_name,
            
            # Enable monitoring and review
            enable_review_ui=True,
            enable_inference_table=True,
            
            # Resource configuration
            workload_size=config["WORKLOAD_SIZE"],
            scale_to_zero_enabled=(config["ENVIRONMENT"] != "prod"),
            
            # Environment configuration
            environment_vars={
                "TAVILY_API_KEY": "{{secrets/msh/TAVILY_API_KEY}}",
                "BRAVE_API_KEY": "{{secrets/msh/BRAVE_API_KEY}}",
                "VECTOR_SEARCH_INDEX": os.getenv("VECTOR_SEARCH_INDEX", "main.msh.docs_index"),
                "ENVIRONMENT": config["ENVIRONMENT"],
                # Response format settings for Databricks schema compatibility
                "DATABRICKS_RESPONSE_FORMAT": "adaptive",
                "AGENT_RETURN_FORMAT": "dict",
                # Serving endpoint indicators for proper detection
                "DATABRICKS_MODEL_SERVING": "true",
                "REQUIRE_DATABRICKS_SCHEMA_COMPLIANCE": "true",
                "DATABRICKS_ENDPOINT_NAME": config["ENDPOINT_NAME"],
                "DATABRICKS_MODEL_NAME": config["UC_MODEL_NAME"],
                "DEPLOYMENT_CONTEXT": "serving"
            }
        )
        
        log_step("Agent Deployment", "SUCCESS", f"Endpoint: {endpoint_name}")
        logger.info(f"Query URL: {deployment.query_endpoint}")
        
        return deployment
        
    except Exception as e:
        handle_error(e, f"Agent Deployment for {endpoint_name}")
        raise

# Execute deployment
try:
    deployment = deploy_agent(DEPLOY_CONFIG, registered_model.version)
except Exception as e:
    error_msg, error_traceback = handle_error(e, "Agent Deployment")
    
    # Create a failure result for agent deployment
    deployment_failure = {
        "Agent Deployment": {
            "success": False,
            "error": error_msg,
            "traceback": error_traceback
        }
    }
    
    # Exit gracefully with error information
    exit_with_error(deployment_failure, "Agent Deployment")

# COMMAND ----------