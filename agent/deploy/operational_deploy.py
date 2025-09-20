#!/usr/bin/env python3
"""
Operational deployment script that combines MLflow model registration with endpoint deployment.
This is designed to be fully operational and complete the entire deployment pipeline.
"""

import json
import os
import sys
import time
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from deploy.command_executor import CommandExecutor
from deep_research_agent.constants import AGENT_CONFIG_FILENAME

def main():
    """Execute complete operational deployment pipeline."""
    
    print("🚀 OPERATIONAL DEPLOYMENT PIPELINE")
    print("=" * 60)
    
    # Load configuration using new ConfigLoader
    try:
        from deep_research_agent.config_loader import ConfigLoader
        deploy_config = ConfigLoader.load_deployment("dev")
        config = deploy_config.model_dump()
        config["ENVIRONMENT"] = "dev"
        config["UC_MODEL_NAME"] = f"{config['model']['catalog']}.{config['model']['schema']}.{config['model']['name']}"
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return False
    
    print(f"Environment: {config.get('ENVIRONMENT', 'dev')}")
    print(f"Endpoint: {config['endpoint']['name']}")
    print(f"Model: {config['UC_MODEL_NAME']}")
    
    # Create command executor
    try:
        executor = CommandExecutor(config)
        print("✅ Command executor initialized")
    except Exception as e:
        print(f"❌ Failed to initialize executor: {e}")
        return False
    
    # Extract configuration values
    uc_model_name = config["UC_MODEL_NAME"]  
    endpoint_name = config["endpoint"]["name"]
    workload_size = config["endpoint"]["workload_size"]
    environment = config.get("ENVIRONMENT", "dev")
    workspace_path = config["workspace_path"]
    
    print("\n📝 PHASE 1: MLflow Model Registration")
    print("-" * 40)
    
    # Create exact MLflow registration using working backup notebook approach
    mlflow_code = '''
import json
import mlflow
import time
import os
import sys
from pkg_resources import get_distribution

try:
    print("🔧 Starting MLflow model registration...")
    
    # Configuration
    model_name = "{}"
    environment = "{}"
    
    # Change to workspace directory where files are synced (CRITICAL for file paths)
    workspace_path = "{}"
    print("Current directory before change: " + os.getcwd())
    print("Changing directory to workspace: " + workspace_path)
    try:
        os.chdir(workspace_path)
        print("✅ Successfully changed to workspace directory: " + workspace_path)
    except Exception as e:
        print("❌ Failed to change to workspace directory: " + str(e))
        print("Trying to continue with current directory...")

    print("Current directory after change: " + os.getcwd())
    sys.path.insert(0, os.getcwd())
    
    # Set MLflow registry to Unity Catalog
    mlflow.set_registry_uri("databricks-uc")
    
    # Set MLflow experiment
    experiment_path = "/Users/michael.shtelma@databricks.com/langgraph-agent-dev"
    try:
        mlflow.set_experiment(experiment_path)
        print("✅ MLflow experiment set")
    except Exception:
        mlflow.create_experiment(experiment_path, tags={{"environment": environment}})
        mlflow.set_experiment(experiment_path)
        print("✅ MLflow experiment created and set")
    
    # Install required dependencies - EXACT three pip calls from backup notebook
    print("📦 Installing required dependencies...")
    import subprocess
    import sys
    
    # First pip install call
    pip_call_1 = ["backoff", "databricks-langchain", "langgraph==0.6.6", "databricks-agents", "mlflow-skinny[databricks]"]
    # Second pip install call  
    pip_call_2 = ["unitycatalog-ai[databricks]", "unitycatalog-langchain[databricks]"]
    # Third pip install call
    pip_call_3 = ["tavily-python>=0.3.0", "requests>=2.31.0", "pydantic>=2.0.0"]
    
    # Execute the three pip install calls exactly as in notebook
    for i, packages in enumerate([pip_call_1, pip_call_2, pip_call_3], 1):
        try:
            print("📦 Pip install call " + str(i) + ": " + ", ".join(packages))
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-U"] + packages)
            print("✅ Pip install call " + str(i) + " completed")
        except subprocess.CalledProcessError as e:
            print("⚠️  Pip install call " + str(i) + " failed: " + str(e))
    
    print("✅ Dependencies installation completed")
    
    with mlflow.start_run(run_name="operational-deployment-" + str(int(time.time()))) as run:
        print("🏗️  Creating operational model...")
        
        # Debug current directory and available files
        import os
        print("Current working directory: " + os.getcwd())
        print("Files in current directory:")
        for item in os.listdir("."):
            print("  " + item)
        
        if os.path.exists("deep_research_agent"):
            print("deep_research_agent directory contents:")
            for item in os.listdir("deep_research_agent"):
                print("  deep_research_agent/" + item)
        else:
            print("deep_research_agent directory does not exist")
        
        agent_file = "deep_research_agent/databricks_compatible_agent.py"
        print("Agent file exists: " + str(os.path.exists(agent_file)))
        
        # Check config file exists
        config_file = "deep_research_agent/" + AGENT_CONFIG_FILENAME
        print("Config file (" + config_file + ") exists: " + str(os.path.exists(config_file)))
        
        # Test agent import before attempting log_model
        print("🔍 Testing agent import in deployment environment...")
        try:
            from deep_research_agent.databricks_compatible_agent import DatabricksCompatibleAgent
            test_agent = DatabricksCompatibleAgent()
            print("✅ Agent import and initialization successful")
        except Exception as import_error:
            print("❌ Agent import failed: " + str(import_error))
            import traceback
            print("Import traceback:")
            print(traceback.format_exc())
        
        # Test minimal code-based logging first
        print("🔍 Testing minimal code-based logging...")
        try:
            minimal_logged = mlflow.pyfunc.log_model(
                artifact_path="minimal_test",
                python_model="deep_research_agent/databricks_compatible_agent.py"
            )
            print("✅ Minimal code-based logging successful")
        except Exception as minimal_error:
            print("❌ Minimal code-based logging failed: " + str(minimal_error))
            import traceback
            print("Minimal logging traceback:")
            print(traceback.format_exc())
        
        # EXACT MLflow logging from backup notebook
        input_example = {{
            "input": [{{"role": "user", "content": "What is 6*7 in Python?"}}]
        }}
        #+ ([AGENT_CONFIG_FILENAME] if os.path.exists(AGENT_CONFIG_FILENAME) else [])
        print("🔍 Attempting full code-based logging with all parameters...")
        logged_model_info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model="deep_research_agent/databricks_compatible_agent.py",
            code_paths=[
                "deep_research_agent"
            ] ,
            input_example=input_example,
            pip_requirements=[
                "databricks-langchain",
                "langgraph==0.6.6",
                "backoff",
                "unitycatalog-ai[databricks]",
                "unitycatalog-langchain[databricks]",
                "tavily-python>=0.3.0",
                "requests>=2.31.0",
                "pydantic>=2.0.0"
            ],
            metadata={{
                "description": "LangGraph multi-step research agent using Databricks Foundation Models",
                "capabilities": ["web_search", "vector_search", "multi_step_reasoning", 
                              "citation_generation", "python_execution"],
                "tools": ["tavily", "brave", "vector_search", "unity_catalog_functions"],
                "search_providers": ["tavily", "brave"],
                "author": "Databricks Migration",
                "environment": environment,
                "framework": "langgraph",
                "deployment_type": "responses_agent"
            }}
        )
        
        print("✅ Model artifacts created successfully")
        
        # Register the model
        registered_model = mlflow.register_model(
            model_uri=logged_model_info.model_uri,
            name=model_name,
            tags={{"environment": environment, "deployment": "operational"}}
        )
        
        model_version = registered_model.version
        model_uri = "models:/" + model_name + "/" + str(model_version)
        
        print("✅ Model registered: " + model_name + "/v" + str(model_version))
        
        # Return success with model information
        print(json.dumps({{
            "success": True,
            "model_name": model_name,
            "model_version": model_version,
            "model_uri": model_uri,
            "run_id": run.info.run_id
        }}))
        
except Exception as e:
    print("❌ MLflow registration failed: " + str(e))
    import traceback
    print("Traceback: " + traceback.format_exc())
    print(json.dumps({{
        "success": False,
        "error": str(e)
    }}))
'''.format(uc_model_name, environment, workspace_path)
    
    # Execute MLflow registration
    print("📊 Registering model in MLflow...")
    mlflow_result = executor.execute_python(mlflow_code, "Register MLflow model")
    
    if not mlflow_result.success:
        print(f"❌ MLflow registration failed: {mlflow_result.error}")
        if mlflow_result.output:
            print(f"Output: {mlflow_result.output}")
        return False
    
    # Parse MLflow results - search through all output for JSON
    mlflow_output_lines = mlflow_result.output.strip().split('\n')
    mlflow_json_line = None
    for line in mlflow_output_lines:
        line = line.strip()
        if line.startswith('{"success":'):
            mlflow_json_line = line
            break
    
    if not mlflow_json_line:
        print("❌ Could not parse MLflow registration results")
        print(f"Full output: {mlflow_result.output}")
        return False
    
    try:
        mlflow_data = json.loads(mlflow_json_line)
        if not mlflow_data.get("success"):
            print(f"❌ MLflow registration failed: {mlflow_data.get('error')}")
            return False
        
        model_version = mlflow_data["model_version"]
        print(f"✅ Model registered: {mlflow_data['model_name']}/v{model_version}")
        
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse MLflow JSON: {e}")
        return False
    
    print("\\n🚀 PHASE 2: Endpoint Deployment") 
    print("-" * 40)
    
    # EXACT agent deployment from backup notebook
    endpoint_code = '''
import json
import os
import time

from databricks import agents

try:
    print("🔧 Starting agent deployment...")
    
    # Configuration
    model_name = "{}"
    endpoint_name = "{}"
    model_version = "{}"
    workload_size = "{}"
    environment = "{}"
    
    print("📡 Deploying endpoint: " + endpoint_name)
    print("📦 Using model: " + model_name + "/v" + model_version)
    
    # Debug: Check what's available in databricks package
    import databricks
    print("databricks module contents: " + str(dir(databricks)))
    
    
    # EXACT deployment call from backup notebook
    deployment = agents.deploy(
        model_name=model_name,
        model_version=model_version,
        name=endpoint_name,
        
        # Enable monitoring and review
        enable_review_ui=True,
        enable_inference_table=True,
        
        # Resource configuration
        workload_size=workload_size,
        scale_to_zero_enabled=(environment != "prod"),
        
        # Environment configuration - EXACT from backup
        environment_vars={{
            "BRAVE_API_KEY": "{{{{secrets/msh/BRAVE_API_KEY}}}}",
            "VECTOR_SEARCH_INDEX": os.getenv("VECTOR_SEARCH_INDEX", "main.msh.docs_index"),
            "ENVIRONMENT": environment,
            # Response format settings for Databricks schema compatibility
            "DATABRICKS_RESPONSE_FORMAT": "adaptive",
            "AGENT_RETURN_FORMAT": "dict",
            # Serving endpoint indicators for proper detection
            "DATABRICKS_MODEL_SERVING": "true",
            "REQUIRE_DATABRICKS_SCHEMA_COMPLIANCE": "true",
            "DATABRICKS_ENDPOINT_NAME": endpoint_name,
            "DATABRICKS_MODEL_NAME": model_name,
            "DEPLOYMENT_CONTEXT": "serving"
        }}
    )
    
    print("✅ Agent deployed successfully")
    print("📊 Query URL: " + str(deployment.query_endpoint))
    
    # Return success response
    print(json.dumps({{
        "success": True,
        "endpoint_name": endpoint_name,
        "model_name": model_name,
        "model_version": model_version,
        "query_url": str(deployment.query_endpoint),
        "status": "deployed"
    }}))
        
except Exception as e:
    print("❌ Agent deployment failed: " + str(e))
    import traceback
    full_traceback = traceback.format_exc()
    print("Full Traceback:")
    print(full_traceback)
    print("Error Type: " + str(type(e)))
    print("Error Args: " + str(e.args))
    print(json.dumps({{
        "success": False,
        "error": str(e),
        "error_type": str(type(e)),
        "traceback": full_traceback
    }}))
'''.format(uc_model_name, endpoint_name, model_version, workload_size, environment)
    
    # Execute endpoint deployment
    print("🎯 Creating serving endpoint...")
    endpoint_result = executor.execute_python(endpoint_code, "Deploy endpoint")
    
    if not endpoint_result.success:
        print(f"❌ Endpoint deployment failed: {endpoint_result.error}")
        if endpoint_result.output:
            print("=== FULL DEPLOYMENT OUTPUT WITH STACKTRACE ===")
            print(endpoint_result.output)
            print("=== END FULL OUTPUT ===")
        return False
    
    # Parse endpoint results - search through all output for JSON
    endpoint_output_lines = endpoint_result.output.strip().split('\n')
    endpoint_json_line = None
    for line in endpoint_output_lines:
        line = line.strip()
        if line.startswith('{"success":'):
            endpoint_json_line = line
            break
    
    if not endpoint_json_line:
        print("❌ Could not parse endpoint deployment JSON results")
        print("=== DEBUGGING: FULL ENDPOINT OUTPUT ===")
        print(endpoint_result.output)
        print("=== END DEBUGGING OUTPUT ===")
        print("Looking for any JSON in output...")
        # Try to find any JSON in the output
        for i, line in enumerate(endpoint_output_lines):
            line_stripped = line.strip()
            if line_stripped.startswith('{') and line_stripped.endswith('}'):
                print(f"Found JSON-like line {i+1}: {line_stripped}")
                try:
                    test_json = json.loads(line_stripped)
                    print(f"Successfully parsed JSON: {test_json}")
                    endpoint_json_line = line_stripped
                    break
                except:
                    print(f"Failed to parse as JSON: {line_stripped}")
        
        if not endpoint_json_line:
            return False
    
    try:
        endpoint_data = json.loads(endpoint_json_line)
        if not endpoint_data.get("success"):
            print(f"❌ Endpoint deployment failed: {endpoint_data.get('error')}")
            return False
        
        print(f"✅ Endpoint operational: {endpoint_data['endpoint_name']}")
        print(f"📦 Model: {endpoint_data['model_name']}/v{endpoint_data['model_version']}")
        print(f"🌐 URL: {endpoint_data.get('endpoint_url', 'N/A')}")
        print(f"📊 State: {endpoint_data.get('state', 'Unknown')}")
        
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse endpoint JSON: {e}")
        return False
    
    print("\\n🎉 DEPLOYMENT SUCCESSFUL!")
    print("=" * 60)
    print(f"✅ Model: {uc_model_name}/v{model_version}")
    print(f"✅ Endpoint: {endpoint_name}")
    print(f"✅ Status: Operational")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)