#!/usr/bin/env python3
"""
Direct endpoint deployment script that bypasses MLflow registration.
This is a test script to verify endpoint deployment works.
"""

import json
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from deploy.command_executor import CommandExecutor
from deploy.config import load_config

def main():
    """Deploy endpoint directly using existing model."""
    
    # Load configuration
    config_path = Path(__file__).parent / "config.yaml"
    config = load_config(config_path, "dev")
    
    print("🚀 DIRECT ENDPOINT DEPLOYMENT")
    print("=" * 50)
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
    
    # Use a mock model version for testing
    model_version = "1"
    
    print(f"📡 Creating endpoint '{endpoint_name}' for model {uc_model_name} version {model_version}")
    
    # Create endpoint deployment code
    deploy_code = """
import json
import os
import time

try:
    print("🔧 Setting up serving endpoint using Databricks Agent Framework...")
    
    # Import agents module for deployment
    from databricks import agents
    
    model_name = "{}"
    endpoint_name = "{}"  
    model_version = "{}"
    workload_size = "{}"
    
    print("📡 Deploying agent endpoint '" + endpoint_name + "' for model " + model_name + " version " + model_version)
    
    # Deploy using agents.deploy instead of SDK methods
    deployment = agents.deploy(
        model_name=model_name,
        model_version=model_version,
        name=endpoint_name,
        
        # Enable monitoring and review
        enable_review_ui=True,
        enable_inference_table=True,
        
        # Resource configuration
        workload_size=workload_size,
        scale_to_zero_enabled=True,
        
        # Environment configuration
        environment_vars={{
            "TAVILY_API_KEY": "{{{{secrets/msh/TAVILY_API_KEY}}}}",
            "BRAVE_API_KEY": "{{{{secrets/msh/BRAVE_API_KEY}}}}",
            "VECTOR_SEARCH_INDEX": os.getenv("VECTOR_SEARCH_INDEX", "main.msh.docs_index"),
            "ENVIRONMENT": "dev",
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
    
    print("✅ Endpoint '" + endpoint_name + "' deployed successfully")
    print("Query URL: " + str(deployment.query_endpoint))
    
    print(json.dumps({{
        "success": True,
        "endpoint_name": endpoint_name,
        "model_name": model_name,
        "model_version": model_version,
        "query_endpoint": str(deployment.query_endpoint)
    }}))
    
except Exception as e:
    print("❌ Endpoint deployment failed: " + str(e))
    import traceback
    print("Full stacktrace:")
    traceback.print_exc()
    print(json.dumps({{
        "success": False,
        "error": str(e),
        "traceback": traceback.format_exc()
    }}))
""".format(
        uc_model_name,
        endpoint_name,
        model_version, 
        workload_size
    )
    
    # Execute endpoint deployment
    print("🎯 Deploying serving endpoint...")
    deploy_result = executor.execute_python(deploy_code, "Deploy endpoint")
    
    if not deploy_result.success:
        print(f"❌ Endpoint deployment failed: {deploy_result.error}")
        if deploy_result.output:
            print("=== FULL DEPLOYMENT OUTPUT WITH STACKTRACE ===")
            print(deploy_result.output)
            print("=== END FULL OUTPUT ===")
        return False
    
    # Parse JSON from output
    output_lines = deploy_result.output.strip().split('\n')
    json_line = None
    for line in reversed(output_lines):
        line = line.strip()
        if line.startswith('{') and line.endswith('}'):
            json_line = line
            break
    
    if not json_line:
        print("❌ Could not find JSON result in endpoint deployment output")
        print(f"Full output: {deploy_result.output}")
        return False
    
    try:
        deploy_data = json.loads(json_line)
        if deploy_data.get("success"):
            print(f"✅ Endpoint deployed: {deploy_data['endpoint_name']}")
            print(f"   Model: {deploy_data['model_name']} v{deploy_data['model_version']}")
            return True
        else:
            print(f"❌ Endpoint deployment failed: {deploy_data.get('error')}")
            return False
            
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse endpoint deployment JSON: {e}")
        print(f"JSON line: {json_line}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)