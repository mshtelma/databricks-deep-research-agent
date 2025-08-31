"""
Main entry point for the deployment package.

This module orchestrates the complete deployment workflow using Command Execution API.
"""

import sys
import time
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.live import Live
from rich.text import Text

from .cli import parse_arguments, configure_logging, get_deployment_config, print_configuration_summary
from .command_executor import CommandExecutor, CommandExecutionError
from .workspace_manager import WorkspaceManager, SyncError
from .test_runner import TestRunner
from .endpoint_manager import EndpointManager, EndpointError
from .validator import Validator
from .error_reporter import ErrorReporter, ErrorContext
from .deployment_steps import DeploymentSteps

logger = logging.getLogger(__name__)
console = Console()


class DeploymentOrchestrator:
    """Main deployment orchestrator using Command Execution API."""
    
    def __init__(self, args_config: Dict[str, Any]):
        """Initialize deployment orchestrator."""
        self.args_config = args_config
        self.config = self.load_configuration()
        self.start_time = time.time()
        
        # Initialize components (will be created when needed)
        self.executor: Optional[CommandExecutor] = None
        self.workspace_manager: Optional[WorkspaceManager] = None
        self.test_runner: Optional[TestRunner] = None
        self.endpoint_manager: Optional[EndpointManager] = None
        self.validator: Optional[Validator] = None
    
    def load_configuration(self) -> Dict[str, Any]:
        """Load deployment configuration from YAML file."""
        config_path = Path(self.args_config["config_file"])
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                full_config = yaml.safe_load(f)
            
            environment = self.args_config["environment"]
            if environment not in full_config["environments"]:
                available = list(full_config["environments"].keys())
                raise ValueError(f"Environment '{environment}' not found. Available: {available}")
            
            # Merge environment config with global config
            env_config = full_config["environments"][environment].copy()
            env_config.update(full_config.get("global", {}))
            
            # Add derived values
            env_config["ENVIRONMENT"] = environment
            env_config["UC_MODEL_NAME"] = f"{env_config['model']['catalog']}.{env_config['model']['schema']}.{env_config['model']['name']}"
            
            # Override with CLI arguments
            if self.args_config.get("endpoint_name"):
                env_config["endpoint"]["name"] = self.args_config["endpoint_name"]
            
            if self.args_config.get("cluster_id"):
                env_config.setdefault("command_execution", {})["cluster_id"] = self.args_config["cluster_id"]
            
            return env_config
            
        except Exception as e:
            raise ValueError(f"Failed to load configuration: {e}")
    
    def get_command_executor(self) -> CommandExecutor:
        """Get or create command executor."""
        if not self.executor:
            try:
                self.executor = CommandExecutor(self.config)
                logger.info("✅ Command executor initialized")
            except CommandExecutionError as e:
                logger.error(f"❌ Failed to initialize command executor: {e}")
                raise
        return self.executor
    
    def get_workspace_manager(self) -> WorkspaceManager:
        """Get or create workspace manager."""
        if not self.workspace_manager:
            self.workspace_manager = WorkspaceManager(self.config)
            logger.info("✅ Workspace manager initialized")
        return self.workspace_manager
    
    def get_test_runner(self) -> TestRunner:
        """Get or create test runner."""
        if not self.test_runner:
            executor = self.get_command_executor()
            self.test_runner = TestRunner(executor)
            logger.info("✅ Test runner initialized")
        return self.test_runner
    
    def get_endpoint_manager(self) -> EndpointManager:
        """Get or create endpoint manager."""
        if not self.endpoint_manager:
            self.endpoint_manager = EndpointManager(self.config)
            logger.info("✅ Endpoint manager initialized")
        return self.endpoint_manager
    
    def get_validator(self) -> Validator:
        """Get or create validator."""
        if not self.validator:
            executor = self.get_command_executor()
            self.validator = Validator(executor)
            logger.info("✅ Validator initialized")
        return self.validator
    
    def run_management_operations(self) -> bool:
        """Handle management operations and return True if should exit."""
        
        # List endpoints
        if self.args_config.get("list_endpoints"):
            endpoint_manager = self.get_endpoint_manager()
            endpoints = endpoint_manager.list_endpoints()
            
            table = Table(title="📋 Serving Endpoints")
            table.add_column("Name", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Creator", style="dim")
            
            if endpoints:
                for endpoint in endpoints:
                    status = "✅ Ready" if endpoint.get("ready") else "⏳ Not Ready"
                    table.add_row(
                        endpoint['name'],
                        status,
                        endpoint.get("creator", "Unknown")
                    )
            else:
                table.add_row("No endpoints found", "", "")
            
            console.print(table)
            return True
        
        # Check endpoint status
        if self.args_config.get("endpoint_status"):
            endpoint_name = self.args_config["endpoint_status"]
            endpoint_manager = self.get_endpoint_manager()
            status = endpoint_manager.get_endpoint_status(endpoint_name)
            
            print(f"\\n📊 ENDPOINT STATUS: {endpoint_name}")
            print("=" * 50)
            print(json.dumps(status, indent=2))
            print("=" * 50)
            return True
        
        # Cleanup old endpoints
        if self.args_config.get("cleanup_old_endpoints"):
            pattern = self.args_config["cleanup_old_endpoints"]
            max_age = self.args_config["cleanup_max_age"]
            endpoint_manager = self.get_endpoint_manager()
            
            # First run dry run
            dry_result = endpoint_manager.cleanup_old_endpoints(pattern, max_age, dry_run=True)
            print(f"\\n🧹 ENDPOINT CLEANUP (DRY RUN)")
            print("=" * 50)
            print(json.dumps(dry_result, indent=2))
            
            if dry_result.get("candidates_found", 0) > 0:
                if self.args_config.get("force") or input("\\nProceed with cleanup? [y/N]: ").lower() in ['y', 'yes']:
                    cleanup_result = endpoint_manager.cleanup_old_endpoints(pattern, max_age, dry_run=False)
                    print("\\n🗑️  CLEANUP RESULTS")
                    print("=" * 50)
                    print(json.dumps(cleanup_result, indent=2))
                else:
                    print("Cleanup cancelled")
            
            print("=" * 50)
            return True
        
        return False
    
    def execute_deployment_pipeline(self) -> bool:
        """Execute the main deployment pipeline."""
        
        # Display deployment info in a panel
        info_table = Table.grid()
        info_table.add_row("Environment:", Text(self.config['ENVIRONMENT'], style="bold cyan"))
        info_table.add_row("Endpoint:", Text(self.config['endpoint']['name'], style="bold green"))
        info_table.add_row("Model:", Text(self.config['UC_MODEL_NAME'], style="bold yellow"))
        
        console.print(Panel(info_table, title="🚀 Deployment Pipeline", expand=False))
        
        try:
            # Stage 1: Workspace preparation
            if not self.prepare_workspace():
                return False
            
            # Stage 2: Test execution (if requested)
            if self.args_config.get("run_tests") and not self.args_config.get("sync_only"):
                if not self.execute_tests():
                    return False
            
            # Exit early for test-only or sync-only modes
            if self.args_config.get("test_only"):
                print("\\n✅ TEST-ONLY MODE COMPLETED")
                return True
            
            if self.args_config.get("sync_only"):
                print("\\n✅ SYNC-ONLY MODE COMPLETED")
                return True
            
            # Stage 3: Handle existing endpoint
            if not self.handle_existing_endpoint():
                return False
            
            # Stage 4: Deploy model and endpoint
            if not self.deploy_model_and_endpoint():
                return False
            
            # Stage 5: Validation (if not skipped)
            if not self.args_config.get("skip_validation"):
                if not self.validate_deployment():
                    return False
            
            self.print_success_summary()
            return True
            
        except KeyboardInterrupt:
            print("\\n⏹️  Deployment cancelled by user")
            return False
        except Exception as e:
            logger.error(f"Unexpected deployment error: {e}")
            print(f"\\n❌ Unexpected error: {e}")
            return False
    
    def prepare_workspace(self) -> bool:
        """Prepare workspace by cleaning and syncing files."""
        console.print(Panel("📁 Workspace Preparation", style="bold blue"))
        
        try:
            workspace_manager = self.get_workspace_manager()
            
            # Clean workspace if requested
            if self.args_config.get("clean_workspace"):
                print("🧹 Cleaning workspace...")
                if self.args_config.get("dry_run"):
                    print("   (DRY RUN: would clean workspace)")
                else:
                    # If --clean-workspace is specified, automatically confirm deletion
                    workspace_manager.clean_deployment_folder(confirm=True)
            
            # Sync files
            print("📤 Syncing files...")
            if self.args_config.get("dry_run"):
                print("   (DRY RUN: would sync files)")
                # Get workspace info for dry run
                info = workspace_manager.get_workspace_info()
                print(f"   Local files: {info.get('local_source_files', 0)} Python files")
                print(f"   Target: {info.get('workspace_path', 'unknown')}")
            else:
                workspace_manager.sync_files()
                
                # Verify sync
                verification = workspace_manager.verify_sync()
                if verification.get("summary", {}).get("sync_success", False):
                    print("✅ File sync verified successfully")
                else:
                    print("⚠️  File sync verification had issues")
                    if not self.args_config.get("force"):
                        if input("Continue anyway? [y/N]: ").lower() not in ['y', 'yes']:
                            return False
            
            return True
            
        except SyncError as e:
            print(f"❌ Workspace preparation failed: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected workspace error: {e}")
            return False
    
    def execute_tests(self) -> bool:
        """Execute test suite on Databricks."""
        print("\\n🧪 TEST EXECUTION")
        print("-" * 30)
        
        try:
            test_runner = self.get_test_runner()
            
            if self.args_config.get("dry_run"):
                print("   (DRY RUN: would run tests)")
                print(f"   Markers: {self.args_config.get('test_markers', 'default')}")
                return True
            
            # Validate test environment
            env_validation = test_runner.validate_test_environment()
            if not env_validation.get("packages_available", False):
                print("📦 Installing missing test dependencies...")
                if not test_runner.install_test_dependencies():
                    print("❌ Failed to install test dependencies")
                    return False
            
            # Run tests
            test_results = test_runner.run_all_tests(
                markers=self.args_config.get("test_markers"),
                fail_fast=self.args_config.get("fail_fast", True)
            )
            
            # Generate and display results
            if test_results.passed:
                print("\\n✅ ALL TESTS PASSED!")
                print(f"Total: {test_results.total_passed} passed")
                for phase_name, phase_results in test_results.phases.items():
                    print(f"  {phase_name}: {phase_results.passed}P/{phase_results.failed}F ({phase_results.duration:.1f}s)")
                return True
            else:
                # Generate detailed error report
                error_report = ErrorReporter.report_test_failures(test_results)
                print(error_report)
                return False
            
        except Exception as e:
            context = ErrorContext(
                stage="test_execution",
                component="test_runner", 
                operation="run_tests",
                suggestion="Check test configuration and try running tests locally first"
            )
            print(ErrorReporter.report_deployment_failure("test_execution", {"error": str(e)}))
            return False
    
    def handle_existing_endpoint(self) -> bool:
        """Handle existing endpoint deletion if requested."""
        if not self.args_config.get("delete_existing_endpoint"):
            return True
        
        print("\\n🗑️  ENDPOINT MANAGEMENT")
        print("-" * 30)
        
        try:
            endpoint_manager = self.get_endpoint_manager()
            endpoint_name = self.config["endpoint"]["name"]
            
            if self.args_config.get("dry_run"):
                exists = endpoint_manager.endpoint_exists(endpoint_name)
                if exists:
                    print(f"   (DRY RUN: would delete endpoint '{endpoint_name}')")
                else:
                    print(f"   (DRY RUN: endpoint '{endpoint_name}' does not exist)")
                return True
            
            if endpoint_manager.endpoint_exists(endpoint_name):
                print(f"Deleting existing endpoint: {endpoint_name}")
                return endpoint_manager.delete_endpoint(
                    endpoint_name, 
                    force=self.args_config.get("force", False)
                )
            else:
                print(f"No existing endpoint to delete: {endpoint_name}")
                return True
                
        except EndpointError as e:
            print(f"❌ Endpoint management failed: {e}")
            return False
    
    def deploy_model_and_endpoint(self) -> bool:
        """Deploy model to MLflow and create serving endpoint."""
        print("\\n🚀 MODEL DEPLOYMENT")
        print("-" * 30)
        
        if self.args_config.get("dry_run"):
            print("   (DRY RUN: would deploy model and endpoint)")
            print(f"   Model: {self.config['UC_MODEL_NAME']}")
            print(f"   Endpoint: {self.config['endpoint']['name']}")
            return True
        
        try:
            executor = self.get_command_executor()
            
            # Step 1: Install dependencies
            print("📦 Installing dependencies...")
            install_code = """
import json
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

print("📦 Installing packages...")
for package in packages:
    print(f"Installing {package}...")
    result = subprocess.run([
        sys.executable, '-m', 'pip', 'install', '-U', package
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Failed to install {package}: {result.stderr}")
        print(json.dumps({"success": False, "error": f"Failed to install {package}: {result.stderr}"}))
        exit(1)
    else:
        print(f"✅ Installed {package}")

print(f"📦 Successfully installed {len(packages)} packages")

# Print success before restarting Python
print(json.dumps({
    "success": True,
    "packages_installed": len(packages),
    "packages": packages
}))

# Now restart Python - this will kill the execution context but that's OK
print("🔄 Restarting Python kernel...")
if 'dbutils' in globals():
    dbutils.library.restartPython()
"""
            install_result = executor.execute_python(install_code, "Install dependencies")
            
            if not install_result.success:
                print(f"❌ Dependency installation failed: {install_result.error}")
                if install_result.output:
                    print(f"Output: {install_result.output}")
                return False
            
            # Parse JSON from output - look for the JSON line before restart
            output_lines = install_result.output.strip().split('\n')
            json_line = None
            for line in reversed(output_lines):  # Check from end backwards
                line = line.strip()
                if line.startswith('{') and line.endswith('}'):
                    json_line = line
                    break
            
            if not json_line:
                print(f"❌ Could not find JSON result in output")
                print(f"Full output: {install_result.output}")
                return False
            
            try:
                install_data = json.loads(json_line)
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse JSON: {e}")
                print(f"JSON line: {json_line}")
                return False
            
            if not install_data.get("success"):
                print(f"❌ Dependencies failed: {install_data.get('error')}")
                return False
            
            print(f"✅ Installed {install_data.get('packages_installed', 0)} packages")
            
            # Step 2: Create new execution context and test agent setup
            print("🤖 Testing agent setup...")
            # We need a new execution context since Python was restarted
            executor = self.get_command_executor()  # This will create a fresh context
            
            test_code = f"""
import json
import os
import sys

# Change to workspace directory where files are synced
workspace_path = "{self.config['workspace_path']}"
print(f"Changing directory from {{os.getcwd()}} to {{workspace_path}}")
try:
    os.chdir(workspace_path)
    print(f"✅ Successfully changed to workspace directory: {{workspace_path}}")
except Exception as e:
    print(f"❌ Failed to change to workspace directory: {{e}}")
    print("Trying to continue with current directory...")

current_dir = os.getcwd()
sys.path.insert(0, current_dir)

print("=" * 60)
print("DETAILED AGENT IMPORT DEBUGGING")
print("=" * 60)
print(f"Current directory: {{current_dir}}")
print(f"Python path (first 5): {{sys.path[:5]}}")

# List files in current directory to verify structure
print("Files in current directory:")
for item in sorted(os.listdir('.')):
    if os.path.isdir(item):
        print(f"  📁 {{item}}/")
        if item == "deep_research_agent":
            # Show contents of deep_research_agent directory
            print("    Contents:")
            try:
                for subitem in sorted(os.listdir(item)):
                    print(f"      {{subitem}}")
            except Exception as e:
                print(f"      ❌ Error listing contents: {{e}}")
    else:
        print(f"  📄 {{item}}")

print("\\n🔧 Testing agent import and setup...")

try:
    print("Step 1: Importing deep_research_agent package...")
    import deep_research_agent
    print(f"✅ deep_research_agent imported from: {{deep_research_agent.__file__}}")
    
    print("Step 2: Importing databricks_compatible_agent module...")
    from deep_research_agent import databricks_compatible_agent
    print(f"✅ databricks_compatible_agent module imported")
    
    print("Step 3: Importing DatabricksCompatibleAgent class...")
    from deep_research_agent.databricks_compatible_agent import DatabricksCompatibleAgent
    print("✅ DatabricksCompatibleAgent class imported successfully")
    
    print("Step 4: Importing UnifiedConfigManager...")
    from deep_research_agent.core.unified_config import UnifiedConfigManager
    print("✅ UnifiedConfigManager imported successfully")
    
    print("✅ Successfully imported agent classes")
    
    # Test agent initialization
    print("🧪 Testing agent initialization...")
    agent = DatabricksCompatibleAgent()
    print("✅ Agent initialized successfully")
    
    print(json.dumps({{
        "success": True,
        "tests_passed": 2,
        "tests_total": 2
    }}))
    
except ImportError as e:
    print(f"❌ Import error: {{e}}")
    print(f"Error details: {{str(e)}}")
    print("Attempting step-by-step diagnosis...")
    
    try:
        print("  Checking if deep_research_agent directory exists...")
        if os.path.exists('deep_research_agent'):
            print("  ✅ deep_research_agent directory exists")
            
            print("  Checking __init__.py file...")
            if os.path.exists('deep_research_agent/__init__.py'):
                print("  ✅ __init__.py exists")
                print(f"  __init__.py size: {{os.path.getsize('deep_research_agent/__init__.py')}} bytes")
            else:
                print("  ❌ __init__.py missing!")
                
            print("  Checking databricks_compatible_agent.py file...")
            if os.path.exists('deep_research_agent/databricks_compatible_agent.py'):
                print("  ✅ databricks_compatible_agent.py exists")
                print(f"  File size: {{os.path.getsize('deep_research_agent/databricks_compatible_agent.py')}} bytes")
            else:
                print("  ❌ databricks_compatible_agent.py missing!")
        else:
            print("  ❌ deep_research_agent directory does not exist!")
            
        print("  Checking sys.modules...")
        modules = [name for name in sys.modules.keys() if 'deep_research' in name]
        print(f"  Found modules: {{modules}}")
        
    except Exception as diag_e:
        print(f"  ❌ Error during diagnosis: {{diag_e}}")
    
    print(json.dumps({{
        "success": False,
        "error": f"Import error: {{str(e)}}",
        "error_type": "ImportError"
    }}))
except Exception as e:
    print(f"❌ Error during agent setup: {{e}}")
    print(json.dumps({{
        "success": False,
        "error": f"Agent setup error: {{str(e)}}"
    }}))
"""
            test_result = executor.execute_python(test_code, "Test agent setup")
            
            if not test_result.success:
                print(f"❌ Agent setup failed: {test_result.error}")
                if test_result.output:
                    print(f"Output: {test_result.output}")
                return False
            
            # Parse JSON from output - look for the JSON line
            output_lines = test_result.output.strip().split('\n')
            json_line = None
            for line in reversed(output_lines):
                line = line.strip()
                if line.startswith('{') and line.endswith('}'):
                    json_line = line
                    break
            
            if not json_line:
                print(f"❌ Could not find JSON result in agent test output")
                print(f"Full output: {test_result.output}")
                return False
            
            try:
                test_data = json.loads(json_line)
                # Always show full output for debugging
                print("🔍 FULL AGENT TEST OUTPUT:")
                print("=" * 60)
                print(test_result.output)
                print("=" * 60)
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse agent test JSON: {e}")
                print(f"JSON line: {json_line}")
                return False
            
            if not test_data.get("success"):
                print(f"❌ Agent tests failed: {test_data.get('error')}")
                return False
            
            print(f"✅ Agent functional: {test_data.get('tests_passed')}/{test_data.get('tests_total')} tests passed")
            
            # Step 3: Log model
            print("📝 Logging model to MLflow...")
            
            # Extract configuration values to avoid f-string issues
            uc_model_name = self.config['UC_MODEL_NAME']
            environment = self.config.get("ENVIRONMENT", "dev")
            mlflow_config = self.config.get('mlflow', {})
            experiment_path = mlflow_config.get('experiment_path', '/Users/michael.shtelma@databricks.com/langgraph-agent-experiments/dev')
            experiment_description = mlflow_config.get('experiment_description', 'LangGraph Research Agent Experiments')
            
            # Build the Python code using simple string concatenation - NO F-STRINGS!
            log_code = """
import json
import mlflow

print("🔧 Minimal MLflow test - starting...")

try:
    print("🔧 Logging model to MLflow...")
    
    # Import the agent
    from deep_research_agent.databricks_compatible_agent import DatabricksCompatibleAgent
    
    model_name = "{}"
    environment = "{}"
    experiment_path = "{}"
    experiment_description = "{}"
    
    print("📝 Logging agent to MLflow as " + model_name + "...")
    
    # Log the model using MLflow
    import time
    
    # Set MLflow registry to Unity Catalog
    mlflow.set_registry_uri("databricks-uc")
    
    print("📊 Setting MLflow experiment: " + experiment_path)
    try:
        mlflow.set_experiment(experiment_path)
        print("✅ MLflow experiment set successfully")
    except Exception as e:
        print("⚠️  Creating new experiment: " + str(e))
        mlflow.create_experiment(
            experiment_path,
            artifact_location=None,  # Use default location
            tags={{"description": experiment_description, "environment": environment}}
        )
        mlflow.set_experiment(experiment_path)
        print("✅ MLflow experiment created and set")
    
    with mlflow.start_run(run_name="langgraph-agent-" + str(int(time.time()))) as run:
        # Log deployment parameters
        mlflow.log_params({{
            "agent_type": "langgraph_research",
            "framework": "langgraph", 
            "deployment_type": "responses_agent",
            "environment": environment,
            "model_name": str(model_name)
        }})
        
        # Create input example
        input_example = {{
            "input": [{{"role": "user", "content": "What is 6*7 in Python?"}}]
        }}
        
        # Debug: Test each component separately to isolate the error
        print("🔍 Testing MLflow components...")
        
        try:
            print("  Testing input_example...")
            test_input = input_example
            print("  ✅ input_example OK: " + str(type(test_input)))
            
            print("  Testing pip_requirements...")
            pip_reqs = [
                "databricks-langchain",
                "langgraph==0.6.6", 
                "backoff",
                "unitycatalog-ai[databricks]",
                "unitycatalog-langchain[databricks]", 
                "tavily-python>=0.3.0",
                "requests>=2.31.0",
                "pydantic>=2.0.0"
            ]
            for i, req in enumerate(pip_reqs):
                print("    " + str(i) + ": " + req + " (" + str(type(req)) + ")")
            print("  ✅ pip_requirements OK")
            
            print("  Testing metadata...")
            meta_data = {{
                "description": "LangGraph multi-step research agent using Databricks Foundation Models",
                "capabilities": str(["web_search", "vector_search", "multi_step_reasoning", "citation_generation"]),
                "tools": str(["tavily", "brave", "vector_search", "unity_catalog_functions"]),
                "author": "Databricks Migration",
                "environment": environment
            }}
            for k, v in meta_data.items():
                print("    " + str(k) + ": " + str(v) + " (" + str(type(v)) + ")")
            print("  ✅ metadata OK")
            
            print("  Testing minimal log_model...")
            logged_model_info = mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model="deep_research_agent/databricks_compatible_agent.py"
            )
            print("  ✅ minimal log_model OK")
            
        except Exception as e:
            print("❌ Debug failed at component test: " + str(e))
            import traceback
            traceback.print_exc()
            raise
        
        print("🔍 All components OK, proceeding with hybrid model logging approach...")
        
        # Use a hybrid approach: full dependencies but optimized execution
        print("📦 Creating production-ready model with essential dependencies...")
        
        # Optimize pip requirements for faster installation
        essential_pip_reqs = [
            "databricks-langchain",
            "langgraph==0.6.6",
            "backoff",
            "pydantic>=2.0.0"
        ]
        
        # Create streamlined metadata
        essential_metadata = {{
            "description": "LangGraph research agent for Databricks",
            "framework": "langgraph", 
            "environment": environment
        }}
        
        try:
            print("🚀 Logging production model with essential dependencies...")
            logged_model_info_prod = mlflow.pyfunc.log_model(
                artifact_path="model_production",
                python_model="deep_research_agent/databricks_compatible_agent.py",
                code_paths=["deep_research_agent"],
                input_example=input_example,
                pip_requirements=essential_pip_reqs,
                metadata=essential_metadata
            )
            
            print("✅ Production model logged successfully")
            final_model_uri = logged_model_info_prod.model_uri
            
        except Exception as e:
            print("⚠️  Production model logging failed: " + str(e))
            print("🔄 Using minimal model as fallback for endpoint deployment")
            final_model_uri = logged_model_info.model_uri
        
        # Register the model (either full or minimal)
        registered_model = mlflow.register_model(
            model_uri=final_model_uri,
            name=model_name
        )
        
        print("✅ Model registered: " + model_name + "/" + str(registered_model.version))
        
    model_uri = "models:/" + model_name + "/" + str(registered_model.version)
    
    print("✅ Model deployment completed successfully: " + model_uri)
    
    print(json.dumps({{
        "success": True,
        "model_uri": model_uri,
        "model_version": registered_model.version
    }}))
    
except Exception as e:
    print("❌ Model logging failed: " + str(e))
    import traceback
    print("Traceback: " + traceback.format_exc())
    print(json.dumps({{
        "success": False,
        "error": str(e)
    }}))
""".format(
                uc_model_name,
                environment, 
                experiment_path,
                experiment_description
            )
            log_result = executor.execute_python(log_code, "Log model to MLflow")
            
            if not log_result.success:
                print(f"❌ Model logging failed: {log_result.error}")
                if log_result.output:
                    print(f"Output: {log_result.output}")
                return False
            
            # Parse JSON from output
            output_lines = log_result.output.strip().split('\n')
            json_line = None
            for line in reversed(output_lines):
                line = line.strip()
                if line.startswith('{') and line.endswith('}'):
                    json_line = line
                    break
            
            if not json_line:
                print(f"❌ Could not find JSON result in model logging output")
                print(f"Full output: {log_result.output}")
                return False
            
            try:
                log_data = json.loads(json_line)
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse model logging JSON: {e}")
                print(f"JSON line: {json_line}")
                return False
            
            if not log_data.get("success"):
                print(f"❌ Model logging failed: {log_data.get('error')}")
                return False
            
            model_uri = log_data["model_uri"]
            model_version = log_data["model_version"]
            print(f"✅ Model logged: {model_uri}")
            
            # Step 4: Register model (skip - already handled by agents.log_model)
            print("✅ Model already registered during logging step")
            # Use the model_version from the previous step
            
            # Step 5: Deploy endpoint
            print("🎯 Deploying serving endpoint...")
            
            # Extract configuration values to avoid f-string issues  
            uc_model_name = self.config["UC_MODEL_NAME"]
            endpoint_name = self.config["endpoint"]["name"]
            workload_size = self.config["endpoint"]["workload_size"]
            
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
    environment = "{}"
    
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
        scale_to_zero_enabled=(environment != "prod"),
        
        # Environment configuration
        environment_vars={{
            "TAVILY_API_KEY": "{{{{secrets/msh/TAVILY_API_KEY}}}}",
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
                workload_size,
                self.config.get("ENVIRONMENT", "dev")
            )
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
                print(f"❌ Could not find JSON result in endpoint deployment output")
                print(f"Full output: {deploy_result.output}")
                return False
            
            try:
                deploy_data = json.loads(json_line)
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse endpoint deployment JSON: {e}")
                print(f"JSON line: {json_line}")
                return False
            
            if not deploy_data.get("success"):
                print(f"❌ Endpoint deployment failed: {deploy_data.get('error')}")
                return False
            
            print(f"✅ Endpoint deployed: {deploy_data['endpoint_name']}")
            
            # Step 6: Wait for endpoint
            print("⏳ Waiting for endpoint to be ready...")
            endpoint_manager = self.get_endpoint_manager()
            ready_timeout = self.config.get("endpoint", {}).get("ready_timeout", 900)  # Default 15 minutes
            check_interval = self.config.get("endpoint", {}).get("check_interval", 30)  # Default 30 seconds
            
            print(f"   Timeout: {ready_timeout} seconds ({ready_timeout/60:.1f} minutes)")
            print(f"   Check interval: every {check_interval} seconds")
            print(f"   Progress updates: every {check_interval * 5} seconds")
            
            wait_result = endpoint_manager.wait_for_endpoint_ready(
                self.config["endpoint"]["name"],
                timeout=ready_timeout,
                check_interval=check_interval
            )
            
            if wait_result.get("success"):
                print(f"✅ Endpoint ready after {wait_result.get('wait_time', 0):.1f}s")
                return True
            else:
                print(f"⚠️  Endpoint may not be ready after {ready_timeout}s: {wait_result.get('error', 'Unknown')}")
                
                # Show current endpoint status
                final_status = wait_result.get('final_status', {})
                if final_status:
                    print(f"   Current state: {final_status.get('config_update', 'unknown')}")
                    print(f"   Ready: {final_status.get('ready', False)}")
                
                print("   Proceeding with validation anyway (it may still succeed)...")
                return True
                
        except Exception as e:
            print(ErrorReporter.report_deployment_failure("model_deployment", {"error": str(e)}))
            return False
    
    def validate_deployment(self) -> bool:
        """Validate the deployed endpoint."""
        print("\\n🧪 DEPLOYMENT VALIDATION")
        print("-" * 30)
        
        try:
            validator = self.get_validator()
            endpoint_name = self.config["endpoint"]["name"]
            endpoint_manager = self.get_endpoint_manager()
            
            if self.args_config.get("dry_run"):
                print(f"   (DRY RUN: would validate endpoint '{endpoint_name}')")
                return True
            
            # First do a health check
            print("🏥 Running endpoint health check...")
            health_status = validator.quick_health_check(endpoint_name)
            
            if not health_status.get('endpoint_exists'):
                print(f"❌ Endpoint '{endpoint_name}' does not exist")
                return False
            
            if not health_status.get('state_ready'):
                print(f"⚠️  Endpoint exists but not ready. State: {health_status.get('config_update', 'unknown')}")
                print("   Proceeding with validation anyway...")
            else:
                print(f"✅ Endpoint is ready")
            
            # Run validation with retry logic
            max_retries = self.config.get("endpoint", {}).get("validation_retries", 2)
            retry_delay = self.config.get("endpoint", {}).get("validation_retry_delay", 60)
            
            for attempt in range(max_retries):
                if attempt > 0:
                    print(f"\\n🔄 Validation retry {attempt}/{max_retries - 1}")
                    print(f"⏳ Waiting {retry_delay} seconds before retry...")
                    import time
                    time.sleep(retry_delay)
                
                # Run validation tests
                validation_result = validator.validate_endpoint(
                    endpoint_name,
                    self.config,
                    timeout=self.args_config.get("validation_timeout", 300)
                )
                
                # Always show test results
                print(f"\\n📊 Validation Results (Attempt {attempt + 1}/{max_retries}):")
                print(f"   Tests: {validation_result.tests_passed}/{validation_result.tests_total} passed")
                
                if validation_result.results:
                    for result in validation_result.results:
                        test_desc = result.get('description', '')
                        if test_desc:
                            test_desc = f" ({test_desc})"
                        
                        if result.get("success"):
                            print(f"   ✓ {result.get('name', 'Test')}{test_desc}: {result.get('response_time', 0):.1f}s")
                        else:
                            error_msg = result.get('error', 'Failed')[:100]  # Truncate long errors
                            print(f"   ✗ {result.get('name', 'Test')}{test_desc}: {error_msg}")
                
                if validation_result.success:
                    print(f"\\n✅ Validation passed!")
                    return True
                
                # Check if we should retry
                if attempt < max_retries - 1:
                    # Check if failure might be due to endpoint not ready
                    timeout_errors = [r for r in validation_result.results 
                                    if 'timeout' in str(r.get('error', '')).lower() or 
                                       'timed out' in str(r.get('error', '')).lower()]
                    
                    if timeout_errors or validation_result.tests_passed == 0:
                        print(f"\\n⚠️  Validation failed (possibly due to endpoint initialization)")
                        print(f"   Error: {validation_result.error}")
                        continue  # Try again
                    else:
                        # Some tests passed, likely a real failure
                        break
            
            # All retries exhausted or decided not to retry
            print(f"\\n❌ Validation failed after {attempt + 1} attempt(s)")
            print(f"   Final error: {validation_result.error}")
            
            # Show detailed failure report
            if validation_result.results:
                failed_tests = [r for r in validation_result.results if not r.get('success', False)]
                if failed_tests:
                    print("\\n📋 Failed Test Details:")
                    for test in failed_tests:
                        print(f"   • {test.get('name', 'Test')}: {test.get('error', 'Unknown error')}")
            
            print(ErrorReporter.report_deployment_failure("endpoint_validation", {
                "error": validation_result.error,
                "results": validation_result.results,
                "attempts": attempt + 1
            }))
            return False
                
        except Exception as e:
            print(ErrorReporter.report_deployment_failure("endpoint_validation", {"error": str(e)}))
            return False
    
    def print_success_summary(self) -> None:
        """Print deployment success summary."""
        duration = time.time() - self.start_time
        endpoint_manager = self.get_endpoint_manager()
        status = endpoint_manager.get_endpoint_status(self.config["endpoint"]["name"])
        
        # Create summary table
        summary_table = Table.grid()
        summary_table.add_row("Environment:", Text(self.config['ENVIRONMENT'], style="bold cyan"))
        summary_table.add_row("Model:", Text(self.config['UC_MODEL_NAME'], style="bold yellow"))
        summary_table.add_row("Endpoint:", Text(self.config['endpoint']['name'], style="bold green"))
        summary_table.add_row("Duration:", Text(f"{duration:.1f}s", style="bold magenta"))
        
        console.print(Panel(summary_table, title="🎉 Deployment Successful!", style="bold green"))
        
        # Show endpoint URLs if available
        if status.get("serving_url"):
            url_table = Table.grid()
            url_table.add_row("Serving:", Text(status['serving_url'], style="link"))
            url_table.add_row("Management:", Text(status.get('management_url', 'N/A'), style="link"))
            console.print(Panel(url_table, title="🔗 Endpoint URLs", style="blue"))
        
        # Next steps
        steps_text = Text()
        steps_text.append("1. Test the endpoint in Databricks UI\n", style="dim")
        steps_text.append("2. Monitor endpoint performance and usage\n", style="dim")
        steps_text.append("3. Update applications to use the new endpoint", style="dim")
        console.print(Panel(steps_text, title="💡 Next Steps", style="yellow"))
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.executor:
            try:
                self.executor.cleanup()
            except Exception as e:
                logger.warning(f"Failed to cleanup executor: {e}")


def main():
    """Main entry point for deployment."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Configure logging
        configure_logging(args)
        
        # Get deployment configuration
        deployment_config = get_deployment_config(args)
        
        # Print configuration summary (unless quiet)
        if not deployment_config.get("quiet"):
            print_configuration_summary(deployment_config)
        
        # Initialize orchestrator
        orchestrator = DeploymentOrchestrator(deployment_config)
        
        try:
            # Handle management operations
            if orchestrator.run_management_operations():
                return 0
            
            # Execute deployment pipeline
            success = orchestrator.execute_deployment_pipeline()
            return 0 if success else 1
            
        finally:
            orchestrator.cleanup()
            
    except KeyboardInterrupt:
        print("\\n⏹️  Deployment interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        print(f"\\n❌ Deployment failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())