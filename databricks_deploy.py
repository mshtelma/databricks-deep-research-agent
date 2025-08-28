#!/usr/bin/env python3
"""
Simplified Databricks Deployment Script using CLI for file sync
"""

import os
import sys
import yaml
import json
import time
import argparse
import logging
import subprocess
import traceback
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Databricks SDK imports
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config
from databricks.sdk.service import compute
from databricks.sdk.service.jobs import Task, NotebookTask, Source, JobSettings, RunLifeCycleState
from databricks.sdk.service.workspace import ImportFormat, Language

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def strip_ansi_codes(text: str) -> str:
    """Remove ANSI color codes from text."""
    import re
    ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
    return ansi_escape.sub('', text)



class DatabricksDeployment:
    """Simplified deployment handler using databricks CLI for sync."""
    
    def __init__(self, environment: str = "dev", config_path: str = "deploy_config.yaml"):
        self.environment = environment
        self.config_path = config_path
        
        # Load configuration
        self.config = self.load_config()
        self.env_config = self.config["environments"][environment]
        
        # Initialize workspace client
        self.workspace_client = self.create_workspace_client()
        
        # Set up paths
        self.local_source_path = Path(self.config["global"]["local_source_path"])
        self.workspace_base_path = self.env_config["workspace_path"]
        
        # Get profile for CLI
        self.profile = self.env_config.get("profile", "DEFAULT")
        
        logger.info(f"Initialized deployment for environment: {environment}")
        logger.info(f"Workspace: {self.workspace_client.config.host}")
        logger.info(f"Local source: {self.local_source_path}")
        logger.info(f"Workspace path: {self.workspace_base_path}")
    
    def load_config(self) -> Dict[str, Any]:
        """Load deployment configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            # Validate configuration
            required_sections = ["global", "environments"]
            for section in required_sections:
                if section not in config:
                    raise ValueError(f"Missing required section '{section}' in config")
            
            if self.environment not in config["environments"]:
                available = list(config["environments"].keys())
                raise ValueError(f"Environment '{self.environment}' not found. Available: {available}")
            
            return config
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")
    
    def create_workspace_client(self) -> WorkspaceClient:
        """Create Databricks workspace client with profile support."""
        try:
            profile = self.env_config.get("profile")
            if profile:
                logger.info(f"Using Databricks profile: {profile}")
                config = Config(profile=profile)
                return WorkspaceClient(config=config)
            else:
                logger.info("Using environment variable authentication")
                return WorkspaceClient()
                
        except Exception as e:
            logger.error(f"Failed to create workspace client: {e}")
            if profile:
                logger.error(f"Make sure profile '{profile}' is configured:")
                logger.error(f"  databricks configure --profile {profile}")
            raise
    
    def sync_to_workspace(self, dry_run: bool = False) -> bool:
        """
        Sync local source code to Databricks workspace using databricks CLI.
        Uses databricks sync which handles everything including:
        - Delta processing (only syncs changed files)
        - .gitignore support
        - Directory structure preservation
        - Notebook and Python file handling
        """
        logger.info("Syncing project to Databricks workspace using databricks sync...")
        
        if dry_run:
            logger.info("DRY RUN MODE - would sync files")
            return True
        
        try:
            # Ensure workspace directory exists
            try:
                self.workspace_client.workspace.get_status(self.workspace_base_path)
                logger.info(f"Workspace directory exists: {self.workspace_base_path}")
            except Exception:
                logger.info(f"Creating workspace directory: {self.workspace_base_path}")
                self.workspace_client.workspace.mkdirs(self.workspace_base_path)
            
            # Step 1: Sync the deep_research_agent directory
            source_path = str(self.local_source_path)
            
            # Create subdirectory for the agent code
            agent_workspace_path = f"{self.workspace_base_path}/deep_research_agent"
            
            # Use databricks sync to transfer the agent code
            cmd = [
                "databricks", "sync",
                "--profile", self.profile,
                "--full",  # Full sync to ensure consistency
                source_path,
                agent_workspace_path
            ]
            
            logger.info(f"Syncing agent code: {source_path} → {agent_workspace_path}")
            logger.info(f"Command: {' '.join(cmd)}")
            
            # Execute the sync command for agent code
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                error_msg = result.stderr or result.stdout or "Unknown error"
                logger.error(f"✗ Failed to sync agent code")
                logger.error(f"Error: {error_msg}")
                return False
            
            logger.info("✓ Successfully synced agent code")
            
            # Step 2: Upload log_and_deploy.py separately
            log_deploy_path = Path("log_and_deploy.py")
            if not log_deploy_path.exists():
                logger.error(f"✗ log_and_deploy.py not found at {log_deploy_path.absolute()}")
                return False
            
            logger.info(f"Uploading log_and_deploy.py to workspace...")
            
            # Read the file content
            with open(log_deploy_path, 'r') as f:
                content = f.read()
            
            # Upload as a notebook (Databricks will recognize .py files)
            workspace_notebook_path = f"{self.workspace_base_path}/log_and_deploy"
            
            try:
                # Try to delete existing file first (to ensure clean upload)
                try:
                    self.workspace_client.workspace.delete(workspace_notebook_path)
                    logger.debug(f"Deleted existing notebook at {workspace_notebook_path}")
                except:
                    pass  # File might not exist, that's okay
                
                # Upload the notebook
                self.workspace_client.workspace.upload(
                    workspace_notebook_path,
                    content.encode('utf-8'),
                    format=ImportFormat.SOURCE,
                    language=Language.PYTHON,
                    overwrite=True
                )
                logger.info(f"✓ Successfully uploaded log_and_deploy.py to {workspace_notebook_path}")
                
            except Exception as e:
                logger.error(f"✗ Failed to upload log_and_deploy.py: {e}")
                return False
            
            logger.info("✓ Successfully synced all files to workspace")
            return True
            
        except Exception as e:
            logger.error(f"Sync failed with exception: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False
    
    def create_deployment_job(self, cluster_id: str = None) -> Optional[str]:
        """
        Create or update the deployment job.
        Compute is determined by configuration:
        - If cluster_id parameter -> use that existing cluster (override)
        - If existing_cluster_id in config -> use that existing cluster
        - If new_cluster in config -> create new cluster
        - If neither -> use serverless (default)
        """
        job_name = self.env_config["job"]["name"]
        job_config = self.env_config.get("job", {})
        
        logger.info(f"Creating/updating deployment job: {job_name}")
        
        # Define the notebook task
        # The notebook will be at log_and_deploy after syncing (no .py extension in Databricks)
        
        # Simplified base parameters - YAML configuration will be read directly by the notebook
        base_params = {
            "CATALOG": self.env_config["model"]["catalog"],
            "SCHEMA": self.env_config["model"]["schema"],
            "MODEL_NAME": self.env_config["model"]["name"],
            "ENDPOINT_NAME": self.env_config["endpoint"]["name"],
            "ENVIRONMENT": self.environment,
            "WORKLOAD_SIZE": self.env_config["endpoint"]["workload_size"]
        }
        
        notebook_task = NotebookTask(
            notebook_path=f"{self.workspace_base_path}/log_and_deploy",
            source=Source.WORKSPACE,
            base_parameters=base_params
        )
        
        # Configure task
        task_config = {
            "task_key": "deploy_agent",
            "description": f"Deploy LangGraph Research Agent - {self.environment}",
            "notebook_task": notebook_task,
            "timeout_seconds": job_config.get("timeout_seconds", 3600),
            "max_retries": job_config.get("max_retries", 1),
            "min_retry_interval_millis": 60000
        }
        
        # Determine and configure compute
        if cluster_id:
            # Command-line override
            task_config["existing_cluster_id"] = cluster_id
            logger.info(f"Using existing cluster (override): {cluster_id}")
            
        elif "existing_cluster_id" in job_config:
            # Use existing cluster from config
            task_config["existing_cluster_id"] = job_config["existing_cluster_id"]
            logger.info(f"Using existing cluster: {job_config['existing_cluster_id']}")
            
        elif "new_cluster" in job_config:
            # Create new cluster from config
            cluster_spec = job_config["new_cluster"]
            task_config["new_cluster"] = compute.ClusterSpec(
                spark_version=cluster_spec.get("spark_version", "13.3.x-scala2.12"),
                node_type_id=cluster_spec.get("node_type_id", "i3.xlarge"),
                num_workers=cluster_spec.get("num_workers", 0),
                spark_env_vars=cluster_spec.get("spark_env_vars", {})
            )
            logger.info(f"Creating new cluster: {cluster_spec.get('node_type_id')}")
            
        else:
            # Default: serverless (no additional config needed)
            logger.info("Using serverless compute (default)")
        
        # Define job configuration  
        job_settings = {
            "name": job_name,
            "tasks": [Task(**task_config)],
            "max_concurrent_runs": 1,
            "tags": {
                "project": "langgraph-agent",
                "environment": self.environment,
                "deployment_time": datetime.now().isoformat()
            }
        }
        
        # Add email notifications if configured
        email_notifications = job_config.get("email_notifications", [])
        if email_notifications:
            job_settings["email_notifications"] = {
                "on_success": email_notifications,
                "on_failure": email_notifications,
                "no_alert_for_skipped_runs": True
            }
        
        try:
            # Check if job already exists
            existing_jobs = list(self.workspace_client.jobs.list(name=job_name))
            
            job_id = None
            for job in existing_jobs:
                if job.settings and job.settings.name == job_name:
                    # Update existing job
                    logger.info(f"Updating existing job: {job.job_id}")
                    self.workspace_client.jobs.reset(
                        job_id=job.job_id,
                        new_settings=JobSettings(**job_settings)
                    )
                    job_id = str(job.job_id)
                    break
            
            if not job_id:
                # Create new job
                logger.info("Creating new job")
                created_job = self.workspace_client.jobs.create(**job_settings)
                job_id = str(created_job.job_id)
                logger.info(f"Created job with ID: {job_id}")
            
            logger.info(f"Job URL: {self.workspace_client.config.host}/#job/{job_id}")
            return job_id
            
        except Exception as e:
            logger.error(f"Failed to create/update job: {e}")
            return None
    
    def get_run_output(self, run_id: int) -> Dict[str, Any]:
        """Get basic run output information for deployment monitoring."""
        try:
            run_info = self.workspace_client.jobs.get_run(run_id=run_id)
            
            result = {
                "run_id": run_id,
                "state": run_info.state.life_cycle_state,
                "result_state": getattr(run_info.state, 'result_state', None),
                "state_message": getattr(run_info.state, 'state_message', None),
                "job_id": getattr(run_info, 'job_id', None),
                "run_url": f"{self.workspace_client.config.host}/#job/{getattr(run_info, 'job_id', '')}/run/{run_id}",
                "success": False
            }
            
            # Check if deployment succeeded
            if result["result_state"] == "SUCCESS":
                result["success"] = True
            elif hasattr(run_info, 'tasks') and run_info.tasks:
                # Check task-level success for multi-task jobs
                successful_tasks = [
                    task for task in run_info.tasks 
                    if task.state and getattr(task.state, 'result_state', None) == "SUCCESS"
                ]
                result["success"] = len(successful_tasks) > 0
                result["task_count"] = len(run_info.tasks)
                result["successful_tasks"] = len(successful_tasks)
            
            # Get basic error information if available
            if not result["success"]:
                try:
                    run_output = self.workspace_client.jobs.get_run_output(run_id=run_id)
                    if run_output and hasattr(run_output, 'error') and run_output.error:
                        result["error"] = strip_ansi_codes(str(run_output.error))
                except Exception:
                    pass  # Error details not critical for deployment validation
            
            return result
            
        except Exception as e:
            return {
                "run_id": run_id,
                "error": f"Failed to fetch run information: {str(e)}",
                "state": "ERROR",
                "success": False
            }
    
    
    
    
    
    
    
    
    def run_deployment_job(self, job_id: str, wait: bool = True, detailed: bool = False) -> bool:
        """
        Run the deployment job and show error information on failure.
        
        Args:
            job_id: The job ID to run
            wait: Whether to wait for completion (default: True)
            detailed: Whether to show full JSON output on failure (default: False)
        
        On failure, prints error information and optionally full JSON output.
        """
        logger.info(f"Starting deployment job: {job_id}")
        
        try:
            # Trigger the job
            run = self.workspace_client.jobs.run_now(job_id=int(job_id))
            run_id = run.run_id
            
            logger.info(f"Job run started with ID: {run_id}")
            logger.info(f"Monitor run: {self.workspace_client.config.host}/#job/{job_id}/run/{run_id}")
            
            if not wait:
                logger.info("Job started. Use --wait to monitor completion.")
                return True
            
            logger.info("Monitoring job execution...")
            
            # Simple monitoring loop
            while True:
                run_info = self.workspace_client.jobs.get_run(run_id=run_id)
                current_state = run_info.state.life_cycle_state
                
                logger.info(f"Job state: {current_state}")
                
                # Check if job completed
                if current_state in [RunLifeCycleState.TERMINATED, RunLifeCycleState.INTERNAL_ERROR]:
                    # Get basic run output
                    output = self.get_run_output(run_id)
                    
                    if output.get("success"):
                        # SUCCESS: Show summary
                        print("\n✅ JOB SUCCEEDED")
                        print(f"📝 Result state: {output.get('result_state')}")
                        
                        if output.get('task_count'):
                            print(f"📋 Tasks: {output.get('successful_tasks', 0)}/{output.get('task_count', 0)} successful")
                        
                        if output.get('state_message'):
                            print(f"💬 {output['state_message']}")
                        
                        print(f"🔗 View details: {output.get('run_url', 'N/A')}")
                        return True
                    else:
                        # FAILURE: Show error information
                        print("\n❌ JOB FAILED")
                        print(f"📝 Result state: {output.get('result_state')}")
                        
                        if output.get('error'):
                            print(f"🔥 Error: {output['error']}")
                        
                        if output.get('state_message'):
                            print(f"💬 {output['state_message']}")
                        
                        if output.get('task_count'):
                            print(f"📋 Tasks: {output.get('successful_tasks', 0)}/{output.get('task_count', 0)} successful")
                        
                        print(f"🔗 View details: {output.get('run_url', 'N/A')}")
                        
                        if detailed:
                            print("\n" + "="*40)
                            print("FULL OUTPUT:")
                            print("="*40)
                            print(json.dumps(output, indent=2, default=str))
                        
                        return False
                
                time.sleep(10)  # Simple 10-second polling
                
        except Exception as e:
            print(f"\n❌ JOB EXECUTION ERROR:")
            print(f"Error: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            return False
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """Get the status of the deployed agent endpoint."""
        endpoint_name = self.env_config["endpoint"]["name"]
        
        try:
            endpoints = list(self.workspace_client.serving_endpoints.list())
            
            for endpoint in endpoints:
                if endpoint.name == endpoint_name:
                    return {
                        "exists": True,
                        "name": endpoint.name,
                        "state": endpoint.state.ready if endpoint.state else "unknown",
                        "url": f"{self.workspace_client.config.host}/ml/endpoints/{endpoint.name}",
                        "config_update": endpoint.state.config_update if endpoint.state else None
                    }
            
            return {"exists": False, "name": endpoint_name}
            
        except Exception as e:
            logger.error(f"Failed to get deployment status: {e}")
            return {"error": str(e)}
    
    def deploy(self, cluster_id: str = None, sync_only: bool = False, run_only: bool = False, 
               wait: bool = True, dry_run: bool = False, detailed: bool = False) -> bool:
        """
        Execute the complete deployment workflow with simplified flow.
        """
        print(f"🚀 DATABRICKS DEPLOYMENT - {self.environment.upper()}")
        if dry_run:
            print("DRY RUN MODE - No changes will be made")
        
        try:
            # Pre-deployment validation: Check required YAML config exists locally
            yaml_path = self.local_source_path / "agent_config.yaml"
            if not yaml_path.exists():
                error_msg = f"DEPLOYMENT BLOCKED: Required agent_config.yaml not found at {yaml_path}"
                logger.error(error_msg)
                print(f"❌ {error_msg}")
                return False
            else:
                print(f"✅ Required agent_config.yaml found at {yaml_path}")
                logger.info(f"Validated required config file exists: {yaml_path}")
            
            # Step 1: Sync files (unless run-only mode)
            if not run_only:
                print("Step 1: Syncing files to workspace...")
                if not self.sync_to_workspace(dry_run=dry_run):
                    print("❌ File sync failed!")
                    return False
                
                if sync_only:
                    print("✅ Sync complete.")
                    return True
            
            if dry_run:
                print(f"Step 2: Would create job '{self.env_config['job']['name']}'")
                print("✅ Dry run complete.")
                return True
            
            # Step 2: Create/update job
            print("Step 2: Creating/updating deployment job...")
            job_id = self.create_deployment_job(cluster_id=cluster_id)
            if not job_id:
                print("❌ Job creation failed!")
                return False
            
            # Step 3: Run job (with built-in error handling)
            print("Step 3: Running deployment job...")
            if not self.run_deployment_job(job_id, wait=wait, detailed=detailed):
                # Error already printed by run_deployment_job
                return False
            
            # Step 4: Check deployment status
            print("Step 4: Checking deployment status...")
            status = self.get_deployment_status()
            
            if status.get("exists"):
                print(f"✅ Endpoint deployed: {status['name']}")
                print(f"   State: {status.get('state', 'Unknown')}")
                print(f"   URL: {status.get('url', 'N/A')}")
            else:
                print(f"⚠️  Endpoint not found: {status.get('name', 'unknown')}")
            
            print("✅ DEPLOYMENT COMPLETE!")
            return True
            
        except Exception as e:
            # Any unexpected error: print and stop
            print(f"\n❌ UNEXPECTED DEPLOYMENT ERROR:")
            print(f"Error: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            return False


def main():
    """Main entry point for the deployment script."""
    parser = argparse.ArgumentParser(
        description="Simplified Databricks Deployment for LangGraph Research Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Deploy to dev (uses existing cluster 0816-235658-5i3k9jfh as configured)
  python databricks_deploy.py --env dev --run-now

  # Deploy with custom cluster ID override
  python databricks_deploy.py --env dev --cluster-id different-cluster-id --wait

  # Deploy to prod (creates new cluster as configured)
  python databricks_deploy.py --env prod --run-now

  # Sync files only (no job execution)
  python databricks_deploy.py --env dev --sync-only

  # Run existing job without syncing files
  python databricks_deploy.py --env prod --run-only

  # Dry run to see what would be done
  python databricks_deploy.py --env dev --dry-run
  
  # Deploy with full JSON output on failure
  python databricks_deploy.py --env dev --detailed
        """
    )
    
    parser.add_argument(
        "--env",
        choices=["dev", "staging", "prod", "test"],
        default="dev",
        help="Target environment (default: dev)"
    )
    parser.add_argument(
        "--cluster-id",
        help="Override existing cluster ID for deployment job"
    )
    parser.add_argument(
        "--config",
        default="deploy_config.yaml",
        help="Path to deployment configuration file (default: deploy_config.yaml)"
    )
    parser.add_argument(
        "--sync-only",
        action="store_true",
        help="Only sync files to workspace, don't create/run job"
    )
    parser.add_argument(
        "--run-only",
        action="store_true",
        help="Only run existing job, don't sync files"
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        default=True,
        help="Wait for job completion (default: True)"
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Don't wait for job completion"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing"
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Only check deployment status"
    )
    parser.add_argument(
        "--get-run-logs",
        type=int,
        help="Fetch and display logs for a specific run ID"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show full JSON output on job failure (default: show only error summary)"
    )
    
    args = parser.parse_args()
    
    # Handle wait flag logic
    wait = args.wait and not args.no_wait
    
    try:
        # Initialize deployment
        deployment = DatabricksDeployment(
            environment=args.env,
            config_path=args.config
        )
        
        if args.status_only:
            # Just check status
            status = deployment.get_deployment_status()
            print(json.dumps(status, indent=2))
            return
        
        if args.get_run_logs:
            # Fetch logs for a specific run
            print(f"Fetching logs for run ID: {args.get_run_logs}")
            output = deployment.get_run_output(args.get_run_logs)
            if output:
                print("Run Output:")
                print(json.dumps(output, indent=2, default=str))
            else:
                print(f"No output available for run ID: {args.get_run_logs}")
            return
        
        # Run deployment workflow
        success = deployment.deploy(
            cluster_id=args.cluster_id,
            sync_only=args.sync_only,
            run_only=args.run_only,
            wait=wait,
            dry_run=args.dry_run,
            detailed=args.detailed
        )
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logger.info("\n⏹️  Deployment cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Deployment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()