"""
Workspace management for file synchronization and workspace operations.
"""

import os
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config

# Import constants for config file paths
from deep_research_agent.constants import AGENT_CONFIG_PATH

logger = logging.getLogger(__name__)


class SyncError(Exception):
    """Exception raised when file sync operations fail."""
    pass


class WorkspaceManager:
    """Manage Databricks workspace operations including file sync and cleanup."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize workspace manager with configuration."""
        self.config = config
        self.workspace_client = self._create_workspace_client()
        self.local_source_path = Path(".")  # Sync entire project including tests
        self.workspace_base_path = config["workspace_path"]
        self.profile = config.get("profile", "DEFAULT")
        
        logger.info(f"WorkspaceManager initialized")
        logger.info(f"Local source: {self.local_source_path}")
        logger.info(f"Workspace path: {self.workspace_base_path}")
        logger.info(f"Profile: {self.profile}")
    
    def _create_workspace_client(self) -> WorkspaceClient:
        """Create workspace client with profile support."""
        profile = self.config.get("profile")
        if profile:
            logger.info(f"Using Databricks profile: {profile}")
            config = Config(profile=profile)
            return WorkspaceClient(config=config)
        else:
            logger.info("Using environment variable authentication")
            return WorkspaceClient()
    
    def sync_files(self, force_full_sync: bool = True) -> bool:
        """
        Sync local files to Databricks workspace using databricks CLI.
        
        Args:
            force_full_sync: Use --full flag for complete sync
            
        Returns:
            True if sync successful, False otherwise
        """
        logger.info("📤 Syncing files to Databricks workspace...")
        
        if not self.local_source_path.exists():
            raise SyncError(f"Local source path does not exist: {self.local_source_path}")
        
        try:
            # Ensure workspace directory exists
            self._ensure_workspace_dir()
            
            # Prepare sync command - sync entire project to workspace root
            workspace_target = self.workspace_base_path
            
            cmd = ["databricks", "sync"]
            
            # Add flags
            if force_full_sync:
                cmd.append("--full")
            
            cmd.extend([
                "--profile", self.profile,
                str(self.local_source_path),
                workspace_target
            ])
            
            logger.info(f"Sync command: {' '.join(cmd)}")
            logger.info(f"Syncing: {self.local_source_path} → {workspace_target}")
            
            # Execute sync
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                error_msg = result.stderr or result.stdout or "Unknown sync error"
                logger.error(f"Sync failed: {error_msg}")
                raise SyncError(f"Sync failed: {error_msg}")
            
            logger.info("✅ Files synced successfully")
            
            # Log some sync statistics if available in output
            if result.stdout:
                logger.info(f"Sync output: {result.stdout}")
            
            return True
            
        except subprocess.CalledProcessError as e:
            error_msg = f"Databricks CLI sync failed: {e}"
            logger.error(error_msg)
            raise SyncError(error_msg)
        except Exception as e:
            error_msg = f"Sync operation failed: {e}"
            logger.error(error_msg)
            raise SyncError(error_msg)
    
    def _ensure_workspace_dir(self):
        """Ensure workspace base directory exists."""
        try:
            # Check if directory exists
            self.workspace_client.workspace.get_status(self.workspace_base_path)
            logger.info(f"Workspace directory exists: {self.workspace_base_path}")
        except Exception:
            # Create directory
            logger.info(f"Creating workspace directory: {self.workspace_base_path}")
            self.workspace_client.workspace.mkdirs(self.workspace_base_path)
    
    def clean_deployment_folder(self, confirm: bool = False) -> bool:
        """
        Remove existing deployment folder in workspace.
        
        Args:
            confirm: If True, skip confirmation prompt
            
        Returns:
            True if cleaned successfully, False otherwise
        """
        logger.info(f"🧹 Cleaning workspace folder: {self.workspace_base_path}")
        
        try:
            # Check if folder exists
            try:
                self.workspace_client.workspace.get_status(self.workspace_base_path)
                folder_exists = True
            except Exception:
                folder_exists = False
                logger.info(f"Workspace folder doesn't exist: {self.workspace_base_path}")
                return True
            
            if folder_exists:
                if not confirm:
                    response = input(f"Delete {self.workspace_base_path}? [y/N]: ")
                    if response.lower() not in ['y', 'yes']:
                        logger.info("Workspace cleanup cancelled by user")
                        return False
                
                # Delete the folder
                self.workspace_client.workspace.delete(
                    self.workspace_base_path,
                    recursive=True
                )
                
                logger.info(f"✅ Workspace folder deleted: {self.workspace_base_path}")
                return True
            
        except Exception as e:
            logger.warning(f"⚠️  Could not clean workspace folder: {e}")
            # Don't fail the entire deployment for cleanup issues
            return False
    
    def upload_deployment_scripts(self) -> bool:
        """Upload deployment_steps.py to workspace for execution."""
        try:
            # Read deployment_steps.py
            deploy_script_path = Path(__file__).parent / "deployment_steps.py"
            
            if not deploy_script_path.exists():
                raise FileNotFoundError(f"deployment_steps.py not found at {deploy_script_path}")
            
            with open(deploy_script_path, 'r') as f:
                script_content = f.read()
            
            # Upload to workspace
            workspace_script_path = f"{self.workspace_base_path}/deployment_steps.py"
            
            self.workspace_client.workspace.upload(
                workspace_script_path,
                script_content.encode('utf-8'),
                overwrite=True
            )
            
            logger.info(f"✅ Uploaded deployment_steps.py to {workspace_script_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to upload deployment scripts: {e}")
            return False
    
    def list_workspace_contents(self, path: Optional[str] = None) -> Dict[str, Any]:
        """List contents of workspace directory for verification."""
        target_path = path or self.workspace_base_path
        
        try:
            contents = list(self.workspace_client.workspace.list(target_path))
            
            files = []
            directories = []
            
            for item in contents:
                if item.object_type.name == "DIRECTORY":
                    directories.append(item.path)
                else:
                    files.append({
                        "path": item.path,
                        "object_type": item.object_type.name,
                        "size": getattr(item, 'size', None)
                    })
            
            return {
                "path": target_path,
                "directories": directories,
                "files": files,
                "total_items": len(files) + len(directories)
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "path": target_path
            }
    
    def verify_sync(self) -> Dict[str, Any]:
        """Verify that key files were synced correctly."""
        verification_results = {}
        
        # Check for key files that should exist after sync
        expected_files = [
            f"{self.workspace_base_path}/deep_research_agent/__init__.py",
            f"{self.workspace_base_path}/conf/base.yaml",
            f"{self.workspace_base_path}/deep_research_agent/research_agent_refactored.py",
            f"{self.workspace_base_path}/deep_research_agent/databricks_compatible_agent.py",
            f"{self.workspace_base_path}/tests/__init__.py"
        ]
        
        for file_path in expected_files:
            try:
                self.workspace_client.workspace.get_status(file_path)
                verification_results[file_path] = {"exists": True}
            except Exception as e:
                verification_results[file_path] = {"exists": False, "error": str(e)}
        
        # Check directory structure
        try:
            contents = self.list_workspace_contents()
            verification_results["workspace_contents"] = contents
        except Exception as e:
            verification_results["workspace_contents"] = {"error": str(e)}
        
        # Summary
        existing_files = sum(1 for result in verification_results.values() 
                           if isinstance(result, dict) and result.get("exists"))
        total_checked = len(expected_files)
        
        verification_results["summary"] = {
            "files_found": existing_files,
            "files_checked": total_checked,
            "sync_success": existing_files == total_checked
        }
        
        return verification_results
    
    def get_workspace_info(self) -> Dict[str, Any]:
        """Get information about the workspace configuration."""
        try:
            return {
                "workspace_url": self.workspace_client.config.host,
                "workspace_path": self.workspace_base_path,
                "local_source_path": str(self.local_source_path),
                "profile": self.profile,
                "local_source_exists": self.local_source_path.exists(),
                "local_source_files": len(list(self.local_source_path.rglob("*.py"))) if self.local_source_path.exists() else 0
            }
        except Exception as e:
            return {
                "error": str(e),
                "workspace_path": self.workspace_base_path,
                "local_source_path": str(self.local_source_path)
            }