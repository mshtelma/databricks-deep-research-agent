"""
Endpoint lifecycle management for Databricks serving endpoints.
"""

import time
import logging
from typing import Dict, Any, Optional, List

from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config

logger = logging.getLogger(__name__)


class EndpointError(Exception):
    """Exception raised when endpoint operations fail."""
    pass


class EndpointManager:
    """Manage Databricks serving endpoint lifecycle operations."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize endpoint manager with configuration."""
        self.config = config
        self.workspace_client = self._create_workspace_client()
        
        logger.info("EndpointManager initialized")
        logger.info(f"Workspace: {self.workspace_client.config.host}")
    
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
    
    def endpoint_exists(self, endpoint_name: str) -> bool:
        """
        Check if an endpoint exists.
        
        Args:
            endpoint_name: Name of the endpoint to check
            
        Returns:
            True if endpoint exists, False otherwise
        """
        try:
            self.workspace_client.serving_endpoints.get(endpoint_name)
            return True
        except Exception as e:
            logger.debug(f"Endpoint {endpoint_name} does not exist or is not accessible: {e}")
            return False
    
    def get_endpoint_status(self, endpoint_name: str) -> Dict[str, Any]:
        """
        Get detailed status information for an endpoint.
        
        Args:
            endpoint_name: Name of the endpoint
            
        Returns:
            Dictionary with endpoint status information
        """
        try:
            endpoint = self.workspace_client.serving_endpoints.get(endpoint_name)
            
            # Convert complex objects to JSON-serializable format
            ready_state = getattr(endpoint.state, 'ready', None) if endpoint.state else None
            config_update = getattr(endpoint.state, 'config_update', None) if endpoint.state else None
            
            status_info = {
                "exists": True,
                "name": endpoint.name,
                "ready": bool(ready_state) if ready_state is not None else False,
                "config_update": str(config_update) if config_update is not None else None,
                "creation_timestamp": getattr(endpoint, 'creation_timestamp', None),
                "creator": getattr(endpoint, 'creator', None),
                "last_updated_timestamp": getattr(endpoint, 'last_updated_timestamp', None)
            }
            
            # Add endpoint URLs
            base_url = self.workspace_client.config.host
            status_info["management_url"] = f"{base_url}/ml/endpoints/{endpoint_name}"
            status_info["serving_url"] = f"{base_url}/serving-endpoints/{endpoint_name}/invocations"
            
            return status_info
            
        except Exception as e:
            return {
                "exists": False,
                "name": endpoint_name,
                "error": str(e)
            }
    
    def delete_endpoint(self, endpoint_name: str, force: bool = False) -> bool:
        """
        Delete an existing endpoint.
        
        Args:
            endpoint_name: Name of the endpoint to delete
            force: If True, skip confirmation prompt
            
        Returns:
            True if deleted successfully, False otherwise
        """
        if not self.endpoint_exists(endpoint_name):
            logger.info(f"Endpoint {endpoint_name} does not exist, nothing to delete")
            return True
        
        if not force:
            response = input(f"Delete endpoint '{endpoint_name}'? [y/N]: ")
            if response.lower() not in ['y', 'yes']:
                logger.info("Endpoint deletion cancelled by user")
                return False
        
        try:
            logger.info(f"🗑️  Deleting endpoint: {endpoint_name}")
            
            self.workspace_client.serving_endpoints.delete(endpoint_name)
            
            logger.info(f"Deletion request sent for {endpoint_name}")
            
            # Wait for deletion to complete
            deletion_success = self._wait_for_deletion(endpoint_name)
            
            if deletion_success:
                logger.info(f"✅ Endpoint {endpoint_name} deleted successfully")
                return True
            else:
                logger.warning(f"⚠️  Endpoint deletion may not have completed within timeout")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to delete endpoint {endpoint_name}: {e}")
            raise EndpointError(f"Failed to delete endpoint: {e}")
    
    def _wait_for_deletion(self, endpoint_name: str, timeout: int = 300) -> bool:
        """
        Wait for endpoint to be fully deleted.
        
        Args:
            endpoint_name: Name of the endpoint
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if endpoint is deleted, False if timeout
        """
        start_time = time.time()
        check_interval = 5  # Check every 5 seconds
        
        logger.info(f"⏳ Waiting for endpoint deletion: {endpoint_name}")
        
        while time.time() - start_time < timeout:
            if not self.endpoint_exists(endpoint_name):
                elapsed = time.time() - start_time
                logger.info(f"✅ Endpoint deletion confirmed after {elapsed:.1f}s")
                return True
            
            time.sleep(check_interval)
            elapsed = time.time() - start_time
            logger.debug(f"Still waiting for deletion... ({elapsed:.1f}s elapsed)")
        
        logger.warning(f"⚠️  Deletion timeout after {timeout}s")
        return False
    
    def list_endpoints(self, filter_pattern: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all serving endpoints, optionally filtered.
        
        Args:
            filter_pattern: Optional pattern to filter endpoint names
            
        Returns:
            List of endpoint information dictionaries
        """
        try:
            endpoints = list(self.workspace_client.serving_endpoints.list())
            
            endpoint_list = []
            for endpoint in endpoints:
                if filter_pattern and filter_pattern not in endpoint.name:
                    continue
                
                endpoint_info = {
                    "name": endpoint.name,
                    "ready": getattr(endpoint.state, 'ready', False) if endpoint.state else False,
                    "creator": getattr(endpoint, 'creator', None),
                    "creation_timestamp": getattr(endpoint, 'creation_timestamp', None),
                    "last_updated_timestamp": getattr(endpoint, 'last_updated_timestamp', None)
                }
                
                endpoint_list.append(endpoint_info)
            
            return endpoint_list
            
        except Exception as e:
            logger.error(f"Failed to list endpoints: {e}")
            return []
    
    def cleanup_old_endpoints(self, pattern: str, max_age_hours: int = 24, 
                             dry_run: bool = True) -> Dict[str, Any]:
        """
        Clean up old endpoints matching a pattern.
        
        Args:
            pattern: Pattern to match endpoint names (e.g., "test-", "dev-")
            max_age_hours: Maximum age in hours before cleanup
            dry_run: If True, only show what would be deleted
            
        Returns:
            Summary of cleanup operations
        """
        logger.info(f"🧹 Cleanup scan for endpoints matching '{pattern}' older than {max_age_hours}h")
        
        try:
            endpoints = self.list_endpoints(filter_pattern=pattern)
            current_time = time.time()
            cleanup_candidates = []
            
            for endpoint in endpoints:
                if endpoint.get('creation_timestamp'):
                    # Convert timestamp to seconds if needed
                    creation_time = endpoint['creation_timestamp']
                    if creation_time > 1e12:  # Likely in milliseconds
                        creation_time = creation_time / 1000
                    
                    age_hours = (current_time - creation_time) / 3600
                    
                    if age_hours > max_age_hours:
                        cleanup_candidates.append({
                            "name": endpoint["name"],
                            "age_hours": age_hours,
                            "ready": endpoint["ready"]
                        })
            
            if dry_run:
                logger.info(f"DRY RUN: Found {len(cleanup_candidates)} endpoints for cleanup")
                for candidate in cleanup_candidates:
                    logger.info(f"  - {candidate['name']} (age: {candidate['age_hours']:.1f}h)")
                
                return {
                    "dry_run": True,
                    "candidates_found": len(cleanup_candidates),
                    "candidates": cleanup_candidates,
                    "pattern": pattern,
                    "max_age_hours": max_age_hours
                }
            
            # Actual cleanup
            deleted_count = 0
            failed_deletions = []
            
            for candidate in cleanup_candidates:
                try:
                    if self.delete_endpoint(candidate["name"], force=True):
                        deleted_count += 1
                        logger.info(f"✅ Deleted: {candidate['name']}")
                    else:
                        failed_deletions.append(candidate["name"])
                        logger.warning(f"⚠️  Failed to delete: {candidate['name']}")
                except Exception as e:
                    failed_deletions.append(candidate["name"])
                    logger.error(f"❌ Error deleting {candidate['name']}: {e}")
            
            return {
                "dry_run": False,
                "candidates_found": len(cleanup_candidates),
                "deleted_count": deleted_count,
                "failed_deletions": failed_deletions,
                "pattern": pattern,
                "max_age_hours": max_age_hours
            }
            
        except Exception as e:
            logger.error(f"Cleanup operation failed: {e}")
            return {
                "error": str(e),
                "pattern": pattern,
                "max_age_hours": max_age_hours
            }
    
    def wait_for_endpoint_ready(self, endpoint_name: str, timeout: int = 600,
                               check_interval: int = 30) -> Dict[str, Any]:
        """
        Wait for an endpoint to become ready.
        
        Args:
            endpoint_name: Name of the endpoint to wait for
            timeout: Maximum time to wait in seconds
            check_interval: Time between status checks in seconds
            
        Returns:
            Final status information
        """
        logger.info(f"⏳ Waiting for endpoint to be ready: {endpoint_name}")
        logger.info(f"   Timeout: {timeout}s, checking every {check_interval}s")
        
        start_time = time.time()
        last_status = None
        status_changes = []
        check_count = 0
        
        while time.time() - start_time < timeout:
            try:
                status = self.get_endpoint_status(endpoint_name)
                check_count += 1
                
                # Track status changes
                current_status = status.get('config_update')
                if last_status != current_status:
                    elapsed = time.time() - start_time
                    status_changes.append({
                        "timestamp": time.time(),
                        "elapsed": elapsed,
                        "status": current_status,
                        "ready": status.get('ready', False)
                    })
                    # Only log actual status changes
                    logger.info(f"   Status change after {elapsed:.1f}s: {last_status} → {current_status}")
                    last_status = current_status
                
                if status.get('ready', False):
                    elapsed = time.time() - start_time
                    logger.info(f"✅ Endpoint ready after {elapsed:.1f}s ({check_count} status checks)")
                    
                    return {
                        "success": True,
                        "endpoint_name": endpoint_name,
                        "ready": True,
                        "wait_time": elapsed,
                        "status_checks": check_count,
                        "status_changes": status_changes,
                        "final_status": status
                    }
                
                # Don't log every check, it causes blinking
                # Only show progress every 5th check (every 2.5 minutes if check_interval=30)
                if check_count % 5 == 0:
                    elapsed = time.time() - start_time
                    logger.info(f"   Still waiting... ({elapsed:.0f}s elapsed, {check_count} checks, status: {current_status or 'unknown'})")
                
                time.sleep(check_interval)
                
            except Exception as e:
                logger.warning(f"Error checking endpoint status: {e}")
                time.sleep(check_interval)
        
        # Timeout reached
        elapsed = time.time() - start_time
        logger.error(f"❌ Endpoint not ready after {elapsed:.1f}s timeout")
        
        # Get final status
        try:
            final_status = self.get_endpoint_status(endpoint_name)
        except Exception as e:
            final_status = {"error": str(e)}
        
        return {
            "success": False,
            "endpoint_name": endpoint_name,
            "ready": False,
            "timeout": True,
            "wait_time": elapsed,
            "status_checks": len(status_changes),
            "status_changes": status_changes,
            "final_status": final_status,
            "error": f"Endpoint not ready after {timeout} seconds"
        }
    
    def get_endpoint_logs(self, endpoint_name: str, lines: int = 100) -> Dict[str, Any]:
        """
        Get recent logs from an endpoint (if available).
        
        Args:
            endpoint_name: Name of the endpoint
            lines: Number of log lines to retrieve
            
        Returns:
            Log information
        """
        # Note: This is a placeholder - actual log retrieval would depend on 
        # Databricks SDK capabilities and endpoint configuration
        logger.info(f"📋 Getting logs for endpoint: {endpoint_name}")
        
        try:
            status = self.get_endpoint_status(endpoint_name)
            
            if not status.get('exists'):
                return {
                    "success": False,
                    "error": f"Endpoint {endpoint_name} does not exist"
                }
            
            # Placeholder for actual log retrieval
            # In practice, you might use workspace client to get logs from specific locations
            # or use other Databricks APIs
            
            return {
                "success": True,
                "endpoint_name": endpoint_name,
                "logs": f"Log retrieval not implemented - check endpoint at {status.get('management_url')}",
                "management_url": status.get('management_url'),
                "note": "Use Databricks UI to view detailed endpoint logs"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "endpoint_name": endpoint_name
            }