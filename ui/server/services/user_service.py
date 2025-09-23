"""User service for Databricks user operations."""

import logging
from typing import Optional

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.iam import User
from fastapi import Request

logger = logging.getLogger(__name__)


class UserService:
    """Service for managing Databricks user operations."""

    def __init__(self):
        """Initialize the user service with Databricks workspace client."""
        self.client = WorkspaceClient()

    def get_current_user(self) -> User:
        """Get the current authenticated user."""
        return self.client.current_user.me()

    def get_user_info(self) -> dict:
        """Get formatted user information."""
        user = self.get_current_user()
        return {
            "userName": user.user_name or "unknown",
            "displayName": user.display_name,
            "active": user.active or False,
            "emails": [email.value for email in (user.emails or [])],
            "groups": [group.display for group in (user.groups or [])],
        }

    def get_user_workspace_info(self) -> dict:
        """Get user workspace information."""
        user = self.get_current_user()

        # Get workspace URL from the client
        workspace_url = self.client.config.host

        return {
            "user": {
                "userName": user.user_name or "unknown",
                "displayName": user.display_name,
                "active": user.active or False,
            },
            "workspace": {
                "url": workspace_url,
                "deployment_name": workspace_url.split("//")[1].split(".")[0] if workspace_url else None,
            },
        }

    def get_workspace_client(self) -> WorkspaceClient:
        """Get the workspace client instance."""
        return self.client

    def get_user_from_request(self, request: Optional[Request] = None) -> dict:
        """Extract user information from the request context or current session.

        In Databricks Apps, the user identity is available through the SDK.
        For OAuth/service principal auth, we get the current authenticated identity.
        """
        try:
            # Try to get current user from Databricks SDK
            user = self.get_current_user()
            user_info = {
                "username": user.user_name or "service-principal",
                "display_name": user.display_name or "Service Principal",
                "email": user.emails[0].value if user.emails else None,
                "active": user.active or True,
                "auth_type": "databricks",
            }
            logger.debug(f"Extracted user from Databricks SDK: {user_info['username']}")
            return user_info
        except Exception as e:
            logger.warning(f"Could not get user from Databricks SDK: {e}")

            # Fallback: Check request headers for user info
            if request and hasattr(request, "headers"):
                # Check for common auth headers
                auth_header = request.headers.get("authorization", "")
                x_user = request.headers.get("x-databricks-user-name", "")

                if x_user:
                    return {
                        "username": x_user,
                        "display_name": x_user,
                        "email": None,
                        "active": True,
                        "auth_type": "header",
                    }
                elif auth_header:
                    # Could parse JWT token here if needed
                    return {
                        "username": "authenticated-user",
                        "display_name": "Authenticated User",
                        "email": None,
                        "active": True,
                        "auth_type": "token",
                    }

            # Final fallback
            return {
                "username": "unknown",
                "display_name": "Unknown User",
                "email": None,
                "active": True,
                "auth_type": "none",
            }
