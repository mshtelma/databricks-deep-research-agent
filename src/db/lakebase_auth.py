"""Lakebase OAuth credential provider."""

import base64
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from urllib.parse import quote_plus

from databricks.sdk import WorkspaceClient

if TYPE_CHECKING:
    from src.core.config import Settings

logger = logging.getLogger(__name__)

# Token refresh buffer (refresh 5 minutes before expiry)
TOKEN_REFRESH_BUFFER = timedelta(minutes=5)
TOKEN_LIFETIME = timedelta(hours=1)


@dataclass
class LakebaseCredential:
    """OAuth credential for Lakebase connection."""

    token: str
    username: str
    expires_at: datetime

    @property
    def is_expired(self) -> bool:
        """Check if token is expired or about to expire."""
        return datetime.now(UTC) >= (self.expires_at - TOKEN_REFRESH_BUFFER)


class LakebaseCredentialProvider:
    """Provides and refreshes OAuth credentials for Lakebase."""

    def __init__(self, settings: "Settings") -> None:
        """Initialize credential provider.

        Args:
            settings: Application settings.
        """
        self._settings = settings
        self._credential: LakebaseCredential | None = None
        self._workspace_client: WorkspaceClient | None = None
        self._instance_host: str | None = None  # Cached actual hostname

    def _get_workspace_client(self) -> WorkspaceClient:
        """Get or create WorkspaceClient with profile auth."""
        if self._workspace_client is None:
            profile = self._settings.databricks_config_profile
            if not profile:
                raise ValueError(
                    "DATABRICKS_CONFIG_PROFILE is required for Lakebase auth"
                )

            logger.info(f"Creating WorkspaceClient with profile: {profile}")
            self._workspace_client = WorkspaceClient(profile=profile)

        return self._workspace_client

    def _get_instance_host(self) -> str:
        """Get the actual hostname for the Lakebase instance.

        The hostname is NOT {name}.database.cloud.databricks.com but rather
        instance-{uid}.database.cloud.databricks.com (from read_write_dns).

        Returns:
            The actual DNS hostname for the instance.

        Raises:
            ValueError: If instance not found.
        """
        if self._instance_host is not None:
            return self._instance_host

        instance_name = self._settings.lakebase_instance_name
        if not instance_name:
            raise ValueError("LAKEBASE_INSTANCE_NAME is required")

        client = self._get_workspace_client()

        logger.info(f"Looking up hostname for Lakebase instance: {instance_name}")

        for inst in client.database.list_database_instances():
            if inst.name == instance_name:
                if inst.read_write_dns:
                    self._instance_host = inst.read_write_dns
                    logger.info(f"Found Lakebase host: {self._instance_host}")
                    return self._instance_host
                raise ValueError(
                    f"Instance '{instance_name}' has no read_write_dns configured"
                )

        raise ValueError(
            f"Lakebase instance '{instance_name}' not found. "
            f"Check LAKEBASE_INSTANCE_NAME in your configuration."
        )

    def get_credential(self, force_refresh: bool = False) -> LakebaseCredential:
        """Get valid OAuth credential, refreshing if needed.

        Args:
            force_refresh: Force credential refresh even if not expired.

        Returns:
            Valid Lakebase credential.
        """
        if (
            self._credential is None
            or self._credential.is_expired
            or force_refresh
        ):
            self._credential = self._generate_credential()

        return self._credential

    def _generate_credential(self) -> LakebaseCredential:
        """Generate new OAuth credential via Databricks SDK.

        Returns:
            Fresh Lakebase credential.

        Raises:
            ValueError: If instance name not configured.
            RuntimeError: If credential generation fails.
        """
        instance_name = self._settings.lakebase_instance_name
        if not instance_name:
            raise ValueError("LAKEBASE_INSTANCE_NAME is required")

        client = self._get_workspace_client()

        logger.info(f"Generating Lakebase credential for instance: {instance_name}")

        cred_response = client.database.generate_database_credential(
            request_id=str(uuid.uuid4()),
            instance_names=[instance_name],
        )

        # Extract token from response
        if not cred_response.token:
            raise RuntimeError("No token returned from Databricks")

        # Parse expiration time if provided, otherwise use default lifetime
        if cred_response.expiration_time:
            try:
                expires_at = datetime.fromisoformat(
                    cred_response.expiration_time.replace("Z", "+00:00")
                )
            except ValueError:
                expires_at = datetime.now(UTC) + TOKEN_LIFETIME
        else:
            expires_at = datetime.now(UTC) + TOKEN_LIFETIME

        logger.info(f"Generated Lakebase credential, expires at: {expires_at}")

        # Extract username from JWT 'sub' claim (Databricks identity - email or service principal)
        username = self._extract_username_from_token(cred_response.token)

        return LakebaseCredential(
            token=cred_response.token,
            username=username,
            expires_at=expires_at,
        )

    def _extract_username_from_token(self, token: str) -> str:
        """Extract username from JWT token's 'sub' claim.

        Args:
            token: JWT token from Databricks.

        Returns:
            Username (email or service principal ID).
        """
        try:
            # JWT format: header.payload.signature
            payload_b64 = token.split(".")[1]
            # Add padding if needed
            payload_b64 += "=" * (4 - len(payload_b64) % 4)
            payload = json.loads(base64.urlsafe_b64decode(payload_b64))
            username = payload.get("sub", "")
            if username:
                logger.info(f"Extracted username from token: {username}")
                return username
        except Exception as e:
            logger.warning(f"Failed to extract username from token: {e}")

        # Fallback: try to get from workspace client
        try:
            client = self._get_workspace_client()
            me = client.current_user.me()
            if me.user_name:
                logger.info(f"Got username from current_user: {me.user_name}")
                return me.user_name
        except Exception as e:
            logger.warning(f"Failed to get current user: {e}")

        raise ValueError("Could not determine username for Lakebase authentication")

    def build_connection_url(self) -> str:
        """Build PostgreSQL connection URL with OAuth token.

        Returns:
            PostgreSQL connection URL with asyncpg driver.

        Raises:
            ValueError: If required configuration is missing.
        """
        cred = self.get_credential()

        # Get actual hostname from API (not derived from instance name)
        host = self._get_instance_host()
        port = self._settings.lakebase_port
        database = self._settings.lakebase_database

        # URL encode the token and username (may contain special chars)
        encoded_token = quote_plus(cred.token)
        encoded_username = quote_plus(cred.username)

        # Build asyncpg connection URL
        # Note: SSL is configured via connect_args in session.py (asyncpg doesn't accept sslmode)
        url = (
            f"postgresql+asyncpg://{encoded_username}:{encoded_token}"
            f"@{host}:{port}/{database}"
        )

        return url
