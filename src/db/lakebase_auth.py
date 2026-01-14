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
        """Get WorkspaceClient from centralized auth.

        Uses the shared DatabricksAuth singleton which handles auth mode
        detection (direct token, profile, or Databricks Apps automatic).
        """
        from src.core.databricks_auth import get_databricks_auth

        return get_databricks_auth().get_client()

    def _get_instance_host(self) -> str:
        """Get the actual hostname for the Lakebase instance.

        In Databricks Apps environment, PGHOST is automatically injected by the
        platform and should be used directly. Falls back to API lookup for local
        development when PGHOST is not available.

        Returns:
            The actual DNS hostname for the instance.

        Raises:
            ValueError: If hostname cannot be determined.
        """
        if self._instance_host is not None:
            return self._instance_host

        import os

        # Priority 1: Use PGHOST if set (Databricks Apps auto-injects this)
        pghost = os.environ.get("PGHOST")
        if pghost:
            self._instance_host = pghost
            logger.info(f"Using PGHOST from environment: {self._instance_host}")
            return self._instance_host

        # Priority 2: Fall back to API lookup (for local development)
        instance_name = self._settings.lakebase_instance_name
        if not instance_name:
            raise ValueError(
                "LAKEBASE_INSTANCE_NAME is required when PGHOST is not set"
            )

        client = self._get_workspace_client()
        logger.info(f"Looking up hostname for Lakebase instance: {instance_name}")

        try:
            # Use get_database_instance() - matches Databricks Apps Cookbook pattern
            # This only requires permission on the specific instance, not workspace-level list
            inst = client.database.get_database_instance(name=instance_name)

            if not inst.read_write_dns:
                raise ValueError(
                    f"Instance '{instance_name}' has no read_write_dns configured"
                )

            self._instance_host = inst.read_write_dns
            logger.info(f"Found Lakebase host: {self._instance_host}")
            return self._instance_host

        except Exception as e:
            raise ValueError(
                f"Failed to get Lakebase instance '{instance_name}': {e}. "
                f"Check LAKEBASE_INSTANCE_NAME and service principal permissions."
            ) from e

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

    def _get_instance_name(self) -> str:
        """Get the Lakebase instance name.

        In Databricks Apps environment, derives instance name from PGHOST by:
        1. Extracting the UID from PGHOST (format: instance-<uid>.database...)
        2. Looking up the actual instance name via SDK

        Falls back to LAKEBASE_INSTANCE_NAME setting for local development.

        Returns:
            Instance name for credential generation.

        Raises:
            ValueError: If instance name cannot be determined.
        """
        import os
        import re

        # Priority 1: Use explicit setting if available
        if self._settings.lakebase_instance_name:
            return self._settings.lakebase_instance_name

        # Priority 2: Derive from PGHOST via UID lookup
        # PGHOST format: instance-<uuid>.database.cloud.databricks.com
        pghost = os.environ.get("PGHOST")
        if pghost:
            # Extract UID from PGHOST (format: instance-<uid>.database.cloud.databricks.com)
            match = re.match(r"instance-([a-f0-9-]+)\.", pghost)
            if not match:
                raise ValueError(
                    f"Could not extract UID from PGHOST: {pghost}. "
                    "Expected format: instance-<uuid>.database.cloud.databricks.com"
                )

            uid = match.group(1)
            logger.info(f"Extracted UID from PGHOST: {uid}")

            # Look up instance name via SDK
            client = self._get_workspace_client()
            try:
                result = client.database.find_database_instance_by_uid(uid=uid)
                if result and result.name:
                    logger.info(f"Resolved instance name from UID: {result.name}")
                    return result.name
                raise ValueError(
                    f"find_database_instance_by_uid returned no name for UID: {uid}"
                )
            except Exception as e:
                raise ValueError(
                    f"Failed to look up instance by UID '{uid}': {e}. "
                    "Ensure the app's service principal has permission to view database instances. "
                    "Alternatively, set LAKEBASE_INSTANCE_NAME environment variable."
                ) from e

        raise ValueError(
            "Cannot determine Lakebase instance name. "
            "Set LAKEBASE_INSTANCE_NAME or ensure PGHOST is available."
        )

    def _generate_credential(self) -> LakebaseCredential:
        """Generate new OAuth credential via Databricks SDK.

        Returns:
            Fresh Lakebase credential.

        Raises:
            ValueError: If instance name not configured.
            RuntimeError: If credential generation fails.
        """
        instance_name = self._get_instance_name()

        client = self._get_workspace_client()

        logger.info(f"Generating Lakebase credential for instance: {instance_name}")

        cred_response = client.database.generate_database_credential(
            request_id=str(uuid.uuid4()),
            instance_names=[instance_name],
        )

        # Extract token from response
        if not cred_response.token:
            raise RuntimeError("No token returned from Databricks")

        # DEBUG: Log raw SDK response for troubleshooting
        logger.info(f"Raw SDK expiration_time: {cred_response.expiration_time!r}")

        # Parse expiration time if provided, otherwise use default lifetime
        now_utc = datetime.now(UTC)
        if cred_response.expiration_time:
            try:
                expires_at = datetime.fromisoformat(
                    cred_response.expiration_time.replace("Z", "+00:00")
                )
                logger.info(f"Parsed expires_at: {expires_at}, now(UTC): {now_utc}")
                # Sanity check: if expiration is in the past or too close, use default
                if expires_at <= now_utc:
                    logger.warning(
                        f"SDK returned expired/invalid expiration_time: {cred_response.expiration_time}, "
                        f"using default 1-hour lifetime"
                    )
                    expires_at = now_utc + TOKEN_LIFETIME
            except ValueError as e:
                logger.warning(
                    f"Failed to parse expiration_time '{cred_response.expiration_time}': {e}, "
                    f"using default 1-hour lifetime"
                )
                expires_at = now_utc + TOKEN_LIFETIME
        else:
            logger.info("No expiration_time from SDK, using default 1-hour lifetime")
            expires_at = now_utc + TOKEN_LIFETIME

        logger.info(f"Generated Lakebase credential, expires at: {expires_at}")

        # Extract username from JWT 'sub' claim (Databricks identity - email or service principal)
        username = self._extract_username_from_token(cred_response.token)

        return LakebaseCredential(
            token=cred_response.token,
            username=username,
            expires_at=expires_at,
        )

    def _extract_username_from_token(self, token: str) -> str:
        """Get username for Lakebase connection.

        In Databricks Apps environment, PGUSER is automatically injected by the
        platform and should be used directly. Falls back to extracting from JWT
        or workspace client for local development.

        Args:
            token: JWT token from Databricks (used for fallback extraction).

        Returns:
            Username (service principal client ID or email).
        """
        import os

        # Priority 1: Use PGUSER if set (Databricks Apps auto-injects this)
        pguser = os.environ.get("PGUSER")
        if pguser:
            logger.info(f"Using PGUSER from environment: {pguser}")
            return pguser

        # Priority 2: Extract from JWT token's 'sub' claim
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

        # Priority 3: Try to get from workspace client (works for users, not service principals)
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

        In Databricks Apps environment, uses auto-injected PGHOST, PGPORT, PGDATABASE.
        Falls back to settings for local development.

        Returns:
            PostgreSQL connection URL with asyncpg driver.

        Raises:
            ValueError: If required configuration is missing.
        """
        import os

        cred = self.get_credential()

        # Get connection parameters (prefer auto-injected env vars from Databricks Apps)
        host = self._get_instance_host()
        port = int(os.environ.get("PGPORT", self._settings.lakebase_port))
        database = os.environ.get("PGDATABASE", self._settings.lakebase_database)

        logger.info(f"Building connection URL: host={host}, port={port}, database={database}")

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
