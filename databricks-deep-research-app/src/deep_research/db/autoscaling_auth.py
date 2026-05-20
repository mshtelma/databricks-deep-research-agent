"""Lakebase OAuth credential provider (Autoscaling backend).

Uses the `client.postgres.generate_database_credential()` API namespace
which differs from the Provisioned `client.database.*` namespace.
"""

import base64
import json
import logging
import os
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from urllib.parse import quote_plus

from deep_research.db.credential_provider import (
    TOKEN_LIFETIME,
    BaseLakebaseCredentialProvider,
    LakebaseBackend,
    LakebaseCredential,
)

if TYPE_CHECKING:
    from deep_research.core.config import Settings

logger = logging.getLogger(__name__)


class AutoscalingCredentialProvider(BaseLakebaseCredentialProvider):
    """Provides and refreshes OAuth credentials for Lakebase Autoscaling."""

    def __init__(self, settings: "Settings") -> None:
        """Initialize credential provider.

        Args:
            settings: Application settings.

        Raises:
            ValueError: If ENDPOINT_NAME is not configured.
        """
        self._settings = settings
        self._credential: LakebaseCredential | None = None

        self._endpoint_name = settings.endpoint_name or os.environ.get("ENDPOINT_NAME", "")
        if not self._endpoint_name:
            raise ValueError(
                "ENDPOINT_NAME is required for Lakebase Autoscaling. "
                "Format: projects/<id>/branches/<id>/endpoints/<id>"
            )
        self._resolved_host: str | None = None

    def _get_workspace_client(self) -> "object":
        """Get WorkspaceClient from centralized auth."""
        from deep_research.core.databricks_auth import get_databricks_auth

        return get_databricks_auth().get_client()

    def get_credential(self, force_refresh: bool = False) -> LakebaseCredential:
        """Get valid OAuth credential, refreshing if needed."""
        if (
            self._credential is None
            or self._credential.is_expired
            or force_refresh
        ):
            self._credential = self._generate_credential()

        return self._credential

    def _generate_credential(self) -> LakebaseCredential:
        """Generate new OAuth credential via Databricks SDK (Autoscaling API)."""
        client = self._get_workspace_client()

        logger.info(
            "AUTOSCALING_CREDENTIAL_GENERATING endpoint=%s",
            self._endpoint_name,
        )

        # Autoscaling uses client.postgres namespace (not client.database)
        cred_response = client.postgres.generate_database_credential(  # type: ignore[attr-defined]
            endpoint=self._endpoint_name,
        )

        if not cred_response.token:
            raise RuntimeError("No token returned from Databricks Autoscaling API")

        # Calculate expiration (same workaround as Provisioned for timezone bug)
        now_utc = datetime.now(UTC)
        expires_at = now_utc + TOKEN_LIFETIME

        logger.info(
            "AUTOSCALING_CREDENTIAL_GENERATED calculated_expires_at=%s "
            "now_utc=%s token_length=%s",
            expires_at.isoformat(),
            now_utc.isoformat(),
            f"len={len(cred_response.token)}" if cred_response.token else None,
        )

        username = self._extract_username(cred_response.token)

        return LakebaseCredential(
            token=cred_response.token,
            username=username,
            expires_at=expires_at,
        )

    def _extract_username(self, token: str) -> str:
        """Extract the Postgres username for the generated credential.

        Databricks Apps custom Lakebase wiring uses the app service
        principal client id as the database role. Apps expose it as
        ``DATABRICKS_CLIENT_ID`` even when ``PGUSER`` is not explicitly
        configured in the app env.
        """
        pguser = os.environ.get("PGUSER")
        if pguser:
            logger.info(f"Using PGUSER from environment: {pguser}")
            return pguser

        client_id = os.environ.get("DATABRICKS_CLIENT_ID")
        if client_id:
            logger.info("Using DATABRICKS_CLIENT_ID as Lakebase username")
            return client_id

        # Local/profile fallback: extract from JWT token's 'sub' claim.
        try:
            payload_b64 = token.split(".")[1]
            payload_b64 += "=" * (4 - len(payload_b64) % 4)
            payload = json.loads(base64.urlsafe_b64decode(payload_b64))
            username: str = payload.get("sub", "")
            if username:
                logger.info(f"Extracted username from token: {username}")
                return username
        except Exception as e:
            logger.warning(f"Failed to extract username from token: {e}")

        raise ValueError("Could not determine username for Autoscaling authentication")

    def get_host(self) -> str:
        """Get the hostname for the Autoscaling endpoint.

        Priority:
        1. Cached host (from prior resolution)
        2. PGHOST env var (Databricks Apps platform or .env.{target} file)
        3. SDK lookup via endpoint status (for CLI tools like db-reset)

        The resolved host is cached for the lifetime of this provider instance
        (same pattern as LakebaseCredentialProvider._get_instance_host).
        """
        if self._resolved_host is not None:
            return self._resolved_host

        # Priority 1: PGHOST env var
        host = os.environ.get("PGHOST")
        if host:
            self._resolved_host = host
            logger.info("AUTOSCALING_HOST_FROM_ENV host=%s", host)
            return host

        # Priority 2: SDK fallback — resolve from endpoint metadata
        try:
            client = self._get_workspace_client()
            ep = client.postgres.get_endpoint(  # type: ignore[attr-defined]
                name=self._endpoint_name,
            )
            if ep.status and ep.status.hosts:
                resolved = str(ep.status.hosts.host)
                if resolved:
                    self._resolved_host = resolved
                    logger.info(
                        "AUTOSCALING_HOST_FROM_SDK host=%s endpoint=%s",
                        resolved, self._endpoint_name,
                    )
                    return resolved
            logger.warning(
                "AUTOSCALING_HOST_NO_STATUS endpoint=%s", self._endpoint_name
            )
        except Exception:
            logger.warning(
                "AUTOSCALING_HOST_RESOLVE_FAILED endpoint=%s",
                self._endpoint_name,
                exc_info=True,
            )

        raise ValueError(
            f"PGHOST not set and SDK lookup failed for endpoint "
            f"'{self._endpoint_name}'. Either set PGHOST in environment "
            f"or run 'make db-provision' to generate .env file."
        )

    def get_port(self) -> int:
        """Get the port for the Autoscaling endpoint."""
        return int(os.environ.get("PGPORT", "5432"))

    def get_database(self) -> str:
        """Get the database name."""
        return os.environ.get("PGDATABASE", self._settings.lakebase_database)

    def get_backend_type(self) -> LakebaseBackend:
        """Return the backend type identifier."""
        return "autoscaling"

    def build_connection_url(self) -> str:
        """Build PostgreSQL connection URL with OAuth token."""
        cred = self.get_credential()

        host = self.get_host()
        port = self.get_port()
        database = self.get_database()

        logger.info(
            "Building Autoscaling connection URL: host=%s, port=%d, database=%s",
            host, port, database,
        )

        encoded_token = quote_plus(cred.token)
        encoded_username = quote_plus(cred.username)

        return (
            f"postgresql+asyncpg://{encoded_username}:{encoded_token}"
            f"@{host}:{port}/{database}"
        )
