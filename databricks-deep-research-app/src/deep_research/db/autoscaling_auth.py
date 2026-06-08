"""Lakebase OAuth credential provider (Autoscaling backend).

Uses the `client.postgres.generate_database_credential()` API namespace
which differs from the Provisioned `client.database.*` namespace.
"""

import base64
import hashlib
import json
import logging
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
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
        self._last_token_claims: dict[str, Any] = {}
        self._last_username_source: str | None = None

        self._endpoint_name = settings.endpoint_name or os.environ.get(
            "ENDPOINT_NAME",
            "",
        )
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
        if self._credential is None or self._credential.is_expired or force_refresh:
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

        token_claims = self._decode_token_claims(cred_response.token)
        username, username_source = self._extract_username(
            cred_response.token,
            token_claims,
        )
        self._last_token_claims = token_claims
        self._last_username_source = username_source

        logger.info(
            "AUTOSCALING_CREDENTIAL_GENERATED calculated_expires_at=%s now_utc=%s token_length=%s",
            expires_at.isoformat(),
            now_utc.isoformat(),
            f"len={len(cred_response.token)}",
        )
        self._log_credential_diagnostics(
            token=cred_response.token,
            username=username,
            username_source=username_source,
            claims=token_claims,
            now_utc=now_utc,
        )

        return LakebaseCredential(
            token=cred_response.token,
            username=username,
            expires_at=expires_at,
            issued_at=now_utc,
        )

    def _extract_username(
        self,
        token: str,
        claims: Mapping[str, Any],
    ) -> tuple[str, str]:
        """Extract the Postgres username for the generated credential.

        Databricks Apps custom Lakebase wiring uses the app service
        principal client id as the database role. Apps expose it as
        ``DATABRICKS_CLIENT_ID`` even when ``PGUSER`` is not explicitly
        configured in the app env.
        """
        pguser = os.environ.get("PGUSER")
        if pguser:
            logger.info(
                "Using PGUSER from environment: %s",
                self._redact_identity(pguser),
            )
            return pguser, "PGUSER"

        client_id = os.environ.get("DATABRICKS_CLIENT_ID")
        if client_id:
            logger.info("Using DATABRICKS_CLIENT_ID as Lakebase username")
            return client_id, "DATABRICKS_CLIENT_ID"

        # Local/profile fallback: extract from JWT token's 'sub' claim.
        username = self._claim_as_str(claims, "sub")
        if username:
            logger.info(
                "Extracted username from token: %s",
                self._redact_identity(username),
            )
            return username, "jwt_sub"

        if token:
            logger.warning("Failed to extract username from token claims")

        raise ValueError("Could not determine username for Autoscaling authentication")

    def log_auth_failure_diagnostics(self, exc: BaseException) -> None:
        """Log redacted Lakebase auth context after a DB authentication failure."""
        cred = self.current_credential
        if cred is None:
            logger.warning(
                "AUTOSCALING_DB_AUTH_FAILURE_DIAGNOSTIC backend=autoscaling "
                "endpoint=%s credential_exists=False exc_class=%s error_fingerprint=%s",
                self._endpoint_name,
                type(exc).__name__,
                self._message_fingerprint(str(exc)),
            )
            return

        claims = self._last_token_claims or self._decode_token_claims(cred.token)
        username_source = self._last_username_source or self._infer_username_source(
            cred.username,
            claims,
        )
        now_utc = datetime.now(UTC)
        cred_age = cred.age_s
        logger.warning(
            "AUTOSCALING_DB_AUTH_FAILURE_DIAGNOSTIC backend=autoscaling "
            "endpoint=%s username_source=%s username=%s token_sub=%s "
            "token_client_id=%s token_aud=%s token_iss=%s token_exp=%s "
            "token_expires_in_s=%s token_iat=%s token_nbf=%s token_age_s=%s "
            "cred_age_s=%s username_matches_sub=%s exc_class=%s error=%s",
            self._endpoint_name,
            username_source,
            self._redact_identity(cred.username),
            self._redact_claim(claims.get("sub")),
            self._redact_claim(claims.get("client_id")),
            self._redact_claim(claims.get("aud")),
            self._redact_claim(claims.get("iss")),
            claims.get("exp"),
            self._token_expires_in_s(claims, now_utc),
            self._claim_epoch(claims, "iat"),
            self._claim_epoch(claims, "nbf"),
            self._token_age_s(claims, now_utc),
            f"{cred_age:.1f}" if cred_age is not None else None,
            self._username_matches_sub(cred.username, claims),
            type(exc).__name__,
            self._sanitize_auth_error(exc, cred.username, claims),
        )

    def _log_credential_diagnostics(
        self,
        *,
        token: str,
        username: str,
        username_source: str,
        claims: Mapping[str, Any],
        now_utc: datetime,
    ) -> None:
        logger.info(
            "AUTOSCALING_CREDENTIAL_DIAGNOSTIC backend=autoscaling endpoint=%s "
            "username_source=%s username=%s token_sub=%s token_client_id=%s "
            "token_aud=%s token_iss=%s token_exp=%s token_expires_in_s=%s "
            "token_iat=%s token_nbf=%s token_age_s=%s "
            "username_matches_sub=%s token_length=%s",
            self._endpoint_name,
            username_source,
            self._redact_identity(username),
            self._redact_claim(claims.get("sub")),
            self._redact_claim(claims.get("client_id")),
            self._redact_claim(claims.get("aud")),
            self._redact_claim(claims.get("iss")),
            claims.get("exp"),
            self._token_expires_in_s(claims, now_utc),
            self._claim_epoch(claims, "iat"),
            self._claim_epoch(claims, "nbf"),
            self._token_age_s(claims, now_utc),
            self._username_matches_sub(username, claims),
            f"len={len(token)}",
        )

    def log_endpoint_state_diagnostics(self) -> None:
        """Best-effort log of the Autoscaling endpoint's current state.

        Called after an auth failure to test the "transient endpoint event"
        hypothesis: if ``current_state`` is not ACTIVE (or ``update_time`` is
        very recent) when tokens are being rejected, the failure correlates
        with an endpoint maintenance/restart rather than a token race. Fully
        non-fatal — any SDK error is swallowed so this never blocks the
        request or connection path.
        """
        try:
            client = self._get_workspace_client()
            ep = client.postgres.get_endpoint(name=self._endpoint_name)  # type: ignore[attr-defined]
            status = getattr(ep, "status", None)
            logger.warning(
                "AUTOSCALING_ENDPOINT_STATE_DIAGNOSTIC endpoint=%s current_state=%s "
                "disabled=%s update_time=%s",
                self._endpoint_name,
                getattr(status, "current_state", None),
                getattr(status, "disabled", None),
                getattr(ep, "update_time", None),
            )
        except Exception as exc:  # noqa: BLE001 - diagnostic must never raise
            logger.warning(
                "AUTOSCALING_ENDPOINT_STATE_DIAGNOSTIC_FAILED endpoint=%s exc_class=%s",
                self._endpoint_name,
                type(exc).__name__,
            )

    @staticmethod
    def _decode_token_claims(token: str) -> dict[str, Any]:
        try:
            parts = token.split(".")
            if len(parts) < 2:
                raise ValueError("token is not a JWT")
            payload_b64 = parts[1]
            payload_b64 += "=" * (-len(payload_b64) % 4)
            payload = json.loads(base64.urlsafe_b64decode(payload_b64))
            if isinstance(payload, dict):
                return payload
        except Exception as exc:
            logger.warning(
                "AUTOSCALING_CREDENTIAL_CLAIMS_DECODE_FAILED exc_class=%s error=%s",
                type(exc).__name__,
                str(exc)[:200],
            )
        return {}

    @staticmethod
    def _claim_as_str(claims: Mapping[str, Any], name: str) -> str | None:
        value = claims.get(name)
        return value if isinstance(value, str) and value else None

    @classmethod
    def _redact_claim(cls, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, list):
            redacted_items = ", ".join(cls._redact_identity(item) for item in value)
            return f"[{redacted_items}]"
        return cls._redact_identity(value)

    @staticmethod
    def _redact_identity(value: Any) -> str:
        rendered = str(value)
        digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()[:12]
        return (
            f"kind={AutoscalingCredentialProvider._identity_kind(rendered)}:"
            f"sha256={digest}:len={len(rendered)}"
        )

    @staticmethod
    def _identity_kind(value: str) -> str:
        if "@" in value:
            return "email"
        if value.startswith("http://") or value.startswith("https://"):
            return "url"
        if len(value) == 36 and value.count("-") == 4:
            return "uuid"
        if value.isdigit():
            return "numeric"
        return "other"

    @classmethod
    def _sanitize_auth_error(
        cls,
        exc: BaseException,
        username: str,
        claims: Mapping[str, Any],
    ) -> str:
        message = str(exc)[:300]
        candidates = {
            username,
            os.environ.get("PGUSER", ""),
            os.environ.get("DATABRICKS_CLIENT_ID", ""),
            cls._claim_as_str(claims, "sub") or "",
        }
        for candidate in sorted(candidates, key=len, reverse=True):
            if candidate:
                message = message.replace(
                    candidate,
                    cls._redact_identity(candidate),
                )
        return message

    @staticmethod
    def _message_fingerprint(message: str) -> str:
        digest = hashlib.sha256(message.encode("utf-8")).hexdigest()[:12]
        return f"sha256={digest}:len={len(message)}"

    @staticmethod
    def _token_expires_in_s(
        claims: Mapping[str, Any],
        now_utc: datetime,
    ) -> int | None:
        exp = claims.get("exp")
        if isinstance(exp, int | float):
            return int(exp - now_utc.timestamp())
        return None

    @staticmethod
    def _claim_epoch(claims: Mapping[str, Any], name: str) -> int | None:
        value = claims.get(name)
        if isinstance(value, int | float):
            return int(value)
        return None

    @classmethod
    def _token_age_s(
        cls,
        claims: Mapping[str, Any],
        now_utc: datetime,
    ) -> int | None:
        """Seconds between the token's ``iat`` claim and now.

        A near-zero age at the moment Lakebase rejects the token with
        "password authentication failed" is the fingerprint of a
        freshly-minted-credential propagation race (PgBouncer /
        databricks_auth eventual consistency) — as opposed to an expired
        or structurally invalid token. Returns None if ``iat`` is absent.
        """
        iat = cls._claim_epoch(claims, "iat")
        if iat is None:
            return None
        return int(now_utc.timestamp() - iat)

    @classmethod
    def _username_matches_sub(
        cls,
        username: str,
        claims: Mapping[str, Any],
    ) -> bool | None:
        sub = cls._claim_as_str(claims, "sub")
        if sub is None:
            return None
        return username == sub

    @classmethod
    def _infer_username_source(
        cls,
        username: str,
        claims: Mapping[str, Any],
    ) -> str:
        if os.environ.get("PGUSER") == username:
            return "PGUSER"
        if os.environ.get("DATABRICKS_CLIENT_ID") == username:
            return "DATABRICKS_CLIENT_ID"
        if cls._claim_as_str(claims, "sub") == username:
            return "jwt_sub"
        return "unknown"

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
                        resolved,
                        self._endpoint_name,
                    )
                    return resolved
            logger.warning("AUTOSCALING_HOST_NO_STATUS endpoint=%s", self._endpoint_name)
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
            host,
            port,
            database,
        )

        encoded_token = quote_plus(cred.token)
        encoded_username = quote_plus(cred.username)

        return f"postgresql+asyncpg://{encoded_username}:{encoded_token}@{host}:{port}/{database}"
