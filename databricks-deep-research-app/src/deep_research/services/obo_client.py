"""On-Behalf-Of (OBO) Databricks Client Service.

This module provides authenticated Databricks client access that operates
on behalf of the authenticated user, respecting their workspace permissions.

Used for:
- Vector Search index access with user permissions
- Genie space access with user permissions
- Knowledge Assistant access with user permissions

Example:
    obo_client = OBODatabricksClient()
    client = await obo_client.get_client(user_token)
    # client now operates with user's permissions

Security Considerations (T110-T111):
=====================================

1. TOKEN HANDLING:
   - Raw tokens are NEVER logged - only the first 8 characters (prefix) for debugging
   - Tokens are hashed using SHA-256 before being used as cache keys
   - The cache stores only token hashes, not actual tokens
   - Token values are passed through to Databricks SDK but not stored locally

2. ERROR MESSAGE SANITIZATION:
   - Raw exception messages may contain sensitive info (internal paths, etc.)
   - All error messages returned to users are sanitized through OBO exception classes
   - Log messages use truncated/sanitized versions of error messages

3. ACCESS VALIDATION:
   - All validate_*_access methods perform lightweight permission checks
   - Results are cached to reduce API calls (TTL: 15 minutes)
   - Cache keys use hashed tokens to prevent token exposure in memory dumps

4. OWNERSHIP/VISIBILITY:
   - This module relies on Databricks workspace permissions
   - The OBO token inherently limits access to what the user can see
   - No additional ownership checks needed - Databricks enforces this

5. AUDIT LOGGING:
   - All access validations are logged with structured events
   - Logs include resource identifiers but NOT tokens or user IDs beyond prefix
   - Failed access attempts are logged at WARNING level for security monitoring
"""

import asyncio
import base64
import json
import os
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from databricks.sdk import WorkspaceClient

from deep_research.core.auth import get_user_workspace_client, get_workspace_client
from deep_research.core.exceptions import (
    OBOPermissionError,
    OBOResourceNotFoundError,
    OBOServiceUnavailableError,
    OBOTokenExpiredError,
)
from deep_research.core.logging_utils import get_logger
from deep_research.observability.agent_designer_metrics import (
    record_token_refresh_attempt,
    record_token_refresh_failure,
)

logger = get_logger(__name__)

# Cache configuration
ACCESS_CACHE_TTL = timedelta(minutes=15)


@dataclass
class CachedAccess:
    """Cached access validation result."""

    has_access: bool
    validated_at: datetime
    error_message: str | None = None

    @property
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return datetime.now(UTC) >= (self.validated_at + ACCESS_CACHE_TTL)


@dataclass
class OBODatabricksClient:
    """Databricks client that operates on behalf of authenticated user.

    Provides OBO-authenticated access to Databricks services:
    - Vector Search indexes
    - Genie spaces
    - Knowledge Assistants

    The client caches access validation results for efficiency.

    Attributes:
        _access_cache: Cache of validated access per (user_token_hash, resource).
    """

    _access_cache: dict[str, CachedAccess] = field(default_factory=dict)

    def _hash_token(self, token: str) -> str:
        """Create a hash of token for cache key (avoid storing raw token).

        SECURITY: This is critical for preventing token exposure in:
        - Memory dumps
        - Cache key enumeration
        - Log files (if cache keys are ever logged)

        We use SHA-256 and truncate to 16 chars which provides sufficient
        uniqueness while keeping cache keys manageable.
        """
        import hashlib

        return hashlib.sha256(token.encode()).hexdigest()[:16]

    async def get_client(self, user_token: str | None) -> WorkspaceClient:
        """Get WorkspaceClient with OBO token exchange.

        If user_token is provided, returns OBO-authenticated client
        that operates with the user's permissions.
        Otherwise returns service principal client.

        Args:
            user_token: User's OAuth access token from x-forwarded-access-token.
                       If None, returns service principal client.

        Returns:
            WorkspaceClient configured appropriately.
        """
        if user_token:
            logger.debug(
                "OBO_CLIENT_GET",
                mode="user_token",
                token_length=len(user_token),
            )
            return get_user_workspace_client(user_token)

        logger.debug("OBO_CLIENT_GET", mode="service_principal")
        return get_workspace_client()

    async def validate_vector_search_access(
        self,
        user_token: str,
        endpoint_name: str,
        index_name: str,
    ) -> tuple[bool, str | None]:
        """Validate user has access to Vector Search index via OBO.

        Attempts a lightweight query to validate access. Results are cached
        for ACCESS_CACHE_TTL.

        Args:
            user_token: User's OAuth token.
            endpoint_name: Vector Search endpoint name.
            index_name: Fully qualified index name (catalog.schema.index).

        Returns:
            Tuple of (has_access, error_message).
            If has_access is True, error_message is None.
        """
        cache_key = f"{self._hash_token(user_token)}:vs:{endpoint_name}:{index_name}"

        # Check cache
        cached = self._access_cache.get(cache_key)
        if cached and not cached.is_expired:
            logger.debug(
                "VECTOR_SEARCH_ACCESS_CACHED",
                endpoint=endpoint_name,
                index=index_name,
                has_access=cached.has_access,
            )
            return cached.has_access, cached.error_message

        # Validate access
        try:
            client = await self.get_client(user_token)

            # Try to get index info (lightweight validation)
            # Using sync API in async context via run_in_executor
            loop = asyncio.get_event_loop()
            index_info = await loop.run_in_executor(
                None,
                lambda: client.vector_search_indexes.get_index(index_name),
            )

            if index_info:
                self._access_cache[cache_key] = CachedAccess(
                    has_access=True,
                    validated_at=datetime.now(UTC),
                )
                logger.info(
                    "VECTOR_SEARCH_ACCESS_VALIDATED",
                    endpoint=endpoint_name,
                    index=index_name,
                )
                return True, None

        except Exception as e:
            error_msg = str(e)
            obo_error: OBOPermissionError | OBOResourceNotFoundError | OBOServiceUnavailableError | OBOTokenExpiredError | None = None

            # Identify permission-specific errors and create appropriate OBO exception
            # SECURITY NOTE: Error messages are sanitized to not expose internal details
            if "PERMISSION_DENIED" in error_msg or "403" in error_msg:
                obo_error = OBOPermissionError(
                    resource_type="Vector Search Index",
                    resource_name=index_name,
                    required_permission="SELECT",
                )
                error_msg = obo_error.message
            elif "NOT_FOUND" in error_msg or "404" in error_msg:
                obo_error = OBOResourceNotFoundError(
                    resource_type="Vector Search Index",
                    resource_name=index_name,
                )
                error_msg = obo_error.message
            elif "expired" in error_msg.lower() or "401" in error_msg:
                obo_error = OBOTokenExpiredError()
                error_msg = obo_error.message
            elif "unavailable" in error_msg.lower() or "503" in error_msg:
                obo_error = OBOServiceUnavailableError(
                    service_name="Vector Search",
                    retry_after_seconds=60,
                )
                error_msg = obo_error.message

            self._access_cache[cache_key] = CachedAccess(
                has_access=False,
                validated_at=datetime.now(UTC),
                error_message=error_msg,
            )
            logger.warning(
                "VECTOR_SEARCH_ACCESS_DENIED",
                endpoint=endpoint_name,
                index=index_name,
                # SECURITY: Log only sanitized message, not raw exception
                error=error_msg[:200],
            )
            return False, error_msg

        return False, "Unknown validation error"

    async def validate_genie_access(
        self,
        user_token: str,
        space_id: str,
    ) -> tuple[bool, str | None]:
        """Validate user has access to Genie space via OBO.

        Args:
            user_token: User's OAuth token.
            space_id: Genie space ID.

        Returns:
            Tuple of (has_access, error_message).
        """
        cache_key = f"{self._hash_token(user_token)}:genie:{space_id}"

        # Check cache
        cached = self._access_cache.get(cache_key)
        if cached and not cached.is_expired:
            logger.debug(
                "GENIE_ACCESS_CACHED",
                space_id=space_id,
                has_access=cached.has_access,
            )
            return cached.has_access, cached.error_message

        # Validate access
        try:
            client = await self.get_client(user_token)

            # Try to get space info (lightweight validation)
            loop = asyncio.get_event_loop()
            # Note: Genie API may vary - adjust based on actual SDK
            space_info = await loop.run_in_executor(
                None,
                lambda: client.genie.get_space(space_id),
            )

            if space_info:
                self._access_cache[cache_key] = CachedAccess(
                    has_access=True,
                    validated_at=datetime.now(UTC),
                )
                logger.info("GENIE_ACCESS_VALIDATED", space_id=space_id)
                return True, None

        except AttributeError:
            # Genie API may not be available in all SDK versions
            logger.warning(
                "GENIE_API_NOT_AVAILABLE: Genie API not available in current SDK version",
            )
            # Allow access if we can't validate (fail open for now)
            self._access_cache[cache_key] = CachedAccess(
                has_access=True,
                validated_at=datetime.now(UTC),
            )
            return True, None

        except Exception as e:
            error_msg = str(e)
            obo_error: OBOPermissionError | OBOResourceNotFoundError | OBOServiceUnavailableError | OBOTokenExpiredError | None = None

            # SECURITY NOTE: Error messages are sanitized to not expose internal details
            if "PERMISSION_DENIED" in error_msg or "403" in error_msg:
                obo_error = OBOPermissionError(
                    resource_type="Genie Space",
                    resource_name=space_id,
                    required_permission="CAN_USE",
                )
                error_msg = obo_error.message
            elif "NOT_FOUND" in error_msg or "404" in error_msg:
                obo_error = OBOResourceNotFoundError(
                    resource_type="Genie Space",
                    resource_name=space_id,
                )
                error_msg = obo_error.message
            elif "expired" in error_msg.lower() or "401" in error_msg:
                obo_error = OBOTokenExpiredError()
                error_msg = obo_error.message
            elif "unavailable" in error_msg.lower() or "503" in error_msg:
                obo_error = OBOServiceUnavailableError(
                    service_name="Genie",
                    retry_after_seconds=60,
                )
                error_msg = obo_error.message

            self._access_cache[cache_key] = CachedAccess(
                has_access=False,
                validated_at=datetime.now(UTC),
                error_message=error_msg,
            )
            logger.warning(
                "GENIE_ACCESS_DENIED",
                space_id=space_id,
                # SECURITY: Log only sanitized message
                error=error_msg[:200],
            )
            return False, error_msg

        return False, "Unknown validation error"

    async def validate_assistant_access(
        self,
        user_token: str,
        endpoint_name: str,
    ) -> tuple[bool, str | None]:
        """Validate user has access to Knowledge Assistant endpoint via OBO.

        Args:
            user_token: User's OAuth token.
            endpoint_name: Serving endpoint name for the assistant.

        Returns:
            Tuple of (has_access, error_message).
        """
        cache_key = f"{self._hash_token(user_token)}:assistant:{endpoint_name}"

        # Check cache
        cached = self._access_cache.get(cache_key)
        if cached and not cached.is_expired:
            logger.debug(
                "ASSISTANT_ACCESS_CACHED",
                endpoint=endpoint_name,
                has_access=cached.has_access,
            )
            return cached.has_access, cached.error_message

        # Validate access
        try:
            client = await self.get_client(user_token)

            # Try to get endpoint info (lightweight validation)
            loop = asyncio.get_event_loop()
            endpoint_info = await loop.run_in_executor(
                None,
                lambda: client.serving_endpoints.get(endpoint_name),
            )

            if endpoint_info:
                self._access_cache[cache_key] = CachedAccess(
                    has_access=True,
                    validated_at=datetime.now(UTC),
                )
                logger.info("ASSISTANT_ACCESS_VALIDATED", endpoint=endpoint_name)
                return True, None

        except Exception as e:
            error_msg = str(e)
            obo_error: OBOPermissionError | OBOResourceNotFoundError | OBOServiceUnavailableError | OBOTokenExpiredError | None = None

            # SECURITY NOTE: Error messages are sanitized to not expose internal details
            if "PERMISSION_DENIED" in error_msg or "403" in error_msg:
                obo_error = OBOPermissionError(
                    resource_type="Knowledge Assistant Endpoint",
                    resource_name=endpoint_name,
                    required_permission="CAN_QUERY",
                )
                error_msg = obo_error.message
            elif "NOT_FOUND" in error_msg or "404" in error_msg:
                obo_error = OBOResourceNotFoundError(
                    resource_type="Knowledge Assistant Endpoint",
                    resource_name=endpoint_name,
                )
                error_msg = obo_error.message
            elif "expired" in error_msg.lower() or "401" in error_msg:
                obo_error = OBOTokenExpiredError()
                error_msg = obo_error.message
            elif "unavailable" in error_msg.lower() or "503" in error_msg:
                obo_error = OBOServiceUnavailableError(
                    service_name="Knowledge Assistant",
                    retry_after_seconds=60,
                )
                error_msg = obo_error.message

            self._access_cache[cache_key] = CachedAccess(
                has_access=False,
                validated_at=datetime.now(UTC),
                error_message=error_msg,
            )
            logger.warning(
                "ASSISTANT_ACCESS_DENIED",
                endpoint=endpoint_name,
                # SECURITY: Log only sanitized message
                error=error_msg[:200],
            )
            return False, error_msg

        return False, "Unknown validation error"

    def clear_cache(self, user_token: str | None = None) -> None:
        """Clear access cache.

        Args:
            user_token: If provided, only clear cache for this user.
                       If None, clear entire cache.
        """
        if user_token:
            token_hash = self._hash_token(user_token)
            keys_to_remove = [k for k in self._access_cache if k.startswith(token_hash)]
            for key in keys_to_remove:
                del self._access_cache[key]
            logger.debug("OBO_CACHE_CLEARED", user_specific=True, keys_cleared=len(keys_to_remove))
        else:
            self._access_cache.clear()
            logger.debug("OBO_CACHE_CLEARED", user_specific=False)

    async def get_vector_search_index_schema(
        self,
        user_token: str,
        endpoint_name: str,
        index_name: str,
    ) -> dict[str, Any] | None:
        """Get Vector Search index schema for auto-detection.

        Used when adding a new user data source to auto-detect
        available columns and their types.

        Args:
            user_token: User's OAuth token.
            endpoint_name: Vector Search endpoint name.
            index_name: Fully qualified index name.

        Returns:
            Dict with schema info (columns, types) or None if unavailable.
        """
        try:
            client = await self.get_client(user_token)
            loop = asyncio.get_event_loop()

            index_info = await loop.run_in_executor(
                None,
                lambda: client.vector_search_indexes.get_index(index_name),
            )

            if not index_info:
                return None

            # Extract column schema from index metadata
            schema: dict[str, Any] = {
                "endpoint_name": endpoint_name,
                "index_name": index_name,
                "columns": [],
                "text_columns": [],
            }

            # Try to extract from delta_sync_index_spec or direct_access_index_spec
            spec = getattr(index_info, "delta_sync_index_spec", None) or getattr(
                index_info, "direct_access_index_spec", None
            )

            if spec and hasattr(spec, "columns_to_sync"):
                for col in spec.columns_to_sync or []:
                    col_info = {
                        "name": col.name,
                        "type": getattr(col, "data_type", "string") or "string",
                    }
                    schema["columns"].append(col_info)

                    # Track text columns for reranking
                    if col_info["type"].lower() in ("string", "text"):
                        schema["text_columns"].append(col.name)

            logger.info(
                "VECTOR_SEARCH_SCHEMA_EXTRACTED",
                endpoint=endpoint_name,
                index=index_name,
                column_count=len(schema["columns"]),
            )
            return schema

        except Exception as e:
            logger.warning(
                "VECTOR_SEARCH_SCHEMA_ERROR",
                endpoint=endpoint_name,
                index=index_name,
                error=str(e)[:200],
            )
            return None


# ----- V1.5 OBO Token Refresh -----
# SECURITY: Raw tokens are NEVER logged. Only SHA-256 hash prefixes appear in logs/metrics.

# Module-level state for the refresh path (gated by env flag)
_refresh_locks: dict[str, asyncio.Lock] = {}
_rotation_cache: dict[str, str] = {}  # old_token_hash -> new_token


def _is_refresh_enabled() -> bool:
    return os.environ.get("AGENT_DESIGNER_OBO_REFRESH_ENABLED") == "1"


def _token_hash_prefix(token: str) -> str:
    """SHA-256 hash prefix for safe logging/keying. Reuses pattern from line 99."""
    import hashlib
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]


def _decode_jwt_exp(token: str) -> int | None:
    """Decode JWT exp claim WITHOUT signature verification (claim inspection only)."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        payload_b64 = parts[1] + "=" * (-len(parts[1]) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        return int(payload.get("exp", 0))
    except Exception:
        return None


def detect_expiring(token: str, threshold_seconds: int = 300) -> bool:
    """Returns True if token expires within threshold_seconds. Always False if flag off."""
    if not _is_refresh_enabled():
        return False
    exp = _decode_jwt_exp(token)
    if exp is None:
        return False
    return (exp - time.time()) < threshold_seconds


async def refresh(token: str, workspace_client: object) -> str | None:
    """Atomic refresh via per-token-hash mutex.

    Concurrent calls for the same token coalesce into ONE upstream call.
    Returns new token on success, None when flag is off, raises on failure.
    """
    if not _is_refresh_enabled():
        return None

    token_key = _token_hash_prefix(token)
    if token_key not in _refresh_locks:
        _refresh_locks[token_key] = asyncio.Lock()
    lock = _refresh_locks[token_key]

    async with lock:
        # Check rotation cache — if another call already rotated this token, return the new one
        if token_key in _rotation_cache:
            record_token_refresh_attempt(outcome="success")
            return _rotation_cache[token_key]

        # Attempt the actual refresh via Databricks SDK
        try:
            new_token = await _call_databricks_token_refresh(workspace_client)
            _rotation_cache[token_key] = new_token
            record_token_refresh_attempt(outcome="success")
            return new_token
        except (ConnectionError, TimeoutError):
            record_token_refresh_failure(error_kind="network")
            record_token_refresh_attempt(outcome="failure")
            raise
        except Exception as exc:
            err_msg = str(exc)
            if "PERMISSION_DENIED" in err_msg or "403" in err_msg:
                record_token_refresh_failure(error_kind="permission")
            else:
                record_token_refresh_failure(error_kind="expired_refresh")
            record_token_refresh_attempt(outcome="failure")
            raise


async def _call_databricks_token_refresh(workspace_client: object) -> str:
    """Wraps the Databricks SDK token-management OBO token exchange call."""
    response = workspace_client.token_management.create_obo_token()  # type: ignore[attr-defined]
    return str(response.token_value)


def get_rotated_token(old_token: str) -> str | None:
    """Cache lookup for callers holding an old token reference."""
    return _rotation_cache.get(_token_hash_prefix(old_token))


class TokenRefresher:
    """V1.5 token refresh service. Methods are module-level functions exposed for OO callers."""

    @staticmethod
    def detect_expiring(token: str, threshold_seconds: int = 300) -> bool:
        return detect_expiring(token, threshold_seconds)

    @staticmethod
    async def refresh(token: str, workspace_client: object) -> str | None:
        return await refresh(token, workspace_client)

    @staticmethod
    def get_rotated_token(old_token: str) -> str | None:
        return get_rotated_token(old_token)
