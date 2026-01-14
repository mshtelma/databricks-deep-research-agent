"""OAuth credential provider for LLM client.

DEPRECATED: This module is a backwards-compatibility layer.
New code should use src.core.databricks_auth directly.

This module re-exports the centralized authentication types and provides
a compatibility wrapper for existing code that imports from here.
"""

from src.core.databricks_auth import (
    TOKEN_LIFETIME,
    TOKEN_REFRESH_BUFFER,
    DatabricksAuth,
    OAuthCredential,
    get_databricks_auth,
)

# Re-export with original names for backwards compatibility
LLMCredential = OAuthCredential


class LLMCredentialProvider:
    """DEPRECATED: Backwards-compatible wrapper around DatabricksAuth.

    New code should use get_databricks_auth() directly.

    This wrapper maintains the original API for existing code that depends on
    LLMCredentialProvider with profile-based initialization.
    """

    def __init__(self, profile: str | None = None) -> None:
        """Initialize credential provider.

        Args:
            profile: IGNORED - uses centralized DatabricksAuth settings.
        """
        # Profile is ignored - centralized auth determines auth mode from settings
        self._auth = get_databricks_auth()
        self._cached_credential: OAuthCredential | None = None

    def get_credential(self, force_refresh: bool = False) -> OAuthCredential:
        """Get valid OAuth credential, refreshing if needed.

        Args:
            force_refresh: Force credential refresh even if not expired.

        Returns:
            Valid credential with fresh token.
        """
        token = self._auth.get_token(force_refresh=force_refresh)

        # Create a credential object for compatibility
        if self._cached_credential is None or self._cached_credential.token != token:
            from datetime import UTC, datetime

            self._cached_credential = OAuthCredential(
                token=token,
                expires_at=datetime.now(UTC) + TOKEN_LIFETIME,
            )

        return self._cached_credential

    def get_base_url(self) -> str:
        """Get the base URL for LLM endpoints.

        Returns:
            Databricks serving endpoints URL.
        """
        return self._auth.get_base_url()


__all__ = [
    "LLMCredential",
    "LLMCredentialProvider",
    "TOKEN_LIFETIME",
    "TOKEN_REFRESH_BUFFER",
    # New exports
    "DatabricksAuth",
    "OAuthCredential",
    "get_databricks_auth",
]
