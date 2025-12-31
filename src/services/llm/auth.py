"""OAuth credential provider for LLM client.

Provides automatic token refresh for profile-based Databricks authentication.
Follows the same pattern as src/db/lakebase_auth.py for consistency.
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from databricks.sdk import WorkspaceClient

from src.core.logging_utils import get_logger

logger = get_logger(__name__)

# OAuth token configuration
TOKEN_LIFETIME = timedelta(hours=1)
TOKEN_REFRESH_BUFFER = timedelta(minutes=5)


@dataclass
class LLMCredential:
    """OAuth credential for LLM API access."""

    token: str
    expires_at: datetime

    @property
    def is_expired(self) -> bool:
        """Check if token is expired or about to expire.

        Returns True when within TOKEN_REFRESH_BUFFER of expiration,
        allowing proactive refresh before actual expiry.
        """
        return datetime.now(UTC) >= (self.expires_at - TOKEN_REFRESH_BUFFER)


class LLMCredentialProvider:
    """Provides and refreshes OAuth credentials for LLM API access.

    This class manages OAuth token lifecycle for profile-based Databricks
    authentication. Tokens are automatically refreshed when they are within
    5 minutes of expiration.

    Example:
        provider = LLMCredentialProvider(profile="my-profile")
        credential = provider.get_credential()
        # Use credential.token for API calls
        # Call get_credential() again before each request to ensure freshness
    """

    def __init__(self, profile: str) -> None:
        """Initialize credential provider.

        Args:
            profile: Databricks config profile name from ~/.databrickscfg.
        """
        self._profile = profile
        self._credential: LLMCredential | None = None
        self._workspace_client: WorkspaceClient | None = None

    def _get_workspace_client(self) -> WorkspaceClient:
        """Get or create WorkspaceClient with profile auth.

        Returns:
            Configured WorkspaceClient instance.
        """
        if self._workspace_client is None:
            self._workspace_client = WorkspaceClient(profile=self._profile)
        return self._workspace_client

    def get_credential(self, force_refresh: bool = False) -> LLMCredential:
        """Get valid OAuth credential, refreshing if needed.

        Args:
            force_refresh: Force credential refresh even if not expired.

        Returns:
            Valid LLM credential with fresh token.
        """
        if (
            self._credential is None
            or self._credential.is_expired
            or force_refresh
        ):
            self._credential = self._generate_credential()

        return self._credential

    def _generate_credential(self) -> LLMCredential:
        """Generate new OAuth credential via Databricks SDK.

        Returns:
            Fresh LLM credential.
        """
        client = self._get_workspace_client()
        client.config.authenticate()
        token = client.config.oauth_token().access_token

        # Calculate expiration (OAuth tokens are typically 1 hour)
        expires_at = datetime.now(UTC) + TOKEN_LIFETIME

        logger.debug(
            "LLM_CREDENTIAL_GENERATED",
            profile=self._profile,
            expires_at=expires_at.isoformat(),
        )

        return LLMCredential(
            token=token,
            expires_at=expires_at,
        )

    def get_base_url(self) -> str:
        """Get the base URL for LLM endpoints.

        Returns:
            Databricks serving endpoints URL.
        """
        client = self._get_workspace_client()
        return f"{client.config.host}/serving-endpoints"
