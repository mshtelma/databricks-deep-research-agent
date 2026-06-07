"""Abstract base for Lakebase credential providers.

Defines the contract for credential providers that support both
Provisioned and Autoscaling Lakebase backends.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal

LakebaseBackend = Literal["provisioned", "autoscaling"]

# Token refresh buffer (refresh 15 minutes before expiry to handle clock skew)
TOKEN_REFRESH_BUFFER = timedelta(minutes=15)
TOKEN_LIFETIME = timedelta(hours=1)


@dataclass
class LakebaseCredential:
    """OAuth credential for Lakebase connection."""

    token: str
    username: str
    expires_at: datetime
    # Wall-clock time this credential was minted. Used purely for diagnostics:
    # a Lakebase "password authentication failed" on a credential whose
    # ``age_s`` is near zero points at a freshly-minted-token propagation race
    # (PgBouncer / databricks_auth eventual consistency) rather than expiry.
    # Optional so providers that do not set it (or older call sites) still work.
    issued_at: datetime | None = None

    @property
    def age_s(self) -> float | None:
        """Seconds since this credential was minted, or None if unknown."""
        if self.issued_at is None:
            return None
        return (datetime.now(UTC) - self.issued_at).total_seconds()

    @property
    def is_expired(self) -> bool:
        """Check if token is expired or about to expire.

        Returns True if current time >= (expires_at - refresh buffer).
        """
        import logging

        logger = logging.getLogger(__name__)

        now = datetime.now(UTC)
        threshold = self.expires_at - TOKEN_REFRESH_BUFFER
        expired = now >= threshold
        time_until_threshold = (threshold - now).total_seconds() if not expired else 0
        logger.info(
            "LAKEBASE_CREDENTIAL_EXPIRY_CHECK now_utc=%s expires_at=%s threshold=%s "
            "is_expired=%s time_until_threshold=%.1f",
            now.isoformat(),
            self.expires_at.isoformat(),
            threshold.isoformat(),
            expired,
            time_until_threshold,
        )
        return expired


class BaseLakebaseCredentialProvider(ABC):
    """Abstract base for Lakebase credential providers."""

    _credential: LakebaseCredential | None = None

    @property
    def current_credential(self) -> LakebaseCredential | None:
        """Access cached credential without triggering refresh.

        Used by session.py for proactive expiry checks.
        """
        return self._credential

    @abstractmethod
    def get_credential(self, force_refresh: bool = False) -> LakebaseCredential:
        """Get valid OAuth credential, refreshing if needed."""
        ...

    @abstractmethod
    def get_host(self) -> str:
        """Get the hostname for the Lakebase instance."""
        ...

    @abstractmethod
    def get_port(self) -> int:
        """Get the port for the Lakebase instance."""
        ...

    @abstractmethod
    def get_database(self) -> str:
        """Get the database name."""
        ...

    @abstractmethod
    def build_connection_url(self) -> str:
        """Build PostgreSQL connection URL with OAuth token."""
        ...

    @abstractmethod
    def get_backend_type(self) -> LakebaseBackend:
        """Return the backend type identifier."""
        ...
