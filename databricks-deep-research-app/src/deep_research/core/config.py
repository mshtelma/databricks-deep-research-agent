"""Application configuration using Pydantic Settings."""

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, PostgresDsn, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_config_logger = logging.getLogger(__name__)

# Find .env in project root (parent of backend/)
# Path: backend/src/core/config.py -> backend/src/core -> backend/src -> backend -> project root
_this_file = Path(__file__).resolve()
_backend_root = _this_file.parent.parent.parent  # config.py -> core -> src -> backend
_project_root = _backend_root.parent  # backend -> root
_env_file = _project_root / ".env"
_default_model_config = str(_backend_root / "config" / "models.yaml")


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=str(_env_file) if _env_file.exists() else ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "Deep Research Agent"
    app_env: Literal["development", "staging", "production"] = "development"
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # Databricks
    databricks_host: str | None = None
    databricks_token: str | None = None
    databricks_config_profile: str | None = None

    # Databricks Apps (automatically set when running as a Databricks App)
    databricks_app_port: int | None = Field(default=None, alias="DATABRICKS_APP_PORT")
    serve_static: bool = False  # Set to True in production to serve frontend from static/

    # Lakebase (OAuth-authenticated PostgreSQL on Databricks)
    lakebase_instance_name: str | None = None  # Provisioned: e.g., "instance-xxx-yyy"
    lakebase_database: str = "deep_research"  # Custom DB we own (can create schemas/tables)
    lakebase_port: int = 5432

    # Lakebase Autoscaling
    endpoint_name: str | None = None  # e.g., "projects/<id>/branches/<id>/endpoints/<id>"

    # Database (fallback for local development when Lakebase is not configured)
    database_url: PostgresDsn | None = Field(default=None)

    @field_validator("database_url", mode="before")
    @classmethod
    def ensure_asyncpg_driver(cls, v: str | None) -> str | None:
        """Ensure the database URL uses asyncpg driver."""
        if v and "postgresql://" in v and "asyncpg" not in v:
            v = v.replace("postgresql://", "postgresql+asyncpg://")
        return v

    # Brave Search
    brave_api_key: str | None = None
    # SSL Verification (disable behind corporate proxies with SSL inspection)
    brave_verify_ssl: bool = True

    # Database pool sizing
    db_pool_size: int = 10
    db_max_overflow: int = 20

    # Per-connection asyncpg command timeout (seconds). Prevents indefinite
    # wire-protocol hangs. Set to None to disable. 60s is large enough for
    # every query this app currently issues and small enough to surface
    # infrastructure stalls.
    db_command_timeout: float | None = 60.0

    # --- Storage backend (chat-document architecture) -----------------
    # Axes are orthogonal:
    #   * storage_backend chooses the wire (lakebase / sql_warehouse / fake).
    #   * storage_service_impl chooses the service facade (sqlalchemy_legacy
    #     keeps the pre-2026-04 per-row ORM path; cached routes through the
    #     chat-document cache + queue).
    storage_backend: Literal["lakebase", "sql_warehouse", "fake"] = "lakebase"
    storage_service_impl: Literal["sqlalchemy_legacy", "cached"] = "cached"

    # SQL Warehouse parameters (required when storage_backend=sql_warehouse).
    storage_warehouse_id: str | None = None
    storage_catalog: str = "main"
    storage_schema: str = "deep_research_state"
    storage_statement_timeout_sec: float = 30.0

    # Runtime tuning for the async storage stack.
    storage_flush_interval_sec: float = 3.0
    storage_flush_size: int = 200
    storage_cache_idle_ttl_min: int = 30
    storage_cold_cache_ttl_sec: float = 60.0
    storage_cold_cache_max_entries: int = 1000
    storage_max_concurrent_hydrations: int = 5
    storage_event_buffer_cap: int = 10_000

    # Cleanup of soft-deleted chats and orphaned file_chunks.
    storage_cleanup_enabled: bool = True
    storage_cleanup_interval_sec: float = 3600.0
    storage_chat_retention_days: int = 7

    # Migration window flag — prevents `migrate_lakebase.py` from being run
    # against a live database without an explicit opt-in.
    storage_migration_mode: bool = False

    # --- Auth user-sync tuning (see middleware/auth.py) ---------------
    # The auth middleware upserts the current user's identity to
    # `user_documents` on every request, throttled by a process-level
    # cache. These knobs let prod tune latency / retry behavior without
    # a code change.
    user_sync_enabled: bool = True
    # None → use the backend-appropriate default from
    # `effective_user_sync_timeout`.
    user_sync_timeout_sec: float | None = None
    user_sync_failure_ttl_sec: int = 30
    user_sync_success_ttl_sec: int = 300
    user_sync_max_cache: int = 1024
    user_sync_lock_ttl_sec: int = 60

    @property
    def effective_user_sync_timeout(self) -> float:
        """Per-backend default timeout unless overridden.

        Lakebase cold-path covers TLS + OAuth + `PREPARE` plus the single
        `INSERT … ON CONFLICT` statement. 30 s leaves enough headroom for
        laptop-to-Lakebase-autoscaling cold-starts; earlier 15 s was too
        tight and produced `TimeoutError` traces in practice.
        SQL Warehouse statement execution has a multi-second floor with
        larger cold-start tails, so it keeps the wider 45 s budget.
        Override via `USER_SYNC_TIMEOUT_SEC` when prod tuning demands it.
        """
        if self.user_sync_timeout_sec is not None:
            return self.user_sync_timeout_sec
        return 45.0 if self.storage_backend == "sql_warehouse" else 30.0

    @model_validator(mode="after")
    def _validate_storage_backend(self) -> "Settings":
        """Enforce per-backend required fields at startup.

        Failure here surfaces a clear error before any request reaches the
        backend rather than 500'ing mid-flight. Only fires when the cached
        service impl is in use (the legacy impl bypasses the storage stack).
        """
        if self.storage_service_impl != "cached":
            return self
        if self.storage_backend == "sql_warehouse" and not self.storage_warehouse_id:
            raise ValueError(
                "STORAGE_BACKEND=sql_warehouse requires STORAGE_WAREHOUSE_ID "
                "to be set."
            )
        if self.storage_backend == "lakebase" and self.database_url is None and not self.use_lakebase:
            raise ValueError(
                "STORAGE_BACKEND=lakebase requires either LAKEBASE_INSTANCE_NAME "
                "(+ ENDPOINT_NAME for autoscaling) or DATABASE_URL."
            )
        return self

    @model_validator(mode="after")
    def _enforce_ssl_in_production(self) -> "Settings":
        """Force SSL verification in production regardless of config."""
        if not self.brave_verify_ssl and self.app_env == "production":
            _config_logger.warning(
                "SSL_VERIFY_FORCED: brave_verify_ssl forced to True in production"
            )
            self.brave_verify_ssl = True
        return self

    # Deploy-here settings (Section S)
    framework_git_url: str = "https://github.com/mshtelma/databricks-deep-research-agent"
    github_api_token: str | None = None
    deploy_here_reachability_timeout_seconds: float = 300.0
    deploy_here_probe_ttl_seconds: float = 60.0
    deploy_here_framework_tag_preflight: bool = True
    # When True, reject deploys whose ``framework_git_tag`` resolves to a
    # branch (refs/heads/...) rather than an immutable tag (refs/tags/...).
    # Branches can be force-pushed; the shell-app's pyproject pins by ref and
    # would silently pick up new framework code on next install. Defaults to
    # False to allow branch refs during active development; flip to True for
    # production tenants that require immutable framework pins. See plan
    # Phase 3 M3 and the "IMMUTABLE GIT TAG REQUIRED" comment in
    # ``templates/agent-shell-app/pyproject.toml.j2:7-8``.
    deploy_here_require_tag_only: bool = False
    deploy_here_disclose_owner: bool = True
    deploy_here_brave_secret_scope: str = "deep-research-secrets"
    deploy_here_brave_secret_key: str = "BRAVE_API_KEY"

    # MLflow
    mlflow_enabled: bool = True
    mlflow_tracking_uri: str = "databricks"
    mlflow_experiment_name: str = "deep-research-agent"
    mlflow_experiment_id: str | None = None  # Injected by Databricks Apps resource

    # CORS (stored as comma-separated string, accessed via cors_origins_list property)
    cors_origins: str = "http://localhost:5173"

    @property
    def cors_origins_list(self) -> list[str]:
        """Get CORS origins as a list."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    # CSRF / Security
    databricks_app_url: str | None = None  # e.g. "https://deep-research-agent-dev.cloud.databricks.com"
    allowed_origins: str = ""  # Comma-separated additional trusted origins
    csp_report_only: bool = False  # True = Content-Security-Policy-Report-Only

    @property
    def csrf_allowed_origins(self) -> set[str]:
        """Merge all origin sources into a normalized deduplicated set."""
        origins: set[str] = set()
        for source in (self.databricks_app_url or "", self.allowed_origins, self.cors_origins):
            for o in source.split(","):
                normalized = o.strip().lower().rstrip("/")
                if normalized:
                    origins.add(normalized)
        return origins

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.app_env == "development"

    # Rate Limiting
    default_tokens_per_minute: int = 100000

    # Model Configuration
    model_config_path: str = _default_model_config

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.app_env == "production"

    @property
    def is_databricks_app(self) -> bool:
        """Check if running as a Databricks App (DATABRICKS_APP_PORT is set)."""
        return self.databricks_app_port is not None or os.environ.get("DATABRICKS_APP_PORT") is not None

    @property
    def server_port(self) -> int:
        """Get the server port (DATABRICKS_APP_PORT or default 8000)."""
        if self.databricks_app_port is not None:
            return self.databricks_app_port
        return int(os.environ.get("DATABRICKS_APP_PORT", "8000"))

    @property
    def use_lakebase(self) -> bool:
        """Check if Lakebase authentication should be used.

        Lakebase is used when:
        - ENDPOINT_NAME is set (Autoscaling backend)
        - OR PGHOST is set (Databricks Apps auto-injects this for Provisioned)
        - OR instance name is configured AND profile/app auth is available
        """
        # Autoscaling: ENDPOINT_NAME is set
        if self.endpoint_name or os.environ.get("ENDPOINT_NAME"):
            return True

        # Provisioned Priority 1: PGHOST is auto-injected by Databricks Apps
        if os.environ.get("PGHOST"):
            return True

        # Provisioned Priority 2: Manual configuration with appropriate auth
        if not self.lakebase_instance_name:
            return False
        return self.is_databricks_app or bool(self.databricks_config_profile)

    @property
    def lakebase_host(self) -> str | None:
        """Get Lakebase host (either from PGHOST or derived from instance name).

        WARNING: This property is a FALLBACK only and provides incorrect hostnames
        when PGHOST is not set. The correct hostname must be obtained from the
        Databricks API via `inst.read_write_dns` (not derived from instance name).

        For correct hostname resolution, use:
            from deep_research.deployment.lakebase_connection import get_lakebase_host
            host = get_lakebase_host(instance_name, workspace_client)

        This property exists for backwards compatibility with code that expects
        a synchronous lakebase_host property. New code should use the async API
        lookup instead.
        """
        # Priority 1: Use PGHOST if available (CORRECT)
        pghost = os.environ.get("PGHOST")
        if pghost:
            return pghost

        # Priority 2: Derive from instance name
        # WARNING: This pattern is INCORRECT - the actual hostname comes from
        # the API's read_write_dns field (format: instance-<uid>.database.cloud.databricks.com)
        # not from the instance name. This is kept as a fallback for backwards
        # compatibility but WILL NOT WORK for most Lakebase instances.
        if not self.lakebase_instance_name:
            return None
        return f"{self.lakebase_instance_name}.database.cloud.databricks.com"

    @property
    def database_url_sync(self) -> str | None:
        """Get synchronous database URL for Alembic migrations."""
        if self.database_url is None:
            return None
        return str(self.database_url).replace("+asyncpg", "")


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
