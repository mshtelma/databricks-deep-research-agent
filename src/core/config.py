"""Application configuration using Pydantic Settings."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, PostgresDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

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
    lakebase_instance_name: str | None = None  # e.g., "instance-xxx-yyy"
    lakebase_database: str = "deep_research"  # Custom DB we own (can create schemas/tables)
    lakebase_port: int = 5432

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

    # MLflow
    mlflow_tracking_uri: str = "databricks"
    mlflow_experiment_name: str = "deep-research-agent"

    # CORS (stored as comma-separated string, accessed via cors_origins_list property)
    cors_origins: str = "http://localhost:5173"

    @property
    def cors_origins_list(self) -> list[str]:
        """Get CORS origins as a list."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

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
        - PGHOST is set (Databricks Apps auto-injects this for database resources)
        - OR instance name is configured AND profile/app auth is available
        """
        # Priority 1: PGHOST is auto-injected by Databricks Apps
        if os.environ.get("PGHOST"):
            return True

        # Priority 2: Manual configuration with appropriate auth
        if not self.lakebase_instance_name:
            return False
        return self.is_databricks_app or bool(self.databricks_config_profile)

    @property
    def lakebase_host(self) -> str | None:
        """Get Lakebase host (either from PGHOST or derived from instance name)."""
        # Priority 1: Use PGHOST if available
        pghost = os.environ.get("PGHOST")
        if pghost:
            return pghost

        # Priority 2: Derive from instance name (not recommended, use PGHOST)
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
