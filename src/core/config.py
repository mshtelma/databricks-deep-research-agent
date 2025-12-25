"""Application configuration using Pydantic Settings."""

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

    # Lakebase (OAuth-authenticated PostgreSQL on Databricks)
    lakebase_instance_name: str | None = None  # e.g., "instance-xxx-yyy"
    lakebase_database: str = "deep_research"   # Database name
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
    def use_lakebase(self) -> bool:
        """Check if Lakebase authentication should be used."""
        return bool(self.lakebase_instance_name and self.databricks_config_profile)

    @property
    def lakebase_host(self) -> str | None:
        """Derive Lakebase host from instance name."""
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
