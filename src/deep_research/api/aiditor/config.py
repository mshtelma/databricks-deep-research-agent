"""Configuration settings for AIditor backend."""

import os

from pydantic import BaseModel


def _parse_bool(value: str | None) -> bool:
    """Parse a string to a boolean (case-insensitive)."""
    if value is None:
        return False
    return value.lower() in ("true", "1", "yes", "on")


class AIditorConfig(BaseModel):
    """AIditor configuration."""

    default_model: str = "databricks-claude-sonnet-4"
    chat_timeout: int = 30
    mcp_timeout: int = 15
    databricks_host: str | None = os.getenv("DATABRICKS_HOST")
    databricks_token: str | None = os.getenv("DATABRICKS_TOKEN")
    databricks_profile: str | None = os.getenv("DATABRICKS_PROFILE")
    databricks_app_name: str | None = os.getenv("DATABRICKS_APP_NAME")
    enable_auto_discovery: bool = _parse_bool(os.getenv("AIDITOR_ENABLE_AUTO_DISCOVERY"))


aiditor_conf = AIditorConfig()
