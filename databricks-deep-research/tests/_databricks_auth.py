from __future__ import annotations

import os
from dataclasses import dataclass

from openai import AsyncOpenAI


@dataclass(frozen=True)
class DatabricksAuthConfig:
    host: str
    token: str
    auth_source: str


def has_databricks_credential_hint() -> bool:
    token = os.getenv("DATABRICKS_TOKEN")
    host = os.getenv("DATABRICKS_HOST")
    profile = os.getenv("DATABRICKS_CONFIG_PROFILE")
    return bool(profile or (token and host))


def resolve_databricks_auth() -> DatabricksAuthConfig:
    profile = os.getenv("DATABRICKS_CONFIG_PROFILE")
    token = os.getenv("DATABRICKS_TOKEN")
    host = os.getenv("DATABRICKS_HOST")

    profile_error: Exception | None = None
    if profile:
        try:
            from databricks.sdk import WorkspaceClient

            workspace = WorkspaceClient(profile=profile)
            auth_headers = workspace.config.authenticate()
            bearer = auth_headers.get("Authorization", "")
            resolved_token = bearer.removeprefix("Bearer ").strip()
            resolved_host = str(workspace.config.host or "").strip().rstrip("/")
            if not resolved_token:
                raise RuntimeError(
                    f"Profile '{profile}' authenticate() did not return a bearer token"
                )
            if not resolved_host:
                raise RuntimeError(f"Profile '{profile}' did not resolve a Databricks host")
            return DatabricksAuthConfig(
                host=resolved_host,
                token=resolved_token,
                auth_source=f"profile:{profile}",
            )
        except Exception as exc:  # pragma: no cover - exercised in env-specific tests
            profile_error = exc

    if token and host:
        return DatabricksAuthConfig(
            host=host.rstrip("/"),
            token=token,
            auth_source="token",
        )

    details: list[str] = []
    if profile_error is not None:
        details.append(f"profile auth failed: {profile_error}")
    if token and not host:
        details.append("DATABRICKS_TOKEN is set but DATABRICKS_HOST is missing")
    if host and not token:
        details.append("DATABRICKS_HOST is set but DATABRICKS_TOKEN is missing")
    if not details:
        details.append(
            "set either DATABRICKS_CONFIG_PROFILE or both DATABRICKS_TOKEN and DATABRICKS_HOST"
        )
    raise RuntimeError("Databricks auth unavailable: " + "; ".join(details))


def create_async_openai_client() -> AsyncOpenAI:
    auth = resolve_databricks_auth()
    return AsyncOpenAI(api_key=auth.token, base_url=f"{auth.host}/serving-endpoints")
