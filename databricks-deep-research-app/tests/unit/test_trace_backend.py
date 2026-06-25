"""Unit tests for ``resolve_trace_backend()`` MLflow backend selection.

The resolver decides where the complex / scaffold live suites send their
MLflow traces: an explicit ``MLFLOW_TRACKING_URI`` override, the Databricks
workspace experiment when creds are present, or a local MLflow OSS sqlite
store otherwise. These tests pin that precedence without touching MLflow.
"""

from __future__ import annotations

import pytest
from tests.shared import (
    DEFAULT_DATABRICKS_EXPERIMENT,
    LOCAL_TRACE_EXPERIMENT,
    resolve_trace_backend,
)

_TRACE_ENV = (
    "MLFLOW_TRACKING_URI",
    "MLFLOW_EXPERIMENT_NAME",
    "DATABRICKS_TOKEN",
    "DATABRICKS_CONFIG_PROFILE",
    "DATABRICKS_HOST",
)


@pytest.fixture
def clean_trace_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every env var the resolver reads, for a deterministic baseline."""
    for var in _TRACE_ENV:
        monkeypatch.delenv(var, raising=False)


def test_explicit_tracking_uri_wins_over_databricks_creds(
    monkeypatch: pytest.MonkeyPatch, clean_trace_env: None
) -> None:
    # Even with Databricks creds present, an explicit URI takes precedence.
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    monkeypatch.setenv("DATABRICKS_CONFIG_PROFILE", "myprofile")

    uri, experiment = resolve_trace_backend()

    assert uri == "http://localhost:5000"
    assert experiment == DEFAULT_DATABRICKS_EXPERIMENT


def test_explicit_uri_uses_experiment_override(
    monkeypatch: pytest.MonkeyPatch, clean_trace_env: None
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "/Shared/custom-experiment")

    uri, experiment = resolve_trace_backend()

    assert uri == "http://localhost:5000"
    assert experiment == "/Shared/custom-experiment"


def test_databricks_creds_select_workspace_backend(
    monkeypatch: pytest.MonkeyPatch, clean_trace_env: None
) -> None:
    monkeypatch.setenv("DATABRICKS_CONFIG_PROFILE", "myprofile")

    uri, experiment = resolve_trace_backend()

    assert uri == "databricks"
    assert experiment == DEFAULT_DATABRICKS_EXPERIMENT


def test_databricks_token_also_selects_workspace_backend(
    monkeypatch: pytest.MonkeyPatch, clean_trace_env: None
) -> None:
    monkeypatch.setenv("DATABRICKS_TOKEN", "dapi-xxxx")

    uri, experiment = resolve_trace_backend()

    assert uri == "databricks"
    assert experiment == DEFAULT_DATABRICKS_EXPERIMENT


def test_local_sqlite_fallback_when_nothing_configured(
    monkeypatch: pytest.MonkeyPatch, clean_trace_env: None
) -> None:
    uri, experiment = resolve_trace_backend()

    assert uri.startswith("sqlite:///")
    # Absolute path under tests/_runs so it sits next to scaffold artifacts.
    assert uri.endswith("/_runs/mlflow.db")
    assert uri.startswith("sqlite:////")  # leading slash => absolute path
    assert experiment == LOCAL_TRACE_EXPERIMENT
