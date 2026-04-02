"""Tests for FrameworkLLMClient.from_databricks() factory method."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from openai import AsyncOpenAI

from databricks_deep_research.llm.client import FrameworkLLMClient


class TestFromDatabricksDirectToken:
    """Path 1: DATABRICKS_HOST + DATABRICKS_TOKEN env vars."""

    @patch.dict(
        "os.environ",
        {"DATABRICKS_HOST": "https://my-workspace.cloud.databricks.com", "DATABRICKS_TOKEN": "dapi123"},
        clear=False,
    )
    def test_creates_client_with_correct_base_url(self) -> None:
        client = FrameworkLLMClient.from_databricks()
        inner = client._client
        assert isinstance(inner, AsyncOpenAI)
        assert str(inner.base_url).rstrip("/").endswith("/serving-endpoints")
        assert inner.api_key == "dapi123"

    @patch.dict(
        "os.environ",
        {"DATABRICKS_HOST": "https://host.com/", "DATABRICKS_TOKEN": "tok"},
        clear=False,
    )
    def test_strips_trailing_slash_from_host(self) -> None:
        client = FrameworkLLMClient.from_databricks()
        url = str(client._client.base_url)
        assert "//serving" not in url

    @patch.dict(
        "os.environ",
        {"DATABRICKS_HOST": "https://host.com", "DATABRICKS_TOKEN": "tok"},
        clear=False,
    )
    def test_no_client_provider_for_direct_token(self) -> None:
        client = FrameworkLLMClient.from_databricks()
        assert client._client_provider is None


class TestFromDatabricksWorkspaceClient:
    """Path 2: SDK auto-detect via WorkspaceClient."""

    @patch.dict("os.environ", {}, clear=True)
    @patch("databricks.sdk.WorkspaceClient")
    def test_uses_workspace_client_with_token_refresh(
        self, mock_ws_cls: MagicMock
    ) -> None:
        mock_ws = MagicMock()
        mock_ws.config.host = "https://sdk-workspace.cloud.databricks.com"
        mock_ws.config.authenticate.return_value = {"Authorization": "Bearer sdk-token-123"}
        mock_ws_cls.return_value = mock_ws

        client = FrameworkLLMClient.from_databricks()

        assert client._client_provider is not None
        assert isinstance(client._client, AsyncOpenAI)
        assert client._client.api_key == "sdk-token-123"


class TestFromDatabricksModelMapping:
    """model_mapping parameter takes precedence over model."""

    @patch.dict(
        "os.environ",
        {"DATABRICKS_HOST": "https://h.com", "DATABRICKS_TOKEN": "t"},
        clear=False,
    )
    def test_explicit_mapping_overrides_model(self) -> None:
        mapping = {"simple": "model-a", "analytical": "model-b", "complex": "model-c"}
        client = FrameworkLLMClient.from_databricks(model_mapping=mapping)
        assert client._models == mapping

    @patch.dict(
        "os.environ",
        {"DATABRICKS_HOST": "https://h.com", "DATABRICKS_TOKEN": "t"},
        clear=False,
    )
    def test_model_param_maps_all_tiers(self) -> None:
        client = FrameworkLLMClient.from_databricks(model="my-llama")
        assert client._models == {
            "simple": "my-llama",
            "analytical": "my-llama",
            "complex": "my-llama",
        }

    @patch.dict(
        "os.environ",
        {"DATABRICKS_HOST": "https://h.com", "DATABRICKS_TOKEN": "t"},
        clear=False,
    )
    def test_default_model_is_haiku(self) -> None:
        client = FrameworkLLMClient.from_databricks()
        for tier in ("simple", "analytical", "complex"):
            assert client._models[tier] == "databricks-claude-haiku-4-5"


class TestFromDatabricksProfile:
    """profile parameter is forwarded to WorkspaceClient."""

    @patch.dict("os.environ", {}, clear=True)
    @patch("databricks.sdk.WorkspaceClient")
    def test_profile_passed_to_workspace_client(self, mock_ws_cls: MagicMock) -> None:
        mock_ws = MagicMock()
        mock_ws.config.host = "https://profile-workspace.cloud.databricks.com"
        mock_ws.config.authenticate.return_value = {"Authorization": "Bearer tok"}
        mock_ws_cls.return_value = mock_ws

        FrameworkLLMClient.from_databricks(profile="my-profile")

        mock_ws_cls.assert_called_once_with(profile="my-profile")

    @patch.dict("os.environ", {}, clear=True)
    @patch("databricks.sdk.WorkspaceClient")
    def test_none_profile_uses_default_constructor(self, mock_ws_cls: MagicMock) -> None:
        mock_ws = MagicMock()
        mock_ws.config.host = "https://default.cloud.databricks.com"
        mock_ws.config.authenticate.return_value = {"Authorization": "Bearer tok"}
        mock_ws_cls.return_value = mock_ws

        FrameworkLLMClient.from_databricks()

        mock_ws_cls.assert_called_once_with()

    @patch.dict(
        "os.environ",
        {"DATABRICKS_HOST": "https://h.com", "DATABRICKS_TOKEN": "t"},
        clear=False,
    )
    def test_profile_ignored_when_direct_token_set(self) -> None:
        client = FrameworkLLMClient.from_databricks(profile="should-be-ignored")
        assert client._client.api_key == "t"


class TestFromDatabricksNoCredentials:
    """Path 3: no credentials → RuntimeError."""

    @patch.dict("os.environ", {}, clear=True)
    def test_raises_runtime_error_with_helpful_message(self) -> None:
        with patch(
            "databricks.sdk.WorkspaceClient",
            side_effect=RuntimeError("no auth"),
        ), pytest.raises(RuntimeError, match="Could not authenticate"):
            FrameworkLLMClient.from_databricks()
