"""Unit tests for services/deployment/github_tag_probe.py (Section S4a).

Covers:
- 200 tag response → reachable=True.
- 404 tag + 200 branch response → reachable=True.
- 404 tag + 404 branch response → reachable=False, error_kind="framework_tag_unreachable".
- 403 response → reachable=True (fail-open), note="rate_limited_or_unauthorized".
- 5xx response → reachable=True (fail-open), note="probe_unavailable:http_*".
- ConnectionError → reachable=True (fail-open), note="probe_unavailable".
- Timeout → reachable=True (fail-open), note="probe_unavailable".
- Non-GitHub URL → reachable=True, note="non_github_url_skip".
- GitHub URL with .git suffix → parsed correctly.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from deep_research.services.deployment.github_tag_probe import probe_framework_tag


def _make_http_response(status_code: int) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    return resp


def _make_http_client(status_code: int) -> MagicMock:
    """Build a mock http_client whose .get() returns a response with given status."""
    client = MagicMock()
    client.get = AsyncMock(return_value=_make_http_response(status_code))
    return client


def _make_http_client_sequence(*status_codes: int) -> MagicMock:
    """Build a mock http_client whose .get() returns responses in order."""
    client = MagicMock()
    client.get = AsyncMock(
        side_effect=[_make_http_response(status_code) for status_code in status_codes]
    )
    return client


def _make_error_client(exc: Exception) -> MagicMock:
    """Build a mock http_client whose .get() raises exc."""
    client = MagicMock()
    client.get = AsyncMock(side_effect=exc)
    return client


GITHUB_URL = "https://github.com/mshtelma/databricks-deep-research-agent"
GIT_TAG = "v0.3.0"


class TestProbeFrameworkTag:
    @pytest.mark.asyncio
    async def test_200_is_reachable(self) -> None:
        client = _make_http_client(200)
        result = await probe_framework_tag(
            git_url=GITHUB_URL, git_tag=GIT_TAG, http_client=client
        )
        assert result.reachable is True
        assert result.error_kind is None
        assert result.note is None

    @pytest.mark.asyncio
    async def test_404_is_unreachable(self) -> None:
        client = _make_http_client(404)
        result = await probe_framework_tag(
            git_url=GITHUB_URL, git_tag=GIT_TAG, http_client=client
        )
        assert result.reachable is False
        assert result.error_kind == "framework_tag_unreachable"
        assert result.note == f"ref_not_found:{GIT_TAG}"

    @pytest.mark.asyncio
    async def test_branch_ref_is_reachable_when_tag_missing(self) -> None:
        client = _make_http_client_sequence(404, 200)
        result = await probe_framework_tag(
            git_url=GITHUB_URL, git_tag="master", http_client=client
        )
        assert result.reachable is True
        assert result.error_kind is None
        assert result.note is None
        assert client.get.await_count == 2

    @pytest.mark.asyncio
    async def test_403_is_fail_open(self) -> None:
        client = _make_http_client(403)
        result = await probe_framework_tag(
            git_url=GITHUB_URL, git_tag=GIT_TAG, http_client=client
        )
        assert result.reachable is True
        assert result.note == "rate_limited_or_unauthorized"

    @pytest.mark.asyncio
    async def test_401_is_fail_open(self) -> None:
        client = _make_http_client(401)
        result = await probe_framework_tag(
            git_url=GITHUB_URL, git_tag=GIT_TAG, http_client=client
        )
        assert result.reachable is True
        assert result.note == "rate_limited_or_unauthorized"

    @pytest.mark.asyncio
    async def test_500_is_fail_open(self) -> None:
        client = _make_http_client(500)
        result = await probe_framework_tag(
            git_url=GITHUB_URL, git_tag=GIT_TAG, http_client=client
        )
        assert result.reachable is True
        assert result.note is not None and "probe_unavailable" in result.note

    @pytest.mark.asyncio
    async def test_connection_error_is_fail_open(self) -> None:
        client = _make_error_client(ConnectionError("connection refused"))
        result = await probe_framework_tag(
            git_url=GITHUB_URL, git_tag=GIT_TAG, http_client=client
        )
        assert result.reachable is True
        assert result.note == "probe_unavailable"

    @pytest.mark.asyncio
    async def test_timeout_is_fail_open(self) -> None:
        client = _make_error_client(TimeoutError())
        result = await probe_framework_tag(
            git_url=GITHUB_URL, git_tag=GIT_TAG, http_client=client
        )
        assert result.reachable is True
        assert result.note == "probe_unavailable"

    @pytest.mark.asyncio
    async def test_non_github_url_skipped(self) -> None:
        result = await probe_framework_tag(
            git_url="https://gitlab.com/owner/repo",
            git_tag=GIT_TAG,
            http_client=None,  # no client — must not be called
        )
        assert result.reachable is True
        assert result.note == "non_github_url_skip"

    @pytest.mark.asyncio
    async def test_github_url_with_git_suffix(self) -> None:
        client = _make_http_client(200)
        result = await probe_framework_tag(
            git_url="https://github.com/mshtelma/databricks-deep-research-agent.git",
            git_tag=GIT_TAG,
            http_client=client,
        )
        assert result.reachable is True
        # Verify the call was made with the correct URL (owner/repo extracted).
        call_args = client.get.call_args
        url_called = call_args[0][0] if call_args[0] else call_args[1].get("url", "")
        # The URL should contain the owner/repo without the .git suffix
        # Note: the GitHub API URL path itself contains "/git/" — only the
        # repo-name portion should have the .git suffix stripped.
        assert "mshtelma" in url_called
        assert "databricks-deep-research-agent" in url_called
        assert "databricks-deep-research-agent.git" not in url_called

    @pytest.mark.asyncio
    async def test_github_url_with_trailing_slash(self) -> None:
        client = _make_http_client(200)
        result = await probe_framework_tag(
            git_url="https://github.com/mshtelma/databricks-deep-research-agent/",
            git_tag=GIT_TAG,
            http_client=client,
        )
        assert result.reachable is True
