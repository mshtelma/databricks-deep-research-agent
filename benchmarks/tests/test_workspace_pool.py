"""Tests for WorkspacePool — multi-workspace LLM load balancing."""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from benchmarks.core.workspace_pool import WorkspacePool, _suppress_databricks_env


def _mock_client(name: str = "mock") -> MagicMock:
    """Create a mock FrameworkLLMClient with async close."""
    client = MagicMock()
    client.aclose = AsyncMock()
    client._name = name
    return client


# -- _suppress_direct_token_env --------------------------------------------


class TestSuppressDatabricksEnv:
    _KEYS = ("DATABRICKS_HOST", "DATABRICKS_TOKEN", "DATABRICKS_CONFIG_PROFILE")

    def test_suppresses_and_restores(self) -> None:
        os.environ["DATABRICKS_HOST"] = "https://test.cloud.databricks.com"
        os.environ["DATABRICKS_TOKEN"] = "dapi_secret"
        os.environ["DATABRICKS_CONFIG_PROFILE"] = "ais"
        try:
            with _suppress_databricks_env():
                for key in self._KEYS:
                    assert key not in os.environ
            assert os.environ["DATABRICKS_HOST"] == "https://test.cloud.databricks.com"
            assert os.environ["DATABRICKS_TOKEN"] == "dapi_secret"
            assert os.environ["DATABRICKS_CONFIG_PROFILE"] == "ais"
        finally:
            for key in self._KEYS:
                os.environ.pop(key, None)

    def test_restores_on_exception(self) -> None:
        os.environ["DATABRICKS_HOST"] = "https://test.cloud.databricks.com"
        os.environ["DATABRICKS_TOKEN"] = "dapi_secret"
        os.environ["DATABRICKS_CONFIG_PROFILE"] = "ais"
        try:
            with pytest.raises(RuntimeError):
                with _suppress_databricks_env():
                    raise RuntimeError("boom")
            assert os.environ["DATABRICKS_HOST"] == "https://test.cloud.databricks.com"
            assert os.environ["DATABRICKS_TOKEN"] == "dapi_secret"
            assert os.environ["DATABRICKS_CONFIG_PROFILE"] == "ais"
        finally:
            for key in self._KEYS:
                os.environ.pop(key, None)

    def test_noop_when_vars_not_set(self) -> None:
        for key in self._KEYS:
            os.environ.pop(key, None)
        with _suppress_databricks_env():
            for key in self._KEYS:
                assert key not in os.environ
        for key in self._KEYS:
            assert key not in os.environ


# -- WorkspacePool.single ---------------------------------------------------


class TestSingleMode:
    @pytest.mark.asyncio
    async def test_acquire_yields_client(self) -> None:
        client = _mock_client()
        pool = WorkspacePool.single(client)

        async with pool.acquire() as (profile, acquired_client):
            assert profile == "__default__"
            assert acquired_client is client

    @pytest.mark.asyncio
    async def test_no_blocking_on_concurrent_acquire(self) -> None:
        """Single mode allows unlimited concurrent access (no queue gating)."""
        client = _mock_client()
        pool = WorkspacePool.single(client)

        results: list[str] = []

        async def worker(label: str) -> None:
            async with pool.acquire() as (profile, _):
                results.append(f"{label}_start")
                await asyncio.sleep(0.01)
                results.append(f"{label}_end")

        await asyncio.gather(worker("a"), worker("b"))
        # Both start before either ends — proves no blocking.
        assert results[:2] == ["a_start", "b_start"]

    def test_size_and_profiles(self) -> None:
        pool = WorkspacePool.single(_mock_client())
        assert pool.size == 1
        assert pool.profiles == ["__default__"]


# -- WorkspacePool multi-mode -----------------------------------------------


class TestMultiMode:
    def _make_pool(self, names: list[str]) -> WorkspacePool:
        """Create a pool with named mock clients."""
        clients = {name: _mock_client(name) for name in names}
        return WorkspacePool(clients, use_queue=True)

    @pytest.mark.asyncio
    async def test_acquire_release_cycle(self) -> None:
        pool = self._make_pool(["a", "b"])

        async with pool.acquire() as (p1, _):
            assert p1 in ("a", "b")
            async with pool.acquire() as (p2, _):
                assert p2 in ("a", "b")
                assert p1 != p2  # Different workspaces.

    @pytest.mark.asyncio
    async def test_blocks_when_all_checked_out(self) -> None:
        """Third acquire blocks until a workspace is released."""
        pool = self._make_pool(["a", "b"])
        acquired: list[str] = []
        released = asyncio.Event()

        async def hold_workspace(name: str) -> None:
            async with pool.acquire() as (profile, _):
                acquired.append(profile)
                await released.wait()

        async def wait_for_workspace() -> str:
            async with pool.acquire() as (profile, _):
                return profile

        # Hold both workspaces.
        t1 = asyncio.create_task(hold_workspace("holder1"))
        t2 = asyncio.create_task(hold_workspace("holder2"))
        await asyncio.sleep(0.05)
        assert len(acquired) == 2

        # Third acquire should block.
        t3 = asyncio.create_task(wait_for_workspace())
        await asyncio.sleep(0.05)
        assert not t3.done()

        # Release → third unblocks.
        released.set()
        result = await asyncio.wait_for(t3, timeout=1.0)
        assert result in ("a", "b")
        await t1
        await t2

    @pytest.mark.asyncio
    async def test_round_robin_ordering(self) -> None:
        """Profiles are returned in FIFO order (round-robin under load)."""
        pool = self._make_pool(["a", "b", "c"])
        order: list[str] = []

        for _ in range(6):
            async with pool.acquire() as (profile, _):
                order.append(profile)

        assert order == ["a", "b", "c", "a", "b", "c"]

    @pytest.mark.asyncio
    async def test_acquire_releases_on_exception(self) -> None:
        """Workspace is returned to pool even when code inside raises."""
        pool = self._make_pool(["only"])

        with pytest.raises(ValueError):
            async with pool.acquire():
                raise ValueError("deliberate")

        # Workspace should be available again.
        async with pool.acquire() as (profile, _):
            assert profile == "only"

    def test_size_and_profiles(self) -> None:
        pool = self._make_pool(["x", "y", "z"])
        assert pool.size == 3
        assert pool.profiles == ["x", "y", "z"]


# -- WorkspacePool.from_profiles --------------------------------------------


class TestBuildProfileClient:
    """Tests for _build_profile_client and from_profiles — mock at SDK level."""

    def test_empty_profiles_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            WorkspacePool.from_profiles([], model="test-model")

    @patch("benchmarks.core.workspace_pool._build_profile_client")
    def test_creates_client_per_profile(self, mock_build: MagicMock) -> None:
        mock_build.return_value = _mock_client()

        pool = WorkspacePool.from_profiles(["prof_a", "prof_b"], model="m")

        assert pool.size == 2
        assert pool.profiles == ["prof_a", "prof_b"]
        calls = mock_build.call_args_list
        assert len(calls) == 2
        assert calls[0].args == ("prof_a", "m")
        assert calls[1].args == ("prof_b", "m")

    @patch("benchmarks.core.workspace_pool._build_profile_client")
    def test_suppresses_env_vars_during_creation(self, mock_build: MagicMock) -> None:
        os.environ["DATABRICKS_HOST"] = "https://should-be-suppressed"
        os.environ["DATABRICKS_TOKEN"] = "suppressed-token"
        os.environ["DATABRICKS_CONFIG_PROFILE"] = "ais"
        try:
            captured_env: dict[str, str | None] = {}

            def capture_env(profile: str, model: str) -> MagicMock:
                captured_env["host"] = os.environ.get("DATABRICKS_HOST")
                captured_env["token"] = os.environ.get("DATABRICKS_TOKEN")
                captured_env["profile"] = os.environ.get("DATABRICKS_CONFIG_PROFILE")
                return _mock_client()

            mock_build.side_effect = capture_env

            WorkspacePool.from_profiles(["p"], model="m")

            # During creation, all Databricks env vars should have been absent.
            assert captured_env["host"] is None
            assert captured_env["token"] is None
            assert captured_env["profile"] is None
            # After creation, env vars should be restored.
            assert os.environ["DATABRICKS_HOST"] == "https://should-be-suppressed"
            assert os.environ["DATABRICKS_TOKEN"] == "suppressed-token"
            assert os.environ["DATABRICKS_CONFIG_PROFILE"] == "ais"
        finally:
            os.environ.pop("DATABRICKS_HOST", None)
            os.environ.pop("DATABRICKS_TOKEN", None)
            os.environ.pop("DATABRICKS_CONFIG_PROFILE", None)

    @patch("benchmarks.core.workspace_pool._build_profile_client")
    def test_fails_fast_cleans_up_partial_clients(self, mock_build: MagicMock) -> None:
        good_client = _mock_client("good")
        mock_build.side_effect = [good_client, RuntimeError("bad profile")]

        with pytest.raises(RuntimeError, match="bad profile"):
            WorkspacePool.from_profiles(["good", "bad"], model="m")

        good_client.aclose.assert_called_once()

    def test_provider_suppresses_env_vars_on_refresh(self) -> None:
        """The client_provider closure must suppress env vars on EVERY call,
        not just during initial creation. This prevents authenticate() from
        picking up restored DATABRICKS_HOST / DATABRICKS_CONFIG_PROFILE."""
        from benchmarks.core.workspace_pool import _build_profile_client

        # Mock WorkspaceClient at the SDK level
        mock_ws_cls = MagicMock()
        mock_config = MagicMock()
        mock_config.host = "https://test-workspace.cloud.databricks.com"
        mock_config.authenticate.return_value = {
            "Authorization": "Bearer test-token-for-profile"
        }
        mock_ws_instance = MagicMock()
        mock_ws_instance.config = mock_config
        mock_ws_cls.return_value = mock_ws_instance

        os.environ["DATABRICKS_HOST"] = "https://wrong-host.com"
        os.environ["DATABRICKS_CONFIG_PROFILE"] = "ais"
        try:
            with patch("databricks.sdk.WorkspaceClient", mock_ws_cls):
                # Suppress env vars for initial creation (as from_profiles does)
                with _suppress_databricks_env():
                    client = _build_profile_client("test-profile", "test-model")

            # Env vars are now RESTORED — simulates real execution
            assert os.environ["DATABRICKS_HOST"] == "https://wrong-host.com"
            assert os.environ["DATABRICKS_CONFIG_PROFILE"] == "ais"

            # Simulate auth refresh: call the client_provider
            # This should suppress env vars internally
            env_during_refresh: dict[str, str | None] = {}
            original_authenticate = mock_config.authenticate

            def capturing_authenticate() -> dict[str, str]:
                env_during_refresh["host"] = os.environ.get("DATABRICKS_HOST")
                env_during_refresh["profile"] = os.environ.get("DATABRICKS_CONFIG_PROFILE")
                return original_authenticate()

            mock_config.authenticate = capturing_authenticate

            # Call the provider (simulates 403 → refresh)
            new_client = client._client_provider()

            # During refresh, env vars should have been suppressed
            assert env_during_refresh["host"] is None
            assert env_during_refresh["profile"] is None
            # After refresh, env vars should be restored
            assert os.environ["DATABRICKS_HOST"] == "https://wrong-host.com"
            assert os.environ["DATABRICKS_CONFIG_PROFILE"] == "ais"
            # The refreshed client should use the correct base_url
            assert "test-workspace" in str(new_client.base_url)
        finally:
            os.environ.pop("DATABRICKS_HOST", None)
            os.environ.pop("DATABRICKS_CONFIG_PROFILE", None)
            os.environ.pop("DATABRICKS_TOKEN", None)


# -- WorkspacePool.aclose ---------------------------------------------------


class TestAclose:
    @pytest.mark.asyncio
    async def test_closes_all_clients(self) -> None:
        clients = {n: _mock_client(n) for n in ["a", "b", "c"]}
        pool = WorkspacePool(clients, use_queue=True)
        await pool.aclose()

        for client in clients.values():
            client.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_swallows_individual_errors(self) -> None:
        """One client raising on close doesn't prevent others from closing."""
        good = _mock_client("good")
        bad = _mock_client("bad")
        bad.aclose = AsyncMock(side_effect=RuntimeError("close failed"))
        also_good = _mock_client("also_good")

        pool = WorkspacePool(
            {"good": good, "bad": bad, "also_good": also_good},
            use_queue=True,
        )
        await pool.aclose()

        good.aclose.assert_awaited_once()
        bad.aclose.assert_awaited_once()
        also_good.aclose.assert_awaited_once()
