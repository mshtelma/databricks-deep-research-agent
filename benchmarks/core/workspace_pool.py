"""Workspace pool for distributing LLM calls across Databricks workspaces.

When ``workspace_profiles`` in config.yaml lists multiple Databricks CLI
profiles, each profile gets its own ``FrameworkLLMClient``.  An
``asyncio.Queue`` provides exclusive, FIFO (round-robin) checkout so every
workspace handles at most one question at a time.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from collections.abc import AsyncGenerator, Callable, Generator
from contextlib import asynccontextmanager
from typing import Any

from databricks_deep_research import FrameworkLLMClient

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _suppress_databricks_env() -> Generator[None, None, None]:
    """Temporarily remove Databricks env vars so explicit CLI profiles win.

    The Databricks SDK resolves config as:
    explicit constructor args > env vars > ~/.databrickscfg profile values.

    Without suppression:
    - ``DATABRICKS_HOST`` (env, priority 2) overrides the host from the
      profile section (priority 3) — all clients connect to the same workspace.
    - ``DATABRICKS_TOKEN`` triggers the direct-token path in
      ``FrameworkLLMClient.from_databricks()``, bypassing profiles entirely.
    - ``DATABRICKS_CONFIG_PROFILE`` could shadow the explicit ``profile=``
      arg in edge cases.

    This context manager clears all three during profile-based client creation
    and restores them unconditionally in the ``finally`` block.
    """
    saved: dict[str, str] = {}
    for key in ("DATABRICKS_HOST", "DATABRICKS_TOKEN", "DATABRICKS_CONFIG_PROFILE"):
        val = os.environ.pop(key, None)
        if val is not None:
            saved[key] = val
    try:
        yield
    finally:
        os.environ.update(saved)


def _build_profile_client(profile: str, model: str) -> FrameworkLLMClient:
    """Build a ``FrameworkLLMClient`` for a specific CLI profile.

    Unlike ``FrameworkLLMClient.from_databricks()``, the returned client's
    ``client_provider`` closure suppresses Databricks env vars on **every**
    token refresh — not just at creation time.  This prevents
    ``authenticate()`` from picking up ``DATABRICKS_HOST`` /
    ``DATABRICKS_CONFIG_PROFILE`` that were restored after pool creation.

    **Must be called inside** ``_suppress_databricks_env()`` so the initial
    ``WorkspaceClient`` construction reads the correct profile.

    Note: safe under asyncio because ``_fresh_client`` is fully synchronous
    (no ``await`` points) — no other coroutine can interleave with the
    env-var suppress/restore.  Not safe under threading.
    """
    from databricks.sdk import WorkspaceClient
    from openai import AsyncOpenAI

    w = WorkspaceClient(profile=profile)
    sdk_host = (w.config.host or "").rstrip("/")
    if not sdk_host:
        raise RuntimeError(
            f"WorkspaceClient(profile={profile!r}) resolved but host is empty"
        )
    base_url = f"{sdk_host}/serving-endpoints"
    mapping: dict[str, str] = {
        "simple": model,
        "analytical": model,
        "complex": model,
    }

    def _make_provider(
        ws: Any, url: str, prof: str
    ) -> Callable[[], AsyncOpenAI]:
        """Factory: closure suppresses env vars on every call."""

        def _fresh_client() -> AsyncOpenAI:
            with _suppress_databricks_env():
                try:
                    headers = ws.config.authenticate()
                except StopIteration as exc:
                    # SDK credential providers may raise StopIteration
                    # internally; convert to RuntimeError so it propagates
                    # cleanly through asyncio Futures (PEP 479).
                    raise RuntimeError(
                        f"SDK auth raised StopIteration for profile {prof}"
                    ) from exc
                token = headers.get("Authorization", "").removeprefix(
                    "Bearer "
                )
                logger.debug(
                    "WORKSPACE_POOL_TOKEN_REFRESH profile=%s host=%s token_tail=%s",
                    prof,
                    url,
                    token[-8:] if len(token) > 8 else "???",
                )
                return AsyncOpenAI(api_key=token, base_url=url)

        return _fresh_client

    provider = _make_provider(w, base_url, profile)

    return FrameworkLLMClient(
        openai_client=provider(),  # Initial client (env vars suppressed by caller)
        model_mapping=mapping,
        client_provider=provider,  # Every refresh also suppresses env vars
    )


class WorkspacePool:
    """Pool of LLM clients across Databricks workspace profiles.

    **Multi-mode** (``from_profiles``): one ``FrameworkLLMClient`` per profile.
    An ``asyncio.Queue`` provides exclusive, FIFO (round-robin) checkout —
    each workspace handles at most one question at a time.

    **Single-mode** (``single``): wraps one client with no queue gating.
    Multiple questions share the client concurrently (backward-compatible).
    """

    def __init__(
        self,
        clients: dict[str, FrameworkLLMClient],
        *,
        use_queue: bool = True,
    ) -> None:
        self._clients = clients
        # Pre-resolve outside the async generator to avoid next() raising
        # StopIteration inside @asynccontextmanager — interacts badly with
        # asyncio Futures in Python 3.11 (_chain_future / PEP 479).
        self._single_key: str | None = (
            list(clients)[0] if (not use_queue and clients) else None
        )
        self._queue: asyncio.Queue[str] | None = None
        if use_queue and len(clients) > 0:
            self._queue = asyncio.Queue()
            for profile in clients:
                self._queue.put_nowait(profile)

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_profiles(cls, profiles: list[str], model: str) -> WorkspacePool:
        """Create a pool with one LLM client per Databricks CLI profile.

        Fails fast if any profile cannot be authenticated.  Suppresses
        ``DATABRICKS_HOST``, ``DATABRICKS_TOKEN``, and
        ``DATABRICKS_CONFIG_PROFILE`` env vars during initial creation
        **and** on every subsequent token refresh (via the closure built
        by ``_build_profile_client``).
        """
        if not profiles:
            raise ValueError("profiles list must not be empty")

        clients: dict[str, FrameworkLLMClient] = {}
        with _suppress_databricks_env():
            for profile in profiles:
                try:
                    client = _build_profile_client(profile, model)
                    clients[profile] = client
                    host = str(getattr(client._client, "base_url", "?"))
                    logger.info(
                        "WORKSPACE_POOL_CLIENT_READY profile=%s host=%s",
                        profile,
                        host,
                    )
                except Exception as exc:
                    for c in clients.values():
                        try:
                            asyncio.get_event_loop().run_until_complete(c.aclose())
                        except Exception:
                            pass
                    raise RuntimeError(
                        f"Failed to create LLM client for profile '{profile}': {exc}"
                    ) from exc

        logger.info(
            "WORKSPACE_POOL_READY profiles=%s count=%d",
            ",".join(profiles),
            len(clients),
        )
        return cls(clients, use_queue=True)

    @classmethod
    def single(cls, client: FrameworkLLMClient) -> WorkspacePool:
        """Wrap a single client — no queue gating (backward-compatible).

        Multiple questions share the client concurrently; the caller's
        ``asyncio.Semaphore`` controls concurrency as before.
        """
        return cls({"__default__": client}, use_queue=False)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of workspace profiles in the pool."""
        return len(self._clients)

    @property
    def profiles(self) -> list[str]:
        """Profile names in the pool."""
        return list(self._clients.keys())

    # ------------------------------------------------------------------
    # Checkout / release
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def acquire(self) -> AsyncGenerator[tuple[str, FrameworkLLMClient], None]:
        """Check out a workspace.  Yields ``(profile_name, llm_client)``.

        * **Multi-mode**: blocks until a workspace is available, releases on
          exit (including on exception).
        * **Single-mode**: yields immediately with no exclusive checkout.
        """
        if self._queue is None:
            # Single mode — no gating, no next() (PEP 479 safe).
            yield self._single_key, self._clients[self._single_key]  # type: ignore[index]
            return

        profile = await self._queue.get()
        try:
            yield profile, self._clients[profile]
        finally:
            self._queue.put_nowait(profile)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def aclose(self) -> None:
        """Close all LLM clients in the pool."""
        for profile, client in self._clients.items():
            try:
                await client.aclose()
            except Exception as exc:
                logger.warning(
                    "WORKSPACE_POOL_CLOSE_ERROR profile=%s error=%s",
                    profile,
                    exc,
                )
