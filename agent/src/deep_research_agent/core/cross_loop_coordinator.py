"""
Cross-event-loop concurrency coordinator.

This module provides optional concurrency limiting that works across
multiple event loops without deadlocks.

Key Features:
- Works across multiple event loops (no asyncio.Semaphore issues)
- Uses loop.call_soon_threadsafe for cross-loop signaling
- FIFO queue prevents starvation
- Handles loop closure gracefully
- Disabled by default (max_permits=0)
"""
import asyncio
import threading
import logging
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class _Waiter:
    """Internal waiter representation."""
    loop: asyncio.AbstractEventLoop
    future: asyncio.Future


class CrossLoopCoordinator:
    """
    Optional concurrency coordinator that works across event loops.

    This provides an alternative to asyncio.Semaphore that doesn't deadlock
    when acquirers and releasers are on different event loops.

    Design:
    - Each endpoint has a permit count and a FIFO wait queue
    - Waiters are signaled via loop.call_soon_threadsafe (cross-loop safe)
    - threading.Lock (not asyncio.Lock) for state mutations
    - Handles cancelled/closed futures gracefully

    Usage:
        # Disabled by default
        coordinator = CrossLoopCoordinator(max_permits=0)

        # Or enable with cap
        coordinator = CrossLoopCoordinator(max_permits=2)

        # Use as async context manager
        async with coordinator.maybe_acquire("my-endpoint"):
            # Only 2 requests to my-endpoint can be here simultaneously
            result = await make_request()
    """

    def __init__(self, max_permits: int):
        """
        Initialize coordinator.

        Args:
            max_permits: Maximum concurrent requests per endpoint
                        0 = disabled (no coordination)
                        >0 = enforce this limit
        """
        self._enabled = max_permits and max_permits > 0
        self._max_permits = max_permits
        # State: (permits_in_use, wait_queue)
        self._states: Dict[str, Tuple[int, Deque[_Waiter]]] = {}
        self._lock = threading.Lock()  # threading.Lock works across event loops

        if self._enabled:
            logger.info(
                f"Initialized CrossLoopCoordinator: max_permits={max_permits}"
            )
        else:
            logger.info("CrossLoopCoordinator disabled (max_permits=0)")

    def maybe_acquire(self, endpoint: str):
        """
        Get an async context manager for optional coordination.

        If coordinator is disabled, returns a no-op context manager.
        If enabled, enforces the permit limit.

        Args:
            endpoint: Endpoint identifier to coordinate

        Returns:
            Async context manager

        Example:
            async with coordinator.maybe_acquire("endpoint"):
                # Protected section
                pass
        """
        if not self._enabled:
            # Disabled - return no-op context manager
            @asynccontextmanager
            async def noop():
                yield
            return noop()

        # Enabled - return coordinating context manager
        loop = asyncio.get_running_loop()

        async def acquire() -> None:
            """Acquire a permit, waiting if necessary."""
            future: Optional[asyncio.Future] = None

            with self._lock:
                permits, waiters = self._states.setdefault(endpoint, (0, deque()))

                if permits < self._max_permits:
                    # Permit available - take it immediately
                    self._states[endpoint] = (permits + 1, waiters)
                    logger.debug(
                        f"Acquired permit for {endpoint} "
                        f"({permits + 1}/{self._max_permits} in use)"
                    )
                    return

                # No permits available - create future and wait
                future = loop.create_future()
                waiters.append(_Waiter(loop=loop, future=future))
                self._states[endpoint] = (permits, waiters)
                logger.debug(
                    f"Waiting for permit for {endpoint} "
                    f"({permits}/{self._max_permits} in use, "
                    f"{len(waiters)} waiting)"
                )

            # Wait for future to be signaled (released from wait queue)
            await future
            logger.debug(f"Wait complete for {endpoint}")

        async def release() -> None:
            """Release a permit, signaling next waiter if any."""
            with self._lock:
                permits, waiters = self._states.setdefault(endpoint, (0, deque()))
                permits = max(permits - 1, 0)

                # Try to wake up next waiter
                while waiters:
                    waiter = waiters.popleft()

                    # Skip cancelled/done futures
                    if waiter.future.cancelled() or waiter.future.done():
                        continue

                    # Try to signal waiter; handle closed event loop gracefully
                    try:
                        waiter.loop.call_soon_threadsafe(waiter.future.set_result, None)
                        permits += 1  # Permit consumed by awoken waiter
                        logger.debug(
                            f"Signaled waiter for {endpoint} "
                            f"({permits}/{self._max_permits} in use)"
                        )
                        break
                    except RuntimeError as e:
                        # Loop closed - try next waiter
                        logger.debug(
                            f"Loop closed for waiter on {endpoint}, skipping: {e}"
                        )
                        continue

                self._states[endpoint] = (permits, waiters)

                if permits == 0 and not waiters:
                    logger.debug(f"Released final permit for {endpoint}")
                else:
                    logger.debug(
                        f"Released permit for {endpoint} "
                        f"({permits}/{self._max_permits} in use, "
                        f"{len(waiters)} waiting)"
                    )

        @asynccontextmanager
        async def manager():
            """Context manager that acquires/releases permits."""
            await acquire()
            try:
                yield
            finally:
                await release()

        return manager()

    def get_stats(self, endpoint: str) -> Dict[str, int]:
        """
        Get coordination statistics for an endpoint.

        Args:
            endpoint: Endpoint to query

        Returns:
            Dictionary with permits_in_use and waiters_count
        """
        with self._lock:
            if endpoint not in self._states:
                return {"permits_in_use": 0, "waiters_count": 0}

            permits, waiters = self._states[endpoint]
            return {
                "permits_in_use": permits,
                "waiters_count": len(waiters),
            }

    def reset(self) -> None:
        """Clear all state (for testing)."""
        with self._lock:
            self._states.clear()
            logger.debug("Reset cross-loop coordinator")
