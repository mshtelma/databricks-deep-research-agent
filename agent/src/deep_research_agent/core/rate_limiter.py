"""
Rate Limiting Infrastructure for Multi-Tier Model System

Provides comprehensive rate limiting to prevent 429 errors across multiple LLM endpoints:
- Token budget tracking with ring buffer (cross-loop safe)
- Exponential backoff with jitter for retries
- Cross-agent request coordination (cross-loop safe)
- Per-endpoint cooldown management
"""

import time
import random
import asyncio
import threading
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Tuple
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
import logging
import re

# Import new cross-loop-safe components
from .token_budget_tracker import TokenBudgetTracker
from .cross_loop_coordinator import CrossLoopCoordinator

logger = logging.getLogger(__name__)

# Constants for rate limiting behavior
DEFAULT_COOLDOWN_SECONDS = 10  # Reduced from 30s for faster recovery
COOLDOWN_SAFETY_MARGIN = 1.1   # Add 10% buffer to prevent immediate re-429
MAX_WAIT_FOR_COOLDOWN = 60     # Never wait more than 60s for cooldown


# LEGACY: Old TokenBudget class replaced by TokenBudgetTracker (cross-loop safe)
# Kept for reference only - do not use
# @dataclass
# class TokenBudget:
#     """DEPRECATED - Use TokenBudgetTracker instead"""
#     pass


class RetryStrategy:
    """
    Exponential backoff with jitter for retry logic.

    Retry delays: 5s → 10s → 20s → 30s (capped)
    """

    def __init__(
        self,
        base_delay: float = 5.0,  # Changed from 2.0 to 5.0
        max_delay: float = 30.0,  # Changed from 60.0 to 30.0
        max_retries: int = 5,
        jitter: float = 0.3
    ):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.max_retries = max_retries
        self.jitter = jitter

    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for retry attempt with exponential backoff and jitter.

        Args:
            attempt: Retry attempt number (0-indexed)

        Returns:
            Delay in seconds
        """
        # Exponential backoff
        delay = min(self.base_delay * (2 ** attempt), self.max_delay)

        # Add jitter: ±30% randomization to prevent thundering herd
        jitter_amount = delay * self.jitter * (random.random() * 2 - 1)
        final_delay = delay + jitter_amount

        return max(0.1, final_delay)  # Minimum 0.1s

    async def wait(self, attempt: int):
        """Async wait for retry delay"""
        delay = self.calculate_delay(attempt)
        logger.info(f"Retry attempt {attempt + 1}/{self.max_retries}, waiting {delay:.2f}s")
        await asyncio.sleep(delay)

    def should_retry(self, attempt: int) -> bool:
        """Check if should retry based on attempt count"""
        return attempt < self.max_retries


# LEGACY: Old RequestCoordinator class replaced by CrossLoopCoordinator (cross-loop safe)
# The old implementation used asyncio.Semaphore which deadlocks across event loops
# Kept for reference only - do not use
# class RequestCoordinator:
#     """DEPRECATED - Use CrossLoopCoordinator instead"""
#     pass


class GlobalRateLimiter:
    """
    Global rate limiter managing token budgets and cooldowns across all endpoints.

    Singleton pattern - one instance shared across all agents.
    """

    _instance: Optional['GlobalRateLimiter'] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            return

        self._initialized = True
        # Use new cross-loop-safe token budget trackers
        self.endpoint_budgets: Dict[str, TokenBudgetTracker] = {}
        self.cooldown_until: Dict[str, float] = {}
        self.retry_strategy = RetryStrategy()
        # Use new cross-loop-safe coordinator (disabled by default)
        self.coordinator = CrossLoopCoordinator(max_permits=0)
        self.total_requests = 0
        self.total_429_errors = 0
        # Use RLock (reentrant) to allow nested lock acquisition
        self.lock = threading.RLock()

        logger.info("GlobalRateLimiter initialized with cross-loop-safe components")

    def configure_endpoint(
        self,
        endpoint: str,
        tokens_per_minute: int,
        safety_margin: float = 0.9,
        window_seconds: int = 60,
        resolution_seconds: int = 1
    ):
        """Configure rate limit for an endpoint using TokenBudgetTracker"""
        with self.lock:
            self.endpoint_budgets[endpoint] = TokenBudgetTracker(
                tokens_per_minute=tokens_per_minute,
                safety_margin=safety_margin,
                window_seconds=window_seconds,
                resolution_seconds=resolution_seconds
            )
            logger.info(
                f"Configured endpoint {endpoint}: "
                f"{tokens_per_minute} tokens/min "
                f"(effective: {int(tokens_per_minute * safety_margin)}, "
                f"window={window_seconds}s)"
            )

    def is_in_cooldown(self, endpoint: str) -> bool:
        """Check if endpoint is in cooldown period"""
        with self.lock:
            if endpoint not in self.cooldown_until:
                return False

            if time.time() < self.cooldown_until[endpoint]:
                return True

            # Cooldown expired, remove it
            del self.cooldown_until[endpoint]
            logger.info(f"Cooldown expired for {endpoint}")
            return False

    def get_cooldown_remaining(self, endpoint: str) -> float:
        """Get remaining cooldown time in seconds"""
        with self.lock:
            if endpoint not in self.cooldown_until:
                return 0.0
            remaining = self.cooldown_until[endpoint] - time.time()
            return max(0, remaining)

    async def wait_for_cooldown(self, endpoint: str):
        """Wait for cooldown to expire with progressive logging"""
        remaining = self.get_cooldown_remaining(endpoint)
        if remaining > 0:
            logger.warning(
                f"⏳ COOLDOWN: Waiting {remaining:.1f}s for {endpoint} to become available"
            )

            # For long waits (>10s), log progress every 10s to show we're not hung
            if remaining > 10:
                elapsed = 0
                while elapsed < remaining:
                    wait_chunk = min(10.0, remaining - elapsed)
                    await asyncio.sleep(wait_chunk)
                    elapsed += wait_chunk
                    if elapsed < remaining:
                        logger.info(
                            f"⏳ COOLDOWN: Still waiting... "
                            f"({elapsed:.0f}s / {remaining:.0f}s elapsed) for {endpoint}"
                        )
                logger.info(f"✅ COOLDOWN: Completed for {endpoint}")
            else:
                await asyncio.sleep(remaining)
                logger.info(f"✅ COOLDOWN: Completed for {endpoint}")

    def report_429(self, endpoint: str, retry_after: Optional[int] = None):
        """
        Record 429 error and set cooldown.

        Args:
            endpoint: Endpoint that returned 429
            retry_after: Value from Retry-After header (seconds)
        """
        with self.lock:
            cooldown_seconds = retry_after if retry_after else DEFAULT_COOLDOWN_SECONDS
            # Add safety margin to prevent immediate re-429
            cooldown_with_margin = cooldown_seconds * COOLDOWN_SAFETY_MARGIN
            self.cooldown_until[endpoint] = time.time() + cooldown_with_margin
            self.total_429_errors += 1

            logger.warning(
                f"⚠️ 429 ERROR on {endpoint}! "
                f"Cooldown for {cooldown_with_margin:.1f}s (base: {cooldown_seconds}s + {(COOLDOWN_SAFETY_MARGIN-1)*100:.0f}% margin). "
                f"Total 429s: {self.total_429_errors}"
            )

    async def reserve_tokens(
        self,
        endpoint: str,
        estimated_tokens: int,
        timeout: float = 120.0
    ):
        """
        Reserve tokens using the new TokenBudgetTracker reservation flow.

        Args:
            endpoint: Model endpoint
            estimated_tokens: Estimated tokens for request
            timeout: Maximum time to wait for budget (default: 120s)

        Returns:
            TokenBucketReservation object for commit/release

        Raises:
            TimeoutError: If reservation times out
        """
        if endpoint not in self.endpoint_budgets:
            # No budget configured, create a fake reservation for consistency
            logger.debug(f"No budget configured for {endpoint}, allowing request")
            from .token_budget_tracker import TokenBucketReservation
            return TokenBucketReservation(
                request_id=f"unbounded-{int(time.time()*1000)}",
                tokens_reserved=estimated_tokens,
                reserved_at=time.time()
            )

        # Wait for cooldown if active
        await self.wait_for_cooldown(endpoint)

        tracker = self.endpoint_budgets[endpoint]
        start_time = time.monotonic()

        while True:
            acquired, wait_time, reservation = tracker.reserve(estimated_tokens)

            if acquired:
                self.total_requests += 1
                usage, limit = tracker.get_current_usage()
                logger.debug(
                    f"Reserved {estimated_tokens} tokens for {endpoint}. "
                    f"Current usage: {usage}/{limit}"
                )
                return reservation

            # Check timeout with remaining time
            elapsed = time.monotonic() - start_time
            remaining = timeout - elapsed

            if remaining <= 0:
                raise TimeoutError(
                    f"Token reservation timed out for {endpoint} after {elapsed:.1f}s. "
                    f"Usage: {tracker.get_current_usage()}"
                )

            # Wait for budget to become available (bounded by remaining timeout)
            sleep_duration = min(wait_time, remaining, 5.0)  # Cap at 5s chunks for progress logging
            logger.info(
                f"Token budget exhausted for {endpoint}. "
                f"Waiting {sleep_duration:.1f}s... "
                f"(elapsed={elapsed:.1f}s, timeout={timeout:.0f}s)"
            )
            await asyncio.sleep(sleep_duration)

    def commit_tokens(self, endpoint: str, reservation, actual_tokens: int):
        """
        Commit actual token usage after request completes.

        Args:
            endpoint: Model endpoint
            reservation: Reservation from reserve_tokens()
            actual_tokens: Actual tokens consumed
        """
        if endpoint not in self.endpoint_budgets:
            return  # No budget tracking

        tracker = self.endpoint_budgets[endpoint]
        tracker.commit(reservation, actual_tokens)

    def release_tokens(self, endpoint: str, reservation):
        """
        Release reservation on request failure.

        Args:
            endpoint: Model endpoint
            reservation: Reservation from reserve_tokens()
        """
        if endpoint not in self.endpoint_budgets:
            return  # No budget tracking

        tracker = self.endpoint_budgets[endpoint]
        tracker.release(reservation)

    def get_available_endpoints(self, endpoints: List[str]) -> List[str]:
        """
        Get list of endpoints not currently in cooldown.

        Args:
            endpoints: List of endpoint names to check

        Returns:
            List of available endpoints (not in cooldown)
        """
        available = []
        with self.lock:
            for endpoint in endpoints:
                if not self.is_in_cooldown(endpoint):
                    available.append(endpoint)

        logger.debug(f"Available endpoints: {len(available)}/{len(endpoints)}")
        return available

    def get_shortest_cooldown(self, endpoints: List[str]) -> Tuple[str, float]:
        """
        Get endpoint with shortest remaining cooldown.

        Args:
            endpoints: List of endpoint names to check

        Returns:
            Tuple of (endpoint_name, cooldown_remaining_seconds)
        """
        with self.lock:
            shortest_endpoint = None
            shortest_time = float('inf')

            for endpoint in endpoints:
                remaining = self.get_cooldown_remaining(endpoint)
                if remaining < shortest_time:
                    shortest_time = remaining
                    shortest_endpoint = endpoint

            if shortest_endpoint is None and endpoints:
                # No cooldowns, return first endpoint with 0 wait
                return endpoints[0], 0.0

            return shortest_endpoint or "", shortest_time

    def get_endpoint_status_summary(
        self,
        endpoints: List[str],
        show_priority: bool = True
    ) -> Tuple[List[Dict], int]:
        """
        Get detailed status for each endpoint.

        Args:
            endpoints: List of endpoint names to check
            show_priority: Whether to include priority numbers (based on order)

        Returns:
            Tuple of (status_list, available_count)
            status_list: [{"endpoint": str, "available": bool, "cooldown_remaining": float, "priority": int}]
            available_count: Number of endpoints not in cooldown
        """
        status = []
        available_count = 0

        with self.lock:
            for idx, endpoint in enumerate(endpoints):
                in_cooldown = self.is_in_cooldown(endpoint)
                cooldown_remaining = self.get_cooldown_remaining(endpoint) if in_cooldown else 0.0

                status.append({
                    "endpoint": endpoint,
                    "available": not in_cooldown,
                    "cooldown_remaining": cooldown_remaining,
                    "priority": idx + 1 if show_priority else None
                })

                if not in_cooldown:
                    available_count += 1

        return status, available_count

    def format_endpoint_status(self, endpoints: List[str]) -> str:
        """
        Format endpoint status as human-readable string with emoji indicators.

        Args:
            endpoints: List of endpoint names to check

        Returns:
            Formatted multi-line string showing status of each endpoint

        Example output:
            📊 Endpoint Status:
               ✅ gpt-oss-20b: Available (priority 1)
               ❄️  databricks-gpt-oss-20b: Cooldown (12.3s remaining)
        """
        status_list, available_count = self.get_endpoint_status_summary(endpoints)

        lines = ["📊 Endpoint Status:"]
        for s in status_list:
            if s["available"]:
                priority_str = f" (priority {s['priority']})" if s['priority'] is not None else ""
                lines.append(f"   ✅ {s['endpoint']}: Available{priority_str}")
            else:
                lines.append(f"   ❄️  {s['endpoint']}: Cooldown ({s['cooldown_remaining']:.1f}s remaining)")

        return "\n".join(lines)

    async def wait_for_any_endpoint(
        self,
        endpoints: List[str],
        timeout: float = MAX_WAIT_FOR_COOLDOWN
    ) -> Optional[str]:
        """
        Wait until ANY endpoint becomes available.

        Args:
            endpoints: List of endpoint names to monitor
            timeout: Maximum seconds to wait

        Returns:
            First endpoint that becomes available, or None if timeout
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            available = self.get_available_endpoints(endpoints)
            if available:
                logger.info(f"Endpoint available after {time.time() - start_time:.1f}s wait")
                return available[0]

            # Check again in 0.5s
            await asyncio.sleep(0.5)

        logger.warning(f"No endpoints became available within {timeout}s timeout")
        return None

    def get_stats(self) -> Dict:
        """Get rate limiter statistics"""
        with self.lock:
            stats = {
                "total_requests": self.total_requests,
                "total_429_errors": self.total_429_errors,
                "endpoints": {}
            }

            for endpoint, budget in self.endpoint_budgets.items():
                in_cooldown = self.is_in_cooldown(endpoint)
                cooldown_remaining = self.get_cooldown_remaining(endpoint)

                stats["endpoints"][endpoint] = {
                    "current_usage": budget.get_current_usage(),
                    "limit": budget.effective_limit,
                    "in_cooldown": in_cooldown,
                    "cooldown_remaining_seconds": cooldown_remaining
                }

            return stats


# Global singleton instance
_global_rate_limiter: Optional[GlobalRateLimiter] = None


def get_global_rate_limiter() -> GlobalRateLimiter:
    """Get or create global rate limiter instance"""
    global _global_rate_limiter
    if _global_rate_limiter is None:
        _global_rate_limiter = GlobalRateLimiter()
    return _global_rate_limiter


def configure_rate_limiting(config: Dict):
    """
    Configure global rate limiter from config dict.

    Example config:
    {
        "models": {
            "simple": {
                "endpoints": ["endpoint1"],
                "tokens_per_minute": 600000
            }
        },
        "rate_limiting": {
            "retry": {...},
            "coordination": {...}
        }
    }
    """
    limiter = get_global_rate_limiter()

    # Configure endpoints from models config
    models_config = config.get("models", {})
    for tier_name, tier_config in models_config.items():
        tokens_per_minute = tier_config.get("tokens_per_minute", 100000)
        safety_margin = tier_config.get("safety_margin", 0.9)

        for endpoint in tier_config.get("endpoints", []):
            limiter.configure_endpoint(
                endpoint=endpoint,
                tokens_per_minute=tokens_per_minute,
                safety_margin=safety_margin
            )

    # Configure retry strategy
    rate_config = config.get("rate_limiting", {})
    retry_config = rate_config.get("retry", {})
    if retry_config:
        limiter.retry_strategy = RetryStrategy(
            base_delay=retry_config.get("base_delay_seconds", 2.0),
            max_delay=retry_config.get("max_delay_seconds", 60.0),
            max_retries=retry_config.get("max_retries", 5),
            jitter=retry_config.get("jitter", 0.3)
        )

    # Configure coordination with CrossLoopCoordinator
    coord_config = rate_config.get("coordination", {})
    if coord_config:
        max_concurrent = coord_config.get("max_concurrent_per_endpoint", 0)
        limiter.coordinator = CrossLoopCoordinator(max_permits=max_concurrent)
        logger.info(f"Coordination configured: max_permits={max_concurrent}")

    logger.info("Rate limiting configured successfully with cross-loop-safe components")
