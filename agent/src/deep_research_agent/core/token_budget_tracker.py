"""
Cross-loop-safe token budget tracker (ring buffer + reservations).

This module provides lock-free token budget tracking for rate limiting
that works correctly across multiple event loops and threads.

Key Features:
- Ring buffer with O(1) amortized rotation
- Reservation system prevents race conditions
- threading.Lock (not asyncio.Lock) for cross-loop safety
- Token estimation accuracy tracking
"""
import time
import threading
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class TokenBucketReservation:
    """Reservation handle returned to callers."""
    request_id: str
    tokens_reserved: int
    reserved_at: float


class TokenBudgetTracker:
    """
    Ring-buffer token tracker that works across threads and event loops.

    Uses a sliding window approach with minimal locking (only for ring buffer operations).
    Safe to use across multiple event loops and threads.

    Design Decisions:
    - Uses threading.Lock (not asyncio.Lock) so it works across event loops
    - Lock is only held for <1ms during ring buffer operations (minimal contention)
    - Token budget is the PRIMARY rate limiting mechanism
    - No artificial concurrency limits - let budget naturally limit parallelism
    - Reservation system prevents race conditions between budget check and actual usage

    Example:
        tracker = TokenBudgetTracker(tokens_per_minute=100000)

        # Reserve tokens before making request
        can_proceed, wait_time, reservation = tracker.reserve(estimated_tokens=5000)

        if not can_proceed:
            await asyncio.sleep(wait_time)
            # Retry reservation...

        try:
            # Make request...
            response = await llm.ainvoke(messages)

            # Commit actual usage
            tracker.commit(reservation, actual_tokens=4500)
        except:
            # Return tokens on failure
            tracker.release(reservation)
            raise
    """

    def __init__(
        self,
        tokens_per_minute: int,
        window_seconds: int = 60,
        safety_margin: float = 0.9,
        resolution_seconds: int = 1
    ):
        """
        Initialize token budget tracker.

        Args:
            tokens_per_minute: Maximum tokens allowed per minute
            window_seconds: Sliding window duration (default: 60s)
            safety_margin: Use only this fraction of budget (default: 0.9 = 90%)
            resolution_seconds: Bucket granularity in seconds (default: 1s)

        Raises:
            ValueError: If configuration is invalid
        """
        if resolution_seconds <= 0:
            raise ValueError("resolution_seconds must be positive")

        self.window_seconds = window_seconds
        self.resolution_seconds = resolution_seconds
        self.bucket_count = window_seconds // resolution_seconds
        if self.bucket_count == 0:
            raise ValueError("window_seconds must be >= resolution_seconds")

        self.effective_limit = int(tokens_per_minute * safety_margin)
        self._buckets = [0] * self.bucket_count
        self._bucket_epoch = [0.0] * self.bucket_count
        self._running_total = 0
        self._pending_tokens = 0
        self._reservations: Dict[str, TokenBucketReservation] = {}
        self._oldest_dirty_idx = 0  # Rotation frontier for O(1) amortized cleanup
        self._estimation_errors = deque(maxlen=100)  # Track estimation accuracy

        # threading.Lock (not asyncio.Lock) works across event loops
        self._lock = threading.Lock()

        logger.info(
            f"Initialized TokenBudgetTracker: "
            f"limit={self.effective_limit:,} tokens/min, "
            f"safety_margin={safety_margin}, "
            f"window={window_seconds}s, "
            f"buckets={self.bucket_count}"
        )

    # --- Internal helpers -------------------------------------------------

    def _bucket_index(self, now: float) -> int:
        """Get bucket index for given timestamp."""
        return int(now // self.resolution_seconds) % self.bucket_count

    def _rotate_if_needed(self, now: float) -> None:
        """
        Rotate expired buckets. Optimized to O(1) amortized via frontier tracking.

        This method scans from the oldest dirty bucket forward, expiring
        buckets until we hit one that's still fresh. This gives O(1) amortized
        complexity instead of scanning all buckets every time.
        """
        cutoff = now - self.window_seconds
        rotated = 0

        # Only scan from oldest dirty bucket (amortized O(1))
        while rotated < self.bucket_count:
            idx = self._oldest_dirty_idx
            if self._bucket_epoch[idx] > cutoff:
                break  # Hit a fresh bucket, stop

            if self._buckets[idx] != 0:
                self._running_total -= self._buckets[idx]
                self._buckets[idx] = 0
                self._bucket_epoch[idx] = now - (now % self.resolution_seconds)

            self._oldest_dirty_idx = (idx + 1) % self.bucket_count
            rotated += 1

    def _compute_wait(self, request_tokens: int, now: float) -> float:
        """
        Compute how long to wait until this request can be afforded.

        Walks buckets chronologically to find when the deficit clears.
        Bounded by window_seconds iterations.
        """
        deficit = (self._running_total + self._pending_tokens + request_tokens) - self.effective_limit
        if deficit <= 0:
            return 0.0

        # Walk buckets in chronological order to find when deficit clears
        wait = 0.0
        future_time = now
        for _ in range(self.bucket_count):
            future_time += self.resolution_seconds
            idx = self._bucket_index(future_time)
            deficit -= self._buckets[idx]
            if deficit <= 0:
                wait = max(0.0, future_time - now)
                break

        return min(wait or self.resolution_seconds, self.window_seconds)

    # --- Public API -------------------------------------------------------

    def reserve(
        self,
        estimated_tokens: int,
        request_id: Optional[str] = None
    ) -> Tuple[bool, float, Optional[TokenBucketReservation]]:
        """
        Reserve tokens for a future request.

        This is the first step in the reserve/commit/release flow.
        Prevents race conditions where multiple callers check budget
        simultaneously then all proceed.

        Args:
            estimated_tokens: Estimated tokens this request will consume
            request_id: Optional request identifier for debugging

        Returns:
            Tuple of (can_proceed, wait_time_seconds, reservation)
            - If can_proceed is True: proceed immediately, use returned reservation
            - If can_proceed is False: wait for wait_time_seconds, then retry

        Example:
            acquired, wait, reservation = tracker.reserve(5000)
            if not acquired:
                await asyncio.sleep(wait)
                # Retry...
        """
        now = time.time()
        with self._lock:
            self._rotate_if_needed(now)

            projected = self._running_total + self._pending_tokens + estimated_tokens
            if projected <= self.effective_limit:
                # Can afford - create reservation
                rid = request_id or f"res-{int(now * 1000)}"
                reservation = TokenBucketReservation(rid, estimated_tokens, now)
                self._reservations[rid] = reservation
                self._pending_tokens += estimated_tokens

                logger.debug(
                    f"Reserved {estimated_tokens:,} tokens for {rid} "
                    f"(pending={self._pending_tokens:,}, "
                    f"total={self._running_total:,})"
                )
                return True, 0.0, reservation

            # Cannot afford - compute wait time
            wait_time = self._compute_wait(estimated_tokens, now)
            logger.debug(
                f"Cannot afford {estimated_tokens:,} tokens "
                f"(usage={self._running_total + self._pending_tokens:,}, "
                f"limit={self.effective_limit:,}, wait={wait_time:.1f}s)"
            )
            return False, wait_time, None

    def commit(self, reservation: TokenBucketReservation, actual_tokens: int) -> None:
        """
        Commit actual token usage after request completes.

        This adjusts the bucket totals to reflect actual usage instead
        of the estimate. Call this in the success path.

        Args:
            reservation: The reservation handle from reserve()
            actual_tokens: Actual tokens consumed by the request

        Example:
            response = await llm.ainvoke(messages)
            actual = count_tokens(response)
            tracker.commit(reservation, actual)
        """
        now = time.time()
        with self._lock:
            self._rotate_if_needed(now)

            # Remove reservation from pending
            self._pending_tokens -= reservation.tokens_reserved
            self._reservations.pop(reservation.request_id, None)

            # Add actual usage to current bucket
            idx = self._bucket_index(now)
            bucket_epoch = now - (now % self.resolution_seconds)
            if self._bucket_epoch[idx] != bucket_epoch:
                # New bucket - zero it first
                self._running_total -= self._buckets[idx]
                self._buckets[idx] = 0
                self._bucket_epoch[idx] = bucket_epoch

            self._buckets[idx] += actual_tokens
            self._running_total += actual_tokens

            logger.debug(
                f"Committed {actual_tokens:,} tokens for {reservation.request_id} "
                f"(estimated={reservation.tokens_reserved:,}, "
                f"total={self._running_total:,})"
            )

            # Track estimation accuracy for observability
            estimated = reservation.tokens_reserved
            if estimated > 0:
                error_pct = abs(actual_tokens - estimated) / estimated * 100
                self._estimation_errors.append(error_pct)

                # Log if consistently inaccurate
                if len(self._estimation_errors) >= 50:
                    avg_error = sum(self._estimation_errors) / len(self._estimation_errors)
                    if avg_error > 20:
                        logger.warning(
                            f"Token estimation avg error: {avg_error:.1f}% "
                            f"(estimated={estimated}, actual={actual_tokens})"
                        )

    def release(self, reservation: TokenBucketReservation) -> None:
        """
        Release a reservation without committing usage.

        Call this in the failure path to return tokens to the budget
        immediately.

        Args:
            reservation: The reservation handle from reserve()

        Example:
            try:
                response = await llm.ainvoke(messages)
                tracker.commit(reservation, actual_tokens)
            except:
                tracker.release(reservation)
                raise
        """
        with self._lock:
            if reservation.request_id in self._reservations:
                self._pending_tokens -= reservation.tokens_reserved
                self._reservations.pop(reservation.request_id, None)
                logger.debug(
                    f"Released {reservation.tokens_reserved:,} tokens for {reservation.request_id}"
                )

    def get_current_usage(self) -> Tuple[int, int]:
        """
        Get current token usage statistics.

        Returns:
            Tuple of (current_usage, effective_limit)
            current_usage includes both committed and pending tokens
        """
        now = time.time()
        with self._lock:
            self._rotate_if_needed(now)
            return self._running_total + self._pending_tokens, self.effective_limit

    def debug_snapshot(self) -> Dict[str, int]:
        """
        Get internal state snapshot for debugging.

        Returns:
            Dictionary with running_total, pending_tokens, and reservation count
        """
        with self._lock:
            return {
                "running_total": self._running_total,
                "pending_tokens": self._pending_tokens,
                "reservations": len(self._reservations),
            }

    def get_estimation_stats(self) -> Dict[str, float]:
        """
        Get token estimation accuracy statistics for observability.

        Returns:
            Dictionary with avg_error_pct, max_error_pct, and sample count
        """
        with self._lock:
            if not self._estimation_errors:
                return {"avg_error_pct": 0.0, "max_error_pct": 0.0, "samples": 0}
            return {
                "avg_error_pct": sum(self._estimation_errors) / len(self._estimation_errors),
                "max_error_pct": max(self._estimation_errors),
                "samples": len(self._estimation_errors),
            }

    def reset(self) -> None:
        """Clear all usage history (for testing)."""
        with self._lock:
            self._buckets = [0] * self.bucket_count
            self._bucket_epoch = [0.0] * self.bucket_count
            self._running_total = 0
            self._pending_tokens = 0
            self._reservations.clear()
            self._oldest_dirty_idx = 0
            self._estimation_errors.clear()
            logger.debug("Reset token budget tracker")
