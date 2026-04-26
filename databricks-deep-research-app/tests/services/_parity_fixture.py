"""Shared parity-test harness for Wave-5 service cutovers.

For each service being moved from `SQLAlchemy*` to `Cached*`, the parity test
exercises both implementations with identical inputs and asserts their
outputs match after normalization (UUIDs, timestamps, ORM vs dict shape).

The harness is intentionally thin: it accepts two factory callables (one for
each impl) and a scenario coroutine that does the service-specific calls.
Individual service tests handle setup (seeding the backend, seeding the
session) and assertion specifics.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

T = TypeVar("T")

ScenarioFn = Callable[[Any], Awaitable[Any]]


@dataclass
class ParityHarness:
    """Container for a parity comparison run."""

    legacy: Any
    cached: Any

    async def assert_same(
        self,
        scenario: ScenarioFn,
        *,
        normalize: Callable[[Any], Any] | None = None,
        msg: str = "",
    ) -> None:
        """Run `scenario(legacy)` and `scenario(cached)`; assert equal."""
        legacy_result = await scenario(self.legacy)
        cached_result = await scenario(self.cached)
        if normalize is not None:
            legacy_result = normalize(legacy_result)
            cached_result = normalize(cached_result)
        if legacy_result != cached_result:
            raise AssertionError(
                f"parity mismatch: {msg}\n"
                f"  legacy: {legacy_result!r}\n"
                f"  cached: {cached_result!r}"
            )


# --- Common normalizers ---------------------------------------------------


def strip_ids(value: Any) -> Any:
    """Remove `id` / `uuid` keys from a dict/list tree so natural-key
    comparisons don't trip on generated UUIDs."""
    if isinstance(value, dict):
        return {k: strip_ids(v) for k, v in value.items() if k not in {"id"}}
    if isinstance(value, list):
        return [strip_ids(item) for item in value]
    return value


def drop_keys(*keys: str) -> Callable[[Any], Any]:
    """Build a normalizer that drops the given keys from any dict in the tree."""
    keyset = set(keys)

    def _inner(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: _inner(v) for k, v in value.items() if k not in keyset}
        if isinstance(value, list):
            return [_inner(item) for item in value]
        return value

    return _inner


def object_to_dict(obj: Any, fields: list[str]) -> dict[str, Any]:
    """Extract named attrs from an ORM/namespace object into a dict."""
    return {f: getattr(obj, f, None) for f in fields}
