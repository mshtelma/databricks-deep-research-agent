"""Contract tests for ``CachedAuditLogService`` (F-OTHER.6).

Exercises the append-only log path. Runs against the parametric ``stack``
fixture from conftest.py (FakeBackend by default; real backends via env vars).
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from deep_research.services.cached.audit_log import CachedAuditLogService


class TestCachedAuditLogServiceContract:
    """Audit log append-only lifecycle."""

    @pytest.mark.asyncio
    async def test_log_does_not_raise(self, stack) -> None:
        """log() must never raise, even with minimal args."""
        svc = CachedAuditLogService(stack)
        await svc.log(user_id="user_abc", action="LOGIN")

    @pytest.mark.asyncio
    async def test_log_with_target_and_details(self, stack) -> None:
        """log() accepts target_id UUID and details dict without error."""
        svc = CachedAuditLogService(stack)
        target = uuid4()
        await svc.log(
            user_id="user_xyz",
            action="DELETE_CHAT",
            target_id=target,
            details={"chat_id": str(target), "reason": "user-requested"},
        )

    @pytest.mark.asyncio
    async def test_log_swallows_enqueue_error(self, stack, monkeypatch) -> None:
        """log() must not propagate WriteQueue errors."""
        svc = CachedAuditLogService(stack)

        def _raise(*_a: object, **_kw: object) -> None:
            raise RuntimeError("simulated queue failure")

        monkeypatch.setattr(svc, "_append_event", _raise)
        # Must not raise
        await svc.log(user_id="u", action="WHATEVER")

    @pytest.mark.asyncio
    async def test_log_concurrent_calls(self, stack) -> None:
        """Concurrent log() calls must complete without error."""
        svc = CachedAuditLogService(stack)
        await asyncio.gather(
            *[
                svc.log(user_id=f"user_{i}", action="CONCURRENT_TEST")
                for i in range(20)
            ]
        )

    @pytest.mark.asyncio
    async def test_make_audit_log_service_factory(self, stack) -> None:
        """make_audit_log_service returns a CachedAuditLogService under cached impl."""
        from unittest.mock import MagicMock

        from deep_research.services._impl_factory import make_audit_log_service
        from deep_research.services._protocols import IAuditLogService

        settings = MagicMock()
        settings.storage_service_impl = "cached"

        svc = make_audit_log_service(settings, stack)
        assert isinstance(svc, IAuditLogService)
        # Must not raise
        await svc.log(user_id="factory_test", action="TEST")
