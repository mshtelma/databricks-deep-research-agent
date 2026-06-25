"""Contract tests for ``CachedSessionService`` (F-OTHER.2).

Exercises session create / get / touch / expire semantics.
Runs against the parametric ``stack`` fixture from conftest.py
(FakeBackend by default; real backends via env vars).
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from deep_research.services.cached.session import CachedSessionService


class TestCachedSessionServiceContract:
    """Session lifecycle tests."""

    @pytest.mark.asyncio
    async def test_get_or_create_creates_new_session(self, stack) -> None:
        svc = CachedSessionService(stack)
        user = f"user_{uuid4().hex[:8]}"

        session, token, is_new = await svc.get_or_create_session(user_id=user)

        assert session is not None
        assert session.id is not None
        assert session.user_id == user
        assert len(token) > 0
        assert is_new is True

    @pytest.mark.asyncio
    async def test_get_or_create_returns_existing_session(self, stack) -> None:
        svc = CachedSessionService(stack)
        user = f"user_{uuid4().hex[:8]}"

        session1, token1, is_new1 = await svc.get_or_create_session(user_id=user)
        assert is_new1 is True

        session2, token2, is_new2 = await svc.get_or_create_session(
            user_id=user, session_token=token1
        )
        assert is_new2 is False
        assert session2.id == session1.id
        assert token2 == token1

    @pytest.mark.asyncio
    async def test_get_by_token_for_user_verifies_ownership(self, stack) -> None:
        svc = CachedSessionService(stack)
        user_a = f"user_{uuid4().hex[:8]}"
        user_b = f"user_{uuid4().hex[:8]}"

        _, token, _ = await svc.get_or_create_session(user_id=user_a)

        # Owner can fetch
        found = await svc.get_by_token_for_user(token, user_a)
        assert found is not None

        # Different user cannot fetch
        not_found = await svc.get_by_token_for_user(token, user_b)
        assert not_found is None

    @pytest.mark.asyncio
    async def test_get_returns_session_by_id(self, stack) -> None:
        svc = CachedSessionService(stack)
        user = f"user_{uuid4().hex[:8]}"

        session, _, _ = await svc.get_or_create_session(user_id=user)
        fetched = await svc.get(session.id)

        assert fetched is not None
        assert fetched.id == session.id
        assert fetched.user_id == user

    @pytest.mark.asyncio
    async def test_touch_extends_ttl(self, stack) -> None:
        svc = CachedSessionService(stack)
        user = f"user_{uuid4().hex[:8]}"

        session, _, _ = await svc.get_or_create_session(user_id=user)
        original_expires = session.expires_at

        # Touch should extend expires_at
        session.touch()
        await svc.update(session)

        fetched = await svc.get(session.id)
        assert fetched is not None
        assert fetched.expires_at >= original_expires

    @pytest.mark.asyncio
    async def test_get_session_status_no_token(self, stack) -> None:
        svc = CachedSessionService(stack)
        user = f"user_{uuid4().hex[:8]}"

        status = await svc.get_session_status(None, user)
        assert status["has_session"] is False
        assert status["chat_count"] == 0
        assert status["expires_at"] is None

    @pytest.mark.asyncio
    async def test_get_session_status_with_valid_token(self, stack) -> None:
        svc = CachedSessionService(stack)
        user = f"user_{uuid4().hex[:8]}"

        _, token, _ = await svc.get_or_create_session(user_id=user)

        status = await svc.get_session_status(token, user)
        assert status["has_session"] is True
        assert status["expires_at"] is not None

    @pytest.mark.asyncio
    async def test_get_session_status_wrong_user(self, stack) -> None:
        svc = CachedSessionService(stack)
        user_a = f"user_{uuid4().hex[:8]}"
        user_b = f"user_{uuid4().hex[:8]}"

        _, token, _ = await svc.get_or_create_session(user_id=user_a)

        status = await svc.get_session_status(token, user_b)
        assert status["has_session"] is False

    @pytest.mark.asyncio
    async def test_cleanup_expired_removes_old_sessions(self, stack) -> None:
        svc = CachedSessionService(stack)
        user = f"user_{uuid4().hex[:8]}"

        # Create a session then manually expire it by updating expires_at
        from datetime import UTC, datetime, timedelta

        session, _, _ = await svc.get_or_create_session(user_id=user)
        session.expires_at = datetime.now(UTC) - timedelta(hours=2)
        await svc.update(session)

        count = await svc.cleanup_expired()
        assert count >= 1

        # Session should be gone
        fetched = await svc.get(session.id)
        assert fetched is None
