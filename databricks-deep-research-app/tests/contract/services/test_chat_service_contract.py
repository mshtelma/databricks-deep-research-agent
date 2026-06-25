"""Contract tests for `CachedChatService` (F-CHAT).

Exercises create / get / list / update / soft_delete / restore semantics and
the search/offset/pagination extensions on both backends via the parametric
`stack` fixture from conftest.py.

These tests run against the `FakeBackend` by default. Set
``STORAGE_TEST_LAKEBASE=1`` / ``STORAGE_TEST_WAREHOUSE=1`` to exercise
real backends.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from deep_research.services.cached.chat import CachedChatService


class TestCachedChatServiceContract:
    """Create / read / list / update / delete lifecycle."""

    @pytest.mark.asyncio
    async def test_create_and_get_by_id(self, stack) -> None:
        svc = CachedChatService(stack)
        user_id = f"user_{uuid4().hex[:8]}"

        chat = await svc.create(user_id=user_id, title="Hello")

        assert chat.id is not None
        assert chat.user_id == user_id
        assert chat.title == "Hello"

        fetched = await svc.get_by_id(chat.id)
        assert fetched is not None
        assert fetched.id == chat.id
        assert fetched.user_id == user_id

    @pytest.mark.asyncio
    async def test_get_for_user_ownership(self, stack) -> None:
        svc = CachedChatService(stack)
        user_a = f"user_{uuid4().hex[:8]}"
        user_b = f"user_{uuid4().hex[:8]}"

        chat = await svc.create(user_id=user_a, title="Mine")

        # Owner can access
        found = await svc.get_for_user(chat.id, user_a)
        assert found is not None

        # Different user cannot access
        not_found = await svc.get_for_user(chat.id, user_b)
        assert not_found is None

    @pytest.mark.asyncio
    async def test_list_returns_only_user_chats(self, stack) -> None:
        svc = CachedChatService(stack)
        user_a = f"user_{uuid4().hex[:8]}"
        user_b = f"user_{uuid4().hex[:8]}"

        await svc.create(user_id=user_a, title="Chat A1")
        await svc.create(user_id=user_a, title="Chat A2")
        await svc.create(user_id=user_b, title="Chat B1")

        chats_a, total_a = await svc.list(user_id=user_a)
        assert total_a == 2
        assert len(chats_a) == 2
        assert all(c.user_id == user_a for c in chats_a)

        chats_b, total_b = await svc.list(user_id=user_b)
        assert total_b == 1

    @pytest.mark.asyncio
    async def test_list_search_case_insensitive(self, stack) -> None:
        svc = CachedChatService(stack)
        user_id = f"user_{uuid4().hex[:8]}"

        await svc.create(user_id=user_id, title="Python Tutorial")
        await svc.create(user_id=user_id, title="JavaScript Guide")
        await svc.create(user_id=user_id, title="python basics")

        results, total = await svc.list(user_id=user_id, search="python")
        assert total == 2
        titles = {c.title for c in results}
        assert "Python Tutorial" in titles
        assert "python basics" in titles

    @pytest.mark.asyncio
    async def test_list_offset_pagination(self, stack) -> None:
        svc = CachedChatService(stack)
        user_id = f"user_{uuid4().hex[:8]}"

        for i in range(5):
            await svc.create(user_id=user_id, title=f"Chat {i}")

        page1, total = await svc.list(user_id=user_id, limit=3, offset=0)
        page2, _ = await svc.list(user_id=user_id, limit=3, offset=3)

        assert total == 5
        assert len(page1) == 3
        assert len(page2) == 2
        # No overlap
        ids1 = {c.id for c in page1}
        ids2 = {c.id for c in page2}
        assert ids1.isdisjoint(ids2)

    @pytest.mark.asyncio
    async def test_update_chat_title(self, stack) -> None:
        svc = CachedChatService(stack)
        user_id = f"user_{uuid4().hex[:8]}"

        chat = await svc.create(user_id=user_id, title="Old Title")
        updated = await svc.update_chat(chat.id, user_id, title="New Title")

        assert updated is not None
        assert updated.title == "New Title"

    @pytest.mark.asyncio
    async def test_soft_delete_hides_chat(self, stack) -> None:
        svc = CachedChatService(stack)
        user_id = f"user_{uuid4().hex[:8]}"

        chat = await svc.create(user_id=user_id, title="To Delete")
        deleted = await svc.soft_delete(chat.id, user_id)
        assert deleted is True

        # get_for_user should return None after soft delete
        found = await svc.get_for_user(chat.id, user_id)
        assert found is None

        # list should not include deleted chat
        chats, total = await svc.list(user_id=user_id)
        assert total == 0
        assert not any(c.id == chat.id for c in chats)

    @pytest.mark.asyncio
    async def test_soft_delete_wrong_user_returns_false(self, stack) -> None:
        svc = CachedChatService(stack)
        user_a = f"user_{uuid4().hex[:8]}"
        user_b = f"user_{uuid4().hex[:8]}"

        chat = await svc.create(user_id=user_a, title="Mine")
        result = await svc.soft_delete(chat.id, user_b)
        assert result is False

        # Chat still exists for owner
        still_there = await svc.get_for_user(chat.id, user_a)
        assert still_there is not None

    @pytest.mark.asyncio
    async def test_update_title_from_message_sets_title_when_empty(self, stack) -> None:
        svc = CachedChatService(stack)
        user_id = f"user_{uuid4().hex[:8]}"

        chat = await svc.create(user_id=user_id, title="")

        await svc.update_title_from_message(chat.id, "Hello, world! This is a message.")
        updated = await svc.get_by_id(chat.id)
        assert updated is not None
        assert updated.title  # not empty

    @pytest.mark.asyncio
    async def test_update_title_from_message_no_op_when_title_set(self, stack) -> None:
        svc = CachedChatService(stack)
        user_id = f"user_{uuid4().hex[:8]}"

        chat = await svc.create(user_id=user_id, title="Existing Title")
        await svc.update_title_from_message(chat.id, "New content that would replace title")

        fetched = await svc.get_by_id(chat.id)
        assert fetched is not None
        assert fetched.title == "Existing Title"

    @pytest.mark.asyncio
    async def test_get_by_id_nonexistent_returns_none(self, stack) -> None:
        svc = CachedChatService(stack)
        result = await svc.get_by_id(uuid4())
        assert result is None

    @pytest.mark.asyncio
    async def test_list_empty_for_new_user(self, stack) -> None:
        svc = CachedChatService(stack)
        user_id = f"user_{uuid4().hex[:8]}"
        chats, total = await svc.list(user_id=user_id)
        assert chats == []
        assert total == 0

    @pytest.mark.asyncio
    async def test_get_full_returns_chat_view(self, stack) -> None:
        svc = CachedChatService(stack)
        user_id = f"user_{uuid4().hex[:8]}"

        chat = await svc.create(user_id=user_id, title="Full Chat")
        full = await svc.get_full(chat.id, user_id)

        assert full is not None
        assert full.id == chat.id
        assert full.title == "Full Chat"
        # messages / sources / research_sessions are empty for a new chat
        assert full.messages == []
        assert full.sources == []

    @pytest.mark.asyncio
    async def test_get_full_wrong_user_returns_none(self, stack) -> None:
        svc = CachedChatService(stack)
        user_a = f"user_{uuid4().hex[:8]}"
        user_b = f"user_{uuid4().hex[:8]}"

        chat = await svc.create(user_id=user_a, title="Private")
        result = await svc.get_full(chat.id, user_b)
        assert result is None


class TestLoadChatsForUser:
    """Backend extension: `load_chats_for_user` single-round-trip."""

    @pytest.mark.asyncio
    async def test_returns_chat_documents_with_state(self, stack) -> None:
        svc = CachedChatService(stack)
        user_id = f"user_{uuid4().hex[:8]}"

        await svc.create(user_id=user_id, title="Doc 1")
        await svc.create(user_id=user_id, title="Doc 2")

        docs = await stack.backend.load_chats_for_user(user_id)
        assert len(docs) == 2
        for doc in docs:
            assert doc.meta.user_id == user_id
            assert doc.state is not None

    @pytest.mark.asyncio
    async def test_search_filter(self, stack) -> None:
        svc = CachedChatService(stack)
        user_id = f"user_{uuid4().hex[:8]}"

        await svc.create(user_id=user_id, title="Alpha chat")
        await svc.create(user_id=user_id, title="Beta chat")

        docs = await stack.backend.load_chats_for_user(user_id, search="alpha")
        assert len(docs) == 1
        assert docs[0].meta.title.lower().startswith("alpha")

    @pytest.mark.asyncio
    async def test_offset_pagination(self, stack) -> None:
        svc = CachedChatService(stack)
        user_id = f"user_{uuid4().hex[:8]}"

        for i in range(4):
            await svc.create(user_id=user_id, title=f"Chat {i}")

        page1 = await stack.backend.load_chats_for_user(user_id, limit=2, offset=0)
        page2 = await stack.backend.load_chats_for_user(user_id, limit=2, offset=2)

        assert len(page1) == 2
        assert len(page2) == 2
        ids1 = {d.meta.chat_id for d in page1}
        ids2 = {d.meta.chat_id for d in page2}
        assert ids1.isdisjoint(ids2)
