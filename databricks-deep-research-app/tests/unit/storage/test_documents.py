"""Unit tests for `deep_research.storage.documents`."""

from __future__ import annotations

from uuid import uuid4

import pytest

from deep_research.storage.documents import (
    CURRENT_SCHEMA_VERSION,
    MAX_MESSAGES,
    MAX_RESEARCH_SESSIONS,
    MAX_SOURCES,
    PLUGIN_EXT_PAYLOAD_MAX_BYTES,
    ChatDocument,
    ChatMeta,
    ChatState,
    Coverage,
    Entity,
    FileMemo,
    Finding,
    Message,
    PluginExtEntry,
    ResearchSessionState,
    Source,
    UploadedFileMeta,
)

# --- ChatDocument.new + schema defaults -----------------------------------


class TestChatDocumentNew:
    def test_new_sets_version_zero_and_title(self) -> None:
        cid = uuid4()
        doc = ChatDocument.new(cid, "alice@x.com", title="Hello")
        assert doc.meta.chat_id == cid
        assert doc.meta.user_id == "alice@x.com"
        assert doc.meta.title == "Hello"
        assert doc.meta.version == 0
        assert doc.meta.deleted_at is None
        assert doc.state.schema_version == CURRENT_SCHEMA_VERSION
        assert doc.state.chat.title == "Hello"
        assert len(doc.state.messages) == 0

    def test_new_defaults_used_when_optional_fields_absent(self) -> None:
        doc = ChatDocument.new(uuid4(), "u")
        assert doc.meta.title == ""
        assert doc.state.chat.title == ""


# --- Mutation helpers -----------------------------------------------------


class TestChatStateMutations:
    def test_add_message_appends(self) -> None:
        doc = ChatDocument.new(uuid4(), "u")
        doc.state.add_message(Message(role="user", content="Hi"))
        assert len(doc.state.messages) == 1
        assert doc.state.messages[0].role == "user"

    def test_upsert_finding_dedup_by_content_hash(self) -> None:
        state = ChatState()
        first = Finding(content_hash="h1", content="A")
        state.upsert_finding(first)
        # Same hash, different content — should replace, same id.
        replacement = Finding(content_hash="h1", content="B")
        state.upsert_finding(replacement)
        assert len(state.memory.findings) == 1
        assert state.memory.findings[0].content == "B"
        assert state.memory.findings[0].id == first.id

    def test_upsert_entity_dedup_by_name(self) -> None:
        state = ChatState()
        state.upsert_entity(Entity(name="Acme"))
        state.upsert_entity(Entity(name="Acme", type="account"))
        assert len(state.memory.entities) == 1
        assert state.memory.entities[0].type == "account"

    def test_upsert_coverage_dedup_by_topic(self) -> None:
        state = ChatState()
        state.upsert_coverage(Coverage(topic="pricing"))
        state.upsert_coverage(Coverage(topic="pricing", status="covered"))
        assert len(state.memory.coverage) == 1
        assert state.memory.coverage[0].status == "covered"

    def test_upsert_file_memo_dedup_by_id(self) -> None:
        fid = uuid4()
        state = ChatState()
        state.upsert_file_memo(FileMemo(id=fid, name="a.pdf"))
        state.upsert_file_memo(FileMemo(id=fid, name="b.pdf"))
        assert len(state.memory.files) == 1
        assert state.memory.files[0].name == "b.pdf"

    def test_upsert_plugin_ext_replaces_payload(self) -> None:
        state = ChatState()
        state.upsert_plugin_ext("sapresalesbot", {"account_brief_markdown": "# A"})
        state.upsert_plugin_ext("sapresalesbot", {"account_brief_markdown": "# B"})
        assert len(state.memory.plugin_ext) == 1
        assert (
            state.memory.plugin_ext["sapresalesbot"].payload["account_brief_markdown"]
            == "# B"
        )

    def test_upsert_plugin_ext_enforces_byte_cap(self) -> None:
        state = ChatState()
        huge = {"blob": "x" * (PLUGIN_EXT_PAYLOAD_MAX_BYTES + 100)}
        with pytest.raises(ValueError, match="exceeds"):
            state.upsert_plugin_ext("bloated", huge)

    def test_add_source_dedup_by_url_and_bumps_last_used(self) -> None:
        state = ChatState()
        state.add_source(Source(url="https://x.com", last_used_step=1))
        state.add_source(Source(url="https://x.com", last_used_step=5))
        assert len(state.sources) == 1
        assert state.sources[0].last_used_step == 5

    def test_add_source_lru_evicts_least_recent(self) -> None:
        state = ChatState()
        overflow = 10
        for i in range(MAX_SOURCES + overflow):
            state.add_source(Source(url=f"https://x.com/{i}", last_used_step=i))
        assert len(state.sources) == MAX_SOURCES
        kept_steps = {s.last_used_step for s in state.sources}
        # Added `MAX_SOURCES + overflow` distinct URLs with steps 0..N-1;
        # LRU keeps the top-MAX_SOURCES, so smallest retained is `overflow`.
        assert min(kept_steps) == overflow

    def test_upsert_research_session_trims_to_max(self) -> None:
        state = ChatState()
        for _ in range(MAX_RESEARCH_SESSIONS + 5):
            state.upsert_research_session(ResearchSessionState())
        assert len(state.research_sessions) == MAX_RESEARCH_SESSIONS


# --- Messages compaction --------------------------------------------------


class TestMessageCompaction:
    def test_messages_trim_to_max_and_fill_summary(self) -> None:
        state = ChatState()
        for i in range(MAX_MESSAGES + 7):
            state.add_message(Message(role="user", content=f"m{i}. Body."))
        assert len(state.messages) == MAX_MESSAGES
        assert state.messages_summary is not None
        assert state.messages_summary.content != ""

    def test_no_compaction_under_threshold(self) -> None:
        state = ChatState()
        for i in range(3):
            state.add_message(Message(role="user", content=f"m{i}"))
        assert state.messages_summary is None


# --- Lazy migration -------------------------------------------------------


class TestLazyMigration:
    def test_pre_versioned_rows_get_stamped(self) -> None:
        # Simulate an old row with no schema_version
        raw = {"chat": {"title": "Legacy"}}
        s = ChatState.model_validate(raw)
        assert s.schema_version == 1
        assert s.chat.title == "Legacy"

    def test_extra_keys_are_ignored_not_raised(self) -> None:
        raw = {"schema_version": 1, "chat": {"title": "T"}, "future_field": {"x": 1}}
        # Should not raise.
        s = ChatState.model_validate(raw)
        assert s.chat.title == "T"

    def test_round_trip_is_idempotent(self) -> None:
        original = ChatState()
        original.add_message(Message(role="user", content="Hi"))
        dumped = original.model_dump()
        restored = ChatState.model_validate(dumped)
        assert restored.messages[0].content == "Hi"


# --- Budgets --------------------------------------------------------------


class TestBudgets:
    def test_byte_size_nonzero(self) -> None:
        state = ChatState()
        assert state.byte_size() > 0

    def test_budget_flags_respect_size(self) -> None:
        state = ChatState()
        assert not state.is_over_hard_budget()
        assert not state.is_over_soft_budget()


# --- ChatMeta.preview_from_state ------------------------------------------


class TestChatMetaPreview:
    def test_preview_from_first_message(self) -> None:
        state = ChatState()
        state.add_message(
            Message(role="user", content="Hello world, this is the first message.")
        )
        preview = ChatMeta.preview_from_state(state)
        assert preview.startswith("Hello world")

    def test_preview_truncates_to_limit(self) -> None:
        state = ChatState()
        state.add_message(Message(role="user", content="x" * 500))
        preview = ChatMeta.preview_from_state(state)
        assert len(preview) == ChatMeta.PREVIEW_MAX_CHARS

    def test_preview_falls_back_to_summary(self) -> None:
        state = ChatState()
        for i in range(MAX_MESSAGES + 5):
            state.add_message(Message(role="user", content=f"Message {i}."))
        preview = ChatMeta.preview_from_state(state)
        # The oldest messages were summarized; preview of first remaining msg.
        assert preview != ""

    def test_preview_empty_when_no_messages(self) -> None:
        assert ChatMeta.preview_from_state(ChatState()) == ""


# --- live_file_ids projection ---------------------------------------------


class TestLiveFileIds:
    def test_returns_ids_in_order(self) -> None:
        state = ChatState()
        ids = [uuid4() for _ in range(3)]
        for i, fid in enumerate(ids):
            state.upsert_uploaded_file(
                UploadedFileMeta(id=fid, name=f"f{i}.pdf", size=10)
            )
        assert state.live_file_ids() == ids

    def test_empty_when_no_files(self) -> None:
        assert ChatState().live_file_ids() == []


# --- PluginExtEntry extra=allow -------------------------------------------


def test_plugin_ext_entry_allows_extra_keys() -> None:
    entry = PluginExtEntry.model_validate(
        {"payload": {"x": 1}, "vendor_meta": "arbitrary"}
    )
    assert entry.payload == {"x": 1}
