"""Unit tests for `tests.fakes.fake_backend.FakeBackend`.

The FakeBackend is the contract reference implementation used everywhere the
production backends aren't available. These tests pin its semantics so the
other storage tests can trust their fixture.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from deep_research.storage import (
    ConflictError,
    PermanentError,
    StorageBackend,
)
from deep_research.storage.documents import (
    ChatDocument,
    ChatMeta,
    ChatState,
    Message,
    PrepJobDocument,
    UploadedFileMeta,
    UserDocument,
)
from tests.fakes.fake_backend import FakeBackend


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


# --- Protocol conformance --------------------------------------------------


def test_satisfies_protocol() -> None:
    assert isinstance(FakeBackend(), StorageBackend)


# --- Chat document round-trip ---------------------------------------------


async def test_load_returns_none_for_unknown_chat() -> None:
    b = FakeBackend()
    assert await b.load_chat(uuid4()) is None


async def test_write_then_load_roundtrip() -> None:
    b = FakeBackend()
    cid = uuid4()
    doc = ChatDocument.new(cid, "u", title="T")
    doc.state.add_message(Message(role="user", content="hi"))
    new_ver = await b.write_chat(doc, expected_version=0)
    assert new_ver == 1
    loaded = await b.load_chat(cid)
    assert loaded is not None
    assert loaded.meta.version == 1
    assert loaded.state.messages[0].content == "hi"


async def test_version_gate_raises_on_stale_write() -> None:
    b = FakeBackend()
    cid = uuid4()
    doc = ChatDocument.new(cid, "u")
    await b.write_chat(doc, expected_version=0)
    with pytest.raises(ConflictError):
        await b.write_chat(doc, expected_version=0)


async def test_deep_copy_on_read_prevents_aliasing() -> None:
    b = FakeBackend()
    cid = uuid4()
    doc = ChatDocument.new(cid, "u")
    doc.state.add_message(Message(role="user", content="first"))
    await b.write_chat(doc, expected_version=0)

    loaded = await b.load_chat(cid)
    assert loaded is not None
    loaded.state.messages.clear()
    reloaded = await b.load_chat(cid)
    assert reloaded is not None
    assert len(reloaded.state.messages) == 1


async def test_soft_delete_hides_from_load() -> None:
    b = FakeBackend()
    cid = uuid4()
    doc = ChatDocument.new(cid, "u")
    doc.meta.deleted_at = _utcnow()
    await b.write_chat(doc, expected_version=0)
    assert await b.load_chat(cid) is None


# --- chat_deleted_files projection ----------------------------------------


async def test_chat_deleted_files_projection_is_maintained() -> None:
    b = FakeBackend()
    cid = uuid4()
    fid1, fid2 = uuid4(), uuid4()
    doc = ChatDocument.new(cid, "u")
    doc.state.upsert_uploaded_file(UploadedFileMeta(id=fid1, name="a.pdf", size=1))
    doc.state.upsert_uploaded_file(UploadedFileMeta(id=fid2, name="b.pdf", size=2))
    await b.write_chat(doc, expected_version=0)
    projected = [r["file_id"] for r in b.raw_events("chat_deleted_files")]
    assert set(projected) == {fid1, fid2}

    # Remove one file; projection must mirror via full overwrite.
    doc.state.uploaded_files = [f for f in doc.state.uploaded_files if f.id != fid1]
    doc.meta.version = 1  # reflect the prior write
    await b.write_chat(doc, expected_version=1)
    projected = [r["file_id"] for r in b.raw_events("chat_deleted_files")]
    assert projected == [fid2]


# --- list_chat_metas projection -------------------------------------------


async def test_list_chat_metas_only_returns_users_chats() -> None:
    b = FakeBackend()
    cid_alice, cid_bob = uuid4(), uuid4()
    await b.write_chat(ChatDocument.new(cid_alice, "alice", title="A"), expected_version=0)
    await b.write_chat(ChatDocument.new(cid_bob, "bob", title="B"), expected_version=0)
    metas = await b.list_chat_metas("alice")
    assert [m.chat_id for m in metas] == [cid_alice]


async def test_list_chat_metas_excludes_soft_deleted_by_default() -> None:
    b = FakeBackend()
    cid1, cid2 = uuid4(), uuid4()
    await b.write_chat(ChatDocument.new(cid1, "u", title="live"), expected_version=0)
    doc2 = ChatDocument.new(cid2, "u", title="dead")
    doc2.meta.deleted_at = _utcnow()
    await b.write_chat(doc2, expected_version=0)
    live = await b.list_chat_metas("u")
    assert [m.chat_id for m in live] == [cid1]
    all_metas = await b.list_chat_metas("u", include_deleted=True)
    assert len(all_metas) == 2


# --- User / PrepJob documents ---------------------------------------------


async def test_user_doc_roundtrip() -> None:
    b = FakeBackend()
    doc = UserDocument(user_id="u", preferences={"theme": "dark"})
    await b.write_user_doc(doc)
    loaded = await b.load_user_doc("u")
    assert loaded is not None and loaded.preferences == {"theme": "dark"}


async def test_prep_job_heartbeat_bypasses_write_prep_job() -> None:
    b = FakeBackend()
    jid = uuid4()
    doc = PrepJobDocument(prep_job_id=jid, account_id="acc")
    await b.write_prep_job(doc)
    new_ts = _utcnow()
    await b.write_prep_heartbeat(jid, new_ts)
    reloaded = await b.load_prep_job(jid)
    assert reloaded is not None
    assert reloaded.heartbeat == new_ts


async def test_prep_heartbeat_on_missing_job_raises() -> None:
    b = FakeBackend()
    with pytest.raises(PermanentError):
        await b.write_prep_heartbeat(uuid4(), _utcnow())


# --- Append-only tables ---------------------------------------------------


async def test_append_events_and_read_chunk() -> None:
    b = FakeBackend()
    fid = uuid4()
    await b.append_events(
        "file_chunks",
        [
            {"file_id": fid, "chunk_index": 0, "content": "a"},
            {"file_id": fid, "chunk_index": 1, "content": "b"},
        ],
    )
    chunks = await b.read_chunk(fid)
    assert [c["chunk_index"] for c in chunks] == [0, 1]
    # Single-chunk read
    single = await b.read_chunk(fid, chunk_index=1)
    assert len(single) == 1 and single[0]["content"] == "b"


# --- List tables ----------------------------------------------------------


async def test_list_rows_and_upsert_delete() -> None:
    b = FakeBackend()
    tid = str(uuid4())
    await b.upsert_row(
        "prompt_templates",
        {"template_id": tid, "owner_id": "u", "name": "T"},
        pk="template_id",
    )
    rows = await b.list_rows("prompt_templates", {"owner_id": "u"})
    assert len(rows) == 1 and rows[0]["name"] == "T"

    # upsert replaces
    await b.upsert_row(
        "prompt_templates",
        {"template_id": tid, "owner_id": "u", "name": "T2"},
        pk="template_id",
    )
    rows = await b.list_rows("prompt_templates", {"owner_id": "u"})
    assert len(rows) == 1 and rows[0]["name"] == "T2"

    await b.delete_row("prompt_templates", tid, pk="template_id")
    assert await b.list_rows("prompt_templates", {"owner_id": "u"}) == []


async def test_list_rows_order_and_limit() -> None:
    b = FakeBackend()
    for i in range(5):
        await b.upsert_row(
            "prompt_templates",
            {"template_id": str(i), "owner_id": "u", "ord": i},
            pk="template_id",
        )
    rows = await b.list_rows(
        "prompt_templates", {"owner_id": "u"}, order_by="-ord", limit=2
    )
    assert [r["ord"] for r in rows] == [4, 3]


# --- Lifecycle ------------------------------------------------------------


async def test_migrate_and_close() -> None:
    b = FakeBackend()
    assert not b.migrated
    await b.migrate()
    assert b.migrated
    await b.close()
    assert b.closed
    with pytest.raises(PermanentError):
        await b.load_chat(uuid4())
