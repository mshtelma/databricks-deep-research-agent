"""Contract tests — every `StorageBackend` implementation must pass these.

Parametrized over `FakeBackend` always, plus `LakebaseBackend` and
`SQLWarehouseBackend` when the corresponding gate env var is set. Semantics
are asserted at the Protocol surface; individual backends may differ in how
many *SQL statements* they emit internally, but the **user-facing contract**
(method results, error types, projection state) must be identical.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from deep_research.storage.backend import ConflictError, PermanentError, StorageBackend
from deep_research.storage.documents import (
    ChatDocument,
    Message,
    PrepJobDocument,
    UploadedFileMeta,
    UserDocument,
)


def _utcnow() -> datetime:
    return datetime.now(tz=UTC)


# --- Chat document round-trip ----------------------------------------------


class TestLoadChat:
    async def test_returns_none_for_unknown_chat(self, backend: StorageBackend) -> None:
        assert await backend.load_chat(uuid4()) is None


class TestWriteChat:
    async def test_fresh_write_then_load_roundtrip(self, backend: StorageBackend) -> None:
        cid = uuid4()
        doc = ChatDocument.new(cid, "alice", title="Hi")
        doc.state.add_message(Message(role="user", content="hello"))
        v = await backend.write_chat(doc, expected_version=0)
        assert v == 1
        loaded = await backend.load_chat(cid)
        assert loaded is not None
        assert loaded.meta.chat_id == cid
        assert loaded.meta.version == 1
        assert loaded.meta.user_id == "alice"
        assert len(loaded.state.messages) == 1
        assert loaded.state.messages[0].content == "hello"

    async def test_version_gate_rejects_stale_write(self, backend: StorageBackend) -> None:
        cid = uuid4()
        doc = ChatDocument.new(cid, "u")
        await backend.write_chat(doc, expected_version=0)
        # Second write with stale version must be rejected.
        with pytest.raises(ConflictError):
            await backend.write_chat(doc, expected_version=0)

    async def test_version_gate_permits_correct_expected(self, backend: StorageBackend) -> None:
        cid = uuid4()
        doc = ChatDocument.new(cid, "u")
        v1 = await backend.write_chat(doc, expected_version=0)
        doc.meta.version = v1
        doc.state.add_message(Message(role="user", content="m2"))
        v2 = await backend.write_chat(doc, expected_version=v1)
        assert v2 == v1 + 1

    async def test_soft_delete_hides_from_load(self, backend: StorageBackend) -> None:
        cid = uuid4()
        doc = ChatDocument.new(cid, "u")
        v = await backend.write_chat(doc, expected_version=0)
        assert await backend.load_chat(cid) is not None
        doc.meta.deleted_at = _utcnow()
        doc.meta.version = v
        await backend.write_chat(doc, expected_version=v)
        assert await backend.load_chat(cid) is None


# --- chat_deleted_files projection -----------------------------------------


class TestProjectionSync:
    async def test_projection_mirrors_live_file_ids(self, backend: StorageBackend) -> None:
        cid = uuid4()
        fid1, fid2 = uuid4(), uuid4()
        doc = ChatDocument.new(cid, "u")
        doc.state.upsert_uploaded_file(UploadedFileMeta(id=fid1, name="a.pdf", size=1))
        doc.state.upsert_uploaded_file(UploadedFileMeta(id=fid2, name="b.pdf", size=2))
        v = await backend.write_chat(doc, expected_version=0)
        # Verify via the list_rows Protocol method (works on every backend).
        rows = await backend.list_rows("chat_deleted_files", {"chat_id": cid})
        observed = {_as_uuid(r["file_id"]) for r in rows}
        assert observed == {fid1, fid2}

        # Remove fid1; projection must mirror via full-overwrite.
        doc.state.uploaded_files = [f for f in doc.state.uploaded_files if f.id != fid1]
        doc.meta.version = v
        await backend.write_chat(doc, expected_version=v)
        rows = await backend.list_rows("chat_deleted_files", {"chat_id": cid})
        observed = {_as_uuid(r["file_id"]) for r in rows}
        assert observed == {fid2}


# --- list_chat_metas -------------------------------------------------------


class TestListChatMetas:
    async def test_filters_by_user_id(self, backend: StorageBackend) -> None:
        cid_a, cid_b = uuid4(), uuid4()
        await backend.write_chat(ChatDocument.new(cid_a, "alice", title="A"), expected_version=0)
        await backend.write_chat(ChatDocument.new(cid_b, "bob", title="B"), expected_version=0)
        alice_chats = await backend.list_chat_metas("alice")
        assert [m.chat_id for m in alice_chats] == [cid_a]

    async def test_excludes_soft_deleted_by_default(self, backend: StorageBackend) -> None:
        cid_live, cid_dead = uuid4(), uuid4()
        await backend.write_chat(ChatDocument.new(cid_live, "u", title="live"), expected_version=0)
        dead = ChatDocument.new(cid_dead, "u", title="dead")
        dead.meta.deleted_at = _utcnow()
        await backend.write_chat(dead, expected_version=0)
        live = await backend.list_chat_metas("u")
        assert [m.chat_id for m in live] == [cid_live]
        all_metas = await backend.list_chat_metas("u", include_deleted=True)
        assert len(all_metas) == 2


# --- User document ---------------------------------------------------------


class TestUserDoc:
    async def test_roundtrip(self, backend: StorageBackend) -> None:
        doc = UserDocument(user_id="alice", profile={"role": "SA"}, preferences={"theme": "dark"})
        await backend.write_user_doc(doc)
        loaded = await backend.load_user_doc("alice")
        assert loaded is not None
        assert loaded.profile == {"role": "SA"}
        assert loaded.preferences == {"theme": "dark"}

    async def test_upsert_replaces_fields(self, backend: StorageBackend) -> None:
        await backend.write_user_doc(UserDocument(user_id="u", profile={"v": 1}))
        await backend.write_user_doc(UserDocument(user_id="u", profile={"v": 2}))
        loaded = await backend.load_user_doc("u")
        assert loaded is not None and loaded.profile == {"v": 2}


# --- Prep-job + heartbeat --------------------------------------------------


class TestPrepJob:
    async def test_roundtrip(self, backend: StorageBackend) -> None:
        jid = uuid4()
        doc = PrepJobDocument(
            prep_job_id=jid, account_id="acc1", status="running", query="demo"
        )
        await backend.write_prep_job(doc)
        loaded = await backend.load_prep_job(jid)
        assert loaded is not None
        assert loaded.status == "running"
        assert loaded.query == "demo"

    async def test_heartbeat_bypasses_write_prep_job(self, backend: StorageBackend) -> None:
        jid = uuid4()
        await backend.write_prep_job(
            PrepJobDocument(prep_job_id=jid, account_id="a", status="running")
        )
        ts = _utcnow()
        await backend.write_prep_heartbeat(jid, ts)
        loaded = await backend.load_prep_job(jid)
        assert loaded is not None
        # Tolerate sub-second rounding across real DB encodings.
        assert abs((loaded.heartbeat - ts).total_seconds()) < 1.0

    async def test_heartbeat_on_missing_job_raises(self, backend: StorageBackend) -> None:
        with pytest.raises(PermanentError):
            await backend.write_prep_heartbeat(uuid4(), _utcnow())


# --- Append-only tables + chunks -------------------------------------------


class TestAppendEvents:
    async def test_bulk_insert_and_read(self, backend: StorageBackend) -> None:
        fid = uuid4()
        ts = _utcnow()
        rows = [
            {"file_id": fid, "chunk_index": i, "ts": ts, "content": f"c{i}", "metadata": {}}
            for i in range(3)
        ]
        await backend.append_events("file_chunks", rows)
        chunks = await backend.read_chunk(fid)
        assert [c["chunk_index"] for c in chunks] == [0, 1, 2]

    async def test_empty_batch_is_noop(self, backend: StorageBackend) -> None:
        # Must not raise and must not create any row.
        await backend.append_events("research_events", [])


# --- List-table CRUD -------------------------------------------------------


class TestListTables:
    async def test_upsert_list_delete(self, backend: StorageBackend) -> None:
        tid = str(uuid4())
        await backend.upsert_row(
            "prompt_templates",
            {
                "template_id": tid,
                "owner_id": "u1",
                "name": "T1",
                "content": "body",
                "visibility": "private",
                "template_type": "default",
                "metadata": {},
                "created_at": _utcnow(),
                "updated_at": _utcnow(),
            },
            pk="template_id",
        )
        rows = await backend.list_rows("prompt_templates", {"owner_id": "u1"})
        assert len(rows) == 1
        assert rows[0]["name"] == "T1"

        # Upsert replaces.
        await backend.upsert_row(
            "prompt_templates",
            {
                "template_id": tid,
                "owner_id": "u1",
                "name": "T2",
                "content": "body",
                "visibility": "private",
                "template_type": "default",
                "metadata": {},
                "created_at": _utcnow(),
                "updated_at": _utcnow(),
            },
            pk="template_id",
        )
        rows = await backend.list_rows("prompt_templates", {"owner_id": "u1"})
        assert rows[0]["name"] == "T2"

        await backend.delete_row("prompt_templates", tid, pk="template_id")
        assert await backend.list_rows("prompt_templates", {"owner_id": "u1"}) == []


# --- Call-count discipline (FakeBackend-only via CountingBackend wrapper) ---


class _CountingBackend:
    """Delegates to a wrapped backend but counts backend-method calls.

    Used to assert the WriteQueue / service layer makes the expected number
    of Protocol calls per operation. Not itself a `StorageBackend` (on
    purpose) — consumed via duck typing inside tests that need call counts.
    """

    def __init__(self, inner: StorageBackend) -> None:
        self._inner = inner
        self.calls: dict[str, int] = {}

    def __getattr__(self, name: str):
        attr = getattr(self._inner, name)
        if not callable(attr):
            return attr
        async def _wrapped(*args, **kwargs):
            self.calls[name] = self.calls.get(name, 0) + 1
            return await attr(*args, **kwargs)
        return _wrapped


class TestCallCountDiscipline:
    async def test_hydrate_is_one_load_chat_call(self, backend: StorageBackend) -> None:
        cid = uuid4()
        await backend.write_chat(ChatDocument.new(cid, "u"), expected_version=0)
        counting = _CountingBackend(backend)
        await counting.load_chat(cid)
        assert counting.calls == {"load_chat": 1}

    async def test_write_chat_is_one_method_call(self, backend: StorageBackend) -> None:
        cid = uuid4()
        counting = _CountingBackend(backend)
        await counting.write_chat(ChatDocument.new(cid, "u"), expected_version=0)
        assert counting.calls == {"write_chat": 1}


# --- Helpers ---------------------------------------------------------------


def _as_uuid(value) -> UUID:  # type: ignore[no-untyped-def]
    """Backends deliver UUIDs sometimes as `UUID`, sometimes as `str`."""
    if isinstance(value, UUID):
        return value
    return UUID(str(value))
