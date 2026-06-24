"""End-to-end lifecycle test against a real Lakebase / Postgres instance.

Gated on `STORAGE_TEST_LAKEBASE=1` + Lakebase credentials already in the
environment (picked up by `Settings.use_lakebase`). Falls through to a
`DATABASE_URL` for local dev runs.

Flow: create temp schema → apply DDL → write chats/messages/files/events →
read back → run cleanup → drop schema. On failure the schema is still
dropped via the `try/finally` fixture.
"""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime

import pytest

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("STORAGE_TEST_LAKEBASE") != "1",
        reason="STORAGE_TEST_LAKEBASE=1 required",
    ),
]


def _utcnow() -> datetime:
    return datetime.now(tz=UTC)


async def test_full_lifecycle() -> None:
    """Migrate → write → read → cleanup — full end-to-end against live DB."""
    from sqlalchemy import text

    from deep_research.core.config import get_settings
    from deep_research.db.session import get_session_maker
    from deep_research.storage.documents import (
        ChatDocument,
        Finding,
        Message,
        UploadedFileMeta,
    )
    from deep_research.storage.lakebase import (
        LakebaseBackend,
        _split_sql,
    )

    settings = get_settings()
    sm = get_session_maker(settings)

    schema_name = f"deep_research_test_{uuid.uuid4().hex[:12]}"

    async def _run_schema(sql: str) -> None:
        async with sm() as session, session.begin():
            await session.execute(text(sql))

    await _run_schema(f'CREATE SCHEMA IF NOT EXISTS "{schema_name}"')

    backend = LakebaseBackend(session_maker=sm)

    try:
        # Apply DDL under the temp schema via `SET search_path`.
        async with sm() as session, session.begin():
            await session.execute(text(f'SET search_path TO "{schema_name}"'))
            for stmt in _split_sql(backend._ddl_path.read_text()):
                await session.execute(text(stmt))

        # Scope every subsequent backend session to the temp schema.

        class _Scoped:
            def __call__(self, *args, **kwargs):
                inner = sm(*args, **kwargs)

                class _CM:
                    async def __aenter__(_self):
                        await inner.__aenter__()
                        await inner.execute(text(f'SET search_path TO "{schema_name}"'))
                        # Commit the implicit transaction opened by `SET
                        # search_path` so the backend's own `session.begin()`
                        # starts fresh.
                        await inner.commit()
                        return inner

                    async def __aexit__(_self, *exc):
                        await inner.__aexit__(*exc)

                return _CM()

        backend._sm = _Scoped()  # type: ignore[assignment]

        cid = uuid.uuid4()
        fid = uuid.uuid4()
        doc = ChatDocument.new(cid, "alice", title="Integration test")
        doc.state.add_message(Message(role="user", content="hello from integration"))
        doc.state.upsert_finding(Finding(content_hash="hash1", content="A finding"))
        doc.state.upsert_uploaded_file(UploadedFileMeta(id=fid, name="doc.pdf", size=123))

        v = await backend.write_chat(doc, expected_version=0)
        assert v == 1

        loaded = await backend.load_chat(cid)
        assert loaded is not None
        assert loaded.state.messages[0].content == "hello from integration"
        assert loaded.state.memory.findings[0].content_hash == "hash1"
        assert [f.id for f in loaded.state.uploaded_files] == [fid]

        # chat_deleted_files projection round-trip.
        rows = await backend.list_rows("chat_deleted_files", {"chat_id": cid})
        assert len(rows) == 1

        # Soft-delete + cleanup (1-day chunk cleanup window is now() - 1 day,
        # so a just-deleted chat's chunks are still within the window — no
        # deletions yet). Verifies the cleanup call completes without error.
        stats = await backend.cleanup_soft_deleted(chat_retention_days=7)
        assert stats.errors == 0
    finally:
        try:
            await backend.close()
        except Exception:
            pass
        await _run_schema(f'DROP SCHEMA IF EXISTS "{schema_name}" CASCADE')
