"""End-to-end lifecycle test against a real Databricks SQL Warehouse.

Gated on `STORAGE_TEST_WAREHOUSE=1` + `STORAGE_WAREHOUSE_ID` + `STORAGE_CATALOG`.
Creates a temp UC schema (`${STORAGE_SCHEMA_PREFIX}_<uuid>`), applies Delta
DDL, exercises the round-trip, and drops the schema in `finally`. See
`docs/storage/ais_setup.md` for prerequisites.
"""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime

import pytest

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("STORAGE_TEST_WAREHOUSE") != "1",
        reason="STORAGE_TEST_WAREHOUSE=1 required",
    ),
    pytest.mark.skipif(
        "STORAGE_WAREHOUSE_ID" not in os.environ,
        reason="STORAGE_WAREHOUSE_ID not set",
    ),
]


def _utcnow() -> datetime:
    return datetime.now(tz=UTC)


async def test_full_lifecycle() -> None:
    from deep_research.storage.documents import (
        ChatDocument,
        Finding,
        Message,
        UploadedFileMeta,
    )
    from deep_research.storage.sql_warehouse import SQLWarehouseBackend

    warehouse_id = os.environ["STORAGE_WAREHOUSE_ID"]
    catalog = os.environ.get("STORAGE_CATALOG", "main")
    prefix = os.environ.get("STORAGE_SCHEMA_PREFIX", "deep_research_test")
    schema_name = f"{prefix}_{uuid.uuid4().hex[:12]}"

    backend = SQLWarehouseBackend(
        warehouse_id=warehouse_id,
        catalog=catalog,
        schema=schema_name,
    )

    try:
        await backend.migrate()

        cid = uuid.uuid4()
        fid = uuid.uuid4()
        doc = ChatDocument.new(cid, "alice", title="Integration test")
        doc.state.add_message(Message(role="user", content="hello from warehouse"))
        doc.state.upsert_finding(Finding(content_hash="hash1", content="A finding"))
        doc.state.upsert_uploaded_file(UploadedFileMeta(id=fid, name="doc.pdf", size=123))

        v = await backend.write_chat(doc, expected_version=0)
        assert v == 1

        loaded = await backend.load_chat(cid)
        assert loaded is not None
        assert loaded.state.messages[0].content == "hello from warehouse"
        assert loaded.state.memory.findings[0].content_hash == "hash1"
        assert len(loaded.state.uploaded_files) == 1

        # Projection round-trip via list_rows (the contract assertion).
        rows = await backend.list_rows("chat_deleted_files", {"chat_id": str(cid)})
        assert len(rows) == 1

        stats = await backend.cleanup_soft_deleted(chat_retention_days=7)
        assert stats.errors == 0
    finally:
        try:
            await backend._execute(f"DROP SCHEMA IF EXISTS {catalog}.{schema_name} CASCADE")
        except Exception:
            pass
        await backend.close()
