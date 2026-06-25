"""Unit test for the legacy (SQLAlchemy) ChatMemoryService.consolidate_from_pool.

Production runs the cached override (tested in
``test_cached_chat_memory_consolidate.py``); this pins the base implementation
used on ``sqlalchemy_legacy`` deployments so both satisfy IChatMemoryService.
``_upsert_finding`` is patched — the real SQL path needs Postgres and is covered
by integration tests; here we assert control flow (two tiers, blank-skipping).
"""

from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from deep_research.services.chat_memory_service import ChatMemoryService

pytestmark = pytest.mark.unit


async def test_base_consolidate_filters_blank_and_counts(mock_db_session) -> None:
    cid = uuid4()
    svc = ChatMemoryService(mock_db_session)
    svc._chat_id = cid  # bypass hydrate()
    svc._upsert_finding = AsyncMock()  # type: ignore[method-assign]

    n = await svc.consolidate_from_pool(
        cid,
        claims=[
            {"claim_text": "A.", "confidence": "high"},
            {"claim_text": "   ", "confidence": "high"},  # blank -> skipped
        ],
        observations=[{"text": "an observation"}],
        research_session_id=None,
        source_step=2,
    )

    assert n == 2  # one claim + one observation; blank dropped
    assert svc._upsert_finding.await_count == 2


async def test_base_consolidate_requires_hydrate(mock_db_session) -> None:
    svc = ChatMemoryService(mock_db_session)
    with pytest.raises(RuntimeError):
        await svc.consolidate_from_pool(
            uuid4(), claims=[], observations=[], research_session_id=None, source_step=1
        )
