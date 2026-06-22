"""Phase 2e-1: Coverage rows carry as_of_turn/updated_at for the freshness gate.

No migration — ChatState is JSON with extra="ignore"; old docs read defaults.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from deep_research.storage.documents import Coverage

pytestmark = pytest.mark.unit


def test_coverage_carries_as_of_turn_and_updated_at() -> None:
    when = datetime(2026, 6, 20, tzinfo=UTC)
    c = Coverage(topic="Acme revenue", status="covered", as_of_turn=3, updated_at=when)
    back = Coverage.model_validate_json(c.model_dump_json())
    assert back.as_of_turn == 3
    assert back.updated_at == when


def test_coverage_defaults_are_backward_compatible() -> None:
    c = Coverage(topic="x")
    assert c.as_of_turn == 0
    assert c.updated_at is None
    # An old doc without the new keys still validates (extra="ignore" + defaults).
    old = Coverage.model_validate({"topic": "y", "status": "gap"})
    assert old.as_of_turn == 0
