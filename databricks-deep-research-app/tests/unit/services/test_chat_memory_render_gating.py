"""Phase 2a-1: consolidated findings must render for every research-driving
agent, not only the synthesizer — otherwise Phase-1's persisted findings reach
no follow-up agent. Also pins the confidence-then-recency ordering.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from deep_research.models.chat_memory_finding import ChatMemoryFinding
from deep_research.services.chat_memory_service import ChatMemoryService

pytestmark = pytest.mark.unit


def _finding(content: str, confidence: str, *, when: datetime | None = None) -> ChatMemoryFinding:
    row = ChatMemoryFinding()
    row.id = uuid4()
    row.content = content
    row.confidence = confidence
    row.origin = "web"
    row.content_hash = content[:16]
    row.entity_ids = []
    row.created_at = when or datetime.now(UTC)
    return row


def _svc_with_findings(*findings: ChatMemoryFinding) -> ChatMemoryService:
    svc = ChatMemoryService(session=None)  # type: ignore[arg-type]
    svc._chat_id = uuid4()
    svc._findings = list(findings)
    return svc


@pytest.mark.parametrize("agent_type", ["researcher", "planner", "coordinator", "synthesizer"])
def test_findings_render_for_research_agents(agent_type: str) -> None:
    svc = _svc_with_findings(_finding("Acme revenue grew 12% in FY24.", "high"))
    out = svc.render(agent_type=agent_type)
    assert "Acme revenue grew 12% in FY24." in out
    assert "Consolidated findings" in out


def test_findings_not_rendered_for_unrelated_agent() -> None:
    # An agent type outside the research set (e.g. crm_context) gets no findings block.
    svc = _svc_with_findings(_finding("Acme revenue grew 12% in FY24.", "high"))
    out = svc.render(agent_type="crm_context")
    assert "Consolidated findings" not in out


def test_findings_ordered_by_confidence_then_recency() -> None:
    old_high = _finding("HIGH OLD fact.", "high", when=datetime(2020, 1, 1, tzinfo=UTC))
    new_high = _finding("HIGH NEW fact.", "high", when=datetime(2026, 1, 1, tzinfo=UTC))
    low = _finding("LOW fact.", "low", when=datetime(2026, 6, 1, tzinfo=UTC))
    svc = _svc_with_findings(low, old_high, new_high)
    out = svc.render(agent_type="researcher")
    # High confidence outranks low (even though low is most recent); within
    # equal confidence, newer first.
    assert out.index("HIGH NEW fact.") < out.index("HIGH OLD fact.") < out.index("LOW fact.")
