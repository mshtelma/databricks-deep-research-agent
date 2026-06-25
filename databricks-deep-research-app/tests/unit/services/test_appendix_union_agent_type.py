"""Phase 2a-2: the orchestrator seeds ONE shared appendix key (harness injects
it into every node), so it must render for the agent type whose render() emits
the UNION of useful blocks — findings AND coverage AND entities. This pins that
contract via the shared ``APPENDIX_AGENT_TYPE`` constant the orchestrator uses.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from deep_research.models.chat_memory_coverage import ChatMemoryCoverage
from deep_research.models.chat_memory_finding import ChatMemoryFinding
from deep_research.services.chat_memory_service import (
    APPENDIX_AGENT_TYPE,
    ChatMemoryService,
)

pytestmark = pytest.mark.unit


def test_union_agent_type_renders_both_findings_and_coverage() -> None:
    svc = ChatMemoryService(session=None)  # type: ignore[arg-type]
    svc._chat_id = uuid4()

    f = ChatMemoryFinding()
    f.id = uuid4()
    f.content = "Acme revenue grew 12% in FY24."
    f.confidence = "high"
    f.origin = "web"
    f.content_hash = "h1"
    f.entity_ids = []
    svc._findings = [f]

    c = ChatMemoryCoverage()
    c.id = uuid4()
    c.topic = "Acme revenue"
    c.status = "covered"
    c.depth = "deep"
    svc._coverage = [c]

    out = svc.render(agent_type=APPENDIX_AGENT_TYPE)
    # The single shared appendix must carry BOTH — a coordinator-only render
    # would omit coverage (Phase-2a-1 widened findings, but coverage is still
    # gated away from "coordinator").
    assert "Consolidated findings" in out
    assert "Research coverage" in out
    assert "Acme revenue grew 12% in FY24." in out
