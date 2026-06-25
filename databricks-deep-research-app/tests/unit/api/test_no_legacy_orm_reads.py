"""Regression fence: the research-session read/write paths must never query the
legacy normalized ``public.*`` tables via SQLAlchemy.

After the ORM->storage-stack migration, research sessions, messages, sources and
research events live exclusively in the event-sourced storage stack
(``deep_research_state.chat_state`` JSONB + ``research_events``) and are read via
the cached services. The legacy ``public.*`` tables are dropped in production
(``scripts/cleanup_legacy_tables.sql``), so a reintroduced ``select(ResearchSession)``
or a direct ``MessageService(db)`` instantiation in these hot paths would turn
into a 500 (UndefinedTable) — exactly the regression this suite guards against.

This is a static source scan (cheap, no DB). It does NOT forbid the ORM model
classes themselves (``ResearchSession`` survives as a response DTO) — only
*queries* against them and direct legacy-service construction.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import deep_research

_SRC = Path(deep_research.__file__).parent

# file (relative to src/deep_research) -> substrings that must NOT appear in it.
_FORBIDDEN: dict[str, list[str]] = {
    "services/job_manager.py": [
        "select(ResearchSession",
        "update(ResearchSession",
        "func.count(ResearchSession",
        "db.get(ResearchSession",
    ],
    "api/v1/citations.py": [
        "select(ResearchSession",
        "Depends(get_db)",
        "verify_message_ownership",
    ],
    "api/v1/research.py": [
        "select(ResearchSession",
        "select(ResearchEvent",
        "func.max(ResearchEvent",
        "db.get(ResearchSession",
        "db.get(Message",
        "ResearchSessionService(",
    ],
    "api/v1/messages.py": [
        "MessageService(",
        "FeedbackService(",
        "select(Message",
        "db.get(Message",
    ],
}


def _strip_comments(source: str) -> str:
    """Drop ``#`` comments so the fence matches code, not prose.

    Cuts each line at its first ``#``. This is intentionally simple — it can
    over-trim a line that contains a literal ``#`` inside a string, but none of
    the forbidden needles legitimately appear after a string-embedded ``#`` in
    these files, and the alternative (full tokenization) is overkill for a fence.
    """
    out: list[str] = []
    for line in source.splitlines():
        hash_idx = line.find("#")
        out.append(line if hash_idx == -1 else line[:hash_idx])
    return "\n".join(out)


@pytest.mark.parametrize(
    ("rel_path", "needle"),
    [(rel, needle) for rel, needles in _FORBIDDEN.items() for needle in needles],
)
def test_no_legacy_orm_reads(rel_path: str, needle: str) -> None:
    source = _strip_comments((_SRC / rel_path).read_text(encoding="utf-8"))
    assert needle not in source, (
        f"{rel_path} contains forbidden legacy-ORM pattern {needle!r}. "
        "Research-session/message/source/event reads must go through the cached "
        "storage-stack services (see scripts/cleanup_legacy_tables.sql / the "
        "ORM->storage migration); the legacy public.* tables are dropped in prod."
    )
