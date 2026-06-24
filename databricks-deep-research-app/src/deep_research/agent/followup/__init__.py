"""Follow-up turn handling for chats bound to a custom agent.

When a chat already has gathered research (a prior completed run with sources),
a new message is routed by *intent*: a conversational follow-up that can be
answered from the data already gathered is answered directly (no expensive
re-run of the agent's workflow), while a genuinely new research request falls
through to the normal workflow.

Public API:
- ``TurnIntent``        — requested per-turn mode (auto / chat / research).
- ``TurnDecision``      — resolved route + reasoning (chat / research / live_search).
- ``decide_turn_intent``— resolve the route (prompt-based, domain-agnostic).
- ``stream_chat_about_results`` — stream a grounded answer over prior sources.
- ``stream_live_search_answer`` — bounded live-web-search escape hatch (spec §4.7).
- ``LiveSearchUnavailable`` — sentinel: bounded search yielded no usable answer.
"""

from __future__ import annotations

from deep_research.agent.followup.chat_answer import stream_chat_about_results
from deep_research.agent.followup.intent import (
    FollowupClassification,
    TurnDecision,
    TurnIntent,
    decide_turn_intent,
)
from deep_research.agent.followup.live_search import (
    LiveSearchUnavailable,
    stream_live_search_answer,
)

__all__ = [
    "FollowupClassification",
    "LiveSearchUnavailable",
    "TurnDecision",
    "TurnIntent",
    "decide_turn_intent",
    "stream_chat_about_results",
    "stream_live_search_answer",
]
