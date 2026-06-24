"""Decide whether a follow-up message is answered from prior research or re-runs the agent.

The decision is **prompt-based and domain/topology-agnostic**: the classifier is
given a summary of what was already researched and asked whether the new message
is answerable from that material. It must never encode example-, benchmark-, or
domain-specific rules.

Key correctness point: the signal is ``answerable_from_prior_research`` — whether
the answer plausibly lives in the *gathered* data — NOT whether the model knows
the answer from general knowledge. An obscure fact (e.g. a small company's CEO)
is answerable-from-prior-research when the prior report covered it, even though
it is not common knowledge.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from deep_research.core.logging_utils import get_logger
from deep_research.services.llm.types import ModelTier

if TYPE_CHECKING:
    from deep_research.services.llm.client import LLMClient

logger = get_logger(__name__)


class TurnIntent(StrEnum):
    """Per-turn routing requested by the caller (request field / override)."""

    AUTO = "auto"  # classify intent and route automatically
    CHAT = "chat"  # force "answer from gathered data" (no re-run)
    RESEARCH = "research"  # force a fresh agent run


class FollowupClassification(BaseModel):
    """LLM classification of a follow-up message against already-gathered research."""

    answerable_from_prior_research: bool = False
    follow_up_type: str = "new_topic"  # new_topic | clarification | complex_follow_up
    # When the message is NOT answerable from the prior pool, is it the kind of
    # question a small live web lookup could answer (a current fact, a fresh
    # detail) — as opposed to needing a full multi-step research run? Used only
    # by the bounded live-search escape hatch (spec §4.7); ignored when the
    # caller does not opt in. Default false keeps the legacy routing identical.
    web_searchable: bool = False
    reasoning: str = ""


@dataclass
class TurnDecision:
    """Resolved routing decision for a turn.

    ``live_search`` is the bounded escape hatch (spec §4.7): a small live web
    lookup answers a follow-up that the prior pool cannot, instead of either an
    empty-pool answer or a full research re-run. It is only ever produced when
    the caller passes ``allow_live_search=True``.
    """

    route: Literal["chat", "research", "live_search"]
    reasoning: str
    classification: FollowupClassification | None = None


_CLASSIFIER_SYSTEM_PROMPT = """You are the turn router for an ongoing research assistant conversation.

Earlier in this conversation the user ran a research agent that gathered sources \
and produced a report. Decide whether the user's NEW message can be answered from \
the research ALREADY gathered, or whether it requires a NEW research run that \
fetches fresh external data.

Decide using these principles:
- Re-running research is slow and expensive. Prefer answering from prior research \
whenever the message asks about, clarifies, summarizes, reformats, compares, or \
extracts a detail from what was already gathered — EVEN IF the specific fact is \
obscure or not common knowledge — as long as it plausibly appears in the gathered \
material or the conversation so far.
- Set ``answerable_from_prior_research`` to true when the answer plausibly lives in \
the gathered material. Base this on the gathered material, NOT on whether the fact \
is common knowledge.
- Require new research (``answerable_from_prior_research`` false) only when the \
message introduces a genuinely new topic, entity, or scope, or needs information \
the prior research plausibly did not cover.
- ``follow_up_type``: "clarification" (re-explains/extracts from prior results), \
"complex_follow_up" (builds on prior research but needs new data), or "new_topic" \
(unrelated to the gathered material).
- ``web_searchable``: only relevant when ``answerable_from_prior_research`` is \
false. Set it true when the message is a focused factual lookup that a small \
handful of fresh web results could plausibly answer (a single current fact, a \
recent figure, a quick external detail). Set it false when the message needs a \
broad, multi-step investigation, deep synthesis, or comparison across many \
sources. When answerable from prior research, leave it false.

You must output valid JSON matching the schema provided."""

_CLASSIFIER_USER_PROMPT = """## User's new message
{query}

## Summary of research already gathered in this conversation
{prior_findings}

## Recent conversation
{conversation_history}

## Output schema
{{
  "answerable_from_prior_research": boolean,
  "follow_up_type": "new_topic" | "clarification" | "complex_follow_up",
  "web_searchable": boolean,
  "reasoning": "<one concise sentence>"
}}

Respond with only valid JSON."""


def _format_history(conversation_history: list[dict[str, str]] | None) -> str:
    if not conversation_history:
        return "(No previous conversation)"
    return "\n".join(
        f"{str(msg.get('role', '')).upper()}: {str(msg.get('content', ''))[:400]}"
        for msg in conversation_history[-6:]
    )


async def _classify_followup(
    *,
    query: str,
    conversation_history: list[dict[str, str]] | None,
    prior_findings_summary: str,
    llm: LLMClient,
) -> FollowupClassification:
    """Run the cheap-tier classifier. Raises on LLM/parse failure (caller handles)."""
    messages = [
        {"role": "system", "content": _CLASSIFIER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _CLASSIFIER_USER_PROMPT.format(
                query=query,
                prior_findings=prior_findings_summary or "(No prior findings summary available.)",
                conversation_history=_format_history(conversation_history),
            ),
        },
    ]
    response = await llm.complete(
        messages=messages,
        tier=ModelTier.BULK_ANALYSIS,
        structured_output=FollowupClassification,
    )
    structured = getattr(response, "structured", None)
    if isinstance(structured, FollowupClassification):
        return structured
    return FollowupClassification.model_validate_json(response.content)


async def decide_turn_intent(
    *,
    query: str,
    conversation_history: list[dict[str, str]] | None,
    prior_findings_summary: str,
    has_prior_research: bool,
    requested: TurnIntent,
    llm: LLMClient,
    allow_live_search: bool = False,
) -> TurnDecision:
    """Resolve whether this turn should ``chat``, ``research``, or ``live_search``.

    Explicit overrides win; AUTO classifies. Any failure defaults to ``research``
    (the safe, pre-existing behavior), never silently dropping a real request.

    ``allow_live_search`` (default False → byte-identical legacy routing) opts in
    to the bounded escape hatch: when AUTO decides the message is NOT answerable
    from the prior pool but IS a focused web-searchable lookup, the route becomes
    ``live_search`` instead of a full ``research`` re-run. The caller runs a small
    capped live search and still falls back to ``research`` if that search yields
    nothing — so the signal never silently drops a real request.
    """
    if requested == TurnIntent.RESEARCH:
        return TurnDecision(route="research", reasoning="explicit override: research")
    if requested == TurnIntent.CHAT:
        # Honor the override even if we couldn't detect prior research; the
        # handler degrades gracefully to conversation-only context.
        return TurnDecision(route="chat", reasoning="explicit override: chat")

    # AUTO
    if not has_prior_research:
        return TurnDecision(
            route="research",
            reasoning="no prior research in this chat (first turn) — running the agent",
        )

    try:
        classification = await _classify_followup(
            query=query,
            conversation_history=conversation_history,
            prior_findings_summary=prior_findings_summary,
            llm=llm,
        )
    except Exception as exc:  # noqa: BLE001 — never fail the turn on classification
        logger.warning("FOLLOWUP_CLASSIFY_FAILED", error=str(exc)[:200])
        return TurnDecision(
            route="research",
            reasoning=f"classification failed; defaulting to research: {str(exc)[:120]}",
        )

    answerable = classification.answerable_from_prior_research or (
        classification.follow_up_type == "clarification"
    )
    if answerable:
        return TurnDecision(
            route="chat",
            reasoning=classification.reasoning,
            classification=classification,
        )

    # Not answerable from the prior pool. With the escape hatch enabled, a focused
    # web-searchable lookup gets a bounded live search instead of a full re-run.
    if allow_live_search and classification.web_searchable:
        return TurnDecision(
            route="live_search",
            reasoning=classification.reasoning,
            classification=classification,
        )

    return TurnDecision(
        route="research",
        reasoning=classification.reasoning,
        classification=classification,
    )
