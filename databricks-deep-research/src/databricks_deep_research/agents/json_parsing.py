"""Tolerant LLM-JSON parsing ladder.

Free-text LLM responses that are *not* produced via a forced-tool /
structured-output call frequently arrive wrapped in markdown fences, with
trailing commas, single-quoted keys, or buried inside prose ("Here is the
answer: {...}"). A bare ``json.loads`` of such text raises, and ad-hoc
per-call-site recovery has drifted across the codebase.

``parse_llm_json`` centralizes the recovery ladder:

1. Strip markdown code fences, then ``.strip()``.
2. ``json.loads`` of the stripped text — the clean path (``recovered=False``).
3. ``json_repair.loads`` of the stripped text (trailing commas, single
   quotes, etc.) — ``recovered=True``.
4. Regex-extract the FIRST balanced ``{...}`` or ``[...]`` block (``re.DOTALL``)
   and ``json_repair.loads`` it — handles prose-wrapped JSON — ``recovered=True``.
5. Exhausted — return ``default``.

A ``model`` may be supplied: when a dict/list candidate is produced it is run
through ``model.model_validate``; on :class:`pydantic.ValidationError` the
ladder falls through to ``default``.

The ``recovered`` flag is ``True`` whenever anything past the plain
``json.loads`` produced the value. It is the diagnostic-first signal that the
upstream prompt emitted fragile output — every recovery also logs
``LLM_JSON_RECOVERED`` exactly once so fragile prompts surface in telemetry
and can be fixed at the source rather than masked here.

NOTE: the suspicion guard below is a generic, subtype-AGNOSTIC port of
``databricks_deep_research.agents.harness._is_suspicious_repair``
(harness.py:121). The harness retains its own subtype-aware copy (it
additionally special-cases empty containers for the ``researcher`` subtype and
raises ``WorkflowError``). De-duplicating the two implementations — having the
harness delegate to this module — is a tracked follow-up; for now the shared
constant ``_JSON_REPAIR_MIN_SIZE_RATIO`` is mirrored here so they cannot drift.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any

import json_repair

if TYPE_CHECKING:
    from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Mirror of ``harness._JSON_REPAIR_MIN_SIZE_RATIO``. When a repair produces
# output drastically smaller than a substantive input, treat it as a false
# success (observed prod failure: ~19.8k chars of reasoning collapsed to a
# 59-char list) and fall through to ``default``.
_JSON_REPAIR_MIN_SIZE_RATIO = 0.1

# Inputs at/above this length are subject to the size-collapse check.
_SIZE_COLLAPSE_MIN_INPUT = 500

# Inputs at/above this length must not collapse to an empty container/scalar.
_SUBSTANTIVE_MIN_CHARS = 20

# Leading ```json / ``` fence and a trailing ``` fence.
_FENCE_LEAD_RE = re.compile(r"^\s*```(?:json|JSON)?\s*", re.DOTALL)
_FENCE_TRAIL_RE = re.compile(r"\s*```\s*$", re.DOTALL)

# First balanced {...} or [...] block (greedy from first opener to last
# matching closer). Single regex, ``re.DOTALL`` so it spans newlines.
_OBJECT_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_ARRAY_BLOCK_RE = re.compile(r"\[.*\]", re.DOTALL)


def _strip_fences(text: str) -> str:
    """Remove a leading ```json/``` fence and a trailing ``` fence."""
    stripped = _FENCE_LEAD_RE.sub("", text)
    stripped = _FENCE_TRAIL_RE.sub("", stripped)
    return stripped.strip()


def _has_substantive_text(text: str, *, min_length: int) -> bool:
    """True when ``text`` carries at least ``min_length`` non-whitespace chars.

    Generic port of the harness helper of the same name; kept local so this
    module stays dependency-light.
    """
    return len("".join(text.split())) >= min_length


def _is_suspicious_repair(parsed: Any, content: str) -> bool:
    """Return True when a repaired candidate looks like a false success.

    Subtype-AGNOSTIC port of ``harness._is_suspicious_repair``. Rejects:

    * an empty scalar (``""``/``None``) — ``json_repair`` collapses
      unrecoverable garbage (and empty input) to ``""``, which is never a
      useful parse (generalizes harness rule 1, whose callers only reach it
      with non-empty content); OR
    * an empty container (``{}``/``[]``) when the input carried
      >= ``_SUBSTANTIVE_MIN_CHARS`` non-whitespace chars; OR
    * ``len(str(parsed)) < 0.1 * len(content)`` when ``len(content) >= 500``.
    """
    if parsed in ("", None):
        return True

    is_empty_container = isinstance(parsed, (list, dict)) and not parsed
    if is_empty_container and _has_substantive_text(
        content, min_length=_SUBSTANTIVE_MIN_CHARS
    ):
        return True

    content_len = len(content)
    if content_len >= _SIZE_COLLAPSE_MIN_INPUT:
        parsed_len = len(str(parsed))
        if parsed_len / max(content_len, 1) < _JSON_REPAIR_MIN_SIZE_RATIO:
            return True

    return False


def _coerce_with_model(
    candidate: Any,
    model: type[BaseModel] | None,
    default: Any,
) -> tuple[Any, bool]:
    """Validate ``candidate`` against ``model`` if given.

    Returns ``(value, ok)`` where ``ok`` is False when validation failed and
    the caller should fall through to ``default``.
    """
    if model is None:
        return candidate, True
    if not isinstance(candidate, (dict, list)):
        return default, False
    from pydantic import ValidationError

    try:
        return model.model_validate(candidate), True
    except ValidationError:
        return default, False


def parse_llm_json(
    text: str,
    *,
    model: type[BaseModel] | None = None,
    default: Any = None,
    site: str = "parse_llm_json",
) -> tuple[Any, bool]:
    """Parse ``text`` as JSON with a tolerant recovery ladder.

    Args:
        text: Raw LLM response text (may carry fences / prose / malformation).
        model: Optional Pydantic model; a dict/list candidate is validated via
            ``model.model_validate``. On ``ValidationError`` the result is
            ``default``.
        default: Returned when the ladder is exhausted or validation fails.
        site: Call-site label included in the ``LLM_JSON_RECOVERED`` telemetry
            log so fragile prompts can be traced to their origin.

    Returns:
        ``(value, recovered)``. ``recovered`` is ``True`` whenever anything
        past the plain ``json.loads`` produced the value (including the
        suspicious/exhausted fall-throughs when the input had substantive
        text). ``value`` is the validated model instance when ``model`` is
        supplied and validation succeeds.
    """
    stripped = _strip_fences(text)
    substantive = _has_substantive_text(text, min_length=_SUBSTANTIVE_MIN_CHARS)

    # Stage 1: clean json.loads — the non-recovered happy path.
    try:
        candidate = json.loads(stripped)
    except (ValueError, TypeError):
        candidate = None
    else:
        value, ok = _coerce_with_model(candidate, model, default)
        if ok:
            return value, False
        # Model validation failed on a cleanly-parsed candidate: this is a
        # recovery fall-through to default.
        logger.warning("LLM_JSON_RECOVERED site=%s stage=%s", site, "model_invalid")
        return default, True

    # Stages 2-4 are recoveries. Try each candidate source in order; the first
    # non-suspicious one that also satisfies the model (if any) wins.
    for stage, raw in (
        ("repair", stripped),
        ("regex", _extract_first_block(stripped)),
    ):
        if raw is None:
            continue
        try:
            candidate = json_repair.loads(raw)
        except (ValueError, TypeError):
            continue
        if _is_suspicious_repair(candidate, text):
            continue
        value, ok = _coerce_with_model(candidate, model, default)
        if not ok:
            continue
        logger.warning("LLM_JSON_RECOVERED site=%s stage=%s", site, stage)
        return value, True

    # Stage 5: exhausted. ``recovered`` is True iff the input had substantive
    # text (a real but unparseable payload) — distinguishes "garbage we tried
    # to recover" from "trivially empty input".
    stage = "suspicious" if substantive else "none"
    if substantive:
        logger.warning("LLM_JSON_RECOVERED site=%s stage=%s", site, stage)
    return default, substantive


def _extract_first_block(text: str) -> str | None:
    """Return the first balanced ``{...}`` or ``[...]`` block, else None.

    Whichever opener appears first in ``text`` wins, so prose on either side
    of the JSON is discarded.
    """
    obj = _OBJECT_BLOCK_RE.search(text)
    arr = _ARRAY_BLOCK_RE.search(text)
    if obj and arr:
        return obj.group() if obj.start() <= arr.start() else arr.group()
    if obj:
        return obj.group()
    if arr:
        return arr.group()
    return None
