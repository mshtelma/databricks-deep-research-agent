"""Structured-output slot filling for agent surfaces (per-slot wires).

After a run produces a report + evidence, this module fills the slots declared
by the agent's surface output components (Table, MetricGrid, KeyFindings,
Chart, List) with ONE small structured LLM call per slot, generated over
ranked EVIDENCE (sources pool + verified claims; report excerpt as context).
``source_refs`` emitted by the model index the evidence list as rendered in
the wire prompt; the persisted envelope carries its own resolution legend
(``meta.sources``).

Everything here is FAIL-SOFT by contract: a wire failure marks ITS slot
``failed`` in ``meta.slots`` (siblings keep their data). Text cells may carry
``[Key]`` citation markers; markers not backed by a known claim key are
stripped and recorded — never silently kept.

This is the shared generation core imported by BOTH the main app (which adds
its own persistence) and the standalone shell-app. The one dependency it takes
from its caller is an LLM client satisfying :class:`StructuredCompletionClient`
— both the app ``LLMClient`` and the framework ``FrameworkLLMClient`` do.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from pydantic import BaseModel

from databricks_deep_research.llm.client import ModelTier
from databricks_deep_research.surface.evidence import (
    EvidenceItem,
    build_evidence,
    build_legend,
    render_evidence_block,
)
from databricks_deep_research.surface.output_schema import (
    SlotSpec,
    build_slot_wire_model,
    wire_slot_docs,
)
from databricks_deep_research.surface.tolerant import (
    WireValidationError,
    coerce_citation_ref,
    json_repair_structured,
    unwrap_placeholder_envelope,
    validate_lenient,
)

logger = logging.getLogger(__name__)


class StructuredCompletionClient(Protocol):
    """Minimal LLM surface the wire generator needs.

    Satisfied structurally by BOTH the app ``LLMClient`` and the framework
    ``FrameworkLLMClient``. ``tier`` is typed ``Any`` because the two clients
    use distinct (same-valued) ``ModelTier`` StrEnums; the generator passes the
    framework ``ModelTier``, which both accept at runtime (``.value``).
    """

    async def complete(
        self,
        *,
        messages: list[dict[str, str]],
        tier: Any,
        structured_output: type[BaseModel] | None = None,
    ) -> Any: ...


# Same grammar the report/citation pipeline uses — kept local so this module
# has no import edge into persistence.
CITATION_MARKER_RE = re.compile(r"\[([A-Za-z][A-Za-z0-9-]*(?:-\d+)?)\]")

MAX_CLAIMS = 80
MAX_CLAIM_CHARS = 200
PAYLOAD_MAX_BYTES = 200 * 1024

WIRE_TIMEOUT_S = 60.0
WIRE_CONCURRENCY = 4
WIRE_MAX_CLAIMS = 40
WIRE_REPORT_CHARS = 6_000
# Trimmed inputs for the retry after a first-attempt timeout.
WIRE_TRIM_EVIDENCE_CHARS = 12_000
WIRE_TRIM_CLAIMS = 20
WIRE_TRIM_REPORT_CHARS = 3_000

_WIRE_SYSTEM_PROMPT = """\
You extract structured data for ONE section of a research dashboard from
research evidence. Return ONLY a JSON object matching the provided schema —
no prose, no code fences, no extra keys.

Grounding rules:
- Base every item on the EVIDENCE SOURCES and VERIFIED CLAIMS below. The
  report excerpt is supporting context only. Never invent facts; fewer items
  beats fabrication.
- Every table row and text item MUST include source_refs: the index strings
  of the sources that support it, e.g. ["1", "3"]. Items you cannot support
  with at least one source ref must be OMITTED. Metrics should carry
  source_refs when possible.
- Text fields may embed citation markers like [Key], ONLY from the allowed
  keys. Numeric fields carry no markers.
- Respect every limit (item counts, text lengths); keep the most
  decision-relevant items when trimming."""


def _claim_get(claim: Any, key: str) -> Any:
    """Read a claim attribute from both shapes: ClaimInfo objects at run
    time, plain ``verification_data["claims"]`` dicts on the retry path."""
    if isinstance(claim, dict):
        return claim.get(key)
    return getattr(claim, key, None)


def _known_citation_keys(claims: list[Any]) -> set[str]:
    keys: set[str] = set()
    for claim in claims:
        primary = _claim_get(claim, "citation_key")
        if isinstance(primary, str) and primary:
            keys.add(primary)
        extra = _claim_get(claim, "citation_keys")
        if isinstance(extra, list):
            keys.update(k for k in extra if isinstance(k, str) and k)
    return keys


def _strip_unknown_markers(
    value: Any, known: set[str], stripped: set[str]
) -> Any:
    """Recursively remove ``[Key]`` markers not backed by a known claim."""
    if isinstance(value, str):

        def _sub(match: re.Match[str]) -> str:
            key = match.group(1)
            if key in known:
                return match.group(0)
            stripped.add(key)
            return ""

        return CITATION_MARKER_RE.sub(_sub, value).replace("  ", " ").strip()
    if isinstance(value, list):
        return [_strip_unknown_markers(item, known, stripped) for item in value]
    if isinstance(value, dict):
        return {
            k: _strip_unknown_markers(v, known, stripped)
            for k, v in value.items()
        }
    return value


def _payload_bytes(payload: dict[str, Any]) -> int:
    return len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))


def _truncate_payload(
    payload: dict[str, Any], truncated: list[str]
) -> dict[str, Any]:
    """Halve the largest list slot until the payload fits the byte cap.

    Never silent: every touched slot is recorded in *truncated*.
    """
    guard = 0
    while _payload_bytes(payload) > PAYLOAD_MAX_BYTES and guard < 32:
        guard += 1
        largest_slot: str | None = None
        largest_len = 0
        for slot, value in payload.items():
            if isinstance(value, list) and len(value) > largest_len:
                largest_slot = slot
                largest_len = len(value)
        if largest_slot is None or largest_len <= 1:
            break
        payload[largest_slot] = payload[largest_slot][: max(1, largest_len // 2)]
        if largest_slot not in truncated:
            truncated.append(largest_slot)
    return payload


def _claims_block(claims: list[Any], limit: int = MAX_CLAIMS) -> str:
    lines: list[str] = []
    for claim in claims[:limit]:
        key = _claim_get(claim, "citation_key") or "?"
        verdict = _claim_get(claim, "verification_verdict") or "unverified"
        text = str(_claim_get(claim, "claim_text") or "")[:MAX_CLAIM_CHARS]
        lines.append(f"- [{key}] ({verdict}) {text}")
    return "\n".join(lines) if lines else "(no verified claims)"


@dataclass
class SlotOutcome:
    """Result of one slot's wire call (post guards/strip)."""

    status: str = "failed"  # "ok" | "empty" | "failed"
    items: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    attempts: int = 0
    duration_ms: float = 0.0
    dropped_unsourced: int = 0
    used_refs: set[str] = field(default_factory=set)
    warnings: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class SlotWireResults:
    """All slot outcomes of one wires run."""

    outcomes: dict[str, SlotOutcome] = field(default_factory=dict)
    used_refs: set[str] = field(default_factory=set)
    stripped_citation_keys: list[str] = field(default_factory=list)
    duration_ms: float = 0.0


def apply_slot_guards(
    kind: str,
    items: list[dict[str, Any]],
    valid_refs: set[str],
    *,
    enforce: bool = True,
) -> tuple[list[dict[str, Any]], int, set[str]]:
    """Canonicalize refs and drop unsourced items for evidenced kinds.

    Returns ``(kept_items, dropped_count, used_refs)``. Fail-open: with an
    empty evidence list (``valid_refs`` empty) or ``enforce=False`` (e.g. a
    Table whose user-declared columns claim the ``source_refs`` name),
    nothing is dropped and refs are cleared/left alone respectively.
    """
    used: set[str] = set()
    if not enforce:
        return items, 0, used
    if not valid_refs:
        for item in items:
            item["source_refs"] = []
        return items, 0, used

    kept: list[dict[str, Any]] = []
    dropped = 0
    evidenced = kind in ("table", "strings")
    for item in items:
        raw_refs = item.get("source_refs")
        canonical: list[str] = []
        if isinstance(raw_refs, list):
            for ref in raw_refs:
                canon = coerce_citation_ref(ref)
                if canon in valid_refs and canon not in canonical:
                    canonical.append(canon)
        item["source_refs"] = canonical
        if evidenced and not canonical:
            dropped += 1
            continue
        used.update(canonical)
        kept.append(item)
    return kept, dropped, used


def _build_wire_messages(
    slot_name: str,
    spec: SlotSpec,
    evidence_block: str,
    known_keys: set[str],
    claims: list[Any],
    report: str,
    *,
    trimmed: bool,
) -> list[dict[str, Any]]:
    claims_limit = WIRE_TRIM_CLAIMS if trimmed else WIRE_MAX_CLAIMS
    report_limit = WIRE_TRIM_REPORT_CHARS if trimmed else WIRE_REPORT_CHARS
    user = (
        "## Section to fill\n"
        f"{wire_slot_docs(slot_name, spec)}\n\n"
        "## Evidence sources (cite by index)\n"
        f"{evidence_block or '(no evidence sources — leave source_refs empty)'}\n\n"
        "## Allowed citation keys\n"
        f"{', '.join(sorted(known_keys)) if known_keys else '(none — use no markers)'}\n\n"
        "## Verified claims\n"
        f"{_claims_block(claims, claims_limit)}\n\n"
        "## Report excerpt (context)\n"
        f"{report[:report_limit]}"
    )
    return [
        {"role": "system", "content": _WIRE_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


async def _run_one_wire(
    *,
    slot_name: str,
    spec: SlotSpec,
    evidence_block: str,
    trimmed_evidence_block: str,
    known_keys: set[str],
    claims: list[Any],
    report: str,
    client: StructuredCompletionClient,
    timeout_s: float,
) -> SlotOutcome:
    """One slot's LLM call: timeout→trimmed retry, invalid→feedback retry."""
    outcome = SlotOutcome()
    started = time.monotonic()
    try:
        model_cls = build_slot_wire_model(slot_name, spec)
    except Exception as exc:  # noqa: BLE001 — fail-soft per slot
        outcome.error = f"wire schema build failed: {exc}"[:300]
        outcome.duration_ms = (time.monotonic() - started) * 1000
        return outcome

    messages = _build_wire_messages(
        slot_name, spec, evidence_block, known_keys, claims, report,
        trimmed=False,
    )
    structured: BaseModel | None = None
    for attempt in (1, 2):
        outcome.attempts = attempt
        try:
            response = await asyncio.wait_for(
                client.complete(
                    messages=messages,
                    tier=ModelTier.analytical,
                    structured_output=model_cls,
                ),
                timeout=timeout_s,
            )
        except TimeoutError:
            outcome.error = (
                f"wire timed out after {timeout_s:.0f}s (attempt {attempt})"
            )
            logger.warning(
                "WIRE_TIMEOUT slot=%s attempt=%d timeout_s=%.0f",
                slot_name, attempt, timeout_s,
            )
            if attempt == 1:
                messages = _build_wire_messages(
                    slot_name, spec, trimmed_evidence_block, known_keys,
                    claims, report, trimmed=True,
                )
                continue
            break
        except Exception as exc:  # noqa: BLE001 — fail-soft per slot
            outcome.error = str(exc)[:500]
            break

        candidate = getattr(response, "structured", None)
        if isinstance(candidate, BaseModel):
            structured = candidate
            outcome.error = None
            break

        # The client returns structured=None on strict parse failure —
        # recover through the tolerant ladder on the raw content.
        content = str(getattr(response, "content", "") or "")
        data: Any
        try:
            data = json.loads(content)
        except ValueError:
            data = json_repair_structured(content)
        if isinstance(data, dict):
            unwrapped = unwrap_placeholder_envelope(model_cls, data)
            if unwrapped is not data:
                logger.warning("WIRE_ENVELOPE_UNWRAPPED slot=%s", slot_name)
            try:
                structured, lenient_dropped = validate_lenient(
                    model_cls, unwrapped
                )
                if lenient_dropped:
                    logger.warning(
                        "WIRE_VALIDATION_LENIENT slot=%s dropped=%s",
                        slot_name, lenient_dropped[:10],
                    )
                    outcome.warnings.append({
                        "code": "lenient_validation",
                        "slot": slot_name,
                        "message": (
                            f"dropped {len(lenient_dropped)} invalid "
                            "field(s) during validation"
                        ),
                    })
                outcome.error = None
                break
            except WireValidationError as exc:
                outcome.error = str(exc)[:1000]
        else:
            outcome.error = (
                f"wire returned non-JSON content ({len(content)} chars)"
            )
        if attempt == 1:
            messages = [
                *messages,
                {"role": "assistant", "content": content[:4000]},
                {
                    "role": "user",
                    "content": (
                        "The previous output failed validation:\n"
                        f"{outcome.error}\n"
                        "Return the corrected JSON object only."
                    ),
                },
            ]

    outcome.duration_ms = (time.monotonic() - started) * 1000
    if structured is not None:
        rows = getattr(structured, slot_name, None) or []
        outcome.items = [
            row.model_dump(mode="json") if isinstance(row, BaseModel) else row
            for row in rows
        ]
        outcome.status = "ok"
    return outcome


def _refs_enforced(spec: SlotSpec) -> bool:
    """False when the wire model carries no dedicated source_refs field."""
    if spec.kind == "table":
        return all(col.key != "source_refs" for col in spec.columns)
    return spec.kind in ("table", "strings", "metrics")


async def run_slot_wires(
    *,
    slots: dict[str, SlotSpec],
    evidence: list[EvidenceItem],
    claims: list[Any],
    report: str,
    llm: StructuredCompletionClient,
    only_slots: set[str] | None = None,
    concurrency: int = WIRE_CONCURRENCY,
    wire_timeout_s: float = WIRE_TIMEOUT_S,
) -> SlotWireResults:
    """Run one wire per slot in parallel; guards + marker strip applied."""
    started = time.monotonic()
    known_keys = _known_citation_keys(claims)
    evidence_block = render_evidence_block(evidence)
    trimmed_evidence_block = render_evidence_block(
        evidence, budget_chars=WIRE_TRIM_EVIDENCE_CHARS
    )
    valid_refs = {item.ref for item in evidence}
    target_slots = {
        name: spec
        for name, spec in slots.items()
        if only_slots is None or name in only_slots
    }

    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def _guarded(name: str, spec: SlotSpec) -> SlotOutcome:
        async with semaphore:
            return await _run_one_wire(
                slot_name=name,
                spec=spec,
                evidence_block=evidence_block,
                trimmed_evidence_block=trimmed_evidence_block,
                known_keys=known_keys,
                claims=claims,
                report=report,
                client=llm,
                timeout_s=wire_timeout_s,
            )

    raw_results = await asyncio.gather(
        *(_guarded(name, spec) for name, spec in target_slots.items()),
        return_exceptions=True,
    )

    results = SlotWireResults()
    stripped: set[str] = set()
    for (name, spec), raw in zip(
        target_slots.items(), raw_results, strict=True
    ):
        if isinstance(raw, BaseException):
            results.outcomes[name] = SlotOutcome(
                status="failed", error=str(raw)[:300], attempts=1
            )
            continue
        outcome = raw
        if outcome.status == "ok":
            kept, dropped, slot_used = apply_slot_guards(
                spec.kind,
                outcome.items,
                valid_refs,
                enforce=_refs_enforced(spec),
            )
            if dropped:
                logger.warning(
                    "WIRE_GUARD_DROPPED slot=%s dropped=%d kept=%d",
                    name, dropped, len(kept),
                )
                outcome.warnings.append({
                    "code": "dropped_unsourced",
                    "slot": name,
                    "message": (
                        f"dropped {dropped} item(s) without a resolvable "
                        "source ref"
                    ),
                })
            outcome.items = [
                _strip_unknown_markers(item, known_keys, stripped)
                for item in kept
            ]
            outcome.dropped_unsourced = dropped
            outcome.used_refs = slot_used
            outcome.status = "ok" if outcome.items else "empty"
            results.used_refs |= slot_used
        results.outcomes[name] = outcome

    results.stripped_citation_keys = sorted(stripped)
    results.duration_ms = (time.monotonic() - started) * 1000
    return results


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def build_pending_envelope(
    *,
    binding: str,
    agent_id: str | None,
    surface_etag: str | None,
    slot_names: list[str],
) -> dict[str, Any]:
    """The stub persisted with the run: every slot pending, no data yet."""
    return {
        "version": 2,
        "binding": binding,
        "agent_id": agent_id,
        "surface_etag": surface_etag,
        "generated_at": _now_iso(),
        "data": {},
        "meta": {
            "model_tier": "analytical",
            "duration_ms": 0.0,
            "slots": {name: {"status": "pending"} for name in slot_names},
            "sources": [],
            "evidence": "pending",
            "warnings": [],
            "stripped_citation_keys": [],
            "truncated_slots": [],
        },
    }


def build_envelope_v2(
    *,
    binding: str,
    agent_id: str | None,
    surface_etag: str | None,
    results: SlotWireResults,
    evidence: list[EvidenceItem],
) -> dict[str, Any]:
    """The persisted ``verification_data["structured_output"]`` envelope."""
    data: dict[str, Any] = {}
    slots_meta: dict[str, Any] = {}
    warnings: list[dict[str, Any]] = []
    for name, outcome in results.outcomes.items():
        meta: dict[str, Any] = {
            "status": outcome.status,
            "attempts": outcome.attempts,
            "duration_ms": round(outcome.duration_ms, 1),
        }
        if outcome.error:
            meta["error"] = outcome.error[:300]
        if outcome.dropped_unsourced:
            meta["dropped_unsourced"] = outcome.dropped_unsourced
        slots_meta[name] = meta
        if outcome.status in ("ok", "empty"):
            data[name] = outcome.items
        warnings.extend(outcome.warnings)

    truncated: list[str] = []
    data = _truncate_payload(data, truncated)
    for slot in truncated:
        warnings.append({
            "code": "truncated_slot",
            "message": f"slot '{slot}' was truncated to fit the size cap",
            "slot": slot,
        })

    return {
        "version": 2,
        "binding": binding,
        "agent_id": agent_id,
        "surface_etag": surface_etag,
        "generated_at": _now_iso(),
        "data": data,
        "meta": {
            "model_tier": "analytical",
            "duration_ms": round(results.duration_ms, 1),
            "slots": slots_meta,
            "sources": build_legend(evidence, results.used_refs),
            "evidence": "pool" if evidence else "report_only",
            "warnings": warnings,
            "stripped_citation_keys": results.stripped_citation_keys,
            "truncated_slots": truncated,
        },
    }


def _merge_envelopes(
    prior: dict[str, Any],
    fresh: dict[str, Any],
) -> dict[str, Any]:
    """Merge a partial-rerun envelope over the prior one.

    Prior slots that were NOT rerun keep their data, slot meta, and legend
    refs untouched. Fresh legend entries are merged by URL: an already-known
    URL reuses its prior ref, a new URL gets the next free number — and the
    fresh items' ``source_refs`` are remapped accordingly, so chips stay
    correct on both sides.
    """
    prior_meta = prior.get("meta") or {}
    prior_legend = [
        dict(entry)
        for entry in (prior_meta.get("sources") or [])
        if isinstance(entry, dict)
    ]
    url_to_ref = {
        str(entry.get("url")): str(entry.get("ref"))
        for entry in prior_legend
        if entry.get("url")
    }
    next_num = (
        max(
            (
                int(str(entry.get("ref")))
                for entry in prior_legend
                if str(entry.get("ref")).isdigit()
            ),
            default=0,
        )
        + 1
    )

    fresh_meta = fresh.get("meta") or {}
    remap: dict[str, str] = {}
    merged_legend = prior_legend[:]
    for entry in fresh_meta.get("sources") or []:
        if not isinstance(entry, dict):
            continue
        url = str(entry.get("url"))
        ref = str(entry.get("ref"))
        known = url_to_ref.get(url)
        if known is not None:
            remap[ref] = known
            continue
        new_ref = str(next_num)
        next_num += 1
        remap[ref] = new_ref
        url_to_ref[url] = new_ref
        merged_legend.append({**entry, "ref": new_ref})

    def _remap_refs(items: Any) -> None:
        if not isinstance(items, list):
            return
        for item in items:
            if isinstance(item, dict) and isinstance(
                item.get("source_refs"), list
            ):
                item["source_refs"] = [
                    remap.get(str(ref), str(ref))
                    for ref in item["source_refs"]
                ]

    data = dict(prior.get("data") or {})
    for slot, items in (fresh.get("data") or {}).items():
        _remap_refs(items)
        data[slot] = items

    slots_meta = dict(prior_meta.get("slots") or {})
    slots_meta.update(fresh_meta.get("slots") or {})

    merged = {**prior, **fresh, "data": data}
    merged["meta"] = {
        **prior_meta,
        **fresh_meta,
        "slots": slots_meta,
        "sources": merged_legend,
    }
    return merged


async def build_structured_envelope(
    *,
    binding: str,
    agent_id: str | None,
    surface_etag: str | None,
    slots: dict[str, SlotSpec],
    report: str,
    claims: list[Any],
    sources: list[Any],
    llm: StructuredCompletionClient,
    only_slots: set[str] | None = None,
    prior_envelope: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the structured-output envelope (generation only, no persistence).

    Builds evidence, runs the slot wires, assembles the v2 envelope (merged
    over *prior_envelope* on a partial rerun) and RETURNS it. Callers that
    persist (the main app) do so separately; callers that stream it (the
    shell-app) emit the returned dict directly.
    """
    evidence = build_evidence(sources, claims)
    results = await run_slot_wires(
        slots=slots,
        evidence=evidence,
        claims=claims,
        report=report,
        llm=llm,
        only_slots=only_slots,
    )
    envelope = build_envelope_v2(
        binding=binding,
        agent_id=agent_id,
        surface_etag=surface_etag,
        results=results,
        evidence=evidence,
    )
    if prior_envelope is not None and only_slots:
        envelope = _merge_envelopes(prior_envelope, envelope)
    return envelope


__all__ = [
    "CITATION_MARKER_RE",
    "SlotOutcome",
    "SlotWireResults",
    "StructuredCompletionClient",
    "apply_slot_guards",
    "build_envelope_v2",
    "build_pending_envelope",
    "build_structured_envelope",
    "run_slot_wires",
]
