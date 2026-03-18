"""Shared test diagnostic helpers for rich output.

Used by complex and integration test suites to print human-readable
event timelines, pool contents, search queries, and evaluator decisions.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Event formatting
# ---------------------------------------------------------------------------


def _format_event_detail(e: Any) -> str:
    """Extract human-readable detail from an event."""
    name = type(e).__name__

    if name == "ToolCallEvent":
        args = getattr(e, "arguments", {})
        args_display = {k: str(v)[:200] for k, v in args.items()}
        return f"tool={e.tool_name} args={args_display}"

    if name == "ToolResultEvent":
        return (
            f"tool={e.tool_name} "
            f"result_len={len(e.result_summary)} "
            f"sources={e.source_count}"
        )

    if name == "ItemStartedEvent":
        return f"step {e.item_index + 1}/{e.total_items}: {e.item_summary[:200]}"

    if name == "ItemCompletedEvent":
        return f"step {e.item_index + 1} done (total processed: {e.items_processed})"

    if name == "ItemsExtractedEvent":
        return f"total_items={e.total_items} cycle={e.cycle}"

    if name == "EvaluationDecisionEvent":
        suffix = ""
        if getattr(e, "evidence_sufficiency", None):
            suffix += f" evidence_sufficiency={e.evidence_sufficiency}"
        if getattr(e, "failure_mode", None):
            suffix += f" failure_mode={e.failure_mode}"
        return f"decision={e.decision}{suffix} reasoning={e.reasoning[:500]}"

    if name == "PlanCreatedEvent":
        steps = getattr(e, "steps", [])
        return f"title={getattr(e, 'title', '?')} steps={len(steps)}"

    if name == "CoordinatorClassifiedEvent":
        return f"complexity={e.complexity} is_simple={e.is_simple}"

    if name == "AgentOutputEvent":
        return f"key={e.output_key} preview={e.output_preview[:200]}"

    if name == "NodeStartedEvent":
        return f"type={e.node_type} label={e.label}"

    if name == "NodeCompletedEvent":
        return f"duration={getattr(e, 'duration_ms', '?'):.0f}ms"

    if name == "NodeErrorEvent":
        return f"error={e.error_message[:300]} retry={e.will_retry}"

    if name == "NodeSkippedEvent":
        return f"reason={e.reason[:80]}"

    if name == "PlanAndExecuteExitEvent":
        planned = getattr(e, "total_planned", "?")
        suffix = ""
        if getattr(e, "completion_mode", None):
            suffix += f" completion_mode={e.completion_mode}"
        if getattr(e, "evidence_sufficiency", None):
            suffix += f" evidence_sufficiency={e.evidence_sufficiency}"
        if getattr(e, "failure_mode", None):
            suffix += f" failure_mode={e.failure_mode}"
        return (
            f"reason={e.reason}{suffix} "
            f"items={e.total_items_processed}/{planned} "
            f"replans={e.replan_cycles}"
        )

    if name == "WorkflowStartedEvent":
        return f"workflow={e.workflow_name}"

    if name == "WorkflowCompletedEvent":
        return f"duration={e.duration_ms:.0f}ms sources={e.total_sources}"

    if name == "WorkflowFailedEvent":
        return (
            f"duration={e.duration_ms:.0f}ms "
            f"error_type={e.error_type} error={e.error_message[:120]}"
        )

    if name == "BackgroundCompletedEvent":
        return f"sources_discovered={e.sources_discovered}"

    if name == "ReflectionDecisionEvent":
        suffix = ""
        if getattr(e, "evidence_sufficiency", None):
            suffix += f" evidence_sufficiency={e.evidence_sufficiency}"
        if getattr(e, "failure_mode", None):
            suffix += f" failure_mode={e.failure_mode}"
        return f"decision={e.decision}{suffix} reasoning={e.reasoning[:500]}"

    if name == "LoopIterationEvent":
        return f"iteration={e.iteration}/{e.max_iterations}"

    if name == "LoopExitEvent":
        return f"reason={e.reason} iterations={e.total_iterations}"

    if name == "BranchSelectedEvent":
        return f"branch={e.branch_index} {e.condition_summary}"

    if name == "ToolCacheHitEvent":
        return f"tool={e.tool_name} (cached)"

    if name == "ReplanTriggeredEvent":
        return f"cycle={e.cycle} remaining={e.items_remaining}"

    # Fallback
    fields = {
        k: v
        for k, v in vars(e).items()
        if k not in ("node_id", "timestamp", "event_type") and v
    }
    return str(fields)[:120] if fields else ""


def print_event_timeline(events: list[Any]) -> None:
    """Print a chronological, human-readable event timeline."""
    print("\n--- EVENT TIMELINE ---")
    for i, e in enumerate(events):
        name = type(e).__name__
        detail = _format_event_detail(e)
        print(f"  [{i:03d}] {name}: {detail}")


# ---------------------------------------------------------------------------
# Event counting
# ---------------------------------------------------------------------------


def event_summary(events: list[Any]) -> dict[str, int]:
    """Count events by type for diagnostics."""
    counts: dict[str, int] = {}
    for e in events:
        name = type(e).__name__
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def assert_terminal_plan_exit(events: list[Any]) -> Any:
    """Assert that plan-and-execute terminated via a valid terminal state.

    Accepts both legacy normal completions and evidence-aware degraded
    completions introduced for exhausted-but-insufficient research.
    Returns the final `PlanAndExecuteExitEvent` for further assertions.
    """
    from databricks_deep_research.events.types import PlanAndExecuteExitEvent

    exit_events = [e for e in events if isinstance(e, PlanAndExecuteExitEvent)]
    assert exit_events, "Expected at least one PlanAndExecuteExitEvent"

    exit_event = exit_events[-1]
    reason = getattr(exit_event, "reason", None)
    completion_mode = getattr(exit_event, "completion_mode", None)
    evidence_sufficiency = getattr(exit_event, "evidence_sufficiency", None)
    failure_mode = getattr(exit_event, "failure_mode", None)

    normal_reasons = {"items_exhausted", "evaluator_complete"}
    if reason in normal_reasons:
        return exit_event

    assert reason == "insufficient_evidence_exhausted", (
        "Unexpected plan-and-execute exit reason: "
        f"reason={reason} completion_mode={completion_mode} "
        f"evidence_sufficiency={evidence_sufficiency} failure_mode={failure_mode}"
    )
    assert completion_mode == "degraded", (
        "Evidence-exhausted completion must be degraded: "
        f"reason={reason} completion_mode={completion_mode}"
    )
    assert evidence_sufficiency in {"partial", "insufficient"}, (
        "Evidence-exhausted completion must carry weak evidence metadata: "
        f"reason={reason} evidence_sufficiency={evidence_sufficiency}"
    )
    return exit_event


# ---------------------------------------------------------------------------
# Report validation
# ---------------------------------------------------------------------------

_REPORT_FAILURE_PHRASES = [
    "cannot generate",
    "pools are empty",
    "no observations",
    "no gathered observations",
    "no available sources",
    "no sources",
    "unable to generate",
    "i cannot",
    "i don't have",
    "no data",
    "no information",
    "no research findings",
    "please provide",
]


def assert_report_has_substance(
    report_str: str,
    min_length: int = 500,
    required_term_groups: tuple[tuple[str, ...], ...] = (),
) -> None:
    """Assert the report contains real research, not a failure message."""
    report_lower = report_str.lower()
    for phrase in _REPORT_FAILURE_PHRASES:
        assert phrase not in report_lower, (
            f"Report contains failure indicator '{phrase}'. "
            f"This means the synthesizer did not receive pool data.\n"
            f"Report preview: {report_str[:300]}"
        )
    assert len(report_str) >= min_length, (
        f"Report too short ({len(report_str)} chars, minimum {min_length}). "
        f"Preview: {report_str[:300]}"
    )
    for group in required_term_groups:
        assert any(term in report_lower for term in group), (
            "Report is missing a required concept group. "
            f"Expected one of: {group}. Preview: {report_str[:300]}"
        )


# ---------------------------------------------------------------------------
# Rich diagnostic printers
# ---------------------------------------------------------------------------


def print_research_plan(events: list[Any]) -> None:
    """Print the research plan from PlanCreatedEvent(s)."""
    from databricks_deep_research.events.types import PlanCreatedEvent

    plan_events = [e for e in events if isinstance(e, PlanCreatedEvent)]
    if not plan_events:
        print("\n--- RESEARCH PLAN ---\n  (no plan created)")
        return

    plan = plan_events[0]
    print(f"\n--- RESEARCH PLAN: {plan.title} ---")
    for i, step in enumerate(plan.steps):
        step_str = (
            step.get("description", str(step))
            if isinstance(step, dict)
            else str(step)
        )
        print(f"  Step {i + 1}: {step_str[:120]}")


def print_step_execution(events: list[Any]) -> None:
    """Print per-step execution details from Item events."""
    from databricks_deep_research.events.types import (
        ItemCompletedEvent,
        ItemStartedEvent,
    )

    items_started = [e for e in events if isinstance(e, ItemStartedEvent)]
    items_completed = [e for e in events if isinstance(e, ItemCompletedEvent)]

    if not items_started:
        print("\n--- STEP EXECUTION ---\n  (no steps executed)")
        return

    print("\n--- STEP EXECUTION ---")
    for started in items_started:
        completed = next(
            (c for c in items_completed if c.item_index == started.item_index),
            None,
        )
        print(f"  Step {started.item_index + 1}: {started.item_summary[:200]}")
        if completed:
            print(f"    -> Completed (total processed: {completed.items_processed})")

    # Show skipped steps info
    from databricks_deep_research.events.types import PlanCreatedEvent

    plan_events = [e for e in events if isinstance(e, PlanCreatedEvent)]
    if plan_events:
        total_planned = len(plan_events[0].steps)
        if total_planned > len(items_started):
            skipped = total_planned - len(items_started)
            print(f"  ({skipped} steps skipped by evaluator)")


def tool_calls_for_node(events: list[Any], node_id: str) -> list[Any]:
    """Return tool calls emitted by a specific workflow node."""
    from databricks_deep_research.events.types import ToolCallEvent

    return [
        event for event in events
        if isinstance(event, ToolCallEvent) and getattr(event, "node_id", "") == node_id
    ]


def tool_calls_for_node_prefix(events: list[Any], node_prefix: str) -> list[Any]:
    """Return tool calls emitted by nodes whose ids share a prefix."""
    from databricks_deep_research.events.types import ToolCallEvent

    return [
        event for event in events
        if isinstance(event, ToolCallEvent)
        and getattr(event, "node_id", "").startswith(node_prefix)
    ]


def assert_plan_executed(events: list[Any], *, min_items: int = 1) -> None:
    """Assert that a plan-and-execute workflow actually executed research items."""
    from databricks_deep_research.events.types import (
        ItemCompletedEvent,
        ItemsExtractedEvent,
        WorkflowCompletedEvent,
    )

    extracted = [event for event in events if isinstance(event, ItemsExtractedEvent)]
    assert extracted, "Expected at least one ItemsExtractedEvent"
    assert max(event.total_items for event in extracted) >= min_items, (
        f"Expected at least {min_items} extracted plan items; "
        f"got {[event.total_items for event in extracted]}"
    )

    completed_items = [event for event in events if isinstance(event, ItemCompletedEvent)]
    assert len(completed_items) >= min_items, (
        f"Expected at least {min_items} completed plan items; "
        f"got {len(completed_items)}"
    )

    workflow_completed = [
        event for event in events if isinstance(event, WorkflowCompletedEvent)
    ]
    assert workflow_completed, "Expected WorkflowCompletedEvent"
    assert workflow_completed[-1].total_steps_executed >= min_items, (
        f"Expected workflow.total_steps_executed >= {min_items}; "
        f"got {workflow_completed[-1].total_steps_executed}"
    )


def print_search_queries(events: list[Any]) -> None:
    """Print all search and crawl queries from ToolCallEvents."""
    from databricks_deep_research.events.types import ToolCallEvent

    tool_calls = [e for e in events if isinstance(e, ToolCallEvent)]
    if not tool_calls:
        print("\n--- SEARCH QUERIES ---\n  (no tool calls)")
        return

    print("\n--- SEARCH QUERIES ---")
    for tc in tool_calls:
        args = getattr(tc, "arguments", {})
        if tc.tool_name == "web_search":
            print(f"  [search] {args.get('query', '?')}")
        elif tc.tool_name == "web_crawl":
            url_idx = args.get("url_index", "?")
            print(f"  [crawl]  url_index={url_idx}")
        else:
            print(f"  [{tc.tool_name}] {args}")


def print_pool_summary(state: Any) -> None:
    """Print pool contents summary from workflow state."""
    pools = getattr(state, "pools", {})
    if not pools:
        print("\n--- POOL CONTENTS ---\n  (no pools)")
        return

    print("\n--- POOL CONTENTS ---")
    for pool_name, pool in pools.items():
        count = pool.count()
        print(f"  {pool_name} ({count} items):")
        recent = pool.get_recent(5)
        for j, item in enumerate(recent):
            if isinstance(item, dict):
                title = item.get("title", "?")[:60]
                url = item.get("url", "?")[:80]
                print(f"    [{j + 1}] {title} — {url}")
            else:
                print(f"    [{j + 1}] {str(item)[:120]}")


def print_evaluator_decisions(events: list[Any]) -> None:
    """Print evaluation/reflection decisions timeline."""
    from databricks_deep_research.events.types import (
        EvaluationDecisionEvent,
        ReflectionDecisionEvent,
    )

    eval_events = [
        e
        for e in events
        if isinstance(e, EvaluationDecisionEvent | ReflectionDecisionEvent)
    ]
    if not eval_events:
        print("\n--- EVALUATOR DECISIONS ---\n  (no evaluations)")
        return

    print("\n--- EVALUATOR DECISIONS ---")
    for ev in eval_events:
        print(f"  [{ev.decision.upper()}] {ev.reasoning[:500]}")


def print_pool_operations(state: Any) -> None:
    """Print pool operation summary: adds, dedup rejections, evictions."""
    pools = getattr(state, "pools", {})
    if not pools:
        print("\n--- POOL OPERATIONS ---\n  (no pools)")
        return

    print("\n--- POOL OPERATIONS ---")
    for pool_name, pool in pools.items():
        total = pool.count()
        stats = getattr(pool, "stats", None)
        if stats is not None:
            print(
                f"  {pool_name}: {total} items, "
                f"attempted={stats.attempted}, added={stats.added}, "
                f"dup_key={stats.rejected_duplicate_key}, dup_hash={stats.rejected_duplicate_hash}"
            )
        else:
            dedup_keys = len(pool.seen_keys)
            dedup_hashes = len(pool.seen_hashes)
            print(
                f"  {pool_name}: {total} items, "
                f"{dedup_keys} seen_keys, {dedup_hashes} seen_hashes"
            )


def _verdict_icon(verdict: str | None) -> str:
    """Map a verification verdict to a label for display."""
    if verdict is None:
        return "(abstained)"
    mapping = {
        "supported": "SUPPORTED",
        "partial": "PARTIAL",
        "unsupported": "UNSUPPORTED",
        "contradicted": "CONTRADICTED",
    }
    return mapping.get(verdict.lower(), verdict.upper())


def _extract_report_context(
    report: str, start: int, end: int, claim_text: str, context_chars: int = 60,
) -> str:
    """Extract the claim's surrounding text from the report with markers."""
    # Validate offsets; fall back to find() if stale
    if start < 0 or end > len(report) or report[start:end] != claim_text:
        pos = report.find(claim_text)
        if pos < 0:
            return "(claim text not found in report)"
        start = pos
        end = pos + len(claim_text)

    ctx_start = max(0, start - context_chars)
    ctx_end = min(len(report), end + context_chars)

    prefix = ("..." if ctx_start > 0 else "") + report[ctx_start:start]
    suffix = report[end:ctx_end] + ("..." if ctx_end < len(report) else "")
    return f"{prefix}>>>{claim_text}<<<{suffix}"


def _build_event_index(
    events: list[Any],
) -> tuple[dict[int, Any], dict[int, list[Any]], dict[int, Any]]:
    """Build lookup dicts from verification events keyed by claim_index."""
    from databricks_deep_research.events.types import (
        CitationCorrectedEvent,
        ClaimVerifiedEvent,
        NumericClaimDetectedEvent,
    )

    verified_map: dict[int, Any] = {}
    corrected_map: dict[int, list[Any]] = {}
    numeric_map: dict[int, Any] = {}

    for e in events:
        if isinstance(e, ClaimVerifiedEvent):
            verified_map[e.claim_index] = e
        elif isinstance(e, CitationCorrectedEvent):
            corrected_map.setdefault(e.claim_index, []).append(e)
        elif isinstance(e, NumericClaimDetectedEvent):
            numeric_map[e.claim_index] = e

    return verified_map, corrected_map, numeric_map


def print_verification_summary(state: Any, events: list[Any]) -> None:
    """Print a UI-matching verification summary."""
    from databricks_deep_research.events.types import CitationCorrectedEvent

    summary = state.get("verification_summary")
    if not summary:
        details = state.get("verification_details")
        if isinstance(details, dict):
            summary = details.get("verification_summary", {})

    if not summary or not isinstance(summary, dict):
        print("\n--- VERIFICATION SUMMARY (UI View) ---\n  (no summary available)")
        return

    total = int(summary.get("total_claims", 0))
    denominator = int(summary.get("fact_rate_denominator", 0) or total)

    def _pct(n: int) -> str:
        if denominator == 0:
            return " 0.0%"
        return f"{n / denominator * 100:5.1f}%"

    supported = summary.get("supported_count", 0)
    partial = summary.get("partial_count", 0)
    unsupported = summary.get("unsupported_count", 0)
    contradicted = summary.get("contradicted_count", 0)
    abstained = summary.get("abstained_count", 0)
    overall = summary.get("overall_confidence", 0.0)
    warning = summary.get("warning", False)

    correction_events = [e for e in events if isinstance(e, CitationCorrectedEvent)]
    replace_count = sum(1 for e in correction_events if e.action == "replace")
    remove_count = sum(1 for e in correction_events if e.action == "remove")
    corrections_total = len(correction_events)

    correction_detail = ""
    if corrections_total > 0:
        parts = []
        if replace_count:
            parts.append(f"{replace_count} replace")
        if remove_count:
            parts.append(f"{remove_count} remove")
        other = corrections_total - replace_count - remove_count
        if other:
            parts.append(f"{other} other")
        correction_detail = f" ({', '.join(parts)})"

    print(f"\n{'=' * 60}")
    print("  VERIFICATION SUMMARY (UI View)")
    print(f"{'=' * 60}")
    print(f"  Total Claims:      {total:3d}")
    if denominator != total:
        print(f"  Fact Denominator:  {denominator:3d}")
    print(f"  Supported:         {supported:3d}   ({_pct(supported)})")
    print(f"  Partial:           {partial:3d}   ({_pct(partial)})")
    print(f"  Unsupported:       {unsupported:3d}   ({_pct(unsupported)})")
    print(f"  Contradicted:      {contradicted:3d}   ({_pct(contradicted)})")
    print(f"  Abstained:         {abstained:3d}")
    print(f"  Overall Confidence: {overall:.2f}")
    print(f"  Warning:            {'Yes' if warning else 'No'}")
    print(f"  Corrections:        {corrections_total}{correction_detail}")
    print(f"{'=' * 60}")


def print_citation_details(state: Any, events: list[Any]) -> None:
    """Print per-claim citation details matching the UI display."""
    report = state.get("report")
    report_str = str(report) if report else ""

    claims: list[dict[str, Any]] = []
    raw_claims = state.get("claims")
    if isinstance(raw_claims, list):
        claims = raw_claims
    else:
        details = state.get("verification_details")
        if isinstance(details, dict):
            claims = details.get("claims", [])

    if not claims:
        print("\n--- CITATION DETAILS (UI View) ---\n  (no claims available)")
        return

    verified_map, corrected_map, numeric_map = _build_event_index(events)

    print("\n--- CITATION DETAILS (UI View) ---")
    for i, claim in enumerate(claims):
        claim_text = claim.get("claim_text", "")
        claim_type = claim.get("claim_type", "?")
        verdict = claim.get("verification_verdict")
        reasoning = claim.get("verification_reasoning", "")
        confidence_level = claim.get("confidence_level", "")
        abstained = claim.get("abstained", False)
        evidence = claim.get("evidence")
        from_free = claim.get("from_free_block", False)
        citation_keys = claim.get("citation_keys", [])
        keys_display = ",".join(str(k) for k in citation_keys) if citation_keys else "?"

        # Cross-reference event confidence score
        verified_event = verified_map.get(i)
        confidence_score = ""
        if verified_event is not None:
            confidence_score = f" ({verified_event.confidence:.2f})"

        print(f"\n  Claim #{i + 1} [{claim_type}]{' ':>40s}[{keys_display}]")
        print(f"  {'-' * 64}")
        print(f"  Text:       {claim_text[:800]}")

        if abstained:
            print("  Verdict:    (abstained)")
        else:
            print(f"  Verdict:    {_verdict_icon(verdict)}")

        if reasoning:
            print(f"  Reasoning:  {reasoning[:800]}")
        print(f"  Confidence: {confidence_level or '?'}{confidence_score}")

        # Evidence
        if from_free and evidence is None:
            print("  Evidence:   (no evidence -- from free block)")
        elif evidence is None:
            print("  Evidence:   (none)")
        else:
            src_url = evidence.get("source_url", "?")
            quote = evidence.get("quote_text", "")
            relevance = evidence.get("relevance_score")
            section = evidence.get("section_heading")
            print("  Evidence:")
            print(f"    Source:    {src_url}")
            if quote:
                print(f"    Quote:     \"{quote[:800]}\"")
            if relevance is not None:
                print(f"    Relevance: {relevance:.2f}")
            if section:
                print(f"    Section:   {section}")

        # Report context
        if report_str and claim_text:
            pos_start = claim.get("position_start", -1)
            pos_end = claim.get("position_end", -1)
            context = _extract_report_context(
                report_str, pos_start, pos_end, claim_text,
            )
            print("  Report context:")
            print(f"    {context[:800]}")

        # Corrections for this claim
        corrections = corrected_map.get(i, [])
        for corr in corrections:
            print(
                f"  Correction: {corr.action} "
                f"(original={corr.original_key} -> corrected={corr.corrected_key})"
            )

        # Numeric info
        numeric_event = numeric_map.get(i)
        if numeric_event is not None:
            print(
                f"  Numeric:    value={numeric_event.numeric_value} "
                f"status={numeric_event.verification_status}"
            )

    print()


def print_full_diagnostics(events: list[Any], state: Any) -> None:
    """Print all diagnostic sections in order."""
    print_research_plan(events)
    print_step_execution(events)
    print_search_queries(events)
    print_pool_summary(state)
    print_pool_operations(state)
    print_evaluator_decisions(events)
    print_event_timeline(events)
    print_verification_summary(state, events)
    print_citation_details(state, events)
