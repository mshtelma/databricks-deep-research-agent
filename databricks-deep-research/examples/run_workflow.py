#!/usr/bin/env python3
"""Run any framework example workflow with streaming output.

Usage:
    uv run examples/run_workflow.py                             # list available workflows
    uv run examples/run_workflow.py simple_research.yaml        # default query
    uv run examples/run_workflow.py single_agent "What is quantum computing?"
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.WARNING, format="%(message)s")

FRAMEWORK_DIR = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = Path(__file__).resolve().parent


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for name in (".env.test", ".env"):
        candidate = FRAMEWORK_DIR / name
        if candidate.exists():
            load_dotenv(candidate, override=False)
            return


_load_dotenv()


def _list_workflows() -> None:
    """Print available workflows and exit."""
    import yaml

    yamls = sorted(EXAMPLES_DIR.glob("*.yaml"))
    if not yamls:
        print("No YAML workflows found in examples/")
        sys.exit(1)

    print("\nAvailable workflows in examples/:\n")
    for path in yamls:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        name = raw.get("name", path.stem)
        desc = raw.get("description", "")
        line = f"  {path.name:<45s} {name}"
        if desc:
            line += f" -- {desc}"
        print(line)

    print('\nUsage: uv run examples/run_workflow.py <workflow.yaml> ["your query"]\n')


# ---------------------------------------------------------------------------
# Event formatting
# ---------------------------------------------------------------------------

_T0: float = 0.0
DEFAULT_QUERY = "What are the latest advances in quantum computing in 2024?"


def _ts() -> str:
    elapsed = time.monotonic() - _T0
    minutes, seconds = divmod(int(elapsed), 60)
    return f"{minutes:02d}:{seconds:02d}"


def _format_event(event) -> str | None:  # type: ignore[no-untyped-def]
    """Format a StreamEvent into a single display line, or None to skip."""
    ts = _ts()
    et = event.event_type

    if et == "workflow_started":
        return f"[{ts}] START {event.workflow_name}"
    if et == "workflow_completed":
        dur = event.duration_ms / 1000
        return f"[{ts}] DONE  {dur:.1f}s, {event.total_tokens} tokens, {event.total_sources} sources"
    if et == "workflow_failed":
        return f"[{ts}] FAIL  {event.error_type}: {event.error_message}"
    if et == "coordinator_classified":
        return f"[{ts}] QUERY complexity={event.complexity}, depth={event.recommended_depth}"
    if et == "node_error":
        retry = f" [retry {event.retry_attempt}]" if event.will_retry else ""
        return f"[{ts}] ERROR {event.node_id}: {event.error_message[:120]}{retry}"
    if et == "node_started":
        return f"[{ts}] NODE  {event.node_id} -- started ({event.node_type})"
    if et == "node_completed":
        return f"[{ts}] NODE  {event.node_id} -- completed ({event.duration_ms / 1000:.1f}s)"
    if et == "tool_call":
        query = event.arguments.get("query") or event.arguments.get("url", "")
        args = f'("{query[:80]}")' if query else ""
        return f"[{ts}] TOOL  {event.tool_name}{args}"
    if et == "tool_result":
        status = "ok" if event.tool_success else f"FAIL: {event.tool_error[:80]}"
        return f"[{ts}] TOOL  {event.tool_name} -- {event.source_count} sources [{status}]"
    if et == "plan_created":
        labels = [s.get("label", s.get("title", "?"))[:40] for s in event.steps]
        return f"[{ts}] PLAN  {len(event.steps)} steps: {labels}"
    if et == "item_started":
        return f"[{ts}] ITEM  Step {event.item_index + 1}/{event.total_items}: {event.item_summary[:80]}"
    if et in ("evaluation_decision", "reflection_decision"):
        return f"[{ts}] EVAL  {event.decision} -- {event.reasoning[:100]}"
    if et == "verification_summary":
        return (
            f"[{ts}] VERIFY {event.verified_claims}/{event.total_claims} claims, "
            f"confidence={event.overall_confidence:.2f}"
        )
    return None


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


async def main(yaml_name: str, query: str) -> None:
    global _T0

    yaml_path = EXAMPLES_DIR / yaml_name
    if not yaml_path.exists():
        print(f"Error: {yaml_path} not found")
        sys.exit(1)

    from databricks_deep_research import WorkflowError, WorkflowRunner, WorkflowValidationError

    try:
        runner = WorkflowRunner.from_databricks()
    except RuntimeError as exc:
        print(f"Auth error: {exc}\nHint: create a .env.test file with credentials.")
        sys.exit(1)

    print(f"\nWorkflow: {yaml_name}")
    print(f"Query:    {query}\n")

    _T0 = time.monotonic()
    completed_stats: dict[str, Any] | None = None
    try:
        async for event in runner.stream(str(yaml_path), query=query):
            if event.event_type == "workflow_completed":
                completed_stats = {
                    "tokens": event.total_tokens,
                    "sources": event.total_sources,
                    "steps": event.total_steps_executed,
                }
            line = _format_event(event)
            if line:
                print(line)
    except KeyboardInterrupt:
        print("\n\nInterrupted.")
    except WorkflowValidationError as exc:
        print(f"\nValidation failed:")
        for err in exc.errors:
            print(f"  - {err}")
        sys.exit(1)
    except WorkflowError as exc:
        print(f"\nWorkflow error: {exc}")
        sys.exit(1)
    finally:
        await runner.aclose()

    wall = time.monotonic() - _T0
    result = runner.last_result
    if result is None:
        return

    # One-line summary
    summary = f"Completed in {wall:.1f}s"
    if completed_stats:
        summary += (
            f" ({completed_stats['tokens']} tokens, "
            f"{completed_stats['sources']} sources, "
            f"{completed_stats['steps']} steps)"
        )
    print(f"\n{summary}")

    # Print report
    if result.output:
        sep = "=" * 60
        print(f"\n{sep}\n OUTPUT ({len(result.output)} chars)\n{sep}")
        print(result.output)
        print()
    else:
        print("[No output produced]")

    # Print numbered sources
    if result.sources:
        print("Sources:")
        for i, item in enumerate(result.sources[:20], 1):
            url = item.get("url", "") if isinstance(item, dict) else getattr(item, "url", "")
            title = item.get("title", "") if isinstance(item, dict) else getattr(item, "title", "")
            if url:
                print(f"  {i:>2}. {title or url}")


if __name__ == "__main__":
    args = sys.argv[1:]

    if not args:
        _list_workflows()
        sys.exit(0)

    yaml_name = args[0]
    if not yaml_name.endswith(".yaml"):
        yaml_name += ".yaml"

    query = args[1] if len(args) > 1 else DEFAULT_QUERY
    asyncio.run(main(yaml_name, query))
