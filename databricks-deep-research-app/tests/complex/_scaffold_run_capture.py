"""Capture helper for the scaffold-and-run integration test.

Writes designer + runner artifacts to a per-run directory so we can inspect
exactly what the LLM produced and what each workflow step did. Used only by
``test_scaffold_and_run.py``.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import sys
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

logger = logging.getLogger(__name__)


def _bound_tool_names(ast: dict[str, Any]) -> set[str]:
    """Return the set of tool names referenced inside workflow nodes.

    Walks the AST recursively.  ``body`` is a full NODE (has
    ``type``/``config``/``children``), so it is descended directly.
    ``planner`` and ``evaluator`` are agent-config dicts, not nodes, so
    they are wrapped in a fake node shell to reuse the same walk.
    """
    bound: set[str] = set()

    def walk(node: Any) -> None:
        if not isinstance(node, dict):
            return
        config = node.get("config") if isinstance(node.get("config"), dict) else {}
        for name in config.get("tools") or []:
            if isinstance(name, str):
                bound.add(name)
        # body is a NODE: walk it directly so its children/config are visited normally.
        body = config.get("body")
        if isinstance(body, dict):
            walk(body)
        # planner / evaluator are agent-config DICTS, not nodes — wrap as a fake
        # node so the same walk picks up their `tools` list.
        for cfg_key in ("planner", "evaluator"):
            cfg = config.get(cfg_key)
            if isinstance(cfg, dict):
                walk({"config": cfg})
        for child in node.get("children") or []:
            walk(child)

    walk(ast.get("root"))
    return bound


# Cap printed report lines so a huge report does not saturate the terminal.
# The full report still goes to ``runner/output.md`` for offline review.
_REPORT_LINE_CAP = 5000
# Banner width for console-report separators. Fits a typical 100-col terminal.
_BANNER_WIDTH = 100


def make_run_dir(root: Path, case_id: str) -> Path:
    """Create ``<root>/<ISO-ts>-<case_id>/`` with designer/ and runner/ subdirs."""
    ts = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = root / f"{ts}-{case_id}"
    (run_dir / "designer").mkdir(parents=True, exist_ok=True)
    (run_dir / "runner").mkdir(parents=True, exist_ok=True)
    return run_dir


def _to_jsonable(obj: Any) -> Any:
    """Best-effort conversion of Pydantic / dataclass / arbitrary objects to JSON-safe."""
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump(mode="json")
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        return {k: _to_jsonable(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
    return obj


def append_jsonl(path: Path, obj: Any) -> None:
    """Append one JSON-serialized object as a single line to ``path``."""
    payload = _to_jsonable(obj)
    with path.open("a") as f:
        f.write(json.dumps(payload, default=str, ensure_ascii=False))
        f.write("\n")


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(_to_jsonable(obj), indent=2, default=str, ensure_ascii=False))


def write_yaml(path: Path, obj: Any) -> None:
    path.write_text(yaml.safe_dump(_to_jsonable(obj), sort_keys=False))


def write_text(path: Path, text: str) -> None:
    path.write_text(text or "")


def summarize_events(events: list[Any]) -> dict[str, int]:
    """Histogram framework StreamEvents by event_type (or designer SSE events by type)."""
    counts: dict[str, int] = {}
    for ev in events:
        kind = getattr(ev, "event_type", None) or getattr(ev, "type", None) or type(ev).__name__
        counts[str(kind)] = counts.get(str(kind), 0) + 1
    return counts


def write_summary_md(
    run_dir: Path,
    *,
    case: dict[str, Any],
    designer_wall_s: float,
    designer_event_counts: dict[str, int],
    workflow_summary: dict[str, int] | None,
    validation_errors: list[dict[str, Any]],
    advice: list[dict[str, Any]],
    runner_wall_s: float,
    runner_event_counts: dict[str, int],
    output_chars: int,
    source_count: int,
) -> None:
    """Human-readable summary of the run."""
    lines: list[str] = [
        f"# Scaffold-and-Run: {case['id']}",
        "",
        f"- **Intent**: {case['intent'].strip().splitlines()[0]} ...",
        f"- **Query**: `{case['query']}`",
        "",
        "## Designer phase",
        "",
        f"- Wall time: **{designer_wall_s:.1f}s**",
        f"- Event histogram: `{designer_event_counts}`",
    ]
    if workflow_summary is not None:
        lines.append(
            f"- AST summary: nodes={workflow_summary.get('node_count')} "
            f"tools={workflow_summary.get('tool_count')} "
            f"sources={workflow_summary.get('source_count')}"
        )
    lines.append(f"- Validation errors: **{len(validation_errors)}**")
    if validation_errors:
        for err in validation_errors[:10]:
            lines.append(f"  - `{err.get('path') or '/'}`: {err.get('message')}")
    lines.append(f"- Quality advice items: **{len(advice)}**")
    if advice:
        for item in advice[:10]:
            lines.append(f"  - `{item.get('path') or '/'}` ({item.get('kind')}): {item.get('message')}")
    lines += [
        "",
        "## Runner phase",
        "",
        f"- Wall time: **{runner_wall_s:.1f}s**",
        f"- Event histogram: `{runner_event_counts}`",
        f"- Output: **{output_chars} chars** (see `runner/output.md`)",
        f"- Sources cited: **{source_count}** (see `runner/sources.json`)",
        "",
        "## Files",
        "",
        "- `intent.txt`, `query.txt`, `case.json`",
        "- `designer/events.jsonl`, `designer/workflow.json`, `designer/workflow.yaml`,",
        "  `designer/validation.json`, `designer/advice.json`, `designer/messages.json`",
        "- `runner/events.jsonl`, `runner/result.json`, `runner/output.md`, `runner/sources.json`",
    ]
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n")


def _format_source_entry(idx: int, source: Any, *, title_width: int = 90) -> str:
    """Render a single source entry for the console-report block.

    Accepts dicts, Pydantic models, plain objects with ``.title``/``.url``
    attributes, or bare strings (treated as URL-only). Always returns a
    string; never raises.
    """
    if isinstance(source, str):
        return f"  [{idx:>2}] {source}"
    if isinstance(source, BaseModel):
        # Avoid pulling the whole model into the title; fish out the two
        # fields we care about and stringify the model if neither is present.
        title = getattr(source, "title", None)
        url = getattr(source, "url", None)
        if title is None and url is None:
            return f"  [{idx:>2}] {source!s}"
        title = title or "(no title)"
        url = url or ""
    elif isinstance(source, dict):
        title = source.get("title") or "(no title)"
        url = source.get("url") or ""
    else:
        title = getattr(source, "title", None) or "(no title)"
        url = getattr(source, "url", None) or ""
    title_text = str(title)
    if len(title_text) > title_width:
        title_text = title_text[: title_width - 1] + "…"
    return f"  [{idx:>2}] {title_text}  —  {url}"


def _format_state_value(value: Any) -> str:
    """Render a top-level state value for console printing.

    Pydantic models are dumped as indented JSON; dicts/lists are dumped as
    JSON; strings pass through; everything else falls back to ``str(value)``.
    Never raises — best-effort, log-and-fall-back on serialization errors.
    """
    if value is None:
        return "(none)"
    if isinstance(value, str):
        return value
    if isinstance(value, BaseModel):
        try:
            return value.model_dump_json(indent=2)
        except Exception:  # noqa: BLE001 — last-resort serialization fallback
            return str(value)
    if isinstance(value, (dict, list, tuple)):
        try:
            return json.dumps(_to_jsonable(value), indent=2, default=str, ensure_ascii=False)
        except Exception:  # noqa: BLE001
            return str(value)
    return str(value)


def emit_console_report(
    *,
    case_id: str,
    report: str | None,
    sources: list[Any] | None,
    runtime_state: Any | None,
    extra_keys: tuple[str, ...] = ("coverage_review", "directives"),
    banner_width: int = _BANNER_WIDTH,
    line_cap: int = _REPORT_LINE_CAP,
    stream: Any | None = None,
) -> None:
    """Print a workflow result to stdout in a banner-delimited format.

    Designed for interactive runs via ``pytest -s``. Flushes after every
    chunk so output is visible in real time during ~10-minute runs.

    Never raises — any failure to introspect ``runtime_state`` is logged and
    the dump continues with whatever fields were successfully extracted.

    Parameters
    ----------
    case_id : str
        Human-readable case identifier (e.g., ``"investment_research"``).
    report : str | None
        Final markdown report (``WorkflowRunResult.output``). Empty/``None``
        is replaced with the literal ``"(empty report)"`` placeholder.
    sources : list[Any] | None
        Citeable source records; each item may be a dict, a Pydantic model,
        a plain object with ``.title``/``.url`` attributes, or a bare URL
        string.
    runtime_state : Any | None
        ``WorkflowRunResult.runtime_state``. If present, its ``.values``
        dict is scanned for any key in ``extra_keys`` and printed when
        present. The lookup tolerates shape drift via ``getattr`` +
        ``try/except``.
    extra_keys : tuple[str, ...]
        Top-level state keys to surface. Defaults cover the two-pass
        reflector workflow but the caller can pass more.
    banner_width : int
        Width of separator lines.
    line_cap : int
        Maximum number of report lines to print. Reports longer than this
        are truncated with a footer pointing to the artifact directory.
    stream : Any | None
        Optional file-like object to print into (mostly for tests via
        ``capsys``); defaults to ``sys.stdout``.
    """
    out = stream if stream is not None else sys.stdout
    banner = "=" * banner_width

    def _emit(line: str = "") -> None:
        print(line, file=out, flush=True)

    _emit("")
    _emit(banner)
    _emit(f"GENERATED REPORT  case={case_id}  chars={len(report or '')}")
    _emit(banner)
    report_text = report if report else "(empty report)"
    report_lines = report_text.splitlines() or [""]
    if len(report_lines) > line_cap:
        truncated_lines = report_lines[:line_cap]
        for line in truncated_lines:
            _emit(line)
        _emit("…")
        _emit(
            f"[truncated at {line_cap} lines of {len(report_lines)} — "
            "see runner/output.md for the full report]"
        )
    else:
        for line in report_lines:
            _emit(line)
    _emit(banner)

    sources_list = sources or []
    _emit(f"SOURCES  count={len(sources_list)}")
    _emit(banner)
    for i, src in enumerate(sources_list, 1):
        try:
            _emit(_format_source_entry(i, src))
        except Exception as exc:  # noqa: BLE001 — never let one bad source kill the dump
            logger.warning(
                "EMIT_CONSOLE_REPORT source_format_failed idx=%d err=%s",
                i, exc,
            )
            _emit(f"  [{i:>2}] (unprintable source: {type(src).__name__})")
    _emit(banner)

    if runtime_state is not None:
        try:
            state_values = getattr(runtime_state, "values", None) or {}
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "EMIT_CONSOLE_REPORT skip_runtime_state reason=%s", exc,
            )
            state_values = {}

        present_keys = [k for k in extra_keys if isinstance(state_values, dict) and k in state_values]
        for key in present_keys:
            value = state_values.get(key)
            if value in (None, "", [], {}):
                continue
            _emit(f"STATE  key={key}")
            _emit(banner)
            _emit(_format_state_value(value))
            _emit(banner)

    _emit("")
