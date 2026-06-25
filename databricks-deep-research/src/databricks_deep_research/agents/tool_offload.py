"""MemEx-first tool I/O: offload large tool outputs to the compute scratchpad.

Pure, fully-typed helpers (no LangChain, no imports from ``react_loop`` to avoid
import cycles). When a non-builtin research tool returns text above a budget, the
full result is stored as a Python *object* in the compute namespace and the
model-visible content is replaced with a compact preview + a handle name the
model can operate on via ``compute`` code.

Spec: ``specs/unified-agent-architecture-plan.md`` §1.1.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from databricks_deep_research.agents.config import ToolOutputBudgetConfig


class ComputeSink(Protocol):
    """Structural type for the compute scratchpad's variable-injection seam.

    Lets this module stay decoupled from the concrete
    ``databricks_deep_research.tools.builtins.compute.PythonComputeTool`` class.
    """

    def inject_variable(self, name: str, value: Any) -> None: ...


def coerce_to_object(content: str) -> Any:
    """Best-effort conversion of tool text into a structured Python object.

    Returns a ``dict``/``list`` when the content parses as JSON of that shape,
    the raw string when it looks tabular (≥2 lines containing ``|``), and the
    original string otherwise. Never raises.
    """
    try:
        parsed = json.loads(content)
    except (ValueError, TypeError):
        parsed = None
    if isinstance(parsed, (dict, list)):
        return parsed

    pipe_lines = sum(1 for line in content.split("\n") if "|" in line)
    if pipe_lines >= 2:
        return content

    return content


def describe_object(obj: Any) -> str:
    """Return a short, bounded descriptor of an offloaded object for the preview."""
    if isinstance(obj, dict):
        return f"dict keys={list(obj)[:20]}"
    if isinstance(obj, list):
        elem = type(obj[0]).__name__ if obj else "empty"
        return f"list len={len(obj)} of {elem}"
    if isinstance(obj, str):
        return f"str chars={len(obj)}"
    return type(obj).__name__


def snap_to_line_boundary(text: str, length: int, *, from_end: bool = False) -> str:
    """Return the head (or tail) of ``text`` cut at a newline near ``length``.

    When ``from_end`` is False, returns the first ~``length`` chars, preferring a
    cut at the last newline within that window. When True, returns the last
    ~``length`` chars, preferring a cut at the first newline within that window.
    Falls back to a hard character cut when no nearby newline exists.
    """
    if length <= 0:
        return ""
    if len(text) <= length:
        return text

    if from_end:
        window = text[-length:]
        newline = window.find("\n")
        if newline != -1 and newline + 1 < len(window):
            return window[newline + 1 :]
        return window

    window = text[:length]
    newline = window.rfind("\n")
    if newline > 0:
        return window[:newline]
    return window


def build_preview(content: str, handle: str, obj: Any, cfg: ToolOutputBudgetConfig) -> str:
    """Compose a compact preview: head + handle marker + tail."""
    head = snap_to_line_boundary(content, cfg.preview_head_chars)
    tail = snap_to_line_boundary(content, cfg.preview_tail_chars, from_end=True)
    marker = (
        f"\n\n[Full result ({len(content)} chars) stored as compute variable "
        f"`{handle}` — {describe_object(obj)}. "
        f"Use `compute` to read/operate on it.]\n\n"
    )
    return head + marker + tail


def should_offload(
    content: str,
    *,
    tool: str,
    mode: str,
    cfg: ToolOutputBudgetConfig,
) -> bool:
    """Return whether ``content`` from ``tool`` should be offloaded under ``mode``."""
    if mode == "off":
        return False
    if tool in cfg.exempt_tools:
        return False
    threshold = cfg.tool_overrides.get(tool, cfg.externalize_min_chars)
    return len(content) > threshold


def maybe_offload(
    result_content: str,
    *,
    tool: str,
    idx: int,
    mode: str,
    compute: ComputeSink | None,
    cfg: ToolOutputBudgetConfig,
) -> tuple[str, str | None]:
    """Offload ``result_content`` to the compute scratchpad when policy allows.

    Returns ``(model_visible_text, handle)``. When no offload happens (no compute
    sink, mode off, exempt tool, or under threshold) the original content is
    returned unchanged with a ``None`` handle.
    """
    if compute is None or not should_offload(result_content, tool=tool, mode=mode, cfg=cfg):
        return result_content, None
    obj = coerce_to_object(result_content)
    handle = f"{tool}_{idx}"
    compute.inject_variable(handle, obj)
    return build_preview(result_content, handle, obj, cfg), handle


# ---------------------------------------------------------------------------
# Budget ladder — truncation rungs (spec §1.2)
#
# Single source of truth for the two non-offload compaction strategies that
# ``react_loop._compact_old_tool_results`` routes through:
#   - ``line_preserving_truncate`` (the ``mask`` heuristic) keeps structural /
#     numeric / unit lines and drops narrative.
#   - ``hard_clip`` is the ``truncate`` default: a verbatim head slice plus a
#     ``...[truncated from N chars]`` suffix.
#
# Both are byte-for-byte ports of the pre-1.2 ``react_loop`` implementations and
# MUST stay output-identical (Codex F7: OfficeQA depends on the ``mask`` output).
# ---------------------------------------------------------------------------


def _is_structural_line(line: str) -> bool:
    """Identify lines providing structural context for data interpretation.

    These lines help the LLM understand *what* the numbers represent even
    though they may not contain data values themselves (e.g., table titles,
    document metadata, section headings).
    """
    lower = line.lower()
    # Document/file metadata
    if lower.startswith(("document:", "file:", "bulletin date:", "source:")):
        return True
    # Table/section titles
    if lower.startswith(("table ", "section ", "part ", "exhibit ")):
        return True
    # Key-value metadata from tool formatting
    if "chunk_type=" in lower or "page_info=" in lower or "file_name=" in lower:
        return True
    # Markdown table alignment row (keeps column structure interpretable)
    return line.startswith("| ---") or line.startswith("|---")


_UNIT_INDICATORS = ("million", "thousand", "billion", "in percent")


def line_preserving_truncate(content: str, max_chars: int = 800) -> str:
    """Preserve key data points when compacting a tool result.

    Keeps lines that contain pipe characters (markdown table rows),
    numeric digits (data values), metadata markers (``[...]`` headers),
    structural context (table titles, document metadata, section
    headings), or unit indicators (e.g. "In millions of dollars").
    Discards narrative text and whitespace to fit within *max_chars*.
    """
    lines = content.split("\n")
    kept: list[str] = []
    char_count = 0

    for line in lines:
        stripped = line.strip()
        if not stripped or len(stripped) < 3:
            continue

        has_pipe = "|" in stripped
        has_number = any(c.isdigit() for c in stripped)
        is_metadata = stripped.startswith("[") and "]" in stripped
        is_structural = _is_structural_line(stripped)
        is_unit_line = any(u in stripped.lower() for u in _UNIT_INDICATORS)

        if has_pipe or has_number or is_metadata or is_structural or is_unit_line:
            # Cap structural-only lines to avoid long footnotes bloating output
            if is_structural and not has_pipe and not has_number and not is_unit_line:
                stripped = stripped[:120]
            kept.append(stripped)
            char_count += len(stripped) + 1
            if char_count >= max_chars:
                kept.append("...[additional data truncated]")
                break

    if not kept:
        return f"[Prior results — {len(content)} chars, no tabular data]"

    return (
        f"[Compacted from {len(content)} chars — key data preserved:]\n"
        + "\n".join(kept)
    )


def hard_clip(content: str, max_chars: int) -> str:
    """Hard-truncate ``content`` to ``max_chars`` with a byte-stable suffix.

    The default rung of the budget ladder: a verbatim head slice plus a
    ``...[truncated from N chars]`` footer. Output is byte-identical to the
    pre-1.2 ``react_loop`` truncate branch.
    """
    return content[:max_chars] + f"\n...[truncated from {len(content)} chars]"
