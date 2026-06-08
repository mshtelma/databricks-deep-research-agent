"""Lightweight Google-style docstring parser for ``@tool``-decorated callables.

Extracts the summary line, the ``Args:`` block (parameter → description), and
the ``Returns:`` description. Falls back to summary-only on unrecognized
formats (NumPy, Sphinx, etc.) — no warnings raised.

The output is consumed by :func:`databricks_deep_research.tools.api.tool` to
populate the tool's overall description and per-parameter ``description``
entries in the JSON Schema.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_HEADER_PATTERN = re.compile(
    r"^(Args|Arguments|Parameters|Returns|Return|Raises|Yields|Example|Examples|Note|Notes):\s*$",
    re.IGNORECASE,
)
_ARG_LINE_PATTERN = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:\([^)]*\))?\s*:\s*(.*)$"
)


@dataclass
class ParsedDocstring:
    summary: str = ""
    args: dict[str, str] = field(default_factory=dict)
    returns: str = ""


def parse_google_docstring(doc: str | None) -> ParsedDocstring:
    """Parse a Google-style docstring.

    Args:
        doc: Raw docstring (or ``None``).

    Returns:
        A :class:`ParsedDocstring` with the summary, ``Args`` mapping, and
        ``Returns`` description filled in. Unparseable docstrings yield a
        result whose ``summary`` is the first non-empty line of the input.
    """
    if not doc:
        return ParsedDocstring()

    lines = doc.expandtabs().splitlines()
    summary_lines: list[str] = []
    section: str | None = None
    args: dict[str, str] = {}
    returns_lines: list[str] = []
    current_arg: str | None = None
    current_arg_lines: list[str] = []

    def _flush_current_arg() -> None:
        nonlocal current_arg, current_arg_lines
        if current_arg is not None:
            args[current_arg] = " ".join(p.strip() for p in current_arg_lines).strip()
        current_arg = None
        current_arg_lines = []

    for raw in lines:
        stripped = raw.strip()
        header_match = _HEADER_PATTERN.match(stripped)
        if header_match:
            _flush_current_arg()
            heading = header_match.group(1).lower()
            if heading in {"args", "arguments", "parameters"}:
                section = "args"
            elif heading in {"returns", "return"}:
                section = "returns"
            else:
                section = heading
            continue

        if section is None:
            if stripped or summary_lines:
                summary_lines.append(stripped)
            continue

        if section == "args":
            if not stripped:
                _flush_current_arg()
                continue
            arg_match = _ARG_LINE_PATTERN.match(raw)
            if arg_match:
                _flush_current_arg()
                current_arg = arg_match.group(1)
                first_desc = arg_match.group(2).strip()
                current_arg_lines = [first_desc] if first_desc else []
            elif current_arg is not None:
                current_arg_lines.append(stripped)
        elif section == "returns":
            if stripped:
                returns_lines.append(stripped)

    _flush_current_arg()

    summary = " ".join(s for s in summary_lines if s).strip()
    returns = " ".join(returns_lines).strip()
    return ParsedDocstring(summary=summary, args=args, returns=returns)


__all__ = ["ParsedDocstring", "parse_google_docstring"]
