"""Markdown structured-passage parser.

Splits the markdown into a sectioned ``{"headings": [...], "body": "..."}``
dict. Headings are lines starting with ``#``; body is the remaining text.
Used for ``equation`` / ``figure_caption`` passages — passthrough is fine.
"""

from __future__ import annotations

from typing import Any


def parse_markdown(content: str) -> dict[str, Any]:
    headings: list[str] = []
    body_lines: list[str] = []
    for line in content.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            headings.append(stripped.lstrip("#").strip())
        else:
            body_lines.append(line)
    parsed: dict[str, Any] = {
        "headings": headings,
        "body": "\n".join(body_lines).strip(),
    }
    return {"raw": content, "parsed": parsed, "parser": "markdown"}
