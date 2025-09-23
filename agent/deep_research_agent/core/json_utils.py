"""Utility helpers for working with loosely structured JSON payloads."""

from __future__ import annotations

import json
from typing import Any, Optional


try:
    from json_repair import repair_json as json_repair  # type: ignore
except Exception:  # pragma: no cover - graceful fallback when library missing
    json_repair = None  # type: ignore


def _strip_code_fences(payload: str) -> str:
    """Remove Markdown code fences surrounding a JSON payload."""

    text = payload.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = [line for line in text.splitlines() if line.strip()]
        if len(lines) >= 2:
            # Drop first and last line (code fence markers)
            return "\n".join(lines[1:-1]).strip()
    return text


def robust_json_loads(payload: Any) -> Optional[Any]:
    """Parse JSON content with best-effort repair when necessary.

    Args:
        payload: Raw JSON string or already-parsed object.

    Returns:
        Parsed JSON object or ``None`` if parsing fails.
    """

    if payload is None:
        return None

    if isinstance(payload, (dict, list)):
        return payload

    if not isinstance(payload, str):
        payload = str(payload)

    text = _strip_code_fences(payload)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        if json_repair is None:
            return None

        try:
            repaired = json_repair(text)
        except Exception:
            return None

        try:
            return json.loads(repaired)
        except Exception:
            return None
