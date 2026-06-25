"""JSON structured-passage parser.

Wraps ``json.loads``; converts parse errors into ``ToolErrorException``
with ``ErrorCode.INVALID_BINDING`` so the upstream tool surface treats
the failure like any other binding-level data error.
"""

from __future__ import annotations

import json as _json
from typing import Any

from ..error_codes import ErrorCode, ToolError, ToolErrorException


def parse_json(content: str) -> dict[str, Any]:
    try:
        parsed = _json.loads(content)
    except _json.JSONDecodeError as e:
        raise ToolErrorException(
            ToolError(
                error_code=ErrorCode.INVALID_BINDING,
                message=f"structured passage JSON parser failed: {e.msg}",
                details={"line": e.lineno, "col": e.colno, "pos": e.pos},
            )
        ) from e
    return {"raw": content, "parsed": parsed, "parser": "json"}
