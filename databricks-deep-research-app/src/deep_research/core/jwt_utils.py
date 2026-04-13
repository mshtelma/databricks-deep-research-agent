"""JWT token utility functions."""

import base64
import json
from datetime import UTC, datetime


def extract_jwt_expiry(token: str) -> datetime | None:
    """Extract expiry from JWT token's ``exp`` claim.

    Decodes the payload without signature verification (we only need the
    expiry timestamp, not authenticity — that's the server's job).

    Returns:
        Expiry datetime (UTC) if the token is a valid JWT with an ``exp``
        claim, otherwise ``None``.
    """
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        payload_b64 = parts[1]
        # JWT uses unpadded base64url; add padding for stdlib decoder
        padding = 4 - len(payload_b64) % 4
        if padding != 4:
            payload_b64 += "=" * padding
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        exp = payload.get("exp")
        if isinstance(exp, (int, float)):
            return datetime.fromtimestamp(exp, tz=UTC)
        return None
    except Exception:  # noqa: BLE001 — best-effort, fall back to estimated lifetime
        return None
