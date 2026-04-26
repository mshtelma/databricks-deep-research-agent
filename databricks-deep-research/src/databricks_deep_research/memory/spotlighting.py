"""Spotlighting wrappers for untrusted attached-context content.

Implements Microsoft Spotlighting modes (delimiting, datamarking) as a
defense-in-depth against indirect prompt injection through uploaded file
content. ``datamark`` is the default: a randomized marker token is
interleaved between chunks of attached content so that injected
imperatives inside the text land in a different representation-space
cluster than legitimate agent instructions.

Reference: Microsoft research on Spotlighting (datamarking cited as
0.00% ASR in their controlled benchmark). We treat this as strong
defense-in-depth, not a guarantee; additional defenses (typed Pydantic
summarizer output, raw file text never injected directly, defensive
system-prompt wording) apply in concert.
"""

from __future__ import annotations

import secrets
from typing import Literal

SpotlightingMode = Literal["delimit", "datamark"]
DEFAULT_SPOTLIGHTING_MODE: SpotlightingMode = "datamark"

_NONCE_BYTES = 4  # 8 hex chars


def _new_nonce() -> str:
    """Return a short random hex token used as the datamark / delimiter."""
    return secrets.token_hex(_NONCE_BYTES)


def wrap_attached_context(
    content: str,
    mode: SpotlightingMode = DEFAULT_SPOTLIGHTING_MODE,
) -> str:
    """Wrap untrusted content with spotlighting sentinels + optional datamarks.

    Returns a markdown-safe block that agents are told to treat as DATA,
    not instructions. Empty input returns empty output (callers should
    short-circuit before injection).
    """
    if not content:
        return ""

    nonce = _new_nonce()

    if mode == "delimit":
        return (
            f'<attached_context marker="{nonce}">\n'
            f"{content}\n"
            f'</attached_context>'
        )

    # datamark mode: interleave the marker at paragraph boundaries
    marker = f" ^{nonce}^ "
    # Split on blank-line paragraphs to avoid breaking inline markdown
    paragraphs = content.split("\n\n")
    datamarked = marker.join(paragraphs)
    return (
        f'<attached_context marker="{nonce}" mode="datamark">\n'
        f"{datamarked}\n"
        f'</attached_context>'
    )


def strip_datamark(content: str) -> str:
    """Remove spotlighting wrappers and datamark tokens from content.

    Used when the content is about to leave the prompt surface (e.g.,
    passed to a deterministic tool). Not used during normal prompt
    rendering; agents see the full wrapped block.
    """
    if not content or "<attached_context" not in content:
        return content

    # Strip the outer sentinel wrapper
    start = content.find(">")
    end = content.rfind("</attached_context>")
    if start == -1 or end == -1:
        return content
    inner = content[start + 1 : end].strip()

    # Strip datamark tokens of the form " ^<hex>^ "
    # Match lazily; hex chars are [0-9a-f]{8}
    import re as _re

    return _re.sub(r" \^[0-9a-f]{8}\^ ", "\n\n", inner).strip()
