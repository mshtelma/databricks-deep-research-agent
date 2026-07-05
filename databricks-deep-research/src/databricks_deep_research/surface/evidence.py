"""Evidence list construction for the structured-output wire calls.

The per-slot wires (``surface/generation.py``) generate structured data over
an EVIDENCE list — the research sources, ranked and indexed — rather than over
the narrative report. ``source_refs`` emitted by the model are **1-based
indices into this list as rendered in the wire prompt**; the persisted
envelope carries a self-contained legend (``meta.sources``) so the frontend
never needs the (unordered / cached-mode-empty) per-session source rows to
resolve a chip.

Accepted source shapes (duck-typed):

* run-time pool dicts — ``{url, title|filename, snippet, content,
  relevance_score, source_type|type}``;
* ``SourceInfo``-like objects — the same fields as attributes;
* legacy ORM ``Source`` rows — ``url/title/snippet`` attributes;
* cached ``DocSource`` views — ``url/title`` attributes plus a ``metadata``
  dict holding ``snippet``/``content``/``relevance_score``.

Pure (stdlib + dataclasses): no app/DB imports, so it can run in the
standalone shell-app as well as the main app.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Evidence sizing: enough grounding for one dashboard section without
# blowing the wire prompt budget.
MAX_EVIDENCE_ITEMS = 24
EVIDENCE_BLOCK_BUDGET = 18_000
FULL_CONTENT_TOP_K = 8
SNIPPET_MAX = 400
CONTENT_MAX_TOP = 1200
CONTENT_MAX_REST = 300


@dataclass(frozen=True)
class EvidenceItem:
    """One ranked, indexed evidence source as the wire prompt presents it."""

    ref: str
    url: str
    title: str | None = None
    snippet: str | None = None
    content: str | None = None


@dataclass(frozen=True)
class _RawSource:
    url: str
    title: str | None
    snippet: str | None
    content: str | None
    relevance_score: float


def _normalize_source(raw: Any) -> _RawSource | None:
    """Duck-type one source into a flat shape (see module docstring)."""
    if isinstance(raw, dict):
        url = raw.get("url")
        if not url:
            return None
        title = raw.get("title") or raw.get("filename")
        snippet = raw.get("snippet")
        content = raw.get("content")
        relevance = raw.get("relevance_score")
    else:
        url = getattr(raw, "url", None)
        if not url:
            return None
        title = getattr(raw, "title", None)
        snippet = getattr(raw, "snippet", None)
        content = getattr(raw, "content", None)
        relevance = getattr(raw, "relevance_score", None)
        meta = getattr(raw, "metadata", None)
        if isinstance(meta, dict):
            snippet = snippet or meta.get("snippet")
            content = content or meta.get("content")
            if relevance is None:
                relevance = meta.get("relevance_score")
    return _RawSource(
        url=str(url),
        title=str(title) if title else None,
        snippet=str(snippet) if snippet else None,
        content=content if isinstance(content, str) and content else None,
        relevance_score=(
            float(relevance) if isinstance(relevance, int | float) else 0.0
        ),
    )


def _claim_evidence_url(claim: Any) -> str | None:
    """The claim's supporting-evidence source URL (dict- and object-shaped)."""
    evidence = (
        claim.get("evidence")
        if isinstance(claim, dict)
        else getattr(claim, "evidence", None)
    )
    if evidence is None:
        return None
    if isinstance(evidence, dict):
        url = evidence.get("source_url")
    else:
        url = getattr(evidence, "source_url", None)
    return str(url) if url else None


def build_evidence(
    sources: list[Any],
    claims: list[Any],
    max_items: int = MAX_EVIDENCE_ITEMS,
) -> list[EvidenceItem]:
    """Dedupe, rank, cap, and index the run's sources for the wire prompts.

    Rank: sources cited by a verified claim first, then ``relevance_score``
    descending, then insertion order. Refs are assigned "1".."N" AFTER
    ranking, so the legend and the prompt agree by construction.
    """
    cited_urls = {
        url for claim in claims if (url := _claim_evidence_url(claim)) is not None
    }

    normalized: list[_RawSource] = []
    seen: set[str] = set()
    for raw in sources:
        source = _normalize_source(raw)
        if source is None or source.url in seen:
            continue
        seen.add(source.url)
        normalized.append(source)

    ranked = sorted(
        enumerate(normalized),
        key=lambda pair: (
            0 if pair[1].url in cited_urls else 1,
            -pair[1].relevance_score,
            pair[0],
        ),
    )

    return [
        EvidenceItem(
            ref=str(position + 1),
            url=source.url,
            title=source.title,
            snippet=source.snippet,
            content=source.content,
        )
        for position, (_, source) in enumerate(ranked[:max_items])
    ]


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def render_evidence_block(
    items: list[EvidenceItem],
    budget_chars: int = EVIDENCE_BLOCK_BUDGET,
    full_top_k: int = FULL_CONTENT_TOP_K,
) -> str:
    """Render the ``## Evidence sources`` prompt block.

    Every item ALWAYS gets its ``[ref] title — url`` header line (so every
    ref the guards accept was actually visible to the model); snippets and
    content excerpts are added top-down only while the budget allows, so
    earlier (higher-ranked) items carry the detail.
    """
    lines: list[str] = []
    optional_budget = budget_chars
    # Reserve the mandatory header lines up front.
    headers: list[str] = []
    for item in items:
        title = item.title or item.url
        headers.append(f"[{item.ref}] {title} — {item.url}")
    optional_budget -= sum(len(h) + 1 for h in headers)

    for index, item in enumerate(items):
        lines.append(headers[index])
        if item.snippet and optional_budget > 0:
            snippet = _clip(item.snippet, SNIPPET_MAX)
            if len(snippet) <= optional_budget:
                lines.append(f"    Snippet: {snippet}")
                optional_budget -= len(snippet)
        if item.content and optional_budget > 0:
            cap = CONTENT_MAX_TOP if index < full_top_k else CONTENT_MAX_REST
            content = _clip(" ".join(item.content.split()), cap)
            if len(content) <= optional_budget:
                lines.append(f"    Content: {content}")
                optional_budget -= len(content)
    return "\n".join(lines)


def build_legend(
    items: list[EvidenceItem], used_refs: set[str]
) -> list[dict[str, Any]]:
    """The envelope's ``meta.sources`` legend — only refs actually used."""
    return [
        {"ref": item.ref, "url": item.url, "title": item.title}
        for item in items
        if item.ref in used_refs
    ]


__all__ = [
    "EVIDENCE_BLOCK_BUDGET",
    "FULL_CONTENT_TOP_K",
    "MAX_EVIDENCE_ITEMS",
    "EvidenceItem",
    "build_evidence",
    "build_legend",
    "render_evidence_block",
]
