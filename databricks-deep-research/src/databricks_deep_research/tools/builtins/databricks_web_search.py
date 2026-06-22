"""Databricks built-in web search adapter — satisfies the ``SearchClient`` protocol.

Databricks Model Serving exposes *built-in web search* on pay-per-token foundation
models (https://docs.databricks.com/aws/en/machine-learning/model-serving/web-search):
the model itself searches the live web during generation and returns grounding
citations.  This adapter turns that capability into a plain ``SearchClient`` so it
slots into the existing ``WebSearchTool`` → pool → crawl → citation pipeline exactly
like Brave/Jina, returning ``SearchResult`` rows with ``content=None`` (downstream
``web_crawl`` fetches the real page text).

Two genuinely different transports/parsers, picked by model family (auto-detected
from the endpoint name, overridable):

* **OpenAI** (``databricks-gpt-5*``) — the **Responses API**
  (``client.responses.create(tools=[{"type": "web_search"}])``).  Citations arrive as
  ``message.content[].annotations`` (``url_citation``) with **direct, crawlable URLs**.
* **Gemini** (``databricks-gemini-*``) — the **native generateContent** surface
  (``POST {base_url}/gemini/v1beta/models/{model}:generateContent`` with
  ``tools=[{"google_search": {}}]``).  Citations arrive as
  ``candidates[0].groundingMetadata.groundingChunks[].web.uri`` — these are
  ``vertexaisearch.cloud.google.com/grounding-api-redirect/...`` **redirects** that we
  resolve to their canonical publisher URL via a cheap 302 ``Location`` lookup.

  NOTE: Gemini via the *OpenAI-compatible* ``chat.completions`` path returns **no**
  grounding metadata, so that path is deliberately unsupported for sourcing.

The adapter is intentionally thin — URL registration, domain filtering, and content
truncation are all handled by :class:`~databricks_deep_research.tools.builtins.web_search.WebSearchTool`.

Caveats (surface to operators): built-in search is a *billed model generation* per
query (latency/cost ≫ a search REST API), only on **pay-per-token** endpoints, and is
unavailable on provisioned-throughput / HIPAA-BAA / cross-region-disabled workspaces.

Usage::

    adapter = build_databricks_web_search_adapter(
        client_provider=lambda: async_openai_client,
        model="databricks-gpt-5",
    )
    results = await adapter.search("NVIDIA revenue 2025", count=5)
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

from databricks_deep_research.tools.builtins.web_search import SearchResult

if TYPE_CHECKING:
    import httpx
    from openai import AsyncOpenAI

__all__ = [
    "DatabricksWebSearchAdapter",
    "build_databricks_web_search_adapter",
]

logger = logging.getLogger(__name__)

ModelFamily = Literal["openai", "gemini"]

# Process-wide concurrency cap. Built-in web search is a full (billed) model
# generation, far heavier than a search REST call, so the default is low. Lazily
# initialized so the env var can be set after import (app bootstrap).
_SEMAPHORE: asyncio.Semaphore | None = None
_SEMAPHORE_LIMIT_DEFAULT = 4

# Markdown inline-citation marker emitted by the OpenAI Responses API inside the
# answer text, e.g. ``([postgresql.org](https://www.postgresql.org/...))``. The
# url_citation annotation offsets point AT this marker, not at the supported prose,
# so we strip it when deriving a human-readable snippet.
_CITATION_MARKER_RE = re.compile(r"\(\[[^\]]*\]\([^)]*\)\)")
# Sentence boundary used to backtrack from a citation marker to the claim it supports.
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")
# Tracking query params dropped during URL canonicalisation (dedup hygiene).
_TRACKING_PARAMS = frozenset(
    {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
     "gclid", "fbclid", "mc_cid", "mc_eid", "ref", "ref_src",
     "featured_on"}  # provider-injected attribution params (e.g. OpenAI web_search)
)

_DEFAULT_INSTRUCTION = (
    "Search the web to gather sources about: {query}. "
    "Prefer authoritative, recent pages. Return up to {count} relevant sources."
)

# Max bare domains to push into the OpenAI Responses ``web_search`` filter. The push
# is ALL-OR-NOTHING: if the configured allowlist cannot be reduced to <= this many
# bare registrable domains, push nothing and rely on the instruction hint + the
# post-hoc ``url_allowed`` predicate. OpenAI documents a higher ceiling (~100); we
# stay conservative — over-limit abandons the structured push, never silently
# truncates the allowlist (which would drop domains the user explicitly requested).
_MAX_ALLOWED_DOMAINS = 20

# Cap on patterns enumerated in the natural-language scope hint so a large allowlist
# does not bloat the prompt (billed tokens; instruction-following degrades on long
# lists). The OpenAI structured filter — not the hint — is the real enforcement on
# OpenAI; the post-hoc predicate is the real enforcement everywhere.
_MAX_HINT_DOMAINS = 20


def _pushable_allowed_domains(
    patterns: list[str], *, limit: int = _MAX_ALLOWED_DOMAINS
) -> list[str]:
    """Reduce allowlist *patterns* to bare registrable domains for the OpenAI filter.

    ALL-OR-NOTHING: returns the deduped bare-domain list only when EVERY pattern maps
    cleanly to a bare host and the count is within *limit*; otherwise returns ``[]`` so
    the caller pushes no structured filter and falls back to the instruction hint +
    post-hoc ``url_allowed``. This avoids hard-restricting the engine to a PARTIAL set
    (which would silently drop results from patterns we could not express, e.g.
    ``*.gov``).

    A pattern is pushable when, after stripping one optional leading ``*.``, the
    remainder is a plain host: no other wildcard, no scheme/path/port/credentials, and
    at least two non-empty dot-separated labels (so single-label public suffixes like
    ``gov`` and bare tokens like ``localhost`` are rejected). OpenAI includes
    subdomains automatically, so ``*.example.com`` and ``example.com`` both map to
    ``example.com``.
    """
    out: list[str] = []
    seen: set[str] = set()
    for raw in patterns:
        if not isinstance(raw, str):
            return []
        d = raw.strip().lower().rstrip(".")
        if not d:
            continue
        if d.startswith("*."):
            d = d[2:]
        if any(c in d for c in ("*", "/", ":", "@", " ")):
            return []
        labels = d.split(".")
        if len(labels) < 2 or not all(labels):
            return []
        if d not in seen:
            seen.add(d)
            out.append(d)
    if not out or len(out) > limit:
        return []
    return out


def _domain_scope_clause(patterns: list[str], *, limit: int = _MAX_HINT_DOMAINS) -> str:
    """Soft natural-language domain-scope hint appended to the search instruction.

    Lists the configured *patterns* verbatim (so wildcard allowlists such as ``*.gov``
    still bias the model — including Gemini, which has no structured domain knob).
    Returns ``""`` for an empty list so the default instruction is byte-identical when
    no allowlist is configured. Deliberately SOFT ("prefer"): hard enforcement is the
    OpenAI structured filter or the post-hoc predicate, never this text — a hard
    "return only X" risks the grounded model self-censoring to zero sources.
    """
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in patterns:
        if not isinstance(raw, str):
            continue
        p = raw.strip().lower()
        if p and p not in seen:
            seen.add(p)
            cleaned.append(p)
    if not cleaned:
        return ""
    shown = cleaned[:limit]
    suffix = "" if len(cleaned) <= limit else f" (and {len(cleaned) - limit} more)"
    return (
        f" Strongly prefer sources whose domain matches one of these patterns: "
        f"{', '.join(shown)}{suffix}; if none are available, return the most "
        f"authoritative sources you can find."
    )


# ---------------------------------------------------------------------------
# Shared helpers (abstracted so both backends stay DRY)
# ---------------------------------------------------------------------------


def _get_semaphore() -> asyncio.Semaphore:
    global _SEMAPHORE
    if _SEMAPHORE is None:
        try:
            limit = int(os.environ.get("DBX_WEBSEARCH_MAX_CONCURRENCY", _SEMAPHORE_LIMIT_DEFAULT))
        except ValueError:
            limit = _SEMAPHORE_LIMIT_DEFAULT
        if limit < 1:
            limit = 1
        _SEMAPHORE = asyncio.Semaphore(limit)
        logger.info("DBX_WEBSEARCH_SEMAPHORE_INIT limit=%d", limit)
    return _SEMAPHORE


def _reset_semaphore_for_tests() -> None:
    """Test hook — drop the module-level semaphore so it re-initializes."""
    global _SEMAPHORE
    _SEMAPHORE = None


def _norm_title(title: str | None) -> str:
    """Strip and collapse whitespace (provider titles carry stray newlines)."""
    if not title:
        return ""
    return " ".join(title.split())


def _canonical_url(url: str) -> str:
    """Canonicalise for dedup: lowercase scheme+host, drop tracking params + fragment.

    Path/query case is preserved (paths can be case-sensitive). Returns the input
    unchanged if it cannot be parsed.
    """
    from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return url.strip()
    if not parts.scheme or not parts.netloc:
        return url.strip()
    query = urlencode(
        [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
         if k.lower() not in _TRACKING_PARAMS]
    )
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, query, ""))


def _claim_before(text: str, idx: int) -> str:
    """Return the supported claim: the sentence(s) ending just before *idx*.

    The OpenAI Responses citation offset points at the inline marker, which is
    appended to the END of the sentence it supports — so the useful snippet is the
    prose preceding the marker. Strips any embedded citation markers and clamps
    *idx* into range.
    """
    if not text:
        return ""
    idx = max(0, min(idx, len(text)))
    head = _CITATION_MARKER_RE.sub("", text[:idx]).strip()
    if not head:
        return ""
    # Keep the trailing 1–2 sentences for a focused snippet.
    sentences = _SENTENCE_BOUNDARY_RE.split(head)
    snippet = " ".join(sentences[-2:]).strip()
    return snippet[:500]


def _dedup_keep_order(results: list[SearchResult]) -> list[SearchResult]:
    """Drop duplicate URLs (by canonical form), preserving first-seen order.

    Returns results carrying the **canonical** URL (tracking params stripped) so the
    downstream pool — which dedups by exact URL string — collapses the same page
    cleanly. Upgrades an empty snippet/title from a later duplicate.
    """
    seen: dict[str, SearchResult] = {}
    order: list[str] = []
    for r in results:
        if not r.url:
            continue
        key = _canonical_url(r.url)
        if key not in seen:
            seen[key] = SearchResult(
                url=key, title=r.title, snippet=r.snippet,
                relevance_score=r.relevance_score, content=r.content,
            )
            order.append(key)
        else:
            cur = seen[key]
            new_snippet = cur.snippet or r.snippet
            new_title = cur.title or r.title
            if new_snippet != cur.snippet or new_title != cur.title:
                seen[key] = SearchResult(
                    url=key, title=new_title, snippet=new_snippet,
                    relevance_score=cur.relevance_score, content=cur.content,
                )
    return [seen[k] for k in order]


def _detect_family(model: str) -> ModelFamily:
    """Infer the model family from the endpoint name."""
    name = model.lower()
    if "gemini" in name:
        return "gemini"
    if "gpt" in name or "openai" in name or "o1" in name or "o3" in name:
        return "openai"
    raise ValueError(
        f"Cannot infer model_family from endpoint {model!r}; set model_family "
        f"explicitly ('openai' or 'gemini')."
    )


def _is_retryable(exc: BaseException) -> bool:
    """True for transient errors worth a bounded retry (429 / 5xx / transport)."""
    from openai import APITimeoutError, RateLimitError

    if isinstance(exc, (RateLimitError, APITimeoutError)):
        return True
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status is None:
        status = getattr(exc, "status_code", None)
    if isinstance(status, int) and (status == 429 or status >= 500):
        return True
    # httpx transport errors (connect/read timeouts etc.) — match by module to
    # avoid importing httpx at module import time.
    return type(exc).__module__.startswith("httpx") and "Status" not in type(exc).__name__


# ---------------------------------------------------------------------------
# Per-family backends
# ---------------------------------------------------------------------------


class _OpenAIResponsesBackend:
    """Built-in web search via the OpenAI Responses API (gpt-5 family)."""

    def __init__(self, model: str, *, allowed_domains: list[str] | None = None) -> None:
        self._model = model
        self._allowed_domains = allowed_domains or []
        # Monotonic latch (False -> True only). Set when the serving proxy rejects the
        # ``filters`` field so the rest of THIS run skips it. Safe under concurrent
        # asyncio searches because it only ever transitions one way; the adapter (and
        # thus this backend) is constructed once per run — see DatabricksWebSearchAdapter.
        self._filters_disabled = False

    async def _create(self, client: AsyncOpenAI, instruction: str) -> Any:
        """Call the Responses API, pushing ``filters.allowed_domains`` when available.

        Graceful degradation: if the serving proxy rejects the ``filters`` field with a
        400 (``BadRequestError``), latch it off for the rest of the run and retry once
        WITHOUT filters — so an unsupporting workspace degrades to the instruction hint
        + post-hoc filter instead of failing every search. Detection is type-based and
        gated on filters actually having been sent, so a genuine (filters-unrelated) bad
        request is never masked. Transient errors (429/5xx/transport) are NOT caught
        here; they propagate to the adapter's retry ladder.
        """
        from openai import BadRequestError

        use_filters = bool(self._allowed_domains) and not self._filters_disabled
        # ``list[Any]``: the SDK types ``tools`` as a union of per-tool TypedDicts, and
        # we add ``filters`` conditionally — a plain-dict variable loses the literal
        # context mypy needs to match the TypedDict. The dict shape is exercised by tests
        # and validated by the API (Phase-0 confirmed the proxy accepts ``filters``).
        tool: dict[str, Any] = {"type": "web_search"}
        if use_filters:
            tool["filters"] = {"allowed_domains": self._allowed_domains}
        tools: list[Any] = [tool]
        try:
            return await client.responses.create(
                model=self._model, input=instruction, tools=tools,
                store=False,  # don't persist the (possibly sensitive) query server-side
            )
        except BadRequestError:
            if not use_filters:
                raise  # genuine bad request unrelated to filters — fail soft upstream
            logger.warning(
                "DBX_WEBSEARCH_FILTERS_UNSUPPORTED_FALLBACK model=%s n=%d — "
                "retrying without filters", self._model, len(self._allowed_domains),
            )
            self._filters_disabled = True
            return await client.responses.create(
                model=self._model, input=instruction,
                tools=[{"type": "web_search"}], store=False,
            )

    async def search(self, client: AsyncOpenAI, instruction: str) -> list[SearchResult]:
        resp = await self._create(client, instruction)
        if getattr(resp, "error", None):
            logger.warning("DBX_WEBSEARCH_OPENAI_ERROR model=%s err=%s",
                           self._model, str(resp.error)[:200])
            return []
        status = getattr(resp, "status", None)
        if status not in (None, "completed"):
            # Incomplete responses can still carry usable citations — parse them
            # but record the degraded status.
            logger.warning("DBX_WEBSEARCH_OPENAI_STATUS model=%s status=%s",
                           self._model, status)

        # Cited URLs (rich: real title + a derived snippet) rank first; the bare
        # ``action.sources`` URLs the model also consulted come after as
        # supplementary coverage. Output-item order is irrelevant to this ranking.
        cited: list[SearchResult] = []
        extra: list[SearchResult] = []
        for item in getattr(resp, "output", None) or []:
            itype = getattr(item, "type", None)
            if itype == "message":
                for part in getattr(item, "content", None) or []:
                    if getattr(part, "type", None) != "output_text":
                        continue
                    text = getattr(part, "text", "") or ""
                    for ann in getattr(part, "annotations", None) or []:
                        if getattr(ann, "type", None) != "url_citation":
                            continue
                        url = getattr(ann, "url", "") or ""
                        if not url:
                            continue
                        start = getattr(ann, "start_index", 0) or 0
                        cited.append(SearchResult(
                            url=url,
                            title=_norm_title(getattr(ann, "title", "")),
                            snippet=_claim_before(text, start),
                        ))
            elif itype == "web_search_call":
                # action.sources is the fuller URL list the model consulted
                # (present for `search` actions). Title/snippet often absent.
                action = getattr(item, "action", None)
                for src in getattr(action, "sources", None) or []:
                    url = getattr(src, "url", "") or (src.get("url") if isinstance(src, dict) else "")
                    if url:
                        extra.append(SearchResult(
                            url=url, title=_norm_title(getattr(src, "title", "")), snippet=""
                        ))
        self._usage_log(resp)
        return cited + extra

    def _usage_log(self, resp: Any) -> None:
        usage = getattr(resp, "usage", None)
        if usage is not None:
            logger.info(
                "DBX_WEBSEARCH_USAGE family=openai model=%s input_tokens=%s output_tokens=%s",
                self._model,
                getattr(usage, "input_tokens", "?"),
                getattr(usage, "output_tokens", "?"),
            )


class _GeminiGenerateContentBackend:
    """Built-in web search via the native Gemini generateContent surface."""

    def __init__(self, model: str, *, resolve_redirects: bool, timeout_seconds: float) -> None:
        self._model = model
        self._resolve_redirects = resolve_redirects
        self._timeout = timeout_seconds

    async def search(self, client: AsyncOpenAI, instruction: str) -> list[SearchResult]:
        import httpx

        base_url = str(client.base_url).rstrip("/")
        token = client.api_key
        url = f"{base_url}/gemini/v1beta/models/{self._model}:generateContent"
        body: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": instruction}]}],
            "tools": [{"google_search": {}}],
        }
        async with httpx.AsyncClient(timeout=self._timeout) as h:
            resp = await h.post(url, headers={"Authorization": f"Bearer {token}"}, json=body)
        if resp.status_code == 429 or resp.status_code >= 500:
            resp.raise_for_status()  # retryable — surfaced to the adapter's retry
        if resp.status_code != 200:
            logger.warning("DBX_WEBSEARCH_GEMINI_HTTP model=%s status=%d body=%s",
                           self._model, resp.status_code, resp.text[:200])
            return []

        data = resp.json()
        candidates = data.get("candidates") or []
        if not candidates:
            return []
        gm = candidates[0].get("groundingMetadata") or {}
        chunks = gm.get("groundingChunks") or []
        supports = gm.get("groundingSupports") or []

        # Map chunk index -> concatenated supported sentence(s) for snippets.
        snippet_by_chunk: dict[int, list[str]] = {}
        for sup in supports:
            seg_text = ((sup.get("segment") or {}).get("text") or "").strip()
            if not seg_text:
                continue
            for ci in sup.get("groundingChunkIndices") or []:
                snippet_by_chunk.setdefault(ci, []).append(seg_text)

        raw: list[tuple[int, str, str]] = []  # (chunk_index, uri, title)
        for i, ch in enumerate(chunks):
            web = ch.get("web") or {}
            uri = web.get("uri") or ""
            if uri:
                raw.append((i, uri, _norm_title(web.get("title"))))

        if self._resolve_redirects and raw:
            uris = await self._resolve_all([u for _, u, _ in raw])
        else:
            uris = [u for _, u, _ in raw]

        results: list[SearchResult] = []
        for (ci, _orig, title), final_uri in zip(raw, uris, strict=True):
            snippet = " ".join(snippet_by_chunk.get(ci, []))[:500]
            results.append(SearchResult(url=final_uri, title=title, snippet=snippet))
        return results

    async def _resolve_all(self, uris: list[str]) -> list[str]:
        """Resolve grounding-redirect URIs to canonical publisher URLs.

        A single ``follow_redirects=False`` GET returns a 30x + ``Location``; we read
        that header without downloading the page. Falls back to the original URI on
        any error/timeout so a resolution hiccup never drops a source.
        """
        import httpx

        async def one(h: httpx.AsyncClient, uri: str) -> str:
            try:
                r = await h.get(uri, headers={"User-Agent": "Mozilla/5.0"})
                if r.is_redirect:
                    loc = r.headers.get("location")
                    if loc:
                        return str(loc)
                return str(r.url)
            except Exception:  # noqa: BLE001 — best-effort; keep original on failure
                return uri

        async with httpx.AsyncClient(timeout=min(self._timeout, 8.0), follow_redirects=False) as h:
            return list(await asyncio.gather(*(one(h, u) for u in uris)))


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class DatabricksWebSearchAdapter:
    """Databricks built-in web search client implementing the ``SearchClient`` protocol.

    Parameters:
        client_provider: Returns a fresh ``AsyncOpenAI`` pointed at
            ``{host}/serving-endpoints`` (used directly for the OpenAI Responses path,
            and as the source of host+token for the Gemini httpx path). A *provider*
            rather than a snapshot so each call uses a refreshed token on long runs.
        model: Serving endpoint name (e.g. ``databricks-gpt-5``).
        model_family: ``"openai"`` / ``"gemini"``; auto-detected from *model* if ``None``.
        max_results: Hard cap on returned results.
        timeout_seconds: Per-call wall-clock budget (the call returns ``[]`` on timeout).
        resolve_redirects: Resolve Gemini grounding-redirect URLs to canonical (no-op for OpenAI).
        url_allowed: Optional predicate; results whose URL fails it are dropped. This
            stays the AUTHORITATIVE filter — the push-down below is an optimization in
            front of it.
        restrict_to_domains: Optional RAW allowlist patterns. Drives the soft instruction
            hint (both families) and, when ``push_allowed_domains`` is on and the
            patterns reduce cleanly to bare domains, the OpenAI ``filters.allowed_domains``
            push-down (all-or-nothing). Wildcard/non-pushable patterns still produce the
            hint and rely on ``url_allowed``.
        push_allowed_domains: Gate the OpenAI structured ``filters`` push (default on).
            No effect on Gemini (no API knob) or on the instruction hint.
    """

    def __init__(
        self,
        client_provider: Callable[[], AsyncOpenAI],
        *,
        model: str,
        model_family: ModelFamily | None = None,
        max_results: int = 10,
        timeout_seconds: float = 30.0,
        resolve_redirects: bool = True,
        url_allowed: Callable[[str], bool] | None = None,
        restrict_to_domains: list[str] | None = None,
        push_allowed_domains: bool = True,
        max_retries: int = 2,
    ) -> None:
        self._client_provider = client_provider
        self._model = model
        self._family: ModelFamily = model_family or _detect_family(model)
        self._max_results = max(1, max_results)
        self._timeout = timeout_seconds
        self._url_allowed = url_allowed
        self._max_retries = max(0, max_retries)
        # Allowlist push-down (additive; the post-hoc ``url_allowed`` stays the
        # authoritative filter). ``restrict_to_domains`` are RAW allowlist patterns: the
        # soft instruction hint lists them verbatim, while the OpenAI structured filter
        # uses only the all-or-nothing pushable bare-domain subset. Derived ONCE here —
        # the adapter (and its backend) is constructed per run, which is what makes the
        # backend's filters-unsupported latch safe.
        self._restrict_patterns = [
            p for p in (restrict_to_domains or []) if isinstance(p, str) and p.strip()
        ]
        self._allowed_domains = (
            _pushable_allowed_domains(self._restrict_patterns)
            if push_allowed_domains
            else []
        )
        self._scope_clause = _domain_scope_clause(self._restrict_patterns)
        if self._restrict_patterns:
            if self._allowed_domains:
                logger.info(
                    "DBX_WEBSEARCH_FILTERS_PUSHED family=%s model=%s n=%d",
                    self._family, model, len(self._allowed_domains),
                )
            else:
                logger.info(
                    "DBX_WEBSEARCH_FILTERS_ABORTED family=%s model=%s reason=%s n_patterns=%d",
                    self._family, model,
                    "disabled" if not push_allowed_domains else "non_pushable",
                    len(self._restrict_patterns),
                )
        self._backend: _OpenAIResponsesBackend | _GeminiGenerateContentBackend
        if self._family == "openai":
            self._backend = _OpenAIResponsesBackend(
                model, allowed_domains=self._allowed_domains
            )
        else:
            self._backend = _GeminiGenerateContentBackend(
                model, resolve_redirects=resolve_redirects, timeout_seconds=timeout_seconds
            )

    async def search(
        self,
        query: str,
        *,
        count: int = 10,
        freshness: str | None = None,  # noqa: ARG002 — protocol conformance (ignored)
    ) -> list[SearchResult]:
        """Run built-in web search and return ``SearchResult`` rows (``content=None``).

        Fail-soft: any error (unsupported endpoint/workspace, timeout, transport,
        empty result) logs a single warning and returns ``[]`` — a search miss must
        never crash the research run.
        """
        n = min(max(count, 1), self._max_results)
        instruction = _DEFAULT_INSTRUCTION.format(query=query, count=n) + self._scope_clause
        loop = asyncio.get_event_loop()
        started = loop.time()
        try:
            raw = await self._run_with_retries(instruction)
        except TimeoutError:
            logger.warning("DBX_WEBSEARCH_TIMEOUT family=%s model=%s timeout=%.1fs query=%r",
                           self._family, self._model, self._timeout, query[:120])
            return []
        except Exception as exc:  # noqa: BLE001 — fail-soft for any provider error
            logger.warning("DBX_WEBSEARCH_ERROR family=%s model=%s err=%s:%s query=%r",
                           self._family, self._model, type(exc).__name__, str(exc)[:200],
                           query[:120])
            return []

        if self._url_allowed is not None:
            raw = [r for r in raw if r.url and self._url_allowed(r.url)]
        deduped = _dedup_keep_order(raw)[:n]
        if not deduped:
            logger.warning(
                "DBX_WEBSEARCH_EMPTY family=%s model=%s restrict_active=%s "
                "filters_pushed=%s query=%r",
                self._family, self._model, bool(self._restrict_patterns),
                bool(self._allowed_domains), query[:120],
            )
        logger.info(
            "DBX_WEBSEARCH_DONE family=%s model=%s n_raw=%d n_out=%d latency_ms=%d",
            self._family, self._model, len(raw), len(deduped),
            int((loop.time() - started) * 1000),
        )
        return deduped

    async def _run_with_retries(self, instruction: str) -> list[SearchResult]:
        attempt = 0
        while True:
            try:
                async with _get_semaphore():
                    client = self._client_provider()
                    return await asyncio.wait_for(
                        self._backend.search(client, instruction),
                        timeout=self._timeout,
                    )
            except TimeoutError:
                raise
            except Exception as exc:  # noqa: BLE001
                if attempt < self._max_retries and _is_retryable(exc):
                    delay = 0.5 * (2 ** attempt)
                    logger.warning(
                        "DBX_WEBSEARCH_RETRY family=%s model=%s attempt=%d/%d err=%s sleep=%.2fs",
                        self._family, self._model, attempt + 1, self._max_retries,
                        type(exc).__name__, delay,
                    )
                    await asyncio.sleep(delay)
                    attempt += 1
                    continue
                raise


def build_databricks_web_search_adapter(
    *,
    client_provider: Callable[[], AsyncOpenAI],
    model: str,
    model_family: ModelFamily | None = None,
    max_results: int = 10,
    timeout_seconds: float = 30.0,
    resolve_redirects: bool = True,
    url_allowed: Callable[[str], bool] | None = None,
    restrict_to_domains: list[str] | None = None,
    push_allowed_domains: bool = True,
) -> DatabricksWebSearchAdapter:
    """Single construction point for the adapter (shared by the app + factory paths)."""
    return DatabricksWebSearchAdapter(
        client_provider,
        model=model,
        model_family=model_family,
        max_results=max_results,
        timeout_seconds=timeout_seconds,
        resolve_redirects=resolve_redirects,
        url_allowed=url_allowed,
        restrict_to_domains=restrict_to_domains,
        push_allowed_domains=push_allowed_domains,
    )
