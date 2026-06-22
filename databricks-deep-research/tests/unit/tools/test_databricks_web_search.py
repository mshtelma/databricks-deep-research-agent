"""Tests for the Databricks built-in web search adapter.

Fixtures mirror the real response shapes captured live against Databricks model
serving (gpt-5 Responses API + native Gemini generateContent), so the parsers are
exercised against the actual schemas they must handle.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from databricks_deep_research.tools.builtins import databricks_web_search as dws
from databricks_deep_research.tools.builtins.databricks_web_search import (
    DatabricksWebSearchAdapter,
    _canonical_url,
    _claim_before,
    _dedup_keep_order,
    _detect_family,
    _GeminiGenerateContentBackend,
    _norm_title,
    _OpenAIResponsesBackend,
    build_databricks_web_search_adapter,
)
from databricks_deep_research.tools.builtins.web_search import SearchResult


@pytest.fixture(autouse=True)
def _reset_semaphore() -> None:
    dws._reset_semaphore_for_tests()


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_norm_title_collapses_whitespace(self) -> None:
        assert _norm_title("\n  Spark 4.1.2   released \n") == "Spark 4.1.2 released"
        assert _norm_title(None) == ""

    def test_detect_family(self) -> None:
        assert _detect_family("databricks-gpt-5") == "openai"
        assert _detect_family("databricks-gpt-5-mini") == "openai"
        assert _detect_family("databricks-gemini-3-1-flash-lite") == "gemini"
        with pytest.raises(ValueError):
            _detect_family("some-unknown-endpoint")

    def test_canonical_url_strips_tracking_and_fragment(self) -> None:
        # utm_* and featured_on dropped; host lowercased; fragment + trailing slash gone.
        assert (
            _canonical_url("HTTPS://Example.com/Docs/?utm_source=openai&q=1#frag")
            == "https://example.com/Docs?q=1"
        )
        assert (
            _canonical_url("https://www.postgresql.org/?featured_on=python")
            == "https://www.postgresql.org/"
        )

    def test_claim_before_returns_prose_not_marker(self) -> None:
        # The OpenAI annotation offset points AT the citation marker; the snippet
        # must be the supported prose preceding it, with the marker stripped.
        text = (
            "PostgreSQL 18 was released in September 2025. "
            "It adds async I/O ([postgresql.org](https://www.postgresql.org/x)). Next."
        )
        idx = text.index("([postgresql.org]")
        snip = _claim_before(text, idx)
        assert "It adds async I/O" in snip
        assert "postgresql.org](" not in snip  # marker stripped
        assert "Next." not in snip  # only prose before the marker

    def test_claim_before_clamps_out_of_range(self) -> None:
        assert _claim_before("short", 9999).startswith("short")
        assert _claim_before("", 5) == ""

    def test_dedup_emits_canonical_and_upgrades_empty_fields(self) -> None:
        rows = [
            SearchResult(url="https://x.com/a?utm_source=openai", title="", snippet=""),
            SearchResult(url="https://x.com/a", title="A", snippet="real snippet"),
            SearchResult(url="https://y.com/b", title="B", snippet="b"),
        ]
        out = _dedup_keep_order(rows)
        assert [r.url for r in out] == ["https://x.com/a", "https://y.com/b"]
        # First occurrence kept its slot but inherited title/snippet from the dup.
        assert out[0].title == "A"
        assert out[0].snippet == "real snippet"

    def test_dedup_skips_empty_urls(self) -> None:
        assert _dedup_keep_order([SearchResult(url="", title="t", snippet="s")]) == []


# ---------------------------------------------------------------------------
# OpenAI Responses backend
# ---------------------------------------------------------------------------


def _ann(url: str, title: str, start: int, end: int) -> SimpleNamespace:
    return SimpleNamespace(
        type="url_citation", url=url, title=title, start_index=start, end_index=end
    )


def _openai_response(*, status: str = "completed", error: Any = None) -> SimpleNamespace:
    text = (
        "PostgreSQL 18 is the latest major version, released September 2025"
        "([postgresql.org](https://www.postgresql.org/about/news/x)). "
        "It introduces asynchronous I/O"
        "([postgresql.org](https://www.postgresql.org/docs/18/x))."
    )
    msg = SimpleNamespace(
        type="message",
        content=[
            SimpleNamespace(
                type="output_text",
                text=text,
                annotations=[
                    _ann("https://www.postgresql.org/about/news/x?utm_source=openai",
                         "\n PostgreSQL 18 Released \n", text.index("([postgresql.org]"), 10),
                    _ann("https://www.postgresql.org/docs/18/x",
                         "Docs 18", text.rindex("([postgresql.org]"), 10),
                ],
            )
        ],
    )
    wsc = SimpleNamespace(
        type="web_search_call",
        action=SimpleNamespace(
            type="search",
            sources=[SimpleNamespace(url="https://extra.example.com/page", title="Extra")],
        ),
    )
    return SimpleNamespace(
        status=status,
        error=error,
        output=[SimpleNamespace(type="reasoning"), wsc, msg],
        usage=SimpleNamespace(input_tokens=10, output_tokens=20),
    )


def _fake_openai_client(response: Any) -> Any:
    async def _create(**_kwargs: Any) -> Any:
        return response

    return SimpleNamespace(responses=SimpleNamespace(create=_create))


class TestOpenAIBackend:
    async def test_parses_annotations_with_prose_snippets(self) -> None:
        backend = _OpenAIResponsesBackend("databricks-gpt-5")
        out = await backend.search(_fake_openai_client(_openai_response()), "q")
        urls = [r.url for r in out]
        assert "https://www.postgresql.org/about/news/x?utm_source=openai" in urls
        assert "https://www.postgresql.org/docs/18/x" in urls
        assert "https://extra.example.com/page" in urls  # from action.sources
        first = out[0]
        assert first.title == "PostgreSQL 18 Released"  # whitespace stripped
        assert "latest major version" in first.snippet
        assert "postgresql.org](" not in first.snippet  # marker not leaked

    async def test_error_returns_empty(self) -> None:
        backend = _OpenAIResponsesBackend("databricks-gpt-5")
        resp = _openai_response(error=SimpleNamespace(code="bad"))
        assert await backend.search(_fake_openai_client(resp), "q") == []

    async def test_incomplete_status_still_parses(self) -> None:
        backend = _OpenAIResponsesBackend("databricks-gpt-5")
        resp = _openai_response(status="incomplete")
        assert len(await backend.search(_fake_openai_client(resp), "q")) >= 1

    async def test_malformed_annotation_skipped(self) -> None:
        backend = _OpenAIResponsesBackend("databricks-gpt-5")
        msg = SimpleNamespace(
            type="message",
            content=[SimpleNamespace(
                type="output_text", text="x",
                annotations=[SimpleNamespace(type="url_citation", url="", title="", start_index=0, end_index=0)],
            )],
        )
        resp = SimpleNamespace(status="completed", error=None, output=[msg], usage=None)
        assert await backend.search(_fake_openai_client(resp), "q") == []


# ---------------------------------------------------------------------------
# Gemini generateContent backend (httpx mocked)
# ---------------------------------------------------------------------------

_REDIRECT_A = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AAA"
_REDIRECT_B = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/BBB"
_CANONICAL = {
    _REDIRECT_A: "https://www.postgresql.org/support/versioning/",
    _REDIRECT_B: "https://bucardo.org/postgres_all_versions.html",
}

_GEMINI_BODY = {
    "candidates": [
        {
            "groundingMetadata": {
                "webSearchQueries": ["latest postgres 2026"],
                "groundingChunks": [
                    {"web": {"uri": _REDIRECT_A, "title": "postgresql.org"}},
                    {"web": {"uri": _REDIRECT_B, "title": "bucardo.org"}},
                ],
                "groundingSupports": [
                    {"segment": {"text": "PostgreSQL 18 is the latest major version."},
                     "groundingChunkIndices": [0]},
                    {"segment": {"text": "It was released in September 2025."},
                     "groundingChunkIndices": [0, 1]},
                ],
            }
        }
    ]
}


class _FakeResponse:
    def __init__(self, *, status_code: int = 200, json_body: Any = None,
                 is_redirect: bool = False, location: str | None = None,
                 url: str = "") -> None:
        self.status_code = status_code
        self._json = json_body
        self.is_redirect = is_redirect
        self.headers = {"location": location} if location else {}
        self.url = url
        self.text = "" if json_body is None else "body"

    def json(self) -> Any:
        return self._json

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"status {self.status_code}")


class _FakeHttpxClient:
    """Stands in for httpx.AsyncClient for both POST (generateContent) and the
    redirect-resolution GETs."""

    post_status = 200

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    async def __aenter__(self) -> _FakeHttpxClient:
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        return None

    async def post(self, url: str, **_kwargs: Any) -> _FakeResponse:  # noqa: ARG002
        return _FakeResponse(status_code=type(self).post_status, json_body=_GEMINI_BODY)

    async def get(self, url: str, **_kwargs: Any) -> _FakeResponse:
        return _FakeResponse(is_redirect=True, location=_CANONICAL.get(url, url), url=url)


@pytest.fixture()
def _patch_httpx(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    _FakeHttpxClient.post_status = 200
    monkeypatch.setattr(httpx, "AsyncClient", _FakeHttpxClient)


def _fake_serving_client() -> Any:
    return SimpleNamespace(base_url="https://host/serving-endpoints", api_key="tok")


class TestGeminiBackend:
    async def test_resolves_redirects_and_maps_snippets(self, _patch_httpx: None) -> None:
        backend = _GeminiGenerateContentBackend(
            "databricks-gemini-3-1-flash-lite", resolve_redirects=True, timeout_seconds=10.0
        )
        out = await backend.search(_fake_serving_client(), "q")
        assert [r.url for r in out] == [
            "https://www.postgresql.org/support/versioning/",
            "https://bucardo.org/postgres_all_versions.html",
        ]
        # chunk 0 gets both supports that reference it; chunk 1 only the second.
        assert "latest major version" in out[0].snippet
        assert "September 2025" in out[0].snippet
        assert out[1].snippet == "It was released in September 2025."

    async def test_no_redirect_resolution_keeps_original(self, _patch_httpx: None) -> None:
        backend = _GeminiGenerateContentBackend(
            "databricks-gemini-3-1-flash-lite", resolve_redirects=False, timeout_seconds=10.0
        )
        out = await backend.search(_fake_serving_client(), "q")
        assert out[0].url == _REDIRECT_A

    async def test_non_200_returns_empty(self, _patch_httpx: None) -> None:
        _FakeHttpxClient.post_status = 404
        backend = _GeminiGenerateContentBackend(
            "databricks-gemini-3-1-flash-lite", resolve_redirects=True, timeout_seconds=10.0
        )
        assert await backend.search(_fake_serving_client(), "q") == []


# ---------------------------------------------------------------------------
# Adapter behavior
# ---------------------------------------------------------------------------


class _StubBackend:
    def __init__(self, *, results: list[SearchResult] | None = None,
                 exc: BaseException | None = None, sleep: float = 0.0) -> None:
        self._results = results or []
        self._exc = exc
        self._sleep = sleep
        self.calls = 0

    async def search(self, _client: Any, _instruction: str) -> list[SearchResult]:
        self.calls += 1
        if self._sleep:
            await asyncio.sleep(self._sleep)
        if self._exc is not None:
            raise self._exc
        return list(self._results)


def _adapter_with_backend(backend: Any, **kwargs: Any) -> DatabricksWebSearchAdapter:
    ad = build_databricks_web_search_adapter(
        client_provider=lambda: SimpleNamespace(base_url="https://h/serving-endpoints", api_key="t"),
        model="databricks-gpt-5",
        **kwargs,
    )
    ad._backend = backend
    return ad


class TestAdapter:
    def test_family_autodetect_picks_backend(self) -> None:
        oa = build_databricks_web_search_adapter(client_provider=lambda: None, model="databricks-gpt-5")
        gm = build_databricks_web_search_adapter(client_provider=lambda: None, model="databricks-gemini-3-flash")
        assert isinstance(oa._backend, _OpenAIResponsesBackend)
        assert isinstance(gm._backend, _GeminiGenerateContentBackend)

    async def test_dedup_and_cap(self) -> None:
        rows = [
            SearchResult(url="https://a.com/1", title="a", snippet="s"),
            SearchResult(url="https://a.com/1?utm_source=x", title="a", snippet="s"),
            SearchResult(url="https://b.com/2", title="b", snippet="s"),
            SearchResult(url="https://c.com/3", title="c", snippet="s"),
        ]
        ad = _adapter_with_backend(_StubBackend(results=rows), max_results=2)
        out = await ad.search("q", count=2)
        assert len(out) == 2  # deduped (a.com/1 collapses) then capped
        assert out[0].url == "https://a.com/1"

    async def test_url_allowed_filter(self) -> None:
        rows = [
            SearchResult(url="https://allow.com/1", title="a", snippet="s"),
            SearchResult(url="https://deny.com/2", title="b", snippet="s"),
        ]
        ad = _adapter_with_backend(
            _StubBackend(results=rows), url_allowed=lambda u: "allow.com" in u
        )
        out = await ad.search("q", count=10)
        assert [r.url for r in out] == ["https://allow.com/1"]

    async def test_timeout_returns_empty(self) -> None:
        ad = _adapter_with_backend(_StubBackend(results=[], sleep=5.0), timeout_seconds=0.05)
        assert await ad.search("q") == []

    async def test_provider_error_is_fail_soft(self) -> None:
        ad = _adapter_with_backend(_StubBackend(exc=RuntimeError("boom")))
        assert await ad.search("q") == []

    async def test_retries_then_succeeds_on_transient(self) -> None:
        class _FlakyBackend:
            def __init__(self) -> None:
                self.calls = 0

            async def search(self, _client: Any, _instruction: str) -> list[SearchResult]:
                self.calls += 1
                if self.calls == 1:
                    exc = RuntimeError("throttled")
                    exc.status_code = 503  # type: ignore[attr-defined]
                    raise exc
                return [SearchResult(url="https://ok.com/1", title="ok", snippet="s")]

        flaky = _FlakyBackend()
        ad = _adapter_with_backend(flaky)
        out = await ad.search("q")
        assert flaky.calls == 2
        assert [r.url for r in out] == ["https://ok.com/1"]


# ---------------------------------------------------------------------------
# Factory provider resolution
# ---------------------------------------------------------------------------


class TestFactoryResolution:
    def test_resolve_databricks_provider_builds_adapter(self) -> None:
        from databricks_deep_research.tools.factories.builtin import _resolve_search_provider
        from databricks_deep_research.tools.factory import ToolFactoryContext

        ws = SimpleNamespace(config=SimpleNamespace(host="https://h", authenticate=lambda: {}))
        ctx = ToolFactoryContext(workspace_client=ws, user_token="obo-token")
        tool = _resolve_search_provider("databricks", ctx, {"model": "databricks-gpt-5"})
        assert isinstance(tool, DatabricksWebSearchAdapter)

    def test_resolve_databricks_requires_workspace_client(self) -> None:
        from databricks_deep_research.tools.factories.builtin import _resolve_search_provider
        from databricks_deep_research.tools.factory import ToolFactoryContext

        ctx = ToolFactoryContext(workspace_client=None)
        with pytest.raises(ValueError, match="workspace_client"):
            _resolve_search_provider("databricks", ctx, {"model": "databricks-gpt-5"})

    def test_resolve_databricks_requires_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from databricks_deep_research.tools.factories.builtin import _resolve_search_provider
        from databricks_deep_research.tools.factory import ToolFactoryContext

        monkeypatch.delenv("DATABRICKS_WEB_SEARCH_ENDPOINT", raising=False)
        ws = SimpleNamespace(config=SimpleNamespace(host="https://h", authenticate=lambda: {}))
        ctx = ToolFactoryContext(workspace_client=ws)
        with pytest.raises(ValueError, match="serving endpoint"):
            _resolve_search_provider("databricks", ctx, {})


# ---------------------------------------------------------------------------
# Allowlist push-down: pushable derivation, instruction hint, OpenAI filters,
# graceful 400 fallback, adapter wiring, factory threading
# ---------------------------------------------------------------------------


def _bad_request_error(message: str = "unsupported parameter: filters") -> Any:
    """Construct a real ``openai.BadRequestError`` (status 400) for fallback tests."""
    import httpx
    from openai import BadRequestError

    request = httpx.Request("POST", "https://h/serving-endpoints/responses")
    response = httpx.Response(400, request=request)
    return BadRequestError(message, response=response, body=None)


def _capturing_openai_client(
    response: Any, *, fail_on_filters: bool = False
) -> tuple[Any, list[dict[str, Any]]]:
    """Fake AsyncOpenAI capturing each ``responses.create`` kwargs.

    When ``fail_on_filters`` is set, any call that includes a ``filters`` key on the
    web_search tool raises a 400 (simulating a serving proxy that rejects ``filters``).
    """
    calls: list[dict[str, Any]] = []

    async def _create(**kwargs: Any) -> Any:
        calls.append(kwargs)
        tools = kwargs.get("tools") or []
        has_filters = any(isinstance(t, dict) and "filters" in t for t in tools)
        if fail_on_filters and has_filters:
            raise _bad_request_error()
        return response

    return SimpleNamespace(responses=SimpleNamespace(create=_create)), calls


class _InstructionCapturingBackend:
    """Adapter stub that records the instruction string it was handed."""

    def __init__(self) -> None:
        self.instruction: str | None = None

    async def search(self, _client: Any, instruction: str) -> list[SearchResult]:
        self.instruction = instruction
        return []


class TestPushableAllowedDomains:
    def test_bare_and_wildcard_apex_push(self) -> None:
        assert dws._pushable_allowed_domains(["reuters.com", "bbc.com"]) == [
            "reuters.com", "bbc.com",
        ]
        assert dws._pushable_allowed_domains(["*.reuters.com"]) == ["reuters.com"]

    def test_dedup_and_normalize(self) -> None:
        # case-fold, trailing-dot strip, *. strip → all collapse to one entry.
        assert dws._pushable_allowed_domains(
            ["Reuters.com", "*.reuters.com", "reuters.com."]
        ) == ["reuters.com"]

    @pytest.mark.parametrize(
        "patterns",
        [["*.gov"], ["gov"], ["news.*"], ["*.example.*"], ["localhost"],
         ["http://x.com"], ["x.com/path"], ["x.com:443"], ["a..b"], [".com"], ["a b.com"]],
    )
    def test_non_pushable_aborts_whole_set(self, patterns: list[str]) -> None:
        assert dws._pushable_allowed_domains(patterns) == []

    def test_mixed_pushable_and_non_aborts(self) -> None:
        # ALL-OR-NOTHING: one non-pushable pattern abandons the whole structured push.
        assert dws._pushable_allowed_domains(["reuters.com", "*.gov"]) == []

    def test_over_limit_aborts(self) -> None:
        assert dws._pushable_allowed_domains(["a.com", "b.com", "c.com"], limit=2) == []

    def test_empty(self) -> None:
        assert dws._pushable_allowed_domains([]) == []


class TestDomainScopeClause:
    def test_empty_is_blank(self) -> None:
        assert dws._domain_scope_clause([]) == ""

    def test_lists_patterns_softly(self) -> None:
        clause = dws._domain_scope_clause(["*.gov", "reuters.com"])
        assert "*.gov" in clause and "reuters.com" in clause
        assert clause.startswith(" ")  # appends cleanly onto the instruction
        assert "prefer" in clause.lower()  # SOFT (not a hard "return only")

    def test_caps_long_lists(self) -> None:
        clause = dws._domain_scope_clause([f"d{i}.com" for i in range(30)], limit=5)
        assert "and 25 more" in clause


class TestOpenAIFilters:
    async def test_filters_injected_when_allowed_domains(self) -> None:
        backend = _OpenAIResponsesBackend("databricks-gpt-5", allowed_domains=["reuters.com"])
        client, calls = _capturing_openai_client(_openai_response())
        await backend.search(client, "q")
        assert calls[0]["tools"][0]["filters"]["allowed_domains"] == ["reuters.com"]

    async def test_no_filters_key_without_allowed_domains(self) -> None:
        backend = _OpenAIResponsesBackend("databricks-gpt-5")
        client, calls = _capturing_openai_client(_openai_response())
        await backend.search(client, "q")
        assert "filters" not in calls[0]["tools"][0]

    async def test_graceful_fallback_on_400(self) -> None:
        backend = _OpenAIResponsesBackend("databricks-gpt-5", allowed_domains=["reuters.com"])
        client, calls = _capturing_openai_client(_openai_response(), fail_on_filters=True)
        out = await backend.search(client, "q")
        assert len(calls) == 2  # filtered call 400s → retried WITHOUT filters
        assert "filters" in calls[0]["tools"][0]
        assert "filters" not in calls[1]["tools"][0]
        assert backend._filters_disabled is True
        assert len(out) >= 1  # results come from the fallback call

    async def test_400_without_filters_is_not_masked(self) -> None:
        # A genuine bad request (no filters were sent) must propagate, never retry.
        backend = _OpenAIResponsesBackend("databricks-gpt-5")  # no allowed_domains
        calls: list[dict[str, Any]] = []

        async def _create(**kwargs: Any) -> Any:
            calls.append(kwargs)
            raise _bad_request_error("model not found")

        client = SimpleNamespace(responses=SimpleNamespace(create=_create))
        with pytest.raises(Exception):  # noqa: B017,PT011 — BadRequestError propagates
            await backend.search(client, "q")
        assert len(calls) == 1  # no fallback retry

    async def test_sticky_disable_under_concurrency(self) -> None:
        backend = _OpenAIResponsesBackend("databricks-gpt-5", allowed_domains=["reuters.com"])
        client, _calls = _capturing_openai_client(_openai_response(), fail_on_filters=True)
        outs = await asyncio.gather(*(backend.search(client, "q") for _ in range(4)))
        assert backend._filters_disabled is True  # one-way latch
        assert all(len(o) >= 1 for o in outs)  # every search self-heals via fallback


class TestAdapterPushDown:
    async def test_scope_clause_appended_to_instruction(self) -> None:
        cap = _InstructionCapturingBackend()
        ad = _adapter_with_backend(cap, restrict_to_domains=["reuters.com", "*.gov"])
        await ad.search("q")
        assert cap.instruction is not None
        assert cap.instruction.endswith(ad._scope_clause)
        assert "reuters.com" in cap.instruction and "*.gov" in cap.instruction

    async def test_no_restrict_leaves_instruction_unchanged(self) -> None:
        cap = _InstructionCapturingBackend()
        ad = _adapter_with_backend(cap)
        await ad.search("q", count=3)
        assert ad._scope_clause == ""
        assert cap.instruction == dws._DEFAULT_INSTRUCTION.format(query="q", count=3)

    def test_openai_backend_receives_pushable_domains(self) -> None:
        ad = build_databricks_web_search_adapter(
            client_provider=lambda: None, model="databricks-gpt-5",
            restrict_to_domains=["*.reuters.com"],
        )
        assert isinstance(ad._backend, _OpenAIResponsesBackend)
        assert ad._backend._allowed_domains == ["reuters.com"]
        assert ad._allowed_domains == ["reuters.com"]

    def test_push_disabled_keeps_clause_but_no_filter(self) -> None:
        ad = build_databricks_web_search_adapter(
            client_provider=lambda: None, model="databricks-gpt-5",
            restrict_to_domains=["reuters.com"], push_allowed_domains=False,
        )
        assert ad._allowed_domains == []
        assert isinstance(ad._backend, _OpenAIResponsesBackend)
        assert ad._backend._allowed_domains == []
        assert ad._scope_clause != ""  # hint still applies

    def test_non_pushable_allowlist_no_filter_but_clause(self) -> None:
        ad = build_databricks_web_search_adapter(
            client_provider=lambda: None, model="databricks-gpt-5",
            restrict_to_domains=["*.gov"],
        )
        assert ad._allowed_domains == []
        assert "*.gov" in ad._scope_clause


class TestFactoryPushDown:
    def _ctx(self) -> Any:
        from databricks_deep_research.tools.factory import ToolFactoryContext

        ws = SimpleNamespace(config=SimpleNamespace(host="https://h", authenticate=lambda: {}))
        return ToolFactoryContext(workspace_client=ws, user_token="obo")

    def test_resolve_databricks_pushes_allowlist(self) -> None:
        from databricks_deep_research.tools.factories.builtin import _resolve_search_provider

        tool = _resolve_search_provider(
            "databricks", self._ctx(),
            {"model": "databricks-gpt-5", "domain_filter": ["*.reuters.com"]},
        )
        assert tool._backend._allowed_domains == ["reuters.com"]

    def test_resolve_databricks_push_flag_false(self) -> None:
        from databricks_deep_research.tools.factories.builtin import _resolve_search_provider

        tool = _resolve_search_provider(
            "databricks", self._ctx(),
            {"model": "databricks-gpt-5", "domain_filter": ["reuters.com"],
             "push_allowed_domains": False},
        )
        assert tool._backend._allowed_domains == []
