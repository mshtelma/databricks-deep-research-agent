"""Unit tests for the key-less academic retrievers.

Every test stubs the HTTP seam (:class:`AsyncHttpFetch`) so no network call is
made. Coverage:

* OpenAlex inverted-index abstract reconstruction (pure-function + end-to-end).
* arXiv Atom feed parsing.
* PubMed Central two-step esearch → efetch normalization.
* Semantic Scholar JSON normalization (TLDR + abstract).
* Each retriever normalizes a stubbed payload to a valid source dict with
  ``source_kind == web`` (so results are admitted to the pool).
* Key-less construction (no API key required).
* The ``academic_search`` kind is declarable via :class:`BuiltinToolFactory`
  (supports + catalog_card + safe_probe + provider dispatch).
"""

from __future__ import annotations

from typing import Any

import pytest

from databricks_deep_research.tools.builtins.academic_search import (
    ACADEMIC_PROVIDERS,
    ArxivSearchTool,
    HttpResponse,
    OpenAlexSearchTool,
    PubMedCentralSearchTool,
    SemanticScholarSearchTool,
    reconstruct_inverted_index_abstract,
)
from databricks_deep_research.tools.factories.builtin import BuiltinToolFactory
from databricks_deep_research.tools.factory import ToolFactoryContext
from databricks_deep_research.tools.protocol import (
    SourceKind,
    ToolContext,
    UrlRegistry,
)
from databricks_deep_research.workflow.definition import ToolDeclaration

# ---------------------------------------------------------------------------
# HTTP stub helper
# ---------------------------------------------------------------------------


class _StubFetch:
    """Records calls and returns canned :class:`HttpResponse` objects.

    *responses* is a list consumed in order, one per HTTP GET, so a two-step
    protocol (PMC esearch → efetch) can be scripted.
    """

    def __init__(self, responses: list[HttpResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def __call__(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 30.0,
    ) -> HttpResponse:
        self.calls.append(
            {"url": url, "params": params or {}, "headers": headers or {}}
        )
        if not self._responses:
            raise AssertionError("stub fetch called more times than scripted")
        return self._responses.pop(0)


# ---------------------------------------------------------------------------
# OpenAlex inverted-index reconstruction — pure function
# ---------------------------------------------------------------------------


class TestInvertedIndexReconstruction:
    def test_reconstructs_in_position_order(self) -> None:
        inverted = {
            "Despite": [0],
            "decades": [1],
            "of": [2, 5],
            "research": [3],
            "the": [4],
            "problem": [6],
        }
        assert (
            reconstruct_inverted_index_abstract(inverted)
            == "Despite decades of research the of problem"
        )

    def test_empty_or_none_returns_empty_string(self) -> None:
        assert reconstruct_inverted_index_abstract(None) == ""
        assert reconstruct_inverted_index_abstract({}) == ""

    def test_malformed_positions_are_skipped(self) -> None:
        # Non-list positions and negative indices degrade gracefully.
        inverted = {
            "good": [0],
            "bad": "nope",  # type: ignore[dict-item]
            "neg": [-1],
            "tail": [1],
        }
        assert reconstruct_inverted_index_abstract(inverted) == "good tail"


# ---------------------------------------------------------------------------
# arXiv Atom parsing
# ---------------------------------------------------------------------------


_ARXIV_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2310.12345v1</id>
    <title>Graph Neural Networks for Molecular Property Prediction</title>
    <summary>  We present a graph neural network that predicts molecular
    properties with high accuracy.  </summary>
    <author><name>Ada Lovelace</name></author>
    <author><name>Alan Turing</name></author>
    <link href="http://arxiv.org/abs/2310.12345v1" rel="alternate" type="text/html"/>
    <link href="http://arxiv.org/pdf/2310.12345v1" rel="related" title="pdf"
          type="application/pdf"/>
  </entry>
</feed>
"""


class TestArxiv:
    @pytest.mark.asyncio
    async def test_parses_atom_feed_to_source_dict(self) -> None:
        stub = _StubFetch([HttpResponse(status_code=200, text=_ARXIV_ATOM)])
        tool = ArxivSearchTool(http_fetch=stub)

        args = tool.validate_arguments({"query": "graph neural networks"})
        result = await tool.execute(args, ToolContext(url_registry=UrlRegistry()))

        assert result.success
        assert len(result.sources) == 1
        src = result.sources[0]
        assert src.title == "Graph Neural Networks for Molecular Property Prediction"
        assert src.url == "http://arxiv.org/abs/2310.12345v1"
        assert src.source_kind == SourceKind.web
        assert "graph neural network" in (src.content or "").lower()
        # Author + PDF metadata folded into content.
        assert "Ada Lovelace" in (src.content or "")
        assert "arxiv.org/pdf/2310.12345v1" in (src.content or "")

    @pytest.mark.asyncio
    async def test_uses_query_in_request_params(self) -> None:
        stub = _StubFetch([HttpResponse(status_code=200, text=_ARXIV_ATOM)])
        tool = ArxivSearchTool(http_fetch=stub)
        await tool.execute(
            tool.validate_arguments({"query": "qubits", "count": 3}),
            ToolContext(url_registry=UrlRegistry()),
        )
        assert stub.calls[0]["params"]["search_query"] == "all:qubits"
        assert stub.calls[0]["params"]["max_results"] == 3


# ---------------------------------------------------------------------------
# OpenAlex end-to-end
# ---------------------------------------------------------------------------


_OPENALEX_JSON = """{
  "results": [
    {
      "id": "https://openalex.org/W123",
      "title": "Attention Mechanisms in Deep Learning",
      "doi": "https://doi.org/10.1234/abcd",
      "publication_year": 2021,
      "primary_location": {
        "landing_page_url": "https://example.org/paper",
        "source": {"display_name": "Journal of ML"}
      },
      "abstract_inverted_index": {
        "Attention": [0],
        "improves": [1],
        "model": [2],
        "accuracy": [3]
      }
    }
  ]
}"""


class TestOpenAlex:
    @pytest.mark.asyncio
    async def test_reconstructs_abstract_and_normalizes(self) -> None:
        stub = _StubFetch([HttpResponse(status_code=200, text=_OPENALEX_JSON)])
        tool = OpenAlexSearchTool(http_fetch=stub)

        result = await tool.execute(
            tool.validate_arguments({"query": "attention"}),
            ToolContext(url_registry=UrlRegistry()),
        )

        assert result.success
        assert len(result.sources) == 1
        src = result.sources[0]
        assert src.title == "Attention Mechanisms in Deep Learning"
        # DOI preferred as the URL.
        assert src.url == "https://doi.org/10.1234/abcd"
        assert src.source_kind == SourceKind.web
        # Inverted index reconstructed end-to-end.
        assert "Attention improves model accuracy" in (src.content or "")
        # Venue + year folded in.
        assert "Journal of ML" in (src.content or "")
        assert "2021" in (src.content or "")

    @pytest.mark.asyncio
    async def test_sends_polite_mailto_no_key_required(self) -> None:
        stub = _StubFetch([HttpResponse(status_code=200, text=_OPENALEX_JSON)])
        tool = OpenAlexSearchTool(http_fetch=stub)  # no api_key
        await tool.execute(
            tool.validate_arguments({"query": "x"}),
            ToolContext(url_registry=UrlRegistry()),
        )
        # Polite-pool mailto present; no api_key param sent when none configured.
        assert "mailto" in stub.calls[0]["params"]
        assert "api_key" not in stub.calls[0]["params"]


# ---------------------------------------------------------------------------
# PubMed Central two-step
# ---------------------------------------------------------------------------


_PMC_ESEARCH = '{"esearchresult": {"idlist": ["7654321"]}}'

_PMC_EFETCH = """<?xml version="1.0"?>
<pmc-articleset>
  <article>
    <front>
      <article-meta>
        <article-id pub-id-type="pmc">7654321</article-id>
        <title-group>
          <article-title>CRISPR Screening in Primary Cells</article-title>
        </title-group>
        <abstract>
          <p>We describe a CRISPR screen applied to primary human cells.</p>
        </abstract>
      </article-meta>
    </front>
  </article>
</pmc-articleset>
"""


class TestPubMedCentral:
    @pytest.mark.asyncio
    async def test_esearch_then_efetch_normalizes(self) -> None:
        stub = _StubFetch(
            [
                HttpResponse(status_code=200, text=_PMC_ESEARCH),
                HttpResponse(status_code=200, text=_PMC_EFETCH),
            ]
        )
        tool = PubMedCentralSearchTool(http_fetch=stub)

        result = await tool.execute(
            tool.validate_arguments({"query": "crispr primary cells"}),
            ToolContext(url_registry=UrlRegistry()),
        )

        assert result.success
        assert len(result.sources) == 1
        src = result.sources[0]
        assert src.title == "CRISPR Screening in Primary Cells"
        assert "primary human cells" in (src.content or "")
        assert src.url == "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7654321/"
        assert src.source_kind == SourceKind.web
        # Two HTTP calls: esearch then efetch.
        assert len(stub.calls) == 2
        assert stub.calls[0]["params"]["term"] == "crispr primary cells"
        assert stub.calls[1]["params"]["id"] == "7654321"

    @pytest.mark.asyncio
    async def test_no_ids_returns_empty_without_efetch(self) -> None:
        stub = _StubFetch(
            [HttpResponse(status_code=200, text='{"esearchresult": {"idlist": []}}')]
        )
        tool = PubMedCentralSearchTool(http_fetch=stub)
        result = await tool.execute(
            tool.validate_arguments({"query": "nothing"}),
            ToolContext(url_registry=UrlRegistry()),
        )
        assert result.success
        assert result.sources == []
        assert len(stub.calls) == 1  # efetch skipped


# ---------------------------------------------------------------------------
# Semantic Scholar
# ---------------------------------------------------------------------------


_S2_JSON = """{
  "data": [
    {
      "title": "Self-Supervised Learning Survey",
      "abstract": "A broad survey of self-supervised methods.",
      "tldr": {"text": "SSL learns from unlabeled data."},
      "url": "https://www.semanticscholar.org/paper/abc",
      "year": 2022,
      "venue": "Survey Track",
      "authors": [{"name": "Grace Hopper"}],
      "externalIds": {"DOI": "10.9/xyz"}
    }
  ]
}"""


class TestSemanticScholar:
    @pytest.mark.asyncio
    async def test_normalizes_tldr_and_abstract(self) -> None:
        stub = _StubFetch([HttpResponse(status_code=200, text=_S2_JSON)])
        tool = SemanticScholarSearchTool(http_fetch=stub)

        result = await tool.execute(
            tool.validate_arguments({"query": "self supervised"}),
            ToolContext(url_registry=UrlRegistry()),
        )

        assert result.success
        src = result.sources[0]
        assert src.title == "Self-Supervised Learning Survey"
        assert src.url == "https://www.semanticscholar.org/paper/abc"
        assert src.source_kind == SourceKind.web
        assert "SSL learns from unlabeled data." in (src.content or "")
        assert "A broad survey of self-supervised methods." in (src.content or "")
        assert "Grace Hopper" in (src.content or "")

    @pytest.mark.asyncio
    async def test_no_key_means_no_api_key_header(self) -> None:
        stub = _StubFetch([HttpResponse(status_code=200, text=_S2_JSON)])
        tool = SemanticScholarSearchTool(http_fetch=stub)  # no key
        await tool.execute(
            tool.validate_arguments({"query": "x"}),
            ToolContext(url_registry=UrlRegistry()),
        )
        assert "x-api-key" not in stub.calls[0]["headers"]


# ---------------------------------------------------------------------------
# Shared behavior: URL registration, validation, error handling
# ---------------------------------------------------------------------------


class TestSharedBehavior:
    @pytest.mark.asyncio
    async def test_registers_urls_in_registry(self) -> None:
        stub = _StubFetch([HttpResponse(status_code=200, text=_OPENALEX_JSON)])
        tool = OpenAlexSearchTool(http_fetch=stub)
        registry = UrlRegistry()
        await tool.execute(
            tool.validate_arguments({"query": "x"}),
            ToolContext(url_registry=registry),
        )
        assert len(registry) == 1
        assert registry.resolve(0) == "https://doi.org/10.1234/abcd"

    def test_validate_arguments_rejects_missing_query(self) -> None:
        tool = ArxivSearchTool()
        with pytest.raises(ValueError, match="'query' is required"):
            tool.validate_arguments({})

    def test_validate_arguments_clamps_count(self) -> None:
        tool = ArxivSearchTool()
        with pytest.raises(ValueError, match="between 1 and"):
            tool.validate_arguments({"query": "x", "count": 9999})

    @pytest.mark.asyncio
    async def test_http_error_status_returns_failure(self) -> None:
        stub = _StubFetch([HttpResponse(status_code=503, text="upstream down")])
        tool = OpenAlexSearchTool(http_fetch=stub)
        result = await tool.execute(
            tool.validate_arguments({"query": "x"}),
            ToolContext(url_registry=UrlRegistry()),
        )
        assert not result.success
        assert result.error is not None

    def test_all_retrievers_construct_without_api_key(self) -> None:
        # Key-less: every provider constructs with no key and no http_fetch.
        for cls in ACADEMIC_PROVIDERS.values():
            tool = cls()
            assert tool.definition.source_kind == SourceKind.web


# ---------------------------------------------------------------------------
# Factory wiring — the kind is declarable
# ---------------------------------------------------------------------------


class TestFactoryWiring:
    def test_factory_supports_academic_search(self) -> None:
        assert BuiltinToolFactory().supports("academic_search")

    def test_catalog_card_present(self) -> None:
        card = BuiltinToolFactory.catalog_cards["academic_search"]
        assert card.summary
        assert "academic_search" in BuiltinToolFactory.safe_probes

    @pytest.mark.asyncio
    async def test_create_defaults_to_arxiv(self) -> None:
        factory = BuiltinToolFactory()
        decl = ToolDeclaration(
            name="papers", kind="academic_search", config={}
        )
        tool = await factory.create(decl, ToolFactoryContext())
        assert isinstance(tool, ArxivSearchTool)
        assert tool.definition.name == "papers"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "provider,expected",
        [
            ("openalex", OpenAlexSearchTool),
            ("pubmed_central", PubMedCentralSearchTool),
            ("semantic_scholar", SemanticScholarSearchTool),
            ("arxiv", ArxivSearchTool),
        ],
    )
    async def test_create_dispatches_on_provider(
        self, provider: str, expected: type
    ) -> None:
        factory = BuiltinToolFactory()
        decl = ToolDeclaration(
            name="papers",
            kind="academic_search",
            config={"provider": provider},
        )
        tool = await factory.create(decl, ToolFactoryContext())
        assert isinstance(tool, expected)

    @pytest.mark.asyncio
    async def test_create_rejects_unknown_provider(self) -> None:
        factory = BuiltinToolFactory()
        decl = ToolDeclaration(
            name="papers",
            kind="academic_search",
            config={"provider": "not_a_provider"},
        )
        with pytest.raises(ValueError, match="Unknown academic_search provider"):
            await factory.create(decl, ToolFactoryContext())

    @pytest.mark.asyncio
    async def test_factory_injects_http_fetch_seam_from_extras(self) -> None:
        stub = _StubFetch([HttpResponse(status_code=200, text=_OPENALEX_JSON)])
        factory = BuiltinToolFactory()
        decl = ToolDeclaration(
            name="papers",
            kind="academic_search",
            config={"provider": "openalex"},
        )
        ctx = ToolFactoryContext(extras={"_academic_http_fetch": stub})
        tool = await factory.create(decl, ctx)
        # The injected stub is used — no network.
        result = await tool.execute(
            tool.validate_arguments({"query": "x"}),
            ToolContext(url_registry=UrlRegistry()),
        )
        assert result.success
        assert len(stub.calls) == 1
