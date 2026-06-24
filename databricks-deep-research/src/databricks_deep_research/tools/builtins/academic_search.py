"""Academic retrievers — key-less full-text scientific search as research tools.

Four retrievers query *public, key-less* scholarly APIs and normalize their
heterogeneous responses to the framework's standard source dict
(:class:`~databricks_deep_research.tools.protocol.SourceInfo`):

* :class:`ArxivSearchTool` — the arXiv Atom query API (e-prints in physics,
  math, CS, quantitative biology/finance, statistics).
* :class:`OpenAlexSearchTool` — the OpenAlex works API (cross-discipline
  scholarly metadata; abstracts ship as an *inverted index* that must be
  reconstructed).
* :class:`PubMedCentralSearchTool` — NCBI E-utilities (``esearch`` → ``efetch``)
  over the PubMed Central open-access biomedical corpus.
* :class:`SemanticScholarSearchTool` — the Semantic Scholar Graph API
  ``paper/search`` endpoint (cross-discipline, with abstracts + TLDRs).

All four are *key-less* — none requires an API token. (Semantic Scholar and
NCBI both *offer* optional keys for higher rate limits; the tools accept one if
supplied but never require it.) Results carry ``source_kind = web`` so they flow
through the SAME pool/admission path as web sources (a non-builtin source kind
is admitted; see ``agents/source_aware.py``).

Design notes
------------
* **Mockable network seam.** Every tool is constructed with an injectable
  ``http_fetch`` callable satisfying :class:`AsyncHttpFetch`. The default
  implementation (:func:`_default_http_fetch`) lazily imports ``httpx``; tests
  inject a stub and never touch the network.
* **Heuristics ported, not LangChain.** The normalization logic (OpenAlex
  inverted-index reconstruction, arXiv Atom parsing, PMC efetch XML walking,
  Semantic Scholar JSON shaping) is reimplemented directly against the public
  API response shapes — no third-party retriever wiring is pulled in.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable
from xml.etree import ElementTree as ET

from databricks_deep_research.tools.protocol import (
    SourceInfo,
    SourceKind,
    ToolContext,
    ToolDefinition,
    ToolResult,
)

__all__ = [
    "AcademicSearchTool",
    "ArxivSearchTool",
    "AsyncHttpFetch",
    "HttpResponse",
    "OpenAlexSearchTool",
    "PubMedCentralSearchTool",
    "SemanticScholarSearchTool",
    "reconstruct_inverted_index_abstract",
]

logger = logging.getLogger(__name__)

# Maximum query length accepted by the tools.
_MAX_QUERY_LENGTH = 500
# Hard ceiling / default on result count.
_MAX_RESULT_COUNT = 25
_DEFAULT_MAX_RESULTS = 5
# Cap on per-result abstract/body text kept in the source dict.
_DEFAULT_MAX_CONTENT_CHARS = 8000
_DEFAULT_TIMEOUT_SECONDS = 30.0


# ---------------------------------------------------------------------------
# Mockable HTTP seam
# ---------------------------------------------------------------------------


class HttpResponse:
    """Minimal response wrapper exposed by :class:`AsyncHttpFetch`.

    Carries both the raw text and a lazily-parsed JSON view so a single seam
    serves JSON APIs (OpenAlex, Semantic Scholar) and XML/Atom APIs (arXiv,
    PubMed Central). Tests construct this directly with canned payloads.
    """

    __slots__ = ("status_code", "text")

    def __init__(self, *, status_code: int, text: str) -> None:
        self.status_code = status_code
        self.text = text

    def json(self) -> Any:
        import json

        return json.loads(self.text)


@runtime_checkable
class AsyncHttpFetch(Protocol):
    """Async HTTP GET seam — the only network dependency academic tools need.

    A concrete implementation issues an HTTP GET and returns an
    :class:`HttpResponse`. The default (:func:`_default_http_fetch`) uses
    ``httpx``; tests inject a stub so no network call is made.
    """

    async def __call__(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> HttpResponse:
        """Issue an HTTP GET and return an :class:`HttpResponse`."""
        ...


async def _default_http_fetch(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = _DEFAULT_TIMEOUT_SECONDS,
) -> HttpResponse:
    """Default :class:`AsyncHttpFetch` backed by a per-call ``httpx`` client."""
    import httpx

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.get(url, params=params, headers=headers)
        return HttpResponse(status_code=resp.status_code, text=resp.text)


# ---------------------------------------------------------------------------
# Normalization heuristics (ported — provider-agnostic)
# ---------------------------------------------------------------------------


def reconstruct_inverted_index_abstract(
    inverted_index: dict[str, list[int]] | None,
) -> str:
    """Reconstruct plain-text abstract from an OpenAlex inverted index.

    OpenAlex ships abstracts as ``abstract_inverted_index`` — a mapping of each
    token to the list of positions at which it occurs::

        {"Despite": [0], "decades": [1], "of": [2, 5], "research": [3]}

    The original text is recovered by placing every token at each of its
    positions and joining in position order. Gaps (a missing position) are
    skipped rather than padded so a malformed index degrades gracefully.

    Returns an empty string when *inverted_index* is falsy or malformed.
    """
    if not inverted_index or not isinstance(inverted_index, dict):
        return ""

    position_to_token: dict[int, str] = {}
    for token, positions in inverted_index.items():
        if not isinstance(positions, list):
            continue
        for position in positions:
            if isinstance(position, int) and position >= 0:
                position_to_token[position] = token

    if not position_to_token:
        return ""

    ordered = [position_to_token[i] for i in sorted(position_to_token)]
    return " ".join(ordered).strip()


def _clip(text: str | None, limit: int) -> str | None:
    """Trim *text* to *limit* characters; return ``None`` for empty input."""
    if not text:
        return None
    cleaned = text.strip()
    if not cleaned:
        return None
    return cleaned[:limit]


def _xml_localname(tag: str) -> str:
    """Strip an XML namespace prefix from *tag* (``{ns}name`` → ``name``)."""
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def _xml_find_text(element: ET.Element, localname: str) -> str:
    """Return stripped text of the first direct child whose local name matches."""
    for child in element:
        if _xml_localname(child.tag) == localname:
            return (child.text or "").strip()
    return ""


def _xml_findall(element: ET.Element, localname: str) -> list[ET.Element]:
    """Return all direct children whose local name matches *localname*."""
    return [child for child in element if _xml_localname(child.tag) == localname]


# ---------------------------------------------------------------------------
# Shared base
# ---------------------------------------------------------------------------


class AcademicSearchTool:
    """Base class for key-less academic retrievers implementing ``ResearchTool``.

    Subclasses implement :meth:`_fetch_sources`, returning a list of
    :class:`SourceInfo` for a query. The base supplies the shared
    :class:`ResearchTool` surface: ``definition``, ``validate_arguments``, and
    an ``execute`` that registers URLs, formats output, and wraps failures.

    Dependencies are constructor-injected so the network is mockable:

    * ``http_fetch`` — any :class:`AsyncHttpFetch`; defaults to ``httpx``.
    """

    # Subclasses override these three class attributes.
    _tool_name: str = "academic_search"
    _provider_label: str = "academic"
    _default_description: str = "Search a key-less scholarly corpus for papers."

    def __init__(
        self,
        *,
        http_fetch: AsyncHttpFetch | None = None,
        name: str | None = None,
        description: str = "",
        max_results: int = _DEFAULT_MAX_RESULTS,
        max_content_chars: int = _DEFAULT_MAX_CONTENT_CHARS,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        api_key: str | None = None,
    ) -> None:
        self._http_fetch: AsyncHttpFetch = http_fetch or _default_http_fetch
        self._name = name or self._tool_name
        self._max_results = min(max(max_results, 1), _MAX_RESULT_COUNT)
        self._max_content_chars = max(max_content_chars, 0)
        self._timeout_seconds = timeout_seconds
        self._api_key = api_key

        self._definition = ToolDefinition(
            name=self._name,
            description=description or self._default_description,
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "A focused scholarly search query. Use author names, "
                            "concepts, methods, or a paper title. "
                            "Example: 'graph neural networks molecular property "
                            "prediction'"
                        ),
                    },
                    "count": {
                        "type": "integer",
                        "description": (
                            f"Number of papers to return "
                            f"(default: {self._max_results}, max: {_MAX_RESULT_COUNT})"
                        ),
                        "default": self._max_results,
                    },
                },
                "required": ["query"],
            },
            source_type=self._tool_name,
            source_kind=SourceKind.web,
        )

    # -- ResearchTool protocol ----------------------------------------------

    @property
    def definition(self) -> ToolDefinition:
        """Return tool definition for LLM function calling."""
        return self._definition

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Validate and clean raw LLM arguments into the ``execute`` input dict."""
        errors: list[str] = []

        query = arguments.get("query")
        if not query:
            errors.append("'query' is required")
        elif not isinstance(query, str):
            errors.append("'query' must be a string")
        elif len(query) > _MAX_QUERY_LENGTH:
            errors.append(f"'query' must be {_MAX_QUERY_LENGTH} characters or less")

        raw_count = arguments.get("count")
        if raw_count is not None:
            if not isinstance(raw_count, int) or isinstance(raw_count, bool):
                errors.append("'count' must be an integer")
            elif raw_count < 1 or raw_count > _MAX_RESULT_COUNT:
                errors.append(f"'count' must be between 1 and {_MAX_RESULT_COUNT}")

        if errors:
            raise ValueError("; ".join(errors))

        count = raw_count if isinstance(raw_count, int) else self._max_results
        count = min(max(count, 1), _MAX_RESULT_COUNT)
        return {"query": str(query).strip(), "count": count}

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Fetch + normalize papers and return a :class:`ToolResult`.

        URLs are registered with the shared URL registry so the LLM cites by
        index, never by raw URL.
        """
        query: str = arguments["query"]
        count: int = arguments.get("count", self._max_results)

        try:
            sources = await self._fetch_sources(query, count)
        except Exception as exc:  # noqa: BLE001 — graceful tool-level failure
            logger.error(
                "ACADEMIC_SEARCH_ERROR provider=%s query=%s error=%s",
                self._provider_label,
                query[:80],
                exc,
            )
            return ToolResult(
                content=f"{self._provider_label} search failed: {exc}",
                success=False,
                error=str(exc),
            )

        registry = context.url_registry
        formatted_lines: list[str] = []
        for source in sources:
            if registry is not None and source.url:
                idx = registry.register(source.url)
            else:
                idx = len(formatted_lines)
            body = source.content or source.snippet or ""
            if body:
                formatted_lines.append(f"[{idx}] **{source.title}**\n{body}")
            else:
                formatted_lines.append(f"[{idx}] **{source.title}**")

        if not formatted_lines:
            content = "No papers found. Try different search terms."
        else:
            content = "\n\n".join(formatted_lines)

        logger.info(
            "ACADEMIC_SEARCH_COMPLETE provider=%s query=%s results=%d",
            self._provider_label,
            query[:80],
            len(sources),
        )

        return ToolResult(
            content=content,
            success=True,
            sources=sources,
            data={
                "query": query,
                "provider": self._provider_label,
                "total_results": len(sources),
                "count": count,
                "source_kind": SourceKind.web,
            },
        )

    # -- subclass hook -------------------------------------------------------

    async def _fetch_sources(self, query: str, count: int) -> list[SourceInfo]:
        """Query the provider API and normalize to :class:`SourceInfo` list."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# arXiv
# ---------------------------------------------------------------------------


class ArxivSearchTool(AcademicSearchTool):
    """Search arXiv via its public Atom query API (no key required).

    The arXiv API returns an Atom XML feed of ``<entry>`` elements. Each entry
    carries a title, summary (abstract), authors, an ``id`` (the abstract page
    URL), and ``<link>`` elements (including a PDF link). This implementation
    parses the Atom feed directly — no third-party arXiv client.
    """

    _ARXIV_URL = "http://export.arxiv.org/api/query"
    _tool_name = "academic_search"
    _provider_label = "arxiv"
    _default_description = (
        "Search arXiv e-prints (physics, math, CS, quantitative biology/finance, "
        "statistics). Returns paper titles, abstracts, and links."
    )

    async def _fetch_sources(self, query: str, count: int) -> list[SourceInfo]:
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": count,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        resp = await self._http_fetch(
            self._ARXIV_URL,
            params=params,
            headers={"Accept": "application/atom+xml"},
            timeout=self._timeout_seconds,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"arXiv API returned HTTP {resp.status_code}")
        return self._parse_atom_feed(resp.text, count)

    def _parse_atom_feed(self, xml_text: str, count: int) -> list[SourceInfo]:
        """Parse an arXiv Atom feed into :class:`SourceInfo` objects."""
        if not xml_text.strip():
            return []
        root = ET.fromstring(xml_text)  # noqa: S314 — arXiv is a trusted endpoint

        sources: list[SourceInfo] = []
        for entry in _xml_findall(root, "entry"):
            title = " ".join(_xml_find_text(entry, "title").split())
            summary = " ".join(_xml_find_text(entry, "summary").split())
            entry_id = _xml_find_text(entry, "id")

            url = entry_id
            pdf_url: str | None = None
            for link in _xml_findall(entry, "link"):
                href = link.attrib.get("href", "")
                if link.attrib.get("title") == "pdf" or link.attrib.get(
                    "type"
                ) == "application/pdf":
                    pdf_url = href
                elif link.attrib.get("rel") == "alternate" and href:
                    url = href

            authors = [
                _xml_find_text(author, "name")
                for author in _xml_findall(entry, "author")
            ]
            authors = [a for a in authors if a]

            if not (url or title):
                continue

            snippet = _clip(summary, 300) or ""
            content = _clip(summary, self._max_content_chars)
            metadata_bits: list[str] = []
            if authors:
                metadata_bits.append("Authors: " + ", ".join(authors[:10]))
            if pdf_url:
                metadata_bits.append(f"PDF: {pdf_url}")
            if content and metadata_bits:
                content = (content + "\n\n" + "\n".join(metadata_bits))[
                    : self._max_content_chars
                ]

            sources.append(
                SourceInfo(
                    url=url or entry_id,
                    title=title or url or entry_id,
                    snippet=snippet,
                    content=content,
                    source_type=self._tool_name,
                    source_kind=SourceKind.web,
                )
            )
            if len(sources) >= count:
                break
        return sources


# ---------------------------------------------------------------------------
# OpenAlex
# ---------------------------------------------------------------------------


class OpenAlexSearchTool(AcademicSearchTool):
    """Search OpenAlex works via its public REST API (no key required).

    OpenAlex returns JSON ``results`` where each work carries metadata and an
    ``abstract_inverted_index`` (the abstract encoded as token→positions). This
    tool reconstructs the plain-text abstract via
    :func:`reconstruct_inverted_index_abstract`.

    OpenAlex requests a contact email in a ``mailto`` param for its polite pool;
    a generic project address is sent (no user data, no key).
    """

    _OPENALEX_URL = "https://api.openalex.org/works"
    _POLITE_MAILTO = "deep-research-agent@example.com"
    _tool_name = "academic_search"
    _provider_label = "openalex"
    _default_description = (
        "Search OpenAlex scholarly works across all disciplines. Returns titles, "
        "reconstructed abstracts, venues, and DOIs/landing-page links."
    )

    async def _fetch_sources(self, query: str, count: int) -> list[SourceInfo]:
        params: dict[str, Any] = {
            "search": query,
            "per_page": count,
            "mailto": self._POLITE_MAILTO,
        }
        if self._api_key:
            params["api_key"] = self._api_key
        resp = await self._http_fetch(
            self._OPENALEX_URL,
            params=params,
            headers={"Accept": "application/json"},
            timeout=self._timeout_seconds,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"OpenAlex API returned HTTP {resp.status_code}")
        payload = resp.json()
        return self._normalize(payload, count)

    def _normalize(self, payload: Any, count: int) -> list[SourceInfo]:
        """Normalize an OpenAlex ``works`` JSON payload to source dicts."""
        if not isinstance(payload, dict):
            return []
        results = payload.get("results")
        if not isinstance(results, list):
            return []

        sources: list[SourceInfo] = []
        for work in results:
            if not isinstance(work, dict):
                continue
            title = str(work.get("title") or work.get("display_name") or "").strip()

            # Landing-page URL preference: DOI → primary_location → OpenAlex id.
            doi = work.get("doi")
            url = ""
            if isinstance(doi, str) and doi.strip():
                url = doi.strip()
            if not url:
                primary = work.get("primary_location")
                if isinstance(primary, dict):
                    landing = primary.get("landing_page_url") or primary.get(
                        "pdf_url"
                    )
                    if isinstance(landing, str) and landing.strip():
                        url = landing.strip()
            if not url:
                openalex_id = work.get("id")
                if isinstance(openalex_id, str):
                    url = openalex_id.strip()

            abstract = reconstruct_inverted_index_abstract(
                work.get("abstract_inverted_index")
            )

            venue = ""
            primary = work.get("primary_location")
            if isinstance(primary, dict):
                source_obj = primary.get("source")
                if isinstance(source_obj, dict):
                    venue = str(source_obj.get("display_name") or "").strip()
            year = work.get("publication_year")

            metadata_bits: list[str] = []
            if venue:
                metadata_bits.append(f"Venue: {venue}")
            if isinstance(year, int):
                metadata_bits.append(f"Year: {year}")

            content = _clip(abstract, self._max_content_chars)
            if content and metadata_bits:
                content = (content + "\n\n" + " | ".join(metadata_bits))[
                    : self._max_content_chars
                ]
            elif not content and metadata_bits:
                content = " | ".join(metadata_bits)

            if not (url or title):
                continue

            sources.append(
                SourceInfo(
                    url=url or title,
                    title=title or url,
                    snippet=_clip(abstract, 300) or (venue or ""),
                    content=content,
                    source_type=self._tool_name,
                    source_kind=SourceKind.web,
                )
            )
            if len(sources) >= count:
                break
        return sources


# ---------------------------------------------------------------------------
# PubMed Central (NCBI E-utilities)
# ---------------------------------------------------------------------------


class PubMedCentralSearchTool(AcademicSearchTool):
    """Search PubMed Central via NCBI E-utilities (``esearch`` → ``efetch``).

    The two-step protocol is ported directly: ``esearch`` returns matching PMC
    ids as JSON, then ``efetch`` returns article XML for those ids which is
    walked to extract titles + abstracts. No key required (NCBI offers an
    optional key for higher rate limits; it is forwarded if supplied).
    """

    _ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    _EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    _PMC_ARTICLE_BASE = "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC"
    _tool_name = "academic_search"
    _provider_label = "pubmed_central"
    _default_description = (
        "Search PubMed Central open-access biomedical literature. Returns article "
        "titles, abstracts, and PMC links."
    )

    async def _fetch_sources(self, query: str, count: int) -> list[SourceInfo]:
        esearch_params: dict[str, Any] = {
            "db": "pmc",
            "term": query,
            "retmax": count,
            "retmode": "json",
        }
        if self._api_key:
            esearch_params["api_key"] = self._api_key
        esearch_resp = await self._http_fetch(
            self._ESEARCH_URL,
            params=esearch_params,
            headers={"Accept": "application/json"},
            timeout=self._timeout_seconds,
        )
        if esearch_resp.status_code >= 400:
            raise RuntimeError(
                f"PMC esearch returned HTTP {esearch_resp.status_code}"
            )
        ids = self._parse_esearch_ids(esearch_resp.json())
        if not ids:
            return []

        efetch_params: dict[str, Any] = {
            "db": "pmc",
            "id": ",".join(ids),
            "retmode": "xml",
        }
        if self._api_key:
            efetch_params["api_key"] = self._api_key
        efetch_resp = await self._http_fetch(
            self._EFETCH_URL,
            params=efetch_params,
            headers={"Accept": "application/xml"},
            timeout=self._timeout_seconds,
        )
        if efetch_resp.status_code >= 400:
            raise RuntimeError(
                f"PMC efetch returned HTTP {efetch_resp.status_code}"
            )
        return self._parse_efetch_xml(efetch_resp.text, count)

    @staticmethod
    def _parse_esearch_ids(payload: Any) -> list[str]:
        """Extract the PMC id list from an ``esearch`` JSON payload."""
        if not isinstance(payload, dict):
            return []
        result = payload.get("esearchresult")
        if not isinstance(result, dict):
            return []
        idlist = result.get("idlist")
        if not isinstance(idlist, list):
            return []
        return [str(i).strip() for i in idlist if str(i).strip()]

    def _parse_efetch_xml(self, xml_text: str, count: int) -> list[SourceInfo]:
        """Walk PMC ``efetch`` article XML into :class:`SourceInfo` objects."""
        if not xml_text.strip():
            return []
        root = ET.fromstring(xml_text)  # noqa: S314 — NCBI is a trusted endpoint

        sources: list[SourceInfo] = []
        # Articles appear as <article> elements anywhere in the tree.
        for article in root.iter():
            if _xml_localname(article.tag) != "article":
                continue
            title = self._extract_title(article)
            abstract = self._extract_abstract(article)
            pmcid = self._extract_pmcid(article)

            url = ""
            if pmcid:
                digits = pmcid.lstrip("PMCpmc")
                url = f"{self._PMC_ARTICLE_BASE}{digits}/"

            if not (title or abstract):
                continue

            content = _clip(abstract, self._max_content_chars)
            sources.append(
                SourceInfo(
                    url=url or (pmcid or title),
                    title=title or (pmcid or "PMC article"),
                    snippet=_clip(abstract, 300) or "",
                    content=content,
                    source_type=self._tool_name,
                    source_kind=SourceKind.web,
                )
            )
            if len(sources) >= count:
                break
        return sources

    @staticmethod
    def _extract_title(article: ET.Element) -> str:
        for elem in article.iter():
            if _xml_localname(elem.tag) == "article-title":
                return " ".join("".join(elem.itertext()).split())
        return ""

    @staticmethod
    def _extract_abstract(article: ET.Element) -> str:
        for elem in article.iter():
            if _xml_localname(elem.tag) == "abstract":
                return " ".join("".join(elem.itertext()).split())
        return ""

    @staticmethod
    def _extract_pmcid(article: ET.Element) -> str:
        for elem in article.iter():
            if _xml_localname(elem.tag) != "article-id":
                continue
            if elem.attrib.get("pub-id-type") == "pmc":
                return (elem.text or "").strip()
        return ""


# ---------------------------------------------------------------------------
# Semantic Scholar
# ---------------------------------------------------------------------------


class SemanticScholarSearchTool(AcademicSearchTool):
    """Search Semantic Scholar via the Graph API ``paper/search`` (no key).

    The endpoint returns JSON ``data`` where each paper carries a title,
    abstract, optional ``tldr`` summary, year, venue, and an external URL. A key
    is optional (raises the rate limit) and forwarded via the ``x-api-key``
    header when supplied.
    """

    _S2_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
    _S2_FIELDS = "title,abstract,url,year,venue,authors,externalIds,tldr"
    _tool_name = "academic_search"
    _provider_label = "semantic_scholar"
    _default_description = (
        "Search Semantic Scholar scholarly papers across all disciplines. Returns "
        "titles, abstracts, TLDR summaries, and links."
    )

    async def _fetch_sources(self, query: str, count: int) -> list[SourceInfo]:
        params: dict[str, Any] = {
            "query": query,
            "limit": count,
            "fields": self._S2_FIELDS,
        }
        headers = {"Accept": "application/json"}
        if self._api_key:
            headers["x-api-key"] = self._api_key
        resp = await self._http_fetch(
            self._S2_URL,
            params=params,
            headers=headers,
            timeout=self._timeout_seconds,
        )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Semantic Scholar API returned HTTP {resp.status_code}"
            )
        return self._normalize(resp.json(), count)

    def _normalize(self, payload: Any, count: int) -> list[SourceInfo]:
        """Normalize a Semantic Scholar ``paper/search`` payload to sources."""
        if not isinstance(payload, dict):
            return []
        data = payload.get("data")
        if not isinstance(data, list):
            return []

        sources: list[SourceInfo] = []
        for paper in data:
            if not isinstance(paper, dict):
                continue
            title = str(paper.get("title") or "").strip()
            abstract = str(paper.get("abstract") or "").strip()

            tldr_text = ""
            tldr = paper.get("tldr")
            if isinstance(tldr, dict):
                tldr_text = str(tldr.get("text") or "").strip()

            url = ""
            raw_url = paper.get("url")
            if isinstance(raw_url, str) and raw_url.strip():
                url = raw_url.strip()
            if not url:
                external = paper.get("externalIds")
                if isinstance(external, dict):
                    doi = external.get("DOI")
                    if isinstance(doi, str) and doi.strip():
                        url = f"https://doi.org/{doi.strip()}"

            venue = str(paper.get("venue") or "").strip()
            year = paper.get("year")
            authors = paper.get("authors")
            author_names: list[str] = []
            if isinstance(authors, list):
                for author in authors:
                    if isinstance(author, dict):
                        name = str(author.get("name") or "").strip()
                        if name:
                            author_names.append(name)

            # Body prefers TLDR (curated) then abstract; both kept when present.
            body_parts: list[str] = []
            if tldr_text:
                body_parts.append(f"TLDR: {tldr_text}")
            if abstract:
                body_parts.append(abstract)
            metadata_bits: list[str] = []
            if author_names:
                metadata_bits.append("Authors: " + ", ".join(author_names[:10]))
            if venue:
                metadata_bits.append(f"Venue: {venue}")
            if isinstance(year, int):
                metadata_bits.append(f"Year: {year}")
            if metadata_bits:
                body_parts.append(" | ".join(metadata_bits))

            body = "\n\n".join(body_parts).strip()
            content = _clip(body, self._max_content_chars)
            snippet = _clip(tldr_text or abstract, 300) or (venue or "")

            if not (url or title):
                continue

            sources.append(
                SourceInfo(
                    url=url or title,
                    title=title or url,
                    snippet=snippet,
                    content=content,
                    source_type=self._tool_name,
                    source_kind=SourceKind.web,
                )
            )
            if len(sources) >= count:
                break
        return sources


# ---------------------------------------------------------------------------
# Provider registry — kind ``academic_search`` dispatches on config.provider
# ---------------------------------------------------------------------------

# Maps the ``config.provider`` value to its concrete tool class. The factory
# uses this so a single ``academic_search`` kind covers all four retrievers.
ACADEMIC_PROVIDERS: dict[str, type[AcademicSearchTool]] = {
    "arxiv": ArxivSearchTool,
    "openalex": OpenAlexSearchTool,
    "pubmed_central": PubMedCentralSearchTool,
    "semantic_scholar": SemanticScholarSearchTool,
}

# Default provider when a declaration omits ``config.provider``.
DEFAULT_ACADEMIC_PROVIDER = "arxiv"
