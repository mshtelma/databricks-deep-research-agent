"""Stage-7 verification retrieval is provider-agnostic (not hardcoded to Brave).

The retriever + pipeline accept a generic ``search_client`` (any provider) and a
back-compat ``brave_client`` alias; Stage-7 external search stays gated off when
no client is injected (no Brave assumed).
"""

from __future__ import annotations

from unittest.mock import MagicMock

from deep_research.services.citation.pipeline import CitationVerificationPipeline
from deep_research.services.citation.verification_retriever import (
    VerificationRetriever,
)


class TestRetrieverProviderAgnostic:
    def test_search_client_param_is_stored(self) -> None:
        client = MagicMock()
        r = VerificationRetriever(
            MagicMock(), search_client=client, web_crawler=MagicMock()
        )
        assert r.search_client is client

    def test_brave_client_alias_maps_to_search_client(self) -> None:
        client = MagicMock()
        r = VerificationRetriever(MagicMock(), brave_client=client)
        assert r.search_client is client

    def test_no_client_leaves_stage7_disabled(self) -> None:
        # Guard `if not self.search_client ...` short-circuits external search.
        r = VerificationRetriever(MagicMock())
        assert r.search_client is None


class TestPipelineProviderAgnostic:
    def test_search_client_param_is_stored(self) -> None:
        client = MagicMock()
        p = CitationVerificationPipeline(MagicMock(), search_client=client)
        assert p.search_client is client

    def test_brave_client_alias_maps_to_search_client(self) -> None:
        client = MagicMock()
        p = CitationVerificationPipeline(MagicMock(), brave_client=client)
        assert p.search_client is client

    def test_no_client_by_default(self) -> None:
        p = CitationVerificationPipeline(MagicMock())
        assert p.search_client is None
