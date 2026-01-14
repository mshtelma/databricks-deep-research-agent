"""Integration tests for grey reference detection.

These tests run REAL research queries and verify no grey/orphaned citations
exist after the verification pipeline completes.

Grey references are:
1. Citation markers [Key] in content without matching Claim rows
2. Claims with citation_key but no evidence (claim.evidence is None)
3. Claims with invalid position offsets (out of bounds)
4. Multi-citation sentences where not all keys are resolved

Tests use the ultra-light test config (config/app.test.yaml):
- 1-2 research steps
- Minimal tool calls
- Fast model tier

Run with:
    uv run pytest tests/integration/test_grey_references.py -v -s
"""

import re
from uuid import uuid4

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.agent.orchestrator import OrchestrationConfig, run_research
from src.agent.persistence import persist_complete_research
from src.agent.state import ClaimInfo, ResearchState
from src.agent.tools.web_crawler import WebCrawler
from src.models.citation import Citation
from src.models.claim import Claim
from src.services.llm.client import LLMClient
from src.services.search.brave import BraveSearchClient
from tests.integration.conftest import requires_all_credentials


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------


def extract_citation_keys(content: str) -> set[str]:
    """Extract all [Key] citation markers from content.

    Matches human-readable keys like [Arxiv], [Zhipu], [Github-2].

    Args:
        content: The final report or synthesized content.

    Returns:
        Set of unique citation keys found in content.
    """
    pattern = r"\[([A-Za-z][A-Za-z0-9-]*(?:-\d+)?)\]"
    return set(re.findall(pattern, content))


def get_claim_keys(state: ResearchState) -> set[str]:
    """Get all citation keys from state claims.

    Includes both primary citation_key and all keys from citation_keys array.

    Args:
        state: Research state containing claims.

    Returns:
        Set of all citation keys across claims.
    """
    keys: set[str] = set()
    for claim in state.claims:
        if claim.citation_key:
            keys.add(claim.citation_key)
        if claim.citation_keys:
            keys.update(claim.citation_keys)
    return keys


def find_claims_without_evidence(claims: list[ClaimInfo]) -> list[ClaimInfo]:
    """Find claims that have citation_key but no evidence.

    These are "grey" claims that were marked but have no supporting evidence.

    Args:
        claims: List of claims from research state.

    Returns:
        List of claims with citation_key but no evidence.
    """
    return [c for c in claims if c.citation_key and c.evidence is None and not c.abstained]


def validate_position_offsets(state: ResearchState) -> list[dict[str, str]]:
    """Check that claim positions are within content bounds.

    Args:
        state: Research state with final_report and claims.

    Returns:
        List of error dicts for claims with invalid positions.
    """
    errors: list[dict[str, str]] = []
    content = state.final_report
    content_length = len(content)

    for claim in state.claims:
        if claim.position_start >= content_length:
            errors.append({
                "key": claim.citation_key or "unknown",
                "error": f"position_start ({claim.position_start}) >= content_length ({content_length})",
            })
        if claim.position_end > content_length:
            errors.append({
                "key": claim.citation_key or "unknown",
                "error": f"position_end ({claim.position_end}) > content_length ({content_length})",
            })

    return errors


def find_orphaned_markers(state: ResearchState) -> set[str]:
    """Find citation markers in content without matching claims.

    Args:
        state: Research state with final_report and claims.

    Returns:
        Set of orphaned citation keys.
    """
    content_keys = extract_citation_keys(state.final_report)
    claim_keys = get_claim_keys(state)
    return content_keys - claim_keys


# ---------------------------------------------------------------------------
# Test Class
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestGreyReferenceDetection:
    """Test grey reference detection after full research pipeline.

    These tests use the ultra-light test config (config/app.test.yaml)
    which runs research with 1-2 steps for fast execution.
    """

    @requires_all_credentials
    @pytest.mark.asyncio
    async def test_no_orphaned_citation_markers(
        self,
        llm_client: LLMClient,
        brave_client: BraveSearchClient,
        web_crawler: WebCrawler,
    ) -> None:
        """Verify all citation markers in content have matching Claims.

        An orphaned marker is [Key] appearing in the report but no Claim
        row exists with that citation_key.
        """
        config = OrchestrationConfig(
            research_depth="light",  # Use light mode for fast execution
        )

        result = await run_research(
            query="What is the capital of France?",
            llm=llm_client,
            brave_client=brave_client,
            crawler=web_crawler,
            config=config,
        )

        state = result.state

        # Skip if no report generated
        if not state.final_report:
            pytest.skip("No report generated - cannot verify citations")

        # Find orphaned markers
        orphaned = find_orphaned_markers(state)

        if orphaned:
            content_keys = extract_citation_keys(state.final_report)
            claim_keys = get_claim_keys(state)
            pytest.fail(
                f"Orphaned citation markers found: {orphaned}\n"
                f"Content keys: {content_keys}\n"
                f"Claim keys: {claim_keys}"
            )

    @requires_all_credentials
    @pytest.mark.asyncio
    async def test_all_claims_have_evidence_or_abstained(
        self,
        llm_client: LLMClient,
        brave_client: BraveSearchClient,
        web_crawler: WebCrawler,
    ) -> None:
        """Verify claims with citation_key have evidence or are marked abstained.

        A grey claim has citation_key but no evidence and is not abstained.
        """
        config = OrchestrationConfig(
            research_depth="light",
        )

        result = await run_research(
            query="Who created Python programming language?",
            llm=llm_client,
            brave_client=brave_client,
            crawler=web_crawler,
            config=config,
        )

        state = result.state

        # Find claims without evidence
        grey_claims = find_claims_without_evidence(state.claims)

        if grey_claims:
            details = [
                f"  - [{c.citation_key}]: {c.claim_text[:50]}..."
                for c in grey_claims
            ]
            pytest.fail(
                f"Claims without evidence found (not abstained):\n" + "\n".join(details)
            )

    @requires_all_credentials
    @pytest.mark.asyncio
    async def test_position_offsets_valid(
        self,
        llm_client: LLMClient,
        brave_client: BraveSearchClient,
        web_crawler: WebCrawler,
    ) -> None:
        """Verify claim position offsets are within content bounds.

        Invalid positions indicate a bug in claim parsing where positions
        were calculated from different content than stored in final_report.
        """
        config = OrchestrationConfig(
            research_depth="light",
        )

        result = await run_research(
            query="What year was Python first released?",
            llm=llm_client,
            brave_client=brave_client,
            crawler=web_crawler,
            config=config,
        )

        state = result.state

        # Skip if no claims
        if not state.claims:
            pytest.skip("No claims generated - cannot verify positions")

        errors = validate_position_offsets(state)

        if errors:
            pytest.fail(f"Position offset errors found:\n{errors}")

    @requires_all_credentials
    @pytest.mark.asyncio
    async def test_multi_marker_sentences_resolved(
        self,
        llm_client: LLMClient,
        brave_client: BraveSearchClient,
        web_crawler: WebCrawler,
    ) -> None:
        """Verify sentences with multiple markers have all keys resolved.

        Example: "Paris has 2M people [Arxiv][Wiki]" should have both
        [Arxiv] and [Wiki] in claim_keys and resolved to claims.
        """
        config = OrchestrationConfig(
            research_depth="light",
        )

        # Use comparison query likely to generate multi-citations
        result = await run_research(
            query="Compare the populations of Paris and London",
            llm=llm_client,
            brave_client=brave_client,
            crawler=web_crawler,
            config=config,
        )

        state = result.state

        if not state.final_report:
            pytest.skip("No report generated")

        # Find sentences with multiple markers
        claim_keys = get_claim_keys(state)
        multi_marker_issues: list[dict[str, object]] = []

        # Split by sentence (simple heuristic)
        sentences = state.final_report.replace("\n", " ").split(".")

        for sentence in sentences:
            markers = list(extract_citation_keys(sentence))
            if len(markers) > 1:
                unresolved = [m for m in markers if m not in claim_keys]
                if unresolved:
                    multi_marker_issues.append({
                        "sentence": sentence[:100],
                        "markers": markers,
                        "unresolved": unresolved,
                    })

        if multi_marker_issues:
            pytest.fail(f"Multi-marker resolution issues:\n{multi_marker_issues}")

    @requires_all_credentials
    @pytest.mark.asyncio
    async def test_database_citation_integrity(
        self,
        llm_client: LLMClient,
        brave_client: BraveSearchClient,
        web_crawler: WebCrawler,
        db_session: AsyncSession,
    ) -> None:
        """Verify Citation rows exist in DB for all Claims with citation_key.

        This is the definitive grey reference check at the database level.
        """
        config = OrchestrationConfig(
            research_depth="light",
        )

        # Pre-generate IDs for deferred materialization
        chat_id = uuid4()
        message_id = uuid4()
        research_session_id = uuid4()
        user_id = "test-user-grey-ref"

        result = await run_research(
            query="What is the population of Paris?",
            llm=llm_client,
            brave_client=brave_client,
            crawler=web_crawler,
            config=config,
        )

        state = result.state

        if not state.final_report:
            pytest.skip("No report generated")

        # Persist to database (atomic operation)
        await persist_complete_research(
            db=db_session,
            chat_id=chat_id,
            user_id=user_id,
            user_query="What is the population of Paris?",
            message_id=message_id,
            research_session_id=research_session_id,
            research_depth="light",
            state=state,
        )
        await db_session.flush()

        # Query all claims for this message
        claims_query = select(Claim).where(Claim.message_id == message_id)
        claims_result = await db_session.execute(claims_query)
        claims = claims_result.scalars().all()

        grey_references: list[dict[str, str]] = []

        for claim in claims:
            if claim.citation_key:
                # Check if Citation row exists for this claim
                citation_query = select(func.count()).where(
                    Citation.claim_id == claim.id
                )
                citation_result = await db_session.execute(citation_query)
                citation_count = citation_result.scalar() or 0

                if citation_count == 0:
                    grey_references.append({
                        "claim_id": str(claim.id),
                        "citation_key": claim.citation_key,
                        "claim_text": claim.claim_text[:50] if claim.claim_text else "N/A",
                        "reason": "No Citation rows in DB",
                    })

        if grey_references:
            pytest.fail(
                "Grey references found (claims without DB citations):\n"
                + "\n".join(f"  - [{g['citation_key']}]: {g['claim_text']}..." for g in grey_references)
            )

    @requires_all_credentials
    @pytest.mark.asyncio
    async def test_evidence_span_chain_complete(
        self,
        llm_client: LLMClient,
        brave_client: BraveSearchClient,
        web_crawler: WebCrawler,
        db_session: AsyncSession,
    ) -> None:
        """Verify complete chain: Claim -> Citation -> EvidenceSpan -> Source.

        Checks that all foreign key relationships are satisfied and no
        NULL references exist in the citation chain.
        """
        config = OrchestrationConfig(
            research_depth="light",
        )

        # Pre-generate IDs
        chat_id = uuid4()
        message_id = uuid4()
        research_session_id = uuid4()
        user_id = "test-user-chain"

        result = await run_research(
            query="Who founded Microsoft and when?",
            llm=llm_client,
            brave_client=brave_client,
            crawler=web_crawler,
            config=config,
        )

        state = result.state

        if not state.final_report:
            pytest.skip("No report generated")

        # Persist to database
        await persist_complete_research(
            db=db_session,
            chat_id=chat_id,
            user_id=user_id,
            user_query="Who founded Microsoft and when?",
            message_id=message_id,
            research_session_id=research_session_id,
            research_depth="light",
            state=state,
        )
        await db_session.flush()

        # Query claims with full relationship chain
        claims_query = (
            select(Claim)
            .where(Claim.message_id == message_id)
            .options(
                selectinload(Claim.citations)
                .selectinload(Citation.evidence_span)
            )
        )
        claims_result = await db_session.execute(claims_query)
        claims = claims_result.scalars().all()

        broken_chains: list[dict[str, str]] = []

        for claim in claims:
            if claim.citation_key and claim.citations:
                for citation in claim.citations:
                    if citation.evidence_span_id is None:
                        broken_chains.append({
                            "claim_key": claim.citation_key,
                            "citation_id": str(citation.id),
                            "reason": "Citation.evidence_span_id is NULL",
                        })
                    elif citation.evidence_span is None:
                        broken_chains.append({
                            "claim_key": claim.citation_key,
                            "citation_id": str(citation.id),
                            "reason": "Citation.evidence_span relationship is NULL (FK broken)",
                        })
                    elif citation.evidence_span.source_id is None:
                        broken_chains.append({
                            "claim_key": claim.citation_key,
                            "citation_id": str(citation.id),
                            "reason": "EvidenceSpan.source_id is NULL",
                        })

        if broken_chains:
            pytest.fail(
                "Broken citation chains found:\n"
                + "\n".join(f"  - [{b['claim_key']}]: {b['reason']}" for b in broken_chains)
            )
