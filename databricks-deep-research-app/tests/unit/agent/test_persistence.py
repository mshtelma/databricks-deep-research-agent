"""Unit tests for agent persistence layer.

Tests the citation key extraction fallback for the grey references bug fix.
"""


import pytest

from deep_research.agent.persistence import (
    CITATION_KEY_PATTERN,
    _ensure_citation_key,
)
from deep_research.agent.state import ClaimInfo, EvidenceInfo

pytestmark = pytest.mark.unit


class TestCitationKeyPattern:
    """Tests for the CITATION_KEY_PATTERN regex."""

    def test_matches_simple_key(self):
        """Test matching simple alphabetic keys like [Arxiv]."""
        matches = CITATION_KEY_PATTERN.findall("[Arxiv]")
        assert matches == ["Arxiv"]

    def test_matches_key_with_suffix(self):
        """Test matching keys with numeric suffix like [Arxiv-1]."""
        matches = CITATION_KEY_PATTERN.findall("[Arxiv-1]")
        assert matches == ["Arxiv-1"]

    def test_matches_alphanumeric_key(self):
        """Test matching alphanumeric keys like [Source2]."""
        matches = CITATION_KEY_PATTERN.findall("[Source2]")
        assert matches == ["Source2"]

    def test_matches_hyphenated_key(self):
        """Test matching hyphenated keys like [News-Site]."""
        matches = CITATION_KEY_PATTERN.findall("[News-Site]")
        assert matches == ["News-Site"]

    def test_matches_multiple_keys(self):
        """Test extracting multiple keys from text."""
        text = "This claim [Arxiv] is supported by [Wikipedia-1] and [Nature]."
        matches = CITATION_KEY_PATTERN.findall(text)
        assert matches == ["Arxiv", "Wikipedia-1", "Nature"]

    def test_does_not_match_numeric_only(self):
        """Test that pure numeric citations like [0] are not matched."""
        matches = CITATION_KEY_PATTERN.findall("[0]")
        assert matches == []

    def test_does_not_match_invalid_patterns(self):
        """Test that invalid patterns are not matched."""
        # Must start with letter
        assert CITATION_KEY_PATTERN.findall("[123]") == []
        assert CITATION_KEY_PATTERN.findall("[-Test]") == []
        # Must have closing bracket
        assert CITATION_KEY_PATTERN.findall("[Test") == []


class TestEnsureCitationKey:
    """Tests for _ensure_citation_key function."""

    def test_preserves_existing_citation_key(self):
        """Test that existing citation_key is preserved."""
        claim = ClaimInfo(
            claim_text="AI is transforming healthcare. [Arxiv]",
            claim_type="general",
            position_start=0,
            position_end=40,
            citation_key="Arxiv",
        )

        result = _ensure_citation_key(claim)

        assert result["citation_key"] == "Arxiv"
        assert result["claim_text"] == claim.claim_text

    def test_extracts_missing_citation_key_from_text(self):
        """Test extraction of citation_key when missing."""
        claim = ClaimInfo(
            claim_text="AI is transforming healthcare. [Arxiv]",
            claim_type="general",
            position_start=0,
            position_end=40,
            citation_key=None,  # Missing!
        )

        result = _ensure_citation_key(claim)

        assert result["citation_key"] == "Arxiv"

    def test_extracts_multiple_keys(self):
        """Test extraction when claim has multiple citation markers."""
        claim = ClaimInfo(
            claim_text="This fact [Arxiv] [Wikipedia] is well documented.",
            claim_type="general",
            position_start=0,
            position_end=50,
            citation_key=None,
        )

        result = _ensure_citation_key(claim)

        # First key becomes citation_key
        assert result["citation_key"] == "Arxiv"
        # All keys are stored in citation_keys
        assert result["citation_keys"] == ["Arxiv", "Wikipedia"]

    def test_no_extraction_for_uncited_claim(self):
        """Test that uncited claims (no markers) don't get a key."""
        claim = ClaimInfo(
            claim_text="This is an uncited statement.",
            claim_type="general",
            position_start=0,
            position_end=30,
            citation_key=None,
        )

        result = _ensure_citation_key(claim)

        # No key should be extracted
        assert result["citation_key"] is None

    def test_preserves_evidence_data(self):
        """Test that evidence data is preserved in output."""
        evidence = EvidenceInfo(
            source_url="https://arxiv.org/paper123",
            quote_text="AI research shows...",
        )
        claim = ClaimInfo(
            claim_text="AI research is advancing. [Arxiv]",
            claim_type="general",
            position_start=0,
            position_end=35,
            citation_key="Arxiv",
            evidence=evidence,
        )

        result = _ensure_citation_key(claim)

        assert result["evidence"] is not None
        assert result["evidence"]["source_url"] == "https://arxiv.org/paper123"

    def test_extracts_key_with_numeric_suffix(self):
        """Test extraction of keys with numeric suffix like [Source-1]."""
        claim = ClaimInfo(
            claim_text="Multiple sources confirm this. [Wikipedia-3]",
            claim_type="general",
            position_start=0,
            position_end=45,
            citation_key=None,
        )

        result = _ensure_citation_key(claim)

        assert result["citation_key"] == "Wikipedia-3"

    def test_preserves_other_claim_fields(self):
        """Test that all other claim fields are preserved."""
        claim = ClaimInfo(
            claim_text="Test claim [Source]",
            claim_type="numeric",
            position_start=100,
            position_end=120,
            citation_key=None,
            confidence_level="high",
            verification_verdict="supported",
            verification_reasoning="Evidence found",
            abstained=False,
            from_free_block=True,
        )

        result = _ensure_citation_key(claim)

        assert result["claim_type"] == "numeric"
        assert result["position_start"] == 100
        assert result["position_end"] == 120
        assert result["confidence_level"] == "high"
        assert result["verification_verdict"] == "supported"
        assert result["verification_reasoning"] == "Evidence found"
        assert result["abstained"] is False
        assert result["from_free_block"] is True


class TestBuildVerificationDataStructuredOutput:
    """The agent-surface structured_output envelope rides verification_data."""

    @staticmethod
    def _state(**overrides):
        from types import SimpleNamespace

        defaults: dict = {"claims": [], "verification_summary": None}
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_included_alongside_claims(self):
        from deep_research.agent.persistence import _build_verification_data

        envelope = {"version": 1, "binding": "run", "data": {"slot": []}}
        state = self._state(
            claims=[ClaimInfo(claim_text="c [K1]", claim_type="general", position_start=0, position_end=6, citation_key="K1")],
            structured_output=envelope,
        )
        data = _build_verification_data(state, {})
        assert data is not None
        assert data["structured_output"] == envelope
        assert len(data["claims"]) == 1

    def test_structured_output_alone_still_persists(self):
        from deep_research.agent.persistence import _build_verification_data

        envelope = {"version": 1, "binding": "run", "data": {}}
        data = _build_verification_data(
            self._state(structured_output=envelope), {}
        )
        assert data is not None
        assert data["structured_output"] == envelope
        assert data["claims"] == []

    def test_absent_for_legacy_states_without_attribute(self):
        from deep_research.agent.persistence import _build_verification_data

        # No structured_output attribute at all (legacy ResearchState shape).
        assert _build_verification_data(self._state(), {}) is None

        state = self._state(
            claims=[ClaimInfo(claim_text="c [K1]", claim_type="general", position_start=0, position_end=6, citation_key="K1")]
        )
        data = _build_verification_data(state, {})
        assert data is not None
        assert "structured_output" not in data


class TestUpdateStructuredOutputIndependent:
    """Targeted verification_data['structured_output'] merge, both modes."""

    def _envelope(self, generated_at: str = "2026-01-02T00:00:00+00:00") -> dict:
        return {
            "version": 2,
            "binding": "run",
            "generated_at": generated_at,
            "data": {"s": []},
            "meta": {"slots": {"s": {"status": "ok"}}},
        }

    def test_stale_guard(self) -> None:
        from deep_research.agent.persistence import (
            _should_replace_structured_output,
        )

        newer = self._envelope("2026-01-03T00:00:00+00:00")
        older = self._envelope("2026-01-01T00:00:00+00:00")
        assert _should_replace_structured_output(None, older) is True
        assert _should_replace_structured_output(older, newer) is True
        assert _should_replace_structured_output(newer, older) is False
        different = {**newer, "binding": "other"}
        assert _should_replace_structured_output(different, older) is True

    @pytest.mark.asyncio
    async def test_cached_merge_preserves_siblings_and_flushes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from types import SimpleNamespace
        from unittest.mock import AsyncMock, MagicMock
        from uuid import uuid4

        from deep_research.agent.persistence import (
            update_structured_output_independent,
        )

        rs_id = uuid4()
        chat_id = uuid4()
        rs = SimpleNamespace(
            id=rs_id,
            verification_data={"claims": [{"claim_text": "kept"}]},
        )
        doc = SimpleNamespace(
            state=SimpleNamespace(research_sessions=[rs]),
            meta=SimpleNamespace(updated_at=None),
        )

        async def _mutate(cid, fn, dirty):  # noqa: ANN001, ANN202
            assert cid == chat_id
            fn(doc)

        stack = SimpleNamespace(
            cache=SimpleNamespace(get=AsyncMock(), mutate=AsyncMock(side_effect=_mutate)),
            queue=SimpleNamespace(flush_chat_now=AsyncMock()),
        )
        monkeypatch.setattr(
            "deep_research.core.config.get_settings",
            lambda: MagicMock(storage_service_impl="cached"),
        )

        envelope = self._envelope()
        written = await update_structured_output_independent(
            chat_id=chat_id,
            research_session_id=rs_id,
            envelope=envelope,
            storage_stack=stack,
        )
        assert written is True
        assert rs.verification_data["structured_output"] == envelope
        assert rs.verification_data["claims"] == [{"claim_text": "kept"}]
        stack.queue.flush_chat_now.assert_awaited_once_with(chat_id)

    @pytest.mark.asyncio
    async def test_cached_stale_write_skipped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from types import SimpleNamespace
        from unittest.mock import AsyncMock, MagicMock
        from uuid import uuid4

        from deep_research.agent.persistence import (
            update_structured_output_independent,
        )

        rs_id = uuid4()
        newer = self._envelope("2026-01-05T00:00:00+00:00")
        rs = SimpleNamespace(
            id=rs_id, verification_data={"structured_output": newer}
        )
        doc = SimpleNamespace(
            state=SimpleNamespace(research_sessions=[rs]),
            meta=SimpleNamespace(updated_at=None),
        )

        async def _mutate(cid, fn, dirty):  # noqa: ANN001, ANN202
            fn(doc)

        stack = SimpleNamespace(
            cache=SimpleNamespace(get=AsyncMock(), mutate=AsyncMock(side_effect=_mutate)),
            queue=SimpleNamespace(flush_chat_now=AsyncMock()),
        )
        monkeypatch.setattr(
            "deep_research.core.config.get_settings",
            lambda: MagicMock(storage_service_impl="cached"),
        )

        written = await update_structured_output_independent(
            chat_id=uuid4(),
            research_session_id=rs_id,
            envelope=self._envelope("2026-01-01T00:00:00+00:00"),
            storage_stack=stack,
        )
        assert written is False
        assert rs.verification_data["structured_output"] == newer
        stack.queue.flush_chat_now.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_legacy_merge_preserves_claims(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        from uuid import uuid4

        from deep_research.agent.persistence import (
            update_structured_output_independent,
        )

        rs_id = uuid4()

        class _FakeDB:
            def __init__(self) -> None:
                self.stmts: list = []
                self.committed = False

            async def execute(self, stmt):  # noqa: ANN001, ANN202
                self.stmts.append(stmt)
                if len(self.stmts) == 1:
                    return SimpleNamespace(
                        first=lambda: (rs_id, {"claims": [{"c": 1}]})
                    )
                return SimpleNamespace()

            async def commit(self) -> None:
                self.committed = True

        db = _FakeDB()

        class _FakeMaker:
            def __call__(self):  # noqa: ANN204
                return self

            async def __aenter__(self):  # noqa: ANN204
                return db

            async def __aexit__(self, *args):  # noqa: ANN002, ANN204
                return False

        monkeypatch.setattr(
            "deep_research.core.config.get_settings",
            lambda: MagicMock(storage_service_impl="legacy"),
        )
        monkeypatch.setattr(
            "deep_research.db.session.get_session_maker",
            lambda: _FakeMaker(),
        )

        envelope = self._envelope()
        written = await update_structured_output_independent(
            chat_id=None,
            research_session_id=rs_id,
            envelope=envelope,
            storage_stack=None,
        )
        assert written is True
        assert db.committed is True
        update_stmt = db.stmts[1]
        params = update_stmt.compile().params
        assert params["verification_data"]["claims"] == [{"c": 1}]
        assert params["verification_data"]["structured_output"] == envelope

    @pytest.mark.asyncio
    async def test_legacy_missing_session_returns_false(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        from uuid import uuid4

        from deep_research.agent.persistence import (
            update_structured_output_independent,
        )

        class _FakeDB:
            async def execute(self, stmt):  # noqa: ANN001, ANN202
                return SimpleNamespace(first=lambda: None)

            async def commit(self) -> None:
                raise AssertionError("must not commit")

        class _FakeMaker:
            def __call__(self):  # noqa: ANN204
                return self

            async def __aenter__(self):  # noqa: ANN204
                return _FakeDB()

            async def __aexit__(self, *args):  # noqa: ANN002, ANN204
                return False

        monkeypatch.setattr(
            "deep_research.core.config.get_settings",
            lambda: MagicMock(storage_service_impl="legacy"),
        )
        monkeypatch.setattr(
            "deep_research.db.session.get_session_maker",
            lambda: _FakeMaker(),
        )

        written = await update_structured_output_independent(
            chat_id=None,
            research_session_id=uuid4(),
            envelope=self._envelope(),
            storage_stack=None,
        )
        assert written is False
