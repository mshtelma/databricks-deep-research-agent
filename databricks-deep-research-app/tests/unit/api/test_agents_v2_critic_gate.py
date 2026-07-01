"""Unit tests for the save-time validation gate in agents_v2 (US-103).

The gate delegates to the unified validator service (``workflow_validation``):
ADVISORY by default (never blocks the save on the stochastic LLM verdict),
STRICT mode (``?validation_mode=strict``) blocks on ``verdict == "fail"`` unless
``?force=true``. These tests cover the gate helpers directly; the route
integration is exercised by the broader api unit suite.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from deep_research.agent_designer.workflow_validation import (
    VALIDATOR_VERSION,
    ValidationSource,
    WorkflowValidationResult,
)
from deep_research.api.v1.agents_v2 import (
    _build_critic_llm,
    _critic_warning_header_value,
    _extract_intent_from_definition,
    _raise_if_coverage_blocks,
    _raise_if_validation_blocks,
    _response_with_validation,
    _run_save_validation,
    _stamp_validation,
    _validation_warning_header,
)
from deep_research.models.agent_v2 import AgentV2


def _result(verdict: str, **kw: Any) -> WorkflowValidationResult:
    return WorkflowValidationResult(
        verdict=verdict,  # type: ignore[arg-type]
        summary=f"verdict={verdict}",
        semantic_hash="sh",
        intent_hash="ih",
        validator_version=VALIDATOR_VERSION,
        source=ValidationSource.FRESH,
        **kw,
    )


# ---------------------------------------------------------------------------
# _extract_intent_from_definition
# ---------------------------------------------------------------------------


class TestExtractIntent:
    def test_returns_designer_goal_from_plan_and_execute_metadata(self) -> None:
        wf = {
            "root": {
                "type": "sequence",
                "children": [
                    {
                        "type": "plan_and_execute",
                        "config": {
                            "synthesis_metadata": {
                                "designer_goal": "investment research on NVDA",
                            },
                            "body": {"type": "sequence", "children": []},
                        },
                    }
                ],
            }
        }
        assert (
            _extract_intent_from_definition(wf) == "investment research on NVDA"
        )

    def test_returns_top_level_description_fallback(self) -> None:
        wf = {
            "root": {"type": "agent", "config": {"subtype": "researcher"}},
            "description": "Build an OfficeQA assistant",
        }
        assert _extract_intent_from_definition(wf) == "Build an OfficeQA assistant"

    def test_returns_empty_when_missing(self) -> None:
        wf = {"root": {"type": "agent", "config": {"subtype": "researcher"}}}
        assert _extract_intent_from_definition(wf) == ""

    def test_handles_malformed_input(self) -> None:
        assert _extract_intent_from_definition({}) == ""
        assert _extract_intent_from_definition({"root": None}) == ""
        assert _extract_intent_from_definition(None) == ""  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _raise_if_validation_blocks
# ---------------------------------------------------------------------------


class TestRaiseIfValidationBlocks:
    def test_advisory_fail_does_not_raise(self) -> None:
        # Advisory (default): a 'fail' verdict NEVER hard-blocks the save.
        _raise_if_validation_blocks(_result("fail"), force=False, strict=False)

    def test_strict_fail_raises_422(self) -> None:
        with pytest.raises(HTTPException) as exc_info:
            _raise_if_validation_blocks(_result("fail"), force=False, strict=True)
        assert exc_info.value.status_code == 422
        assert "validation" in exc_info.value.detail
        assert exc_info.value.detail["validation"]["verdict"] == "fail"
        # FE back-compat: the result is also surfaced under "critique" so the
        # frontend AgentCriticError parser (keys on "critique") gets the typed
        # path. Safe because strict only blocks on verdict=="fail".
        assert "critique" in exc_info.value.detail
        assert exc_info.value.detail["critique"]["verdict"] == "fail"
        assert "?force=true" in exc_info.value.detail["message"]

    def test_strict_fail_with_force_does_not_raise(self) -> None:
        _raise_if_validation_blocks(_result("fail"), force=True, strict=True)

    def test_strict_pass_does_not_raise(self) -> None:
        _raise_if_validation_blocks(_result("pass"), force=False, strict=True)

    def test_strict_needs_revision_does_not_raise(self) -> None:
        _raise_if_validation_blocks(
            _result("needs_revision"), force=False, strict=True
        )

    def test_strict_skipped_does_not_raise(self) -> None:
        _raise_if_validation_blocks(_result("skipped"), force=False, strict=True)


# ---------------------------------------------------------------------------
# _build_critic_llm (fail-open)
# ---------------------------------------------------------------------------


class TestBuildCriticLlm:
    def test_none_client_returns_none(self) -> None:
        assert _build_critic_llm(None) is None

    def test_adapter_init_failure_returns_none(self, monkeypatch) -> None:
        import sys
        import types

        fake = types.ModuleType("deep_research.agent_designer.llm_adapter")

        class _Boom:
            def __init__(self, *_: Any, **__: Any) -> None:
                raise RuntimeError("adapter unavailable")

        fake.AppLLMAdapter = _Boom  # type: ignore[attr-defined]
        monkeypatch.setitem(
            sys.modules, "deep_research.agent_designer.llm_adapter", fake
        )
        assert _build_critic_llm(object()) is None

    def test_ok_returns_adapter(self, monkeypatch) -> None:
        import sys
        import types

        fake = types.ModuleType("deep_research.agent_designer.llm_adapter")

        class _OK:
            def __init__(self, *_: Any, **__: Any) -> None: ...

        fake.AppLLMAdapter = _OK  # type: ignore[attr-defined]
        monkeypatch.setitem(
            sys.modules, "deep_research.agent_designer.llm_adapter", fake
        )
        assert isinstance(_build_critic_llm(object()), _OK)


# ---------------------------------------------------------------------------
# _run_save_validation
# ---------------------------------------------------------------------------


class TestRunSaveValidation:
    @pytest.mark.asyncio
    async def test_passes_extracted_intent(self, monkeypatch) -> None:
        captured: dict[str, Any] = {}

        async def _fake(**kwargs: Any) -> WorkflowValidationResult:
            captured.update(kwargs)
            return _result("needs_revision")

        monkeypatch.setattr(
            "deep_research.api.v1.agents_v2.validate_workflow", _fake
        )
        wf = {
            "root": {"type": "agent", "config": {"subtype": "researcher"}},
            "description": "Build a research assistant",
        }
        result = await _run_save_validation(wf, None, AsyncMock())
        assert result.verdict == "needs_revision"
        assert captured["intent"] == "Build a research assistant"


# ---------------------------------------------------------------------------
# _validation_warning_header
# ---------------------------------------------------------------------------


class TestValidationWarningHeader:
    def test_fail_and_needs_revision_emit_header(self) -> None:
        assert "fail" in (_validation_warning_header(_result("fail")) or "")
        assert "needs_revision" in (
            _validation_warning_header(_result("needs_revision")) or ""
        )

    def test_pass_and_skipped_emit_no_header(self) -> None:
        assert _validation_warning_header(_result("pass")) is None
        assert _validation_warning_header(_result("skipped")) is None


# ---------------------------------------------------------------------------
# _stamp_validation
# ---------------------------------------------------------------------------


class TestStampValidation:
    def test_stamps_verdict_hash_and_blob(self) -> None:
        agent = AgentV2()
        _stamp_validation(agent, _result("needs_revision"))
        assert agent.last_validation_verdict == "needs_revision"
        assert agent.last_validation_hash == "sh"
        assert isinstance(agent.last_validation, dict)
        assert agent.last_validation["verdict"] == "needs_revision"


# ---------------------------------------------------------------------------
# _response_with_validation
# ---------------------------------------------------------------------------


class TestResponseWithValidation:
    def test_attaches_validation_to_body(self) -> None:
        from datetime import UTC, datetime
        from uuid import uuid4

        from deep_research.models.visibility import AgentVisibility

        agent = SimpleNamespace(
            id=uuid4(),
            owner_id="u",
            name="n",
            description=None,
            avatar_url=None,
            visibility=AgentVisibility.PRIVATE.value,
            definition={"root": {}},
            schema_version=1,
            etag="e",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        result = _result("fail")
        resp = _response_with_validation(agent, result)  # type: ignore[arg-type]
        assert resp.validation is not None
        assert resp.validation.verdict == "fail"

    def test_none_validation_is_allowed(self) -> None:
        from datetime import UTC, datetime
        from uuid import uuid4

        from deep_research.models.visibility import AgentVisibility

        agent = SimpleNamespace(
            id=uuid4(),
            owner_id="u",
            name="n",
            description=None,
            avatar_url=None,
            visibility=AgentVisibility.PRIVATE.value,
            definition={"root": {}},
            schema_version=1,
            etag="e",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        resp = _response_with_validation(agent, None)  # type: ignore[arg-type]
        assert resp.validation is None


# ---------------------------------------------------------------------------
# _critic_warning_header_value (latin-1 safety — unchanged)
# ---------------------------------------------------------------------------


class TestCriticWarningHeaderValue:
    def test_sanitizes_model_text_for_starlette_header_encoding(self) -> None:
        value = _critic_warning_header_value(
            ["validation verdict=needs_revision: revise — add analysis\nnext line"]
        )
        assert "\n" not in value
        assert "\r" not in value
        assert "?" in value
        value.encode("latin-1")


# ---------------------------------------------------------------------------
# Ordering: validate AFTER the etag check (conflict ⇒ zero LLM calls)
# ---------------------------------------------------------------------------


class TestUpdateValidatesAfterEtagCheck:
    @pytest.mark.asyncio
    async def test_etag_conflict_returns_409_without_validating(
        self, monkeypatch
    ) -> None:
        """A stale-ETag PATCH must 409 from service.update BEFORE the validator
        runs — so a conflicting save never spends an LLM call."""
        from uuid import uuid4

        from fastapi import BackgroundTasks, Response

        import deep_research.api.v1.agents_v2 as mod
        from deep_research.services.agent_v2_service import EtagConflictError

        calls: list[int] = []

        async def _spy(**_: Any) -> WorkflowValidationResult:
            calls.append(1)
            return _result("pass")

        async def _conflict(*_: Any, **__: Any) -> Any:
            raise EtagConflictError(expected="want", actual="have")

        monkeypatch.setattr(mod, "validate_workflow", _spy)
        monkeypatch.setattr(mod.AgentV2Service, "update", _conflict)

        with pytest.raises(HTTPException) as exc_info:
            await mod.update_agent(
                agent_id=uuid4(),
                request=mod.UpdateAgentV2Request(name="rename only"),
                response=Response(),
                user=SimpleNamespace(user_id="u"),
                fastapi_request=SimpleNamespace(),
                background_tasks=BackgroundTasks(),
                if_match="have-not",
                force=False,
                validation_mode="advisory",
                session=AsyncMock(),
            )
        assert exc_info.value.status_code == 409
        assert calls == []  # validator never reached on conflict


# ---------------------------------------------------------------------------
# Decoupled advisory flow (the fix for "Request timed out after 30000ms"):
# the save commits durably + returns instantly; the slow critic runs in a
# background task. A cache hit still returns the verdict inline.
# ---------------------------------------------------------------------------


_DEF_WITH_INTENT = {"root": {}, "description": "Build a research assistant"}
_DEF_NO_INTENT = {"root": {}}


class TestAdvisorySaveProbe:
    """`_advisory_save_probe`: LLM-free, never blocks. Cache hit -> verdict;
    cache miss (intent present) -> signal background; no intent -> skip."""

    @pytest.mark.asyncio
    async def test_cache_hit_returns_verdict_no_background(self, monkeypatch) -> None:
        import deep_research.api.v1.agents_v2 as mod

        cached = _result("needs_revision").model_copy(
            update={"source": ValidationSource.CACHE}
        )

        async def _fake(**_: Any) -> WorkflowValidationResult:
            return cached

        monkeypatch.setattr(mod, "validate_workflow", _fake)
        agent = SimpleNamespace(definition=_DEF_WITH_INTENT)
        validation, needs_bg = await mod._advisory_save_probe(agent, AsyncMock())  # type: ignore[arg-type]
        assert needs_bg is False
        assert validation is not None and validation.source == ValidationSource.CACHE

    @pytest.mark.asyncio
    async def test_cache_miss_signals_background(self, monkeypatch) -> None:
        import deep_research.api.v1.agents_v2 as mod

        async def _fake(**_: Any) -> WorkflowValidationResult:
            return _result("skipped").model_copy(
                update={"source": ValidationSource.SKIPPED}
            )

        monkeypatch.setattr(mod, "validate_workflow", _fake)
        agent = SimpleNamespace(definition=_DEF_WITH_INTENT)
        validation, needs_bg = await mod._advisory_save_probe(agent, AsyncMock())  # type: ignore[arg-type]
        assert validation is None and needs_bg is True

    @pytest.mark.asyncio
    async def test_no_intent_never_consults_validator(self, monkeypatch) -> None:
        import deep_research.api.v1.agents_v2 as mod

        called: list[int] = []

        async def _fake(**_: Any) -> WorkflowValidationResult:
            called.append(1)
            return _result("pass")

        monkeypatch.setattr(mod, "validate_workflow", _fake)
        agent = SimpleNamespace(definition=_DEF_NO_INTENT)
        validation, needs_bg = await mod._advisory_save_probe(agent, AsyncMock())  # type: ignore[arg-type]
        assert validation is None and needs_bg is False
        assert called == []


class _BgFakeSession:
    """Async-context-manager session that records whether it committed."""

    def __init__(self) -> None:
        self.committed = False

    async def __aenter__(self) -> _BgFakeSession:
        return self

    async def __aexit__(self, *_exc: Any) -> bool:
        return False

    async def commit(self) -> None:
        self.committed = True


def _install_bg_session(monkeypatch: Any, session: _BgFakeSession) -> None:
    monkeypatch.setattr(
        "deep_research.api.v1.agents_v2.get_session_maker", lambda: (lambda: session)
    )


class TestValidateInBackground:
    """`_validate_in_background`: own session, stamps only when the agent's
    current definition still matches what we validated, never raises."""

    @pytest.mark.asyncio
    async def test_stamps_when_definition_unchanged(self, monkeypatch) -> None:
        from uuid import uuid4

        import deep_research.api.v1.agents_v2 as mod

        session = _BgFakeSession()
        _install_bg_session(monkeypatch, session)

        async def _fake_validate(**_: Any) -> WorkflowValidationResult:
            return _result("needs_revision")  # semantic_hash == "sh"

        monkeypatch.setattr(mod, "validate_workflow", _fake_validate)
        monkeypatch.setattr(mod, "compute_semantic_hash", lambda *a, **k: "sh")

        agent = AgentV2()
        agent.definition = _DEF_WITH_INTENT

        async def _get_for_user(_self: Any, *_a: Any) -> Any:
            return agent

        monkeypatch.setattr(mod.AgentV2Service, "get_for_user", _get_for_user)

        await mod._validate_in_background(
            agent_id=uuid4(), owner_id="u", definition=_DEF_WITH_INTENT, llm_client=None
        )
        assert session.committed is True
        assert agent.last_validation_verdict == "needs_revision"
        assert agent.last_validation_hash == "sh"

    @pytest.mark.asyncio
    async def test_skips_stamp_when_definition_changed(self, monkeypatch) -> None:
        from uuid import uuid4

        import deep_research.api.v1.agents_v2 as mod

        session = _BgFakeSession()
        _install_bg_session(monkeypatch, session)

        async def _fake_validate(**_: Any) -> WorkflowValidationResult:
            return _result("fail")  # semantic_hash == "sh"

        monkeypatch.setattr(mod, "validate_workflow", _fake_validate)
        # Current definition now hashes DIFFERENTLY -> the result is stale.
        monkeypatch.setattr(mod, "compute_semantic_hash", lambda *a, **k: "CHANGED")

        agent = AgentV2()
        agent.definition = _DEF_WITH_INTENT

        async def _get_for_user(_self: Any, *_a: Any) -> Any:
            return agent

        monkeypatch.setattr(mod.AgentV2Service, "get_for_user", _get_for_user)

        await mod._validate_in_background(
            agent_id=uuid4(), owner_id="u", definition=_DEF_WITH_INTENT, llm_client=None
        )
        assert session.committed is False
        assert agent.last_validation_verdict != "fail"

    @pytest.mark.asyncio
    async def test_timeout_stamps_nonauthoritative_fallback(self, monkeypatch) -> None:
        from uuid import uuid4

        import deep_research.api.v1.agents_v2 as mod

        session = _BgFakeSession()
        _install_bg_session(monkeypatch, session)
        monkeypatch.setattr(mod, "compute_semantic_hash", lambda *a, **k: "sh")

        async def _raise_timeout(coro: Any, timeout: float) -> Any:
            coro.close()  # avoid "coroutine never awaited"
            raise TimeoutError

        monkeypatch.setattr(mod.asyncio, "wait_for", _raise_timeout)

        agent = AgentV2()
        agent.definition = _DEF_WITH_INTENT

        async def _get_for_user(_self: Any, *_a: Any) -> Any:
            return agent

        monkeypatch.setattr(mod.AgentV2Service, "get_for_user", _get_for_user)

        await mod._validate_in_background(
            agent_id=uuid4(), owner_id="u", definition=_DEF_WITH_INTENT, llm_client=None
        )
        # Fallback is stamped so the FE's pending poll resolves; never cached.
        assert session.committed is True
        assert agent.last_validation_verdict == "skipped"
        assert agent.last_validation["source"] == "fallback"

    @pytest.mark.asyncio
    async def test_never_raises_on_session_failure(self, monkeypatch) -> None:
        from uuid import uuid4

        import deep_research.api.v1.agents_v2 as mod

        def _boom() -> Any:
            raise RuntimeError("db down")

        monkeypatch.setattr(mod, "get_session_maker", _boom)
        # Must swallow the error — a background failure cannot affect the save.
        await mod._validate_in_background(
            agent_id=uuid4(), owner_id="u", definition=_DEF_WITH_INTENT, llm_client=None
        )


class TestHydrateGetValidation:
    """`_hydrate_get_validation`: surface the stamped verdict only when it
    matches the current definition; otherwise report pending (intent present)."""

    def test_returns_validation_when_hash_matches(self, monkeypatch) -> None:
        import deep_research.api.v1.agents_v2 as mod

        monkeypatch.setattr(mod, "compute_semantic_hash", lambda *a, **k: "H")
        agent = AgentV2()
        agent.definition = _DEF_WITH_INTENT
        agent.last_validation = _result("pass").model_dump(mode="json")
        agent.last_validation_hash = "H"
        validation, pending = mod._hydrate_get_validation(agent)
        assert pending is False
        assert validation is not None and validation.verdict == "pass"

    def test_pending_when_no_stamp_but_intent(self, monkeypatch) -> None:
        import deep_research.api.v1.agents_v2 as mod

        monkeypatch.setattr(mod, "compute_semantic_hash", lambda *a, **k: "H")
        agent = AgentV2()
        agent.definition = _DEF_WITH_INTENT
        agent.last_validation = None
        agent.last_validation_hash = None
        validation, pending = mod._hydrate_get_validation(agent)
        assert validation is None and pending is True

    def test_not_pending_when_no_intent(self) -> None:
        import deep_research.api.v1.agents_v2 as mod

        agent = AgentV2()
        agent.definition = _DEF_NO_INTENT
        validation, pending = mod._hydrate_get_validation(agent)
        assert validation is None and pending is False

    def test_pending_on_corrupt_stamp(self, monkeypatch) -> None:
        import deep_research.api.v1.agents_v2 as mod

        monkeypatch.setattr(mod, "compute_semantic_hash", lambda *a, **k: "H")
        agent = AgentV2()
        agent.definition = _DEF_WITH_INTENT
        agent.last_validation = {"not": "a valid result"}
        agent.last_validation_hash = "H"
        validation, pending = mod._hydrate_get_validation(agent)
        assert validation is None and pending is True


def _fake_agent_row() -> SimpleNamespace:
    from datetime import UTC, datetime
    from uuid import uuid4

    from deep_research.models.visibility import AgentVisibility

    return SimpleNamespace(
        id=uuid4(),
        owner_id="u",
        name="n",
        description=None,
        avatar_url=None,
        visibility=AgentVisibility.PRIVATE.value,
        definition=_DEF_WITH_INTENT,
        schema_version=1,
        etag="e",
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


class TestCreateAdvisoryDecoupling:
    """`create_agent` advisory path: cache miss -> pending + background task
    scheduled (no inline LLM); cache hit -> verdict inline, no task."""

    async def _call_create(self, monkeypatch: Any, probe_result: Any) -> Any:
        from fastapi import BackgroundTasks, Response

        import deep_research.api.v1.agents_v2 as mod

        agent = _fake_agent_row()

        async def _create(_self: Any, **_: Any) -> Any:
            return agent

        async def _probe(_agent: Any, _session: Any) -> Any:
            return probe_result

        async def _write_rev(_self: Any, *_a: Any, **_k: Any) -> None:
            return None

        monkeypatch.setattr(mod.AgentV2Service, "create", _create)
        monkeypatch.setattr(mod, "_advisory_save_probe", _probe)
        monkeypatch.setattr(
            mod.AgentV2Service, "_write_revision_best_effort", _write_rev
        )

        bt = BackgroundTasks()
        resp = await mod.create_agent(
            request=mod.CreateAgentV2Request.model_construct(
                name="n",
                description=None,
                avatar_url=None,
                visibility="private",
                definition=_DEF_WITH_INTENT,
            ),
            response=Response(),
            user=SimpleNamespace(user_id="u"),
            fastapi_request=SimpleNamespace(
                app=SimpleNamespace(state=SimpleNamespace(llm_client=None))
            ),
            background_tasks=bt,
            force=False,
            validation_mode="advisory",
            session=AsyncMock(),
        )
        return resp, bt

    @pytest.mark.asyncio
    async def test_cache_miss_returns_pending_and_schedules_background(
        self, monkeypatch
    ) -> None:
        import deep_research.api.v1.agents_v2 as mod

        resp, bt = await self._call_create(monkeypatch, (None, True))
        assert resp.validation_pending is True
        assert resp.validation is None
        assert len(bt.tasks) == 1
        assert bt.tasks[0].func is mod._validate_in_background

    @pytest.mark.asyncio
    async def test_cache_hit_returns_verdict_inline_no_background(
        self, monkeypatch
    ) -> None:
        cached = _result("needs_revision").model_copy(
            update={"source": ValidationSource.CACHE}
        )
        resp, bt = await self._call_create(monkeypatch, (cached, False))
        assert resp.validation_pending is False
        assert resp.validation is not None
        assert resp.validation.verdict == "needs_revision"
        assert len(bt.tasks) == 0


class TestRaiseIfCoverageBlocks:
    """Deterministic prompt-term coverage save gate — force-overridable, no LLM."""

    _TERMS = ["fundamentals", "earnings", "competitors"]

    def _definition(self, synth_system_prompt: str) -> dict:
        return {
            "required_prompt_terms": self._TERMS,
            "root": {
                "id": "root",
                "type": "sequence",
                "config": {},
                "children": [
                    {
                        "id": "synth",
                        "type": "agent",
                        "label": "Synthesizer",
                        "config": {
                            "subtype": "synthesizer",
                            "system_prompt": synth_system_prompt,
                        },
                        "children": [],
                    }
                ],
            },
        }

    def test_blocks_uncovered_without_force(self) -> None:
        with pytest.raises(HTTPException) as excinfo:
            _raise_if_coverage_blocks(self._definition("Generic synthesis."), force=False)
        assert excinfo.value.status_code == 422
        assert "coverage_errors" in excinfo.value.detail

    def test_force_bypasses(self) -> None:
        # No raise even though the workflow is uncovered.
        _raise_if_coverage_blocks(self._definition("Generic synthesis."), force=True)

    def test_covered_passes(self) -> None:
        sp = "Synthesize covering fundamentals, earnings, and competitors."
        _raise_if_coverage_blocks(self._definition(sp), force=False)
