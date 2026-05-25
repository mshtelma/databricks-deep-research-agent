"""Plan v2.1 PR-2 — BuildBlueprintTool + RequestSignatureRevisionTool tests.

Covers the two new framework-tools wrappers around the deterministic
blueprint builder and the architect's bounded escape hatch:

* :class:`BuildBlueprintTool` — reads task_signature / intent / assets
  from arguments OR (when not in args) from optional state getters;
  calls :func:`build_blueprint`; returns the AST + fingerprint +
  lane_keys in ToolResult.data; fails-closed on invalid signatures.
* :class:`RequestSignatureRevisionTool` — bounded escape hatch with
  K=2 limit per plan M12; emits ``signature_unresolved`` after
  exhaustion; carries reason + fields_to_reconsider into state.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from databricks_deep_research.tools.protocol import ToolContext

from deep_research.agent_designer.blueprint import (
    DESIGNER_DETERMINISTIC_BLUEPRINT_ENV,
)
from deep_research.agent_designer.framework_tools import (
    BuildBlueprintTool,
    RequestSignatureRevisionTool,
)


@pytest.fixture(autouse=True)
def _enable_deterministic_blueprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Most BuildBlueprintTool tests assert the FLAG-ON behavior.

    The flag defaults OFF in PR-2 (it flips ON in PR-3). For the
    feature-targeted tests below we enable it via env. The one explicit
    no-op test disables it locally.
    """
    monkeypatch.setenv(DESIGNER_DETERMINISTIC_BLUEPRINT_ENV, "1")


def _ctx() -> ToolContext:
    """Minimal ToolContext for invoking tools in tests.

    The tools we test do not consume any ToolContext fields, so a
    bare instance is sufficient.
    """
    return ToolContext()  # type: ignore[call-arg]


def _officeqa_signature() -> dict[str, Any]:
    return {
        "asset_signature": "corpus_only",
        "retrieval_pattern": "pipelined_retrieve_read_compute",
        "question_class": "numeric_aggregation",
        "primary_evidence_kind": "structured_tables",
        "expected_output_shape": "single_number",
        "independent_workstreams_count": 1,
        "step_dependencies_present": True,
        "iteration_required": True,
        "lane_descriptions": ["retrieve then read then compute pipeline"],
    }


def _investment_signature() -> dict[str, Any]:
    return {
        "asset_signature": "web_only",
        "retrieval_pattern": "independent_lanes",
        "question_class": "open_research",
        "primary_evidence_kind": "web_articles",
        "expected_output_shape": "structured_report",
        "independent_workstreams_count": 6,
        "lane_descriptions": [
            "fundamentals",
            "valuation",
            "risk",
            "market trends",
            "earnings",
            "competitors",
        ],
    }


# ---------------------------------------------------------------------------
# Feature-flag gating — PR-2 default-OFF preserves legacy behavior
# ---------------------------------------------------------------------------


def test_build_blueprint_tool_is_inert_when_flag_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PR-2 acceptance: with flag OFF, the tool is a no-op pass-through.

    Returns success + empty data so the YAML can wire the node
    unconditionally. No state writes happen, so the legacy
    architect-authored-AST flow is preserved untouched.
    """
    monkeypatch.setenv(DESIGNER_DETERMINISTIC_BLUEPRINT_ENV, "0")

    state_writes: dict[str, Any] = {}

    def signature_getter() -> Any:
        return _investment_signature()

    def blueprint_setter(v: Any) -> None:
        state_writes["initial_blueprint"] = v

    tool = BuildBlueprintTool(
        signature_getter=signature_getter,
        blueprint_setter=blueprint_setter,
    )
    result = asyncio.run(tool.execute({"intent": "q", "assets": []}, _ctx()))
    assert result.success is not False
    # No state was written even though getters were provided.
    assert "initial_blueprint" not in state_writes
    payload = json.loads(result.content or "{}")
    assert payload.get("skipped") is True
    assert payload.get("reason") == "flag_off"
    # The data dict is intentionally empty — no AST propagation.
    assert (result.data or {}) == {}


# ---------------------------------------------------------------------------
# BuildBlueprintTool — happy path
# ---------------------------------------------------------------------------


def test_build_blueprint_tool_definition_exposes_correct_metadata() -> None:
    tool = BuildBlueprintTool()
    definition = tool.definition
    assert definition.name == "build_blueprint"
    assert definition.source_type == "builtin"
    assert "task_signature" in (definition.parameters.get("properties") or {})
    assert "intent" in (definition.parameters.get("properties") or {})
    assert "assets" in (definition.parameters.get("properties") or {})


def test_build_blueprint_tool_arguments_only_no_state() -> None:
    """Stateless invocation: all inputs come via arguments dict."""
    tool = BuildBlueprintTool()
    result = asyncio.run(
        tool.execute(
            {
                "task_signature": _investment_signature(),
                "intent": "Investment analysis",
                "assets": [],
            },
            _ctx(),
        )
    )
    assert result.success is not False
    data = result.data or {}
    assert "initial_blueprint" in data
    assert "current_ast" in data
    assert "blueprint_fingerprint" in data
    assert "blueprint_lane_keys" in data
    assert isinstance(data["blueprint_fingerprint"], str)
    assert len(data["blueprint_fingerprint"]) == 64  # sha256 hex
    assert len(data["blueprint_lane_keys"]) == 6


def test_build_blueprint_tool_reads_from_state_getters() -> None:
    """When arguments are empty, the tool falls back to state getters."""
    captured: dict[str, Any] = {}

    def signature_getter() -> Any:
        return _officeqa_signature()

    def intent_getter() -> Any:
        return "Sum FY1945 Army expenditures"

    def assets_getter() -> Any:
        return [{"kind": "vector_index", "full_name": "a.b.c"}]

    def blueprint_setter(v: Any) -> None:
        captured["initial_blueprint"] = v

    def ast_setter(v: Any) -> None:
        captured["current_ast"] = v

    def fingerprint_setter(v: Any) -> None:
        captured["fingerprint"] = v

    def lane_keys_setter(v: Any) -> None:
        captured["lane_keys"] = v

    tool = BuildBlueprintTool(
        signature_getter=signature_getter,
        intent_getter=intent_getter,
        assets_getter=assets_getter,
        blueprint_setter=blueprint_setter,
        ast_setter=ast_setter,
        fingerprint_setter=fingerprint_setter,
        lane_keys_setter=lane_keys_setter,
    )
    result = asyncio.run(tool.execute({}, _ctx()))
    assert result.success is not False
    # Both the data dict AND the state setters captured the values.
    data = result.data or {}
    assert data["blueprint_fingerprint"] == captured["fingerprint"]
    assert data["initial_blueprint"] is captured["initial_blueprint"]
    assert data["current_ast"] is captured["current_ast"]
    assert data["blueprint_lane_keys"] == captured["lane_keys"]


def test_build_blueprint_tool_content_carries_ast_with_metadata() -> None:
    """Content is the full blueprint AST as JSON.

    The YAML tool-node executor writes ``content`` to
    ``state.<output_key>`` (executor.py:863); downstream nodes
    deserialize it back into the AST shape. The
    ``structural_fingerprint`` and ``lane_keys`` are embedded as
    top-level fields on the AST itself, so state lookups like
    ``state.initial_blueprint.structural_fingerprint`` work via the
    framework's dot-path resolution.
    """
    tool = BuildBlueprintTool()
    result = asyncio.run(
        tool.execute(
            {
                "task_signature": _investment_signature(),
                "intent": "q",
                "assets": [],
            },
            _ctx(),
        )
    )
    assert result.content is not None
    payload = json.loads(result.content)
    # The AST has standard top-level keys
    assert "root" in payload
    assert "pools" in payload
    # Plus the v2.1 blueprint metadata fields
    assert "structural_fingerprint" in payload
    assert "lane_keys" in payload
    assert isinstance(payload["lane_keys"], dict)
    assert len(payload["lane_keys"]) == 6


# ---------------------------------------------------------------------------
# BuildBlueprintTool — failure-closed
# ---------------------------------------------------------------------------


def test_build_blueprint_tool_no_signature_anywhere_fails() -> None:
    """Plan M11: no signature in args, no signature in state → error result."""
    tool = BuildBlueprintTool()
    result = asyncio.run(
        tool.execute({"intent": "q", "assets": []}, _ctx())
    )
    assert result.success is False
    assert result.error is not None
    assert "build_blueprint failed" in result.error


def test_build_blueprint_tool_invalid_signature_fails_closed() -> None:
    tool = BuildBlueprintTool()
    result = asyncio.run(
        tool.execute(
            {
                "task_signature": {"asset_signature": "not_real"},
                "intent": "q",
            },
            _ctx(),
        )
    )
    assert result.success is False
    assert result.error is not None
    assert "build_blueprint failed" in result.error


def test_build_blueprint_tool_signature_as_json_string_accepted() -> None:
    """LLM may stringify the signature payload; the tool tolerates that.

    Uses a ``web_only`` signature to keep the assertion focused on the
    stringified-JSON tolerance behavior rather than tripping the
    corpus-only fail-closed branch in ``_build_asset_tool_plan``.
    """
    tool = BuildBlueprintTool()
    result = asyncio.run(
        tool.execute(
            {
                "task_signature": json.dumps(_investment_signature()),
                "intent": "q",
                "assets": [],
            },
            _ctx(),
        )
    )
    assert result.success is not False


def test_build_blueprint_tool_state_getter_exception_does_not_crash() -> None:
    """Getters can throw — the tool must downgrade gracefully, not crash."""

    def signature_getter() -> Any:
        raise RuntimeError("simulated state read failure")

    tool = BuildBlueprintTool(signature_getter=signature_getter)
    result = asyncio.run(tool.execute({"intent": "q"}, _ctx()))
    # No signature available → fail-closed with a clean error
    assert result.success is False


# ---------------------------------------------------------------------------
# RequestSignatureRevisionTool — argument validation
# ---------------------------------------------------------------------------


def test_revision_tool_validation_rejects_empty_reason() -> None:
    tool = RequestSignatureRevisionTool()
    with pytest.raises(ValueError, match="reason"):
        tool.validate_arguments({"reason": ""})
    with pytest.raises(ValueError, match="reason"):
        tool.validate_arguments({"reason": "   "})


def test_revision_tool_validation_rejects_non_string_reason() -> None:
    tool = RequestSignatureRevisionTool()
    with pytest.raises(ValueError, match="reason"):
        tool.validate_arguments({"reason": 42})  # type: ignore[arg-type]


def test_revision_tool_validation_rejects_non_list_fields() -> None:
    tool = RequestSignatureRevisionTool()
    with pytest.raises(ValueError, match="fields_to_reconsider"):
        tool.validate_arguments(
            {"reason": "topology", "fields_to_reconsider": "not_a_list"}
        )


def test_revision_tool_validation_normalizes_fields() -> None:
    tool = RequestSignatureRevisionTool()
    out = tool.validate_arguments(
        {
            "reason": "  topology disagrees  ",
            "fields_to_reconsider": ["count", "  ", "iter", 42],
        }
    )
    assert out == {
        "reason": "topology disagrees",
        "fields_to_reconsider": ["count", "iter", "42"],
    }


# ---------------------------------------------------------------------------
# RequestSignatureRevisionTool — K=2 boundary
# ---------------------------------------------------------------------------


def test_revision_first_call_succeeds_and_writes_state() -> None:
    captured: dict[str, Any] = {}

    def count_getter() -> Any:
        return 0

    def count_setter(v: Any) -> None:
        captured["count"] = v

    def request_setter(v: Any) -> None:
        captured["request"] = v

    tool = RequestSignatureRevisionTool(
        revision_count_getter=count_getter,
        revision_count_setter=count_setter,
        revision_request_setter=request_setter,
    )
    args = tool.validate_arguments(
        {
            "reason": "topology disagrees with brief shape",
            "fields_to_reconsider": ["independent_workstreams_count"],
        }
    )
    result = asyncio.run(tool.execute(args, _ctx()))
    assert result.success is not False
    assert captured["count"] == 1
    assert captured["request"]["reason"] == "topology disagrees with brief shape"
    assert captured["request"]["revision_count"] == 1


def test_revision_second_call_succeeds_and_increments_count() -> None:
    state = {"count": 1}

    def count_getter() -> Any:
        return state["count"]

    def count_setter(v: Any) -> None:
        state["count"] = v

    tool = RequestSignatureRevisionTool(
        revision_count_getter=count_getter,
        revision_count_setter=count_setter,
    )
    args = tool.validate_arguments(
        {"reason": "still disagrees"}
    )
    result = asyncio.run(tool.execute(args, _ctx()))
    assert result.success is not False
    assert state["count"] == 2


def test_revision_third_call_fails_with_signature_unresolved() -> None:
    """Plan M12: K=2 exhaustion → signature_unresolved error."""
    captured: dict[str, Any] = {}

    def count_getter() -> Any:
        return 2

    def error_setter(v: Any) -> None:
        captured["error"] = v

    tool = RequestSignatureRevisionTool(
        revision_count_getter=count_getter,
        error_setter=error_setter,
    )
    args = tool.validate_arguments({"reason": "third attempt"})
    result = asyncio.run(tool.execute(args, _ctx()))
    assert result.success is False
    assert result.error is not None
    assert "signature_unresolved" in result.error
    assert captured["error"]["kind"] == "signature_unresolved"


def test_revision_count_string_digit_treated_as_int() -> None:
    """Some state backends serialize integers as strings — tool tolerates that."""

    def count_getter() -> Any:
        return "2"

    tool = RequestSignatureRevisionTool(revision_count_getter=count_getter)
    args = tool.validate_arguments({"reason": "boundary"})
    result = asyncio.run(tool.execute(args, _ctx()))
    # 2 is already the max → third attempt = exhausted.
    assert result.success is False


def test_revision_no_getter_defaults_count_to_zero() -> None:
    """Without a count_getter, the tool assumes 0 — first call succeeds."""
    tool = RequestSignatureRevisionTool()
    args = tool.validate_arguments({"reason": "first attempt"})
    result = asyncio.run(tool.execute(args, _ctx()))
    assert result.success is not False


def test_revision_setters_exceptions_dont_crash_tool() -> None:
    """Setters may throw on misconfigured state; tool should still succeed."""

    def request_setter(v: Any) -> None:
        raise RuntimeError("state write failed")

    tool = RequestSignatureRevisionTool(revision_request_setter=request_setter)
    args = tool.validate_arguments({"reason": "first"})
    result = asyncio.run(tool.execute(args, _ctx()))
    # Tool should still return success (best-effort writes per the pattern)
    assert result.success is not False


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


def test_both_tools_registered_in_builtin_designer_tools() -> None:
    from deep_research.agent_designer.framework_tools import (
        builtin_designer_tools,
    )

    names = {t.definition.name for t in builtin_designer_tools()}
    assert "build_blueprint" in names
    assert "request_signature_revision" in names


# ---------------------------------------------------------------------------
# Asset payload shape — orchestrator stores ``state.designer_assets`` as a
# dict (``{"assets": [...], "count": N}``), the YAML's ``input_mapping``
# threads it as the ``assets`` argument. BuildBlueprintTool must unwrap
# that dict before delegating to build_blueprint; the earlier list-only
# check silently dropped the dict to ``[]``, which broke asset→tool
# wiring for every case whose designer_assets came through state.
# ---------------------------------------------------------------------------


def test_build_blueprint_tool_unwraps_designer_assets_dict_form() -> None:
    """When the YAML wires ``state.designer_assets`` (a dict produced by
    ``asset_context_payload``) into the ``assets`` argument, the tool must
    extract the inner list so ``build_blueprint``'s asset→tool wiring runs
    on real assets, not on an empty list.
    """
    tool = BuildBlueprintTool()
    designer_assets_payload = {
        "assets": [
            {
                "kind": "vector_index",
                "full_name": "vs.example.idx",
                "usage": "required",
                "metadata": {"columns": ["chunk_id"]},
            },
            {
                "kind": "delta_table",
                "full_name": "delta.example.tbl",
                "usage": "required",
                "field_roles": {"primary_key": "id", "content": "content"},
                "metadata": {
                    "warehouse_id": "wh-1",
                    "columns": ["id", "content"],
                },
            },
        ],
        "count": 2,
    }
    result = asyncio.run(
        tool.execute(
            {
                "task_signature": _officeqa_signature(),
                "intent": "corpus pipeline",
                "assets": designer_assets_payload,
            },
            _ctx(),
        )
    )
    assert result.success is not False
    ast = (result.data or {}).get("initial_blueprint") or {}
    declared_kinds = {
        str(t.get("kind") or "")
        for t in (ast.get("tools") or [])
        if isinstance(t, dict)
    }
    assert "vector_search" in declared_kinds, (
        "vector_search must be declared from the wrapped dict's vector_index "
        f"asset; got kinds={sorted(declared_kinds)}"
    )
    assert "delta_read" in declared_kinds
    # No web tools — corpus_only signature must NOT leak public-web defaults.
    assert "web_research" not in declared_kinds
    assert "web_crawl" not in declared_kinds


def test_build_blueprint_tool_accepts_plain_list_form() -> None:
    """Symmetric: when the assets arrive as a plain list (callers that bypass
    ``asset_context_payload`` or pass raw arguments), the tool still wires
    corpus tools."""
    tool = BuildBlueprintTool()
    plain_list = [
        {
            "kind": "vector_index",
            "full_name": "vs.example.idx",
            "usage": "required",
            "metadata": {"columns": ["chunk_id"]},
        },
    ]
    result = asyncio.run(
        tool.execute(
            {
                "task_signature": _officeqa_signature(),
                "intent": "corpus pipeline",
                "assets": plain_list,
            },
            _ctx(),
        )
    )
    ast = (result.data or {}).get("initial_blueprint") or {}
    declared_kinds = {
        str(t.get("kind") or "")
        for t in (ast.get("tools") or [])
        if isinstance(t, dict)
    }
    assert "vector_search" in declared_kinds
    assert "web_research" not in declared_kinds


# ---------------------------------------------------------------------------
# Plan v2.1 generic-robustness — EvaluateSignatureLoopTool
# ---------------------------------------------------------------------------


def test_evaluate_signature_loop_done_when_approved_and_no_revision() -> None:
    """Happy path: critic approved AND no revision request → loop exits."""
    from deep_research.agent_designer.framework_tools import (
        EvaluateSignatureLoopTool,
    )

    tool = EvaluateSignatureLoopTool()
    result = asyncio.run(
        tool.execute(
            {
                "critic_approved": {"critic_approved": True},
                "revision_request": {},
                "revision_count": 0,
            },
            _ctx(),
        )
    )
    data = result.data or {}
    assert data["signature_loop_done"] is True
    assert data["critic_approved"] is True
    assert data["has_revision_request"] is False
    assert data["exhausted"] is False


def test_evaluate_signature_loop_continues_when_revision_pending() -> None:
    """Critic approved BUT architect set a revision_request → re-classify."""
    from deep_research.agent_designer.framework_tools import (
        EvaluateSignatureLoopTool,
    )

    tool = EvaluateSignatureLoopTool()
    result = asyncio.run(
        tool.execute(
            {
                "critic_approved": {"critic_approved": True},
                "revision_request": {
                    "reason": "blueprint suggested 1 lane but intent has 6",
                    "fields_to_reconsider": ["independent_workstreams_count"],
                },
                "revision_count": 0,
            },
            _ctx(),
        )
    )
    data = result.data or {}
    assert data["signature_loop_done"] is False
    assert data["has_revision_request"] is True


def test_evaluate_signature_loop_exhausted_at_k2_even_with_revision() -> None:
    """Plan v2.1 M12: when revision_count == K (=2), exit even if the
    architect is STILL requesting more revisions. The
    RequestSignatureRevisionTool itself returns signature_unresolved at
    this point; the evaluator must agree to exit the outer loop."""
    from deep_research.agent_designer.framework_tools import (
        EvaluateSignatureLoopTool,
    )

    tool = EvaluateSignatureLoopTool()
    result = asyncio.run(
        tool.execute(
            {
                "critic_approved": False,
                "revision_request": {"reason": "still wrong"},
                "revision_count": 2,
            },
            _ctx(),
        )
    )
    data = result.data or {}
    assert data["signature_loop_done"] is True
    assert data["exhausted"] is True


def test_evaluate_signature_loop_exits_when_no_revision_requested() -> None:
    """Critic rejected AND no revision → "no point re-classifying" early
    exit. If the architect didn't escalate via request_signature_revision,
    re-running the classifier won't help (the failure isn't a classifier
    mistake). Exit and surface the architect-side defect."""
    from deep_research.agent_designer.framework_tools import (
        EvaluateSignatureLoopTool,
    )

    tool = EvaluateSignatureLoopTool()
    result = asyncio.run(
        tool.execute(
            {
                "critic_approved": False,
                "revision_request": {},
                "revision_count": 0,
            },
            _ctx(),
        )
    )
    data = result.data or {}
    assert data["signature_loop_done"] is True
    assert data["has_revision_request"] is False


def test_evaluate_signature_loop_empty_revision_payload_treated_as_no_revision() -> None:
    """Defensive: a ``revision_request`` dict without a ``reason`` is treated
    as no real revision request (architect's tool would never persist such a
    shape, but state could carry a partial dict from a different code path)."""
    from deep_research.agent_designer.framework_tools import (
        EvaluateSignatureLoopTool,
    )

    tool = EvaluateSignatureLoopTool()
    result = asyncio.run(
        tool.execute(
            {
                "critic_approved": {"critic_approved": True},
                "revision_request": {"fields_to_reconsider": []},  # no reason
                "revision_count": 0,
            },
            _ctx(),
        )
    )
    data = result.data or {}
    assert data["signature_loop_done"] is True
    assert data["has_revision_request"] is False


def test_request_signature_revision_persists_via_setters() -> None:
    """Plan v2.1 generic-robustness — the architect's tool call must mutate
    the orchestrator-wired state slots so the outer signature_loop can read
    them."""
    captured_count: list[int] = []
    captured_request: list[dict[str, Any]] = []

    def count_getter() -> int:
        return captured_count[-1] if captured_count else 0

    def count_setter(value: Any) -> None:
        captured_count.append(int(value))

    def request_setter(value: Any) -> None:
        captured_request.append(value)

    tool = RequestSignatureRevisionTool(
        revision_count_getter=count_getter,
        revision_count_setter=count_setter,
        revision_request_setter=request_setter,
    )
    args = tool.validate_arguments(
        {
            "reason": "blueprint missed a lane",
            "fields_to_reconsider": ["independent_workstreams_count"],
        }
    )
    result = asyncio.run(tool.execute(args, _ctx()))
    assert result.success is not False
    assert captured_count == [1]
    assert captured_request, "revision_request_setter must be called"
    persisted = captured_request[-1]
    assert isinstance(persisted, dict)
    assert persisted.get("reason") == "blueprint missed a lane"
