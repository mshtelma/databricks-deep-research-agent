"""Static validation tests for workflow condition contracts."""

from __future__ import annotations

import pytest

from databricks_deep_research.errors import WorkflowValidationError
from databricks_deep_research.workflow.loader import load_workflow_from_dict
from databricks_deep_research.workflow.runtime_keys import RUNTIME_INJECTED_KEYS


def _agent(node_id: str, *, subtype: str = "researcher", output_key: str = "out") -> dict:
    return {
        "id": node_id,
        "type": "agent",
        "label": node_id,
        "config": {"subtype": subtype, "output_key": output_key, "output_format": "json"},
    }


def _conditional(condition: dict, *, node_id: str = "router") -> dict:
    return {
        "id": node_id,
        "type": "conditional",
        "label": "Router",
        "config": {"conditions": [condition], "default_branch": 1},
        "children": [
            _agent("branch-a", output_key="branch_a"),
            _agent("branch-b", output_key="branch_b"),
        ],
    }


def _workflow(root: dict, *, output_keys: list[str] | None = None) -> dict:
    return {
        "id": "condition-contract-test",
        "name": "Condition Contract Test",
        "version": 1,
        "required_inputs": ["query"],
        "output_keys": output_keys or ["branch_a", "branch_b", "findings"],
        "root": root,
    }


def _load_errors(workflow: dict) -> list[str]:
    with pytest.raises(WorkflowValidationError) as exc_info:
        load_workflow_from_dict(workflow)
    return exc_info.value.errors


def test_rejects_missing_top_level_condition_key() -> None:
    errors = _load_errors(
        _workflow(
            _conditional(
                {"type": "state", "key": "missing", "operator": "eq", "value": "x"}
            )
        )
    )

    assert any("missing" in error and "Available top-level keys" in error for error in errors)


def test_rejects_nested_path_under_unknown_schema() -> None:
    errors = _load_errors(
        _workflow(
            _conditional(
                {"type": "state", "key": "query.intent", "operator": "eq", "value": "x"}
            )
        )
    )

    assert any("descends into unknown state binding 'query'" in error for error in errors)


def test_accepts_declared_nested_builtin_output_field() -> None:
    root = {
        "id": "root",
        "type": "sequence",
        "label": "Root",
        "config": {},
        "children": [
            _agent("coordinator", subtype="coordinator", output_key="coordination"),
            _conditional(
                {
                    "type": "state",
                    "key": "coordination.complexity",
                    "operator": "eq",
                    "value": "complex",
                }
            ),
        ],
    }

    assert load_workflow_from_dict(_workflow(root)).root.id == "root"


def test_rejects_missing_nested_builtin_output_field() -> None:
    root = {
        "id": "root",
        "type": "sequence",
        "label": "Root",
        "config": {},
        "children": [
            _agent("coordinator", subtype="coordinator", output_key="coordination"),
            _conditional(
                {
                    "type": "state",
                    "key": "coordination.routing_lane",
                    "operator": "eq",
                    "value": "lane_1",
                }
            ),
        ],
    }

    errors = _load_errors(_workflow(root))
    assert any("coordination.routing_lane" in error for error in errors)
    assert any("complexity" in error for error in errors)


def test_rejects_current_step_lane_in_default_plan_and_execute_body() -> None:
    condition = {
        "type": "state",
        "key": "current_step.lane",
        "operator": "eq",
        "value": "lane_1",
    }
    root = {
        "id": "plan-exec",
        "type": "plan_and_execute",
        "label": "Plan and Execute",
        "config": {
            "planner": {
                "subtype": "planner",
                "output_key": "research_plan",
                "output_format": "json",
            },
            "items_path": "steps",
            "item_state_key": "current_step",
            "body": _conditional(condition, node_id="research-lane-router"),
        },
    }

    errors = _load_errors(_workflow(root))
    assert any("current_step.lane" in error for error in errors)
    assert any("id" in error and "user_prompt_template" in error for error in errors)


def test_accepts_current_step_declared_field_in_plan_and_execute_body() -> None:
    condition = {
        "type": "state",
        "key": "current_step.id",
        "operator": "eq",
        "value": "step-1",
    }
    root = {
        "id": "plan-exec",
        "type": "plan_and_execute",
        "label": "Plan and Execute",
        "config": {
            "planner": {
                "subtype": "planner",
                "output_key": "research_plan",
                "output_format": "json",
            },
            "items_path": "steps",
            "item_state_key": "current_step",
            "body": _conditional(condition),
        },
    }

    assert load_workflow_from_dict(_workflow(root)).root.id == "plan-exec"


def test_rejects_enum_value_not_declared_by_output_schema() -> None:
    root = {
        "id": "root",
        "type": "sequence",
        "label": "Root",
        "config": {},
        "children": [
            {
                "id": "classifier",
                "type": "agent",
                "label": "Classifier",
                "config": {
                    "subtype": "classifier",
                    "output_key": "intent",
                    "output_schema": {
                        "type": "object",
                        "required": ["intent_type"],
                        "properties": {
                            "intent_type": {
                                "type": "string",
                                "enum": ["simple", "complex"],
                            }
                        },
                    },
                },
            },
            _conditional(
                {
                    "type": "state",
                    "key": "intent.intent_type",
                    "operator": "eq",
                    "value": "unknown",
                }
            ),
        ],
    }

    errors = _load_errors(_workflow(root))
    assert any("allowed values" in error and "unknown" in error for error in errors)


def test_accepts_enum_value_declared_by_output_schema() -> None:
    root = {
        "id": "root",
        "type": "sequence",
        "label": "Root",
        "config": {},
        "children": [
            {
                "id": "classifier",
                "type": "agent",
                "label": "Classifier",
                "config": {
                    "subtype": "classifier",
                    "output_key": "intent",
                    "output_schema": {
                        "type": "object",
                        "required": ["intent_type"],
                        "properties": {
                            "intent_type": {
                                "type": "string",
                                "enum": ["simple", "complex"],
                            }
                        },
                    },
                },
            },
            _conditional(
                {
                    "type": "state",
                    "key": "intent.intent_type",
                    "operator": "eq",
                    "value": "complex",
                }
            ),
        ],
    }

    assert load_workflow_from_dict(_workflow(root)).root.id == "root"


def test_rejects_llm_condition_template_variable_missing_from_state_scope() -> None:
    root = _conditional(
        {
            "type": "llm",
            "prompt_template": "Should this use {missing_context}?",
            "expected_output": "yes",
        }
    )

    errors = _load_errors(_workflow(root))
    assert any("missing_context" in error for error in errors)


def _load_ok(workflow: dict) -> None:
    """Assert the workflow passes build-time validation (no error raised)."""
    load_workflow_from_dict(workflow)


def test_accepts_subworkflow_inline_config() -> None:
    # api/compile.py embeds a full WorkflowDefinition dump under config.inline for
    # SubAgent-derived subworkflow nodes; SubworkflowNodeConfig must accept it.
    inline = {
        "id": "inner",
        "name": "Inner",
        "version": 1,
        "required_inputs": ["query"],
        "output_keys": ["output"],
        "run_as": "caller",
        "root": _agent("inner-agent", output_key="output"),
    }
    sub = {
        "id": "sub",
        "type": "subworkflow",
        "label": "Sub",
        "config": {"ref": "parent.inner", "inline": inline, "pool_mode": "inherit"},
    }
    root = {"id": "root", "type": "sequence", "label": "Root", "config": {}, "children": [sub]}
    _load_ok(_workflow(root, output_keys=["subworkflow_result"]))


def test_accepts_loop_until_key_produced_in_body() -> None:
    # A loop's until-condition is validated against the post-body scope, so a
    # top-level key produced by the loop body is in scope.
    loop = {
        "id": "loop",
        "type": "loop",
        "label": "Loop",
        "config": {
            "min_iterations": 1,
            "max_iterations": 3,
            "until": {"type": "state", "key": "done", "operator": "exists"},
        },
        "children": [_agent("worker", output_key="done")],
    }
    _load_ok(_workflow(loop, output_keys=["done"]))


def test_accepts_loop_until_nested_boolean_produced_in_body() -> None:
    # Mirrors designer_workflow.yaml: a tool extracts a typed boolean into a
    # top-level key and the loop until reads the nested field with eq/true.
    extractor = {
        "id": "extract",
        "type": "tool",
        "label": "Extract flag",
        "config": {
            "ref": {"type": "builtin", "name": "extract_flag"},
            "output_key": "flag",
            "output_schema": {
                "type": "object",
                "required": ["flag"],
                "properties": {"flag": {"type": "boolean"}},
            },
        },
    }
    loop = {
        "id": "loop",
        "type": "loop",
        "label": "Loop",
        "config": {
            "min_iterations": 1,
            "max_iterations": 3,
            "until": {"type": "state", "key": "flag.flag", "operator": "eq", "value": True},
        },
        "children": [extractor],
    }
    _load_ok(_workflow(loop, output_keys=["flag"]))


def test_accepts_condition_reading_declared_runtime_injected_key() -> None:
    # A condition may read a per-workflow declared runtime_injected_key; the
    # validator seeds these into the root scope (parity with the dataflow check).
    workflow = {
        "id": "rt-declared",
        "name": "RT declared",
        "version": 1,
        "required_inputs": ["query"],
        "runtime_injected_keys": ["external_ctx"],
        "output_keys": ["branch_a", "branch_b"],
        "root": _conditional({"type": "state", "key": "external_ctx", "operator": "exists"}),
    }
    _load_ok(workflow)


def test_accepts_condition_reading_framework_global_runtime_key() -> None:
    # A condition may read a framework-global runtime-injected key (harness
    # template var) without declaring it.
    global_key = "background"
    assert global_key in RUNTIME_INJECTED_KEYS  # guard: keep the test meaningful
    _load_ok(
        _workflow(
            _conditional({"type": "state", "key": global_key, "operator": "exists"}),
            output_keys=["branch_a", "branch_b"],
        )
    )


def test_rejects_condition_reading_undeclared_non_runtime_key() -> None:
    # Negative control: seeding runtime keys must NOT admit arbitrary undeclared,
    # un-produced keys.
    made_up = "totally_made_up_key"
    assert made_up not in RUNTIME_INJECTED_KEYS
    errors = _load_errors(
        _workflow(
            _conditional({"type": "state", "key": made_up, "operator": "exists"}),
            output_keys=["branch_a", "branch_b"],
        )
    )
    assert any(made_up in error and "Available top-level keys" in error for error in errors)
