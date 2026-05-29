"""US-DF7: adversarial false-negative coverage.

Inject one broken workflow per defect class and assert the checker flags each.
This is the measurement the Phase 5 (free-form synthesis) gate requires — evidence
that the checker provably catches dangling reads, dead stores, dangling control
reads, and dangling tool inputs (not just that it stays silent on good input).
"""
from __future__ import annotations

from databricks_deep_research.workflow.dataflow_contracts import (
    detect_dead_stores,
    validate_dataflow_contracts,
)
from databricks_deep_research.workflow.definition import (
    NodeType,
    WorkflowDefinition,
    WorkflowNode,
)


def _agent(
    node_id: str,
    *,
    subtype: str = "researcher",
    input_keys: tuple[str, ...] = (),
    output_key: str = "output",
) -> WorkflowNode:
    return WorkflowNode(
        id=node_id,
        type=NodeType.agent,
        label=node_id,
        config={
            "subtype": subtype,
            "input_keys": list(input_keys),
            "output_key": output_key,
            "system_prompt": "",
            "user_prompt_template": "",
        },
    )


def _seq(node_id: str, children: list[WorkflowNode]) -> WorkflowNode:
    return WorkflowNode(id=node_id, type=NodeType.sequence, label=node_id, children=children)


def _defn(
    root: WorkflowNode,
    *,
    output_keys: tuple[str, ...] = ("findings",),
) -> WorkflowDefinition:
    return WorkflowDefinition(
        id="t", name="t", root=root, required_inputs=["query"], output_keys=list(output_keys)
    )


def test_injected_dangling_data_read_is_error_in_strict() -> None:
    defn = _defn(_seq("root", [_agent("r", input_keys=("query", "ghost"), output_key="findings")]))
    report = validate_dataflow_contracts(defn, strict=True)
    assert any("ghost" in e for e in report.errors)


def test_injected_dead_data_store_is_warning() -> None:
    defn = _defn(
        _seq(
            "root",
            [
                _agent("r", input_keys=("query",), output_key="findings"),
                _agent("aux", input_keys=("query",), output_key="orphan"),
            ],
        )
    )
    diags = detect_dead_stores(defn)
    assert any("orphan" in d.message and d.severity == "warning" for d in diags)


def test_injected_dangling_control_read_is_error_in_strict() -> None:
    loop = WorkflowNode(
        id="loop",
        type=NodeType.loop,
        label="loop",
        config={"until": {"type": "state", "key": "ghost_control", "operator": "exists"}},
        children=[_agent("step", input_keys=("query",), output_key="findings")],
    )
    defn = _defn(_seq("root", [loop]))
    report = validate_dataflow_contracts(defn, strict=True)
    assert any("ghost_control" in e for e in report.errors)


def test_injected_dangling_tool_input_is_error_in_strict() -> None:
    tool = WorkflowNode(
        id="t1",
        type=NodeType.tool,
        label="t1",
        config={
            "ref": {"name": "web"},  # type defaults to "builtin"
            "input_mapping": {"q": "ghost_state"},  # values are outer state keys
            "output_key": "tool_result",
        },
    )
    defn = _defn(
        _seq("root", [_agent("r", input_keys=("query",), output_key="findings"), tool]),
        output_keys=("findings",),
    )
    report = validate_dataflow_contracts(defn, strict=True)
    assert any("ghost_state" in e for e in report.errors)
