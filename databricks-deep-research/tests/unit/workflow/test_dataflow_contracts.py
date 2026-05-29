"""Unit tests for the build-time dataflow checker (dataflow_contracts.py).

Grows story-by-story: US-DF2 (data model + effective_reads), US-DF3 (Pass A
dangling reads), US-DF5 (control edges + Pass B dead stores + fixpoint).
"""
from __future__ import annotations

from databricks_deep_research.agents.config import AgentNodeConfig
from databricks_deep_research.workflow.dataflow_contracts import (
    DataflowReport,
    Diagnostic,
    dangling_reads,
    detect_dead_stores,
    effective_reads,
)
from databricks_deep_research.workflow.definition import (
    NodeType,
    WorkflowDefinition,
    WorkflowNode,
)

# --- US-DF2: data model + effective_reads --------------------------------------


def test_diagnostic_and_report_value_objects() -> None:
    d = Diagnostic(message="x", severity="warning")
    assert d.severity == "warning"
    rep = DataflowReport()
    rep.errors.append("e")
    rep.warnings.append("w")
    assert rep.errors == ["e"] and rep.warnings == ["w"]


def test_effective_reads_union_template_vars_input_keys() -> None:
    cfg = AgentNodeConfig(
        subtype="researcher",
        output_key="findings",
        input_keys=["query", "current_step"],
        user_prompt_template="Investigate {query} for {focus_area}.",
        system_prompt="",
    )
    reads = effective_reads(cfg)
    assert {"query", "current_step", "focus_area"} <= reads


def test_effective_reads_excludes_loop_local_variable() -> None:
    # The {%for s in sources_list%} iterable 'sources_list' IS a read; the loop
    # var 's' (matched as {s} in the body) is a local binding, NOT a state read.
    cfg = AgentNodeConfig(
        subtype="synthesizer",
        output_key="report",
        input_keys=["query"],
        system_prompt="",
        user_prompt_template="{%for s in sources_list%}{s}{%endfor%}",
    )
    reads = effective_reads(cfg)
    assert "sources_list" in reads
    assert "s" not in reads


# --- US-DF3: Pass A dangling reads ---------------------------------------------


def _agent(
    node_id: str,
    *,
    subtype: str = "researcher",
    input_keys: tuple[str, ...] = (),
    output_key: str = "output",
    system_prompt: str = "",
    user_prompt_template: str = "",
) -> WorkflowNode:
    return WorkflowNode(
        id=node_id,
        type=NodeType.agent,
        label=node_id,
        config={
            "subtype": subtype,
            "input_keys": list(input_keys),
            "output_key": output_key,
            "system_prompt": system_prompt,
            "user_prompt_template": user_prompt_template,
        },
    )


def _agent_cfg(
    subtype: str,
    *,
    input_keys: tuple[str, ...] = (),
    output_key: str = "output",
) -> dict:
    return {
        "subtype": subtype,
        "input_keys": list(input_keys),
        "output_key": output_key,
        "system_prompt": "",
        "user_prompt_template": "",
    }


def _seq(node_id: str, children: list[WorkflowNode]) -> WorkflowNode:
    return WorkflowNode(id=node_id, type=NodeType.sequence, label=node_id, children=children)


def _par(node_id: str, children: list[WorkflowNode]) -> WorkflowNode:
    return WorkflowNode(id=node_id, type=NodeType.parallel, label=node_id, children=children)


def _defn(
    root: WorkflowNode,
    *,
    required_inputs: tuple[str, ...] = ("query",),
    output_keys: tuple[str, ...] = ("findings",),
) -> WorkflowDefinition:
    return WorkflowDefinition(
        id="t",
        name="t",
        root=root,
        required_inputs=list(required_inputs),
        output_keys=list(output_keys),
    )


def test_sequence_read_resolves_to_prior_sibling() -> None:
    defn = _defn(
        _seq(
            "root",
            [
                _agent("coordinator", subtype="coordinator", input_keys=("query",), output_key="coordination"),
                _agent("researcher", input_keys=("query", "coordination"), output_key="findings"),
            ],
        )
    )
    assert dangling_reads(defn) == []


def test_unproduced_key_is_dangling() -> None:
    defn = _defn(
        _seq("root", [_agent("researcher", input_keys=("query", "nope"), output_key="findings")])
    )
    assert any("nope" in d for d in dangling_reads(defn))


def test_runtime_injected_key_is_not_dangling() -> None:
    defn = _defn(
        _seq(
            "root",
            [
                _agent(
                    "reflector",
                    subtype="reflector",
                    system_prompt="{plan_summary}",
                    input_keys=("query",),
                    output_key="reflection",
                )
            ],
        ),
        output_keys=("reflection",),
    )
    assert dangling_reads(defn) == []  # plan_summary is runtime-injected


def test_parallel_siblings_are_hidden_from_each_other() -> None:
    defn = _defn(
        _par(
            "root",
            [
                _agent("left", input_keys=("query",), output_key="a"),
                _agent("right", input_keys=("query", "a"), output_key="b"),
            ],
        ),
        output_keys=("a", "b"),
    )
    assert any("a" in d and "right" in d for d in dangling_reads(defn))


def test_current_step_resolves_inside_pae_but_dangles_outside() -> None:
    pae = WorkflowNode(
        id="pae",
        type=NodeType.plan_and_execute,
        label="pae",
        config={
            "planner": _agent_cfg("planner", input_keys=("query",), output_key="research_plan"),
            "body": _agent(
                "researcher",
                input_keys=("query", "current_step", "research_plan"),
                output_key="findings",
            ),
            "evaluator": _agent_cfg("reflector", input_keys=("query", "findings"), output_key="evaluation"),
            "items_path": "steps",
            "item_state_key": "current_step",
        },
    )
    inside = _defn(_seq("root", [pae]))
    assert dangling_reads(inside) == []  # current_step is bound inside the PAE

    outside = _defn(
        _seq("root", [_agent("researcher", input_keys=("query", "current_step"), output_key="findings")])
    )
    assert any("current_step" in d for d in dangling_reads(outside))


# --- US-DF5: control edges + Pass B dead stores + loop-carry fixpoint ----------


def test_unread_data_output_is_warning() -> None:
    defn = _defn(
        _seq(
            "root",
            [
                _agent("researcher", input_keys=("query",), output_key="findings"),
                _agent("aux", input_keys=("query",), output_key="orphan"),
            ],
        )
    )  # default output_keys=("findings",) -> 'orphan' is non-terminal and unread
    diags = detect_dead_stores(defn)
    assert any("orphan" in d.message and d.severity == "warning" for d in diags)


def test_terminal_output_is_exempt() -> None:
    defn = _defn(_seq("root", [_agent("researcher", input_keys=("query",), output_key="findings")]))
    assert not any("findings" in d.message for d in detect_dead_stores(defn))


def test_pool_roundtrip_is_not_dead() -> None:
    writer = WorkflowNode(
        id="w",
        type=NodeType.agent,
        label="w",
        config={
            "subtype": "researcher",
            "input_keys": ["query"],
            "output_key": "findings",
            "pool_writes": [{"pool": "observations", "extract": "findings"}],
        },
    )
    reader = WorkflowNode(
        id="s",
        type=NodeType.agent,
        label="s",
        config={
            "subtype": "synthesizer",
            "input_keys": ["query"],
            "output_key": "report",
            "pool_inject": [{"pool": "observations", "threshold": 0}],
        },
    )
    defn = _defn(_seq("root", [writer, reader]), output_keys=("report",))
    assert not any("observations" in d.message for d in detect_dead_stores(defn))


def test_pae_evaluation_consumed_as_control_is_not_dead() -> None:
    pae = WorkflowNode(
        id="pae",
        type=NodeType.plan_and_execute,
        label="pae",
        config={
            "planner": _agent_cfg("planner", input_keys=("query",), output_key="research_plan"),
            "body": _agent(
                "researcher",
                input_keys=("query", "current_step", "research_plan"),
                output_key="findings",
            ),
            "evaluator": _agent_cfg("reflector", input_keys=("query", "findings"), output_key="evaluation"),
            "items_path": "steps",
            "item_state_key": "current_step",
        },
    )
    defn = _defn(_seq("root", [pae]), output_keys=("findings",))
    # 'evaluation' is consumed as a control edge (the loop branches on it) -> not dead.
    assert not any("evaluation" in d.message for d in detect_dead_stores(defn))


def test_loop_carried_read_is_not_dangling() -> None:
    loop = WorkflowNode(
        id="loop",
        type=NodeType.loop,
        label="loop",
        config={"until": {"type": "state", "key": "done", "operator": "exists"}},
        children=[
            _agent("stepA", input_keys=("query", "carry"), output_key="done"),
            _agent("stepB", input_keys=("query",), output_key="carry"),
        ],
    )
    defn = _defn(_seq("root", [loop]), output_keys=("done",))
    # 'carry' is produced by stepB but read by the earlier stepA — loop-carried,
    # so the 2-pass fixpoint must NOT flag it dangling.
    assert not any("carry" in d for d in dangling_reads(defn))
