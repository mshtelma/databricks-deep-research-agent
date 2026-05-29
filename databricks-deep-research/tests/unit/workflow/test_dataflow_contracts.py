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


def test_effective_reads_excludes_runtime_injected() -> None:
    cfg = AgentNodeConfig(
        subtype="reflector",
        output_key="reflection",
        input_keys=["findings"],  # 'findings' is a real produced key, not runtime-injected
        system_prompt="{plan_summary} {all_observations}",
        user_prompt_template="",
    )
    reads = effective_reads(cfg, exclude_runtime=True)
    # plan_summary/all_observations (and query, if present) are runtime-injected.
    assert "plan_summary" not in reads and "all_observations" not in reads
    # A genuine (non-runtime) read survives.
    assert "findings" in reads


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
