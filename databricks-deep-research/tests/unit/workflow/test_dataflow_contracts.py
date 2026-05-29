"""Unit tests for the build-time dataflow checker (dataflow_contracts.py).

Grows story-by-story: US-DF2 (data model + effective_reads), US-DF3 (Pass A
dangling reads), US-DF5 (control edges + Pass B dead stores + fixpoint).
"""
from __future__ import annotations

from databricks_deep_research.agents.config import AgentNodeConfig
from databricks_deep_research.workflow.dataflow_contracts import (
    DataflowReport,
    Diagnostic,
    effective_reads,
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
