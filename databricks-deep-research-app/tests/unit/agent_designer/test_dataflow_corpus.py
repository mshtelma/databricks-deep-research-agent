"""US-DF6: dataflow false-positive gate.

Builds each designer topology through the REAL loader path and asserts the
dataflow checker emits ZERO warnings on currently-generated output. This is the
gate that must pass before any strict flip (DATAFLOW_CHECK_STRICT).

Parametrized by topology (the structural variation that actually shapes the AST);
domain only changes prompt prose, not dataflow shape, so it is not part of the
matrix (and a hardcoded domain list would violate the no-hardcoded-domains rule).
"""
from __future__ import annotations

import pytest
from databricks_deep_research.workflow.dataflow_contracts import (
    validate_dataflow_contracts,
)
from databricks_deep_research.workflow.loader import load_workflow_from_dict

from deep_research.agent_designer.designer_types import (
    ToolDeclarationSpec,
    ToolPlan,
    WorkflowDesignBrief,
)
from deep_research.agent_designer.workflow_builder import build_web_research_workflow

TOPOLOGIES = ["single_agent", "parallel_lanes", "plan_and_execute"]


def _brief(topology: str) -> WorkflowDesignBrief:
    return WorkflowDesignBrief(
        workflow_name=f"corpus-{topology}",
        topology=topology,
        research_lanes=[
            {
                "description": "Web research lane.",
                "user_prompt_template": "Research {query}.",
            }
        ],
        tool_plan=ToolPlan(tools=[ToolDeclarationSpec(name="web", kind="web_search")]),
    )


@pytest.mark.parametrize("topology", TOPOLOGIES)
def test_generated_workflow_has_no_dataflow_warnings(topology: str) -> None:
    wf_dict = build_web_research_workflow(
        intent="corpus probe", design_brief=_brief(topology)
    )
    defn = load_workflow_from_dict(wf_dict)
    report = validate_dataflow_contracts(defn, strict=False)
    assert report.warnings == [], f"{topology} produced dataflow warnings: {report.warnings}"
