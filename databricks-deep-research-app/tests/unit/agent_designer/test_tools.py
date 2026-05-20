import pytest
from pydantic import ValidationError

from deep_research.agent_designer.tools import (
    CHAT_TOOLS,
    AddBlockArgs,
    DeclareToolArgs,
    ProposeWorkflowArgs,
    parse_tool_args,
)


def test_chat_tools_count() -> None:
    assert len(CHAT_TOOLS) >= 12


def test_chat_tools_shape() -> None:
    for t in CHAT_TOOLS:
        assert t["type"] == "function"
        fn = t["function"]
        assert isinstance(fn["name"], str) and fn["name"]
        assert isinstance(fn["description"], str) and fn["description"]
        assert "parameters" in fn
        assert fn["parameters"]["type"] == "object"


def test_chat_tools_unique_names() -> None:
    names = [t["function"]["name"] for t in CHAT_TOOLS]
    assert len(names) == len(set(names))


def test_chat_tools_includes_required_set() -> None:
    names = {t["function"]["name"] for t in CHAT_TOOLS}
    required = {
        "propose_workflow",
        "add_block",
        "update_block",
        "delete_block",
        "move_block",
        "declare_tool",
        "remove_tool",
        "bind_tool_to_block",
        "set_model_tier",
        "discover_sources",
        "list_node_types",
        "list_tool_kinds",
        "list_modes",
        "validate",
    }
    assert required.issubset(names), f"missing: {required - names}"


def test_pydantic_extra_forbidden() -> None:
    with pytest.raises(ValidationError):
        AddBlockArgs(
            parent_path="root", node_type="agent", label="x", config={}, extra_field="boom"
        )


def test_parse_tool_args_dispatches_correctly() -> None:
    args = parse_tool_args("declare_tool", {"kind": "web_search", "name": "ws", "config": {}})
    assert isinstance(args, DeclareToolArgs)
    assert args.kind == "web_search"
    assert args.name == "ws"


def test_propose_workflow_args_accepts_structured_design_brief() -> None:
    args = parse_tool_args(
        "propose_workflow",
        {
            "intent": "Build an investment research workflow",
            "design_brief": {
                "domain": "Investment Research",
                "research_lanes": ["Valuation", "Earnings calls"],
                "required_outputs": ["Bull and bear thesis"],
                "quality_gates": ["Reject generic company summaries"],
            },
        },
    )
    assert isinstance(args, ProposeWorkflowArgs)
    assert args.design_brief is not None
    assert args.design_brief.domain == "Investment Research"
    # research_lanes is now list[LaneSpec]; legacy list[str] auto-coerces with
    # description=string and empty system_prompt.
    assert [lane.description for lane in args.design_brief.research_lanes] == [
        "Valuation",
        "Earnings calls",
    ]
    assert all(lane.system_prompt == "" for lane in args.design_brief.research_lanes)


def test_propose_workflow_args_accepts_per_lane_system_prompt() -> None:
    args = parse_tool_args(
        "propose_workflow",
        {
            "intent": "Build an investment research workflow",
            "design_brief": {
                "domain": "Investment Research",
                "research_lanes": [
                    {
                        "description": "Valuation",
                        "system_prompt": (
                            "Compute P/E, EV/EBITDA, EV/Sales vs historical "
                            "and peer ranges; flag stretched multiples."
                        ),
                    },
                    {"description": "Earnings calls"},
                ],
            },
        },
    )
    assert isinstance(args, ProposeWorkflowArgs)
    assert args.design_brief is not None
    lane_0, lane_1 = args.design_brief.research_lanes
    assert lane_0.description == "Valuation"
    assert "P/E" in lane_0.system_prompt
    assert lane_1.description == "Earnings calls"
    assert lane_1.system_prompt == ""


def test_parse_tool_args_unknown_tool_raises() -> None:
    with pytest.raises(KeyError):
        parse_tool_args("not_a_real_tool", {})


def test_parse_tool_args_invalid_args_raises() -> None:
    with pytest.raises(ValidationError):
        parse_tool_args("set_model_tier", {"node_path": "root", "tier": ""})


def test_set_model_tier_args_accepts_configured_tier_name() -> None:
    from deep_research.agent_designer.tools import SetModelTierArgs

    valid = SetModelTierArgs(node_path="root", tier="bulk_analysis")
    assert valid.tier == "bulk_analysis"


def test_add_block_args_validates_node_type_literal() -> None:
    from deep_research.agent_designer.tools import AddBlockArgs

    with pytest.raises(ValidationError):
        AddBlockArgs(parent_path="root", node_type="not_a_real_type", label="x", config={})
