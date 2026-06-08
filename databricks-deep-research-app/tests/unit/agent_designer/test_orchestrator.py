"""Tests for DesignerChatOrchestrator using a fake LLM client.

NOTE — Most of the async tests here exercise the LEGACY hand-coded
LLM-tool-call loop that lived inside ``DesignerChatOrchestrator.run_turn``
(orchestrator.py:487-620 before W5c). That loop was replaced by a
thin framework-workflow shim in US-11 (W5c); the shim calls
``WorkflowRunner.stream(designer_workflow.yaml)`` and no longer dispatches
tool calls through ``LLMClientProto.stream``.

The async tests below stay in the codebase as historical references
documenting the legacy contract, but they CANNOT run against the new
shim — fake LLMs that satisfy ``LLMClientProto`` don't provide the
``_ensure_fresh_client`` interface that the framework adapter needs.
We skip them at the module level so the agent_designer unit suite still
passes after the shim landed; the new behavior is covered by
``test_route_shim.py``. The synchronous tests that only exercise private
helper functions (``_propose_initial_ast``, ``WorkflowDesignBrief`` flow)
continue to run as-is.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, cast

import pytest

from deep_research.agent_designer.discovery import DesignerDiscoveryAdapter
from deep_research.agent_designer.orchestrator import (
    MAX_AST_BYTES,
    MAX_DESIGNER_TOOL_ROUNDS,
    MAX_MESSAGES,
    DesignerChatOrchestrator,
    DoneEvent,
    ErrorEvent,
    LLMStreamChunk,
    LLMToolCall,
    MessageEvent,
    MutationProposedEvent,
    RequestTooLargeError,
    ToolCallEvent,
    ToolResultEvent,
    _mutation_event_for_ast_change,
)
from deep_research.agent_designer.workflow_builder import build_web_research_workflow

# Tests below that call ``orchestrator.run_turn`` exercise the legacy
# loop (replaced by the framework shim in US-11 / W5c). They cannot work
# against the new shim without a real framework LLM, so we skip them
# collectively. New behavior is covered by ``test_route_shim.py``.
_LEGACY_LOOP_SKIP = pytest.mark.skip(
    reason=(
        "Legacy hand-coded LLM-tool-call loop replaced by framework workflow "
        "shim in US-11 (W5c). See test_route_shim.py for the new contract."
    )
)

# ---- Fake helpers ----


class _FakeLLM:
    def __init__(self, chunks: list[LLMStreamChunk] | list[list[LLMStreamChunk]]) -> None:
        self._rounds: list[list[LLMStreamChunk]]
        if chunks and isinstance(chunks[0], LLMStreamChunk):
            self._rounds = [cast(list[LLMStreamChunk], chunks)]
        else:
            self._rounds = cast(list[list[LLMStreamChunk]], chunks)
        self.calls: list[tuple[list[Any], list[Any]]] = []

    async def stream(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> AsyncIterator[LLMStreamChunk]:
        round_index = len(self.calls)
        self.calls.append((messages, tools))
        chunks = (
            self._rounds[round_index]
            if round_index < len(self._rounds)
            else [LLMStreamChunk(finish=True)]
        )
        for c in chunks:
            yield c


class _FakeDiscoveryResponse:
    def __init__(self, sources: list[Any]) -> None:
        self.sources = sources


class _FakeDiscoveryService:
    def __init__(self, sources: list[Any] | None = None) -> None:
        self._sources = sources or []

    async def discover_all(
        self,
        user_id: str,
        user_token: str | None = None,
        **kwargs: Any,
    ) -> _FakeDiscoveryResponse:
        return _FakeDiscoveryResponse(self._sources)


def _adapter(sources: list[Any] | None = None) -> DesignerDiscoveryAdapter:
    return DesignerDiscoveryAdapter(_FakeDiscoveryService(sources))


async def _collect(orch: DesignerChatOrchestrator, **kwargs: Any) -> list[Any]:
    return [ev async for ev in orch.run_turn(**kwargs)]


def _agent_configs(ast: dict[str, Any]) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []

    def walk(node: dict[str, Any]) -> None:
        config = node.get("config")
        if node.get("type") == "agent" and isinstance(config, dict):
            configs.append(config)
        if node.get("type") == "plan_and_execute" and isinstance(config, dict):
            planner = config.get("planner")
            evaluator = config.get("evaluator")
            body = config.get("body")
            if isinstance(planner, dict):
                configs.append(planner)
            if isinstance(evaluator, dict):
                configs.append(evaluator)
            if isinstance(body, dict):
                walk(body)
        for child in node.get("children") or []:
            if isinstance(child, dict):
                walk(child)

    walk(ast["root"])
    return configs


def _walk_nodes(node: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = [node]
    config = node.get("config")
    if isinstance(config, dict):
        body = config.get("body")
        if isinstance(body, dict):
            nodes.extend(_walk_nodes(body))
    for child in node.get("children") or []:
        if isinstance(child, dict):
            nodes.extend(_walk_nodes(child))
    return nodes


# ---- Tests ----


def test_research_workflow_scaffold_has_no_finance_specific_defaults() -> None:
    """Designer scaffolds must stay domain-neutral; domains belong in user-authored workflow text."""
    workflow = build_web_research_workflow(
        "Research user-provided launch ideas and produce a readiness brief",
        "Launch readiness workflow",
    )
    serialized = repr(workflow).lower()

    forbidden_defaults = [
        "ticker",
        "investment",
        "bull case",
        "bear case",
        "portfolio",
    ]
    for term in forbidden_defaults:
        assert term not in serialized


def test_mutation_event_normalizes_researcher_prompt_contract() -> None:
    raw_ast = {
        "root": {
            "id": "lane-researcher",
            "type": "agent",
            "label": "Corpus Lane",
            "config": {
                "subtype": "researcher",
                "model_tier": "analytical",
                "tools": ["table_search", "table_read"],
                "user_prompt_template": (
                    "## Investigation Brief\n\n"
                    "You are investigating: **{query}**\n\n"
                    "### Sub-questions\n"
                    "1. Which corpus records address the request?\n"
                    "2. What exact text evidence supports the answer?\n"
                    "3. What structured rows support numeric claims?\n"
                    "4. What calculations are needed?\n"
                    "5. What evidence gaps remain?\n\n"
                    "### Required output structure\n"
                    "- **Evidence-backed findings**: source-backed facts.\n"
                    "- **Coverage and conflicts**: agreements, disagreements, and gaps.\n"
                    "- **Unsupported items**: unavailable or weakly supported claims.\n\n"
                    "### Definition of done\n"
                    "Mark missing evidence as \"Data unavailable\" -- DO NOT improvise."
                ),
            },
            "children": [],
        },
        "tools": [
            {"kind": "table_search", "name": "table_search", "config": {}},
            {"kind": "table_read", "name": "table_read", "config": {}},
        ],
        "pools": [],
    }

    event = _mutation_event_for_ast_change(
        tool_name="propose_workflow",
        tool_call_id="test",
        raw_ast=raw_ast,
        last_ast_seen={},
        normalization_fixes=[],
    )

    assert event is not None
    template = event.new_ast["root"]["config"]["user_prompt_template"]
    assert "### Search strategy" in template
    assert "available corpus retrieval tools" in template
    assert any(
        fix.get("kind") == "researcher_prompt_contract"
        for fix in event.normalization_fixes
    )


@_LEGACY_LOOP_SKIP
async def test_simple_propose_and_done() -> None:
    """propose_workflow yields ToolCallEvent, MutationProposedEvent, DoneEvent."""
    chunks = [
        LLMStreamChunk(
            tool_call=LLMToolCall(
                id="t1", name="propose_workflow", arguments={"intent": "build a researcher"}
            )
        ),
        LLMStreamChunk(finish=True),
    ]
    orch = DesignerChatOrchestrator(_FakeLLM(chunks), _adapter())
    events = await _collect(
        orch,
        messages=[{"role": "user", "content": "go"}],
        current_ast=None,
        session_id=None,
        user_token="t",
    )
    types = [type(e).__name__ for e in events]
    assert "ToolCallEvent" in types
    assert "MutationProposedEvent" in types
    assert types[-1] == "DoneEvent"
    mutation = [e for e in events if isinstance(e, MutationProposedEvent)][0]
    assert mutation.tool_name == "propose_workflow"


@_LEGACY_LOOP_SKIP
async def test_propose_research_workflow_scaffolds_web_research_and_synthesis() -> None:
    """Research intents start from a useful web-research pipeline, not one node.

    Pins the legacy ``plan_and_execute`` topology shape (planner_guidance +
    synthesis_metadata + lane_router body) by explicitly requesting that
    topology in the design brief. The DEFAULT topology is now
    ``parallel_lanes``; tests of the plan_and_execute scaffold must opt in.
    """
    intent = "create a simple research workflow using web search and summarize it"
    chunks = [
        LLMStreamChunk(
            tool_call=LLMToolCall(
                id="t1",
                name="propose_workflow",
                arguments={
                    "intent": intent,
                    "design_brief": {"topology": "plan_and_execute"},
                },
            )
        ),
        LLMStreamChunk(finish=True),
    ]
    orch = DesignerChatOrchestrator(_FakeLLM(chunks), _adapter())

    events = await _collect(
        orch,
        messages=[{"role": "user", "content": "go"}],
        current_ast=None,
        session_id=None,
        user_token="t",
    )

    mutation = [e for e in events if isinstance(e, MutationProposedEvent)][0]
    ast = mutation.new_ast
    children = ast["root"]["children"]

    assert mutation.validation_errors == []
    assert mutation.summary is not None
    assert ast["root"]["type"] == "sequence"
    assert ast["output_keys"] == ["report"]
    assert {tool["name"] for tool in ast["tools"]} >= {"web_research", "web_crawl"}
    assert [child["type"] for child in children] == ["agent", "plan_and_execute", "agent"]
    assert ast["description"] == intent
    assert intent in children[1]["config"]["planner_guidance"]
    assert intent in children[1]["config"]["planner"]["system_prompt"]
    research_body = children[1]["config"]["body"]
    body_agents = [
        node["config"]
        for node in _walk_nodes(research_body)
        if node.get("type") == "agent" and isinstance(node.get("config"), dict)
    ]
    assert any(
        config.get("subtype") == "researcher"
        and {"web_research", "web_crawl"}.issubset(set(config.get("tools", [])))
        for config in body_agents
    )
    assert any(intent in config["system_prompt"] for config in body_agents)
    assert children[-1]["config"]["subtype"] == "synthesizer"
    assert children[-1]["config"]["output_key"] == "report"
    assert intent in children[-1]["config"]["system_prompt"]
    for config in _agent_configs(ast):
        assert config["system_prompt"].strip()
        assert config["user_prompt_template"].strip()
    assert any("current_step" in config["input_keys"] for config in body_agents)
    assert "report" in {config.get("output_key") for config in _agent_configs(ast)}


@_LEGACY_LOOP_SKIP
async def test_research_workflow_preserves_specific_designer_goal_in_runtime_prompts() -> None:
    """Specific Designer goals must survive as executable workflow instructions.

    Pins the plan_and_execute topology shape (synthesis_metadata carries
    designer_goal; planner_guidance reflects the brief). New default is
    parallel_lanes; this test opts into the legacy shape via design_brief.
    """
    intent = "Research user-provided launch ideas and produce a readiness brief"
    chunks = [
        LLMStreamChunk(
            tool_call=LLMToolCall(
                id="t1",
                name="propose_workflow",
                arguments={
                    "intent": intent,
                    "design_brief": {"topology": "plan_and_execute"},
                },
            )
        ),
        LLMStreamChunk(finish=True),
    ]
    orch = DesignerChatOrchestrator(_FakeLLM(chunks), _adapter())

    events = await _collect(
        orch,
        messages=[{"role": "user", "content": "go"}],
        current_ast=None,
        session_id=None,
        user_token="t",
    )

    mutation = [e for e in events if isinstance(e, MutationProposedEvent)][0]
    ast = mutation.new_ast
    plan_and_execute = ast["root"]["children"][1]
    research_body = plan_and_execute["config"]["body"]
    synthesizer = ast["root"]["children"][2]
    body_agents = [
        node["config"]
        for node in _walk_nodes(research_body)
        if node.get("type") == "agent" and isinstance(node.get("config"), dict)
    ]

    assert mutation.validation_errors == []
    assert ast["description"] == intent
    assert intent in plan_and_execute["config"]["planner_guidance"]
    assert intent in plan_and_execute["config"]["synthesis_metadata"]["designer_goal"]
    assert "generic overview" in plan_and_execute["config"]["planner_guidance"]
    assert (
        "Do not omit requested domain-specific sections" in synthesizer["config"]["system_prompt"]
    )
    assert any(intent in config["system_prompt"] for config in body_agents)


def test_investment_research_workflow_compiles_domain_specific_design_brief() -> None:
    """Domain-specific design briefs (supplied by the Designer LLM) must propagate
    through the builder into planner_guidance, synthesis_metadata, and the direct
    plan-and-execute researcher body. Domain keyword matching was removed (no hardcoded domains); domain
    flavor now comes from an explicit ``design_brief`` argument that the LLM
    constructs when calling ``propose_workflow``.
    """
    from deep_research.agent_designer.designer_architect import WorkflowDesignBrief

    intent = (
        "Build an investment research assistant for public companies that covers "
        "valuation, financial statements, earnings calls, competitors, market trends, "
        "news sentiment, risks, and a final bull and bear thesis."
    )
    design_brief = WorkflowDesignBrief(
        workflow_name="Investment Research Assistant",
        domain="Investment Research",
        # Opt into the plan_and_execute topology; the assertions below pin
        # that scaffold's specific shape (planner_guidance, synthesis_metadata,
        # direct researcher body). The new default is parallel_lanes, which has a
        # different shape — when this test is generalized to parallel_lanes
        # the assertions need to change too.
        topology="plan_and_execute",
        research_lanes=[
            {
                "description": "Analyze recent financial performance, revenue drivers, margins, cash flow, and balance sheet strength.",
                "system_prompt": "Investigate revenue trends, gross/operating/net margins, FCF conversion, balance-sheet strength. Cite latest 10-K/10-Q. Flag restated figures.",
            },
            {
                "description": "Assess valuation using market multiples, historical context, and peer comparisons.",
                "system_prompt": "Compute P/E, EV/EBITDA, EV/Sales vs historical and peer ranges. Cite primary financial data providers. Flag stretched multiples.",
            },
            {
                "description": "Review earnings calls, filings, management guidance, and analyst expectations.",
                "system_prompt": "Review the last 4-8 earnings call transcripts. Extract guidance, KPI trends, and analyst Q&A signals. Flag credibility issues.",
            },
            {
                "description": "Compare direct competitors and substitutes, including relative strengths and weaknesses.",
                "system_prompt": "Benchmark against 3-5 named competitors on revenue growth, margins, market share. Cite primary sources.",
            },
            {
                "description": "Evaluate news flow, market sentiment, catalysts, controversies, and downside risks.",
                "system_prompt": "Synthesize recent news and sentiment over last 6 months. Identify catalysts and downside risks with sources.",
            },
            {
                "description": "Produce an investment thesis with bull case, bear case, key risks, confidence, and watch items.",
                "system_prompt": "Reconcile signals across lanes. Build explicit bull case and bear case. State confidence and watch items.",
            },
        ],
        required_outputs=[
            "Executive investment summary.",
            "Financial and valuation analysis.",
            "Competitive and market landscape.",
        ],
        quality_gates=[
            "Do not provide a generic company summary when an investment decision support workflow was requested.",
            "Every final thesis must connect evidence to an investment implication.",
            "Cover both upside and downside cases before synthesizing a recommendation or neutral conclusion.",
        ],
    )

    workflow = build_web_research_workflow(intent, "Investment workflow", design_brief=design_brief)
    plan_and_execute = workflow["root"]["children"][1]
    guidance = plan_and_execute["config"]["planner_guidance"]
    metadata = plan_and_execute["config"]["synthesis_metadata"]
    researcher = plan_and_execute["config"]["body"]

    assert workflow["name"] == "Investment Research Assistant"
    assert workflow["root"]["label"] == "Investment Research Pipeline"
    assert plan_and_execute["label"] == "Plan & Execute Investment Research"
    assert plan_and_execute["config"]["max_iterations"] >= 6
    assert metadata["designer_domain"] == "Investment Research"
    assert "`lane` field" not in guidance
    assert "Do not emit prompt-only routing fields" in guidance
    assert "lane_1" in metadata["designer_lane_ids"]
    assert researcher["type"] == "agent"
    assert researcher["label"] == "Investment Research Researcher"
    assert researcher["config"]["subtype"] == "researcher"
    # Phase 0 (dataflow-enforcement plan): body is the direct researcher agent —
    # no router/conditional, no body reflector.
    assert plan_and_execute["config"]["body"]["type"] == "agent"
    researcher_prompt = (
        researcher["config"]["system_prompt"]
        + "\n"
        + researcher["config"]["user_prompt_template"]
    ).casefold()
    assert "financial performance" in researcher_prompt
    assert "valuation" in researcher_prompt
    assert "earnings" in researcher_prompt
    assert "competitors" in researcher_prompt
    for required in [
        "financial performance",
        "valuation",
        "earnings calls",
        "competitors",
        "bull case",
        "bear case",
    ]:
        assert required in guidance
    assert "Do not provide a generic company summary" in guidance


def test_supplied_brief_without_lanes_does_not_fall_back_to_default_profile() -> None:
    """A Designer-supplied brief must author its semantic lane shape.

    The YAML default profile remains only for legacy no-brief callers. Once
    the Designer provides a design_brief, missing lanes should surface as a
    correction path instead of silently becoming generic profile lanes.
    """
    from deep_research.agent_designer.designer_architect import WorkflowDesignBrief

    intent = (
        "Build an investment research assistant covering valuation, earnings calls, "
        "competitors, market trends, and risks."
    )

    with pytest.raises(ValueError, match="plan_and_execute requires"):
        build_web_research_workflow(
            intent,
            "Investment workflow",
            design_brief=WorkflowDesignBrief(topology="plan_and_execute"),
        )


@_LEGACY_LOOP_SKIP
async def test_propose_workflow_uses_design_brief_and_architect_system_prompt() -> None:
    """LLM-supplied architect/critic briefs drive generated workflow identity and guidance."""
    chunks = [
        LLMStreamChunk(
            tool_call=LLMToolCall(
                id="t1",
                name="propose_workflow",
                arguments={
                    "intent": "Research supplier risk for a target company",
                    "design_brief": {
                        "workflow_name": "Supplier Risk Diligence",
                        "domain": "Supply Chain Risk",
                        # Pin plan_and_execute shape (planner_guidance etc.);
                        # parallel_lanes is the new default.
                        "topology": "plan_and_execute",
                        "research_lanes": [
                            "Map critical suppliers and dependencies.",
                            "Assess disruptions, concentration, and geopolitical exposure.",
                        ],
                        "required_outputs": ["Risk heatmap", "Mitigation watchlist"],
                        "quality_gates": ["Reject generic supplier summaries."],
                    },
                },
            )
        ),
        LLMStreamChunk(finish=True),
    ]
    llm = _FakeLLM(chunks)
    orch = DesignerChatOrchestrator(llm, _adapter())

    events = await _collect(
        orch,
        messages=[{"role": "user", "content": "go"}],
        current_ast=None,
        session_id=None,
        user_token="t",
    )

    llm_messages = llm.calls[0][0]
    mutation = [e for e in events if isinstance(e, MutationProposedEvent)][0]
    ast = mutation.new_ast
    plan_and_execute = ast["root"]["children"][1]
    guidance = plan_and_execute["config"]["planner_guidance"]

    assert llm_messages[0]["role"] == "system"
    assert "Agent Designer architect" in llm_messages[0]["content"]
    assert "Critic" in llm_messages[0]["content"]
    assert ast["name"] == "Supplier Risk Diligence"
    assert ast["root"]["label"] == "Supply Chain Risk Pipeline"
    assert "Map critical suppliers" in guidance
    assert "Reject generic supplier summaries" in guidance


@_LEGACY_LOOP_SKIP
async def test_propose_direct_workflow_materializes_prompt_contract() -> None:
    """Non-research scaffolds are still executable and prompt-explicit."""
    chunks = [
        LLMStreamChunk(
            tool_call=LLMToolCall(
                id="t1",
                name="propose_workflow",
                arguments={"intent": "create a direct answer assistant"},
            )
        ),
        LLMStreamChunk(finish=True),
    ]
    orch = DesignerChatOrchestrator(_FakeLLM(chunks), _adapter())

    events = await _collect(
        orch,
        messages=[{"role": "user", "content": "go"}],
        current_ast=None,
        session_id=None,
        user_token="t",
    )

    mutation = [e for e in events if isinstance(e, MutationProposedEvent)][0]
    ast = mutation.new_ast
    agent_config = ast["root"]["children"][0]["config"]
    assert mutation.validation_errors == []
    assert ast["output_keys"] == ["output"]
    assert agent_config["subtype"] == "custom"
    assert agent_config["system_prompt"].strip()
    assert agent_config["user_prompt_template"] == "{query}"


@_LEGACY_LOOP_SKIP
async def test_message_event_passes_through() -> None:
    """Plain content chunks become MessageEvent instances."""
    chunks = [
        LLMStreamChunk(content="hello "),
        LLMStreamChunk(content="world"),
        LLMStreamChunk(finish=True),
    ]
    orch = DesignerChatOrchestrator(_FakeLLM(chunks), _adapter())
    events = await _collect(
        orch,
        messages=[{"role": "user", "content": "hi"}],
        current_ast=None,
        session_id=None,
        user_token="t",
    )
    assert sum(isinstance(e, MessageEvent) for e in events) == 2


async def test_oversized_messages_raises() -> None:
    """Exceeding MAX_MESSAGES raises RequestTooLargeError before the LLM is called."""
    orch = DesignerChatOrchestrator(_FakeLLM([]), _adapter())
    with pytest.raises(RequestTooLargeError):
        async for _ in orch.run_turn(
            messages=[{"role": "user", "content": "x"}] * (MAX_MESSAGES + 1),
            current_ast=None,
            session_id=None,
            user_token="t",
        ):
            pass


async def test_oversized_ast_raises() -> None:
    """An AST exceeding MAX_AST_BYTES raises RequestTooLargeError."""
    huge_ast: dict[str, Any] = {"root": {"label": "x" * (MAX_AST_BYTES + 1000)}}
    orch = DesignerChatOrchestrator(_FakeLLM([]), _adapter())
    with pytest.raises(RequestTooLargeError):
        async for _ in orch.run_turn(
            messages=[{"role": "user", "content": "x"}],
            current_ast=huge_ast,
            session_id=None,
            user_token="t",
        ):
            pass


@_LEGACY_LOOP_SKIP
async def test_unknown_tool_yields_error() -> None:
    """An unknown tool name yields an ErrorEvent, not an exception."""
    chunks = [
        LLMStreamChunk(tool_call=LLMToolCall(id="t1", name="not_a_tool", arguments={})),
        LLMStreamChunk(finish=True),
    ]
    orch = DesignerChatOrchestrator(_FakeLLM(chunks), _adapter())
    events = await _collect(
        orch,
        messages=[{"role": "user", "content": "x"}],
        current_ast=None,
        session_id=None,
        user_token="t",
    )
    assert any(isinstance(e, ErrorEvent) for e in events)


@_LEGACY_LOOP_SKIP
async def test_invalid_args_yields_error() -> None:
    """Invalid tool args (bad tier value) yields an ErrorEvent."""
    chunks = [
        LLMStreamChunk(
            tool_call=LLMToolCall(
                id="t1",
                name="set_model_tier",
                arguments={"node_path": "root", "tier": "not_a_tier"},
            )
        ),
        LLMStreamChunk(finish=True),
    ]
    orch = DesignerChatOrchestrator(_FakeLLM(chunks), _adapter())
    events = await _collect(
        orch,
        messages=[{"role": "user", "content": "x"}],
        current_ast={
            "root": {"id": "r", "type": "agent", "label": "r", "config": {}, "children": []}
        },
        session_id=None,
        user_token="t",
    )
    assert any(isinstance(e, ErrorEvent) for e in events)


@_LEGACY_LOOP_SKIP
async def test_validate_tool_no_ast_returns_invalid() -> None:
    """validate with no AST (empty dict) returns valid=False in ToolResultEvent."""
    chunks = [
        LLMStreamChunk(tool_call=LLMToolCall(id="t1", name="validate", arguments={})),
        LLMStreamChunk(finish=True),
    ]
    orch = DesignerChatOrchestrator(_FakeLLM(chunks), _adapter())
    events = await _collect(
        orch,
        messages=[{"role": "user", "content": "x"}],
        current_ast=None,
        session_id=None,
        user_token="t",
    )
    tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(tool_results) == 1
    assert tool_results[0].result["valid"] is False


@_LEGACY_LOOP_SKIP
async def test_validate_tool_surfaces_quality_advice_for_unspecialized_agents() -> None:
    """When the AST is structurally valid but agents are still on the default
    scaffold (empty / short system_prompt, missing tools, default tier), the
    chat-side validate tool returns ``advice`` entries naming the suggested
    follow-up tool calls so the LLM can patch the workflow before save."""
    minimal_ast = {
        "id": "wf",
        "name": "wf",
        "version": 1,
        "tools": [],
        "pools": [],
        "sources": [],
        "models": {},
        "required_inputs": ["query"],
        "output_keys": ["output"],
        "token_budget": 0,
        "timeout_seconds": 1800,
        "root": {
            "id": "root",
            "type": "agent",
            "label": "root",
            "config": {"subtype": "researcher"},
            "children": [],
        },
    }
    chunks = [
        LLMStreamChunk(tool_call=LLMToolCall(id="t1", name="validate", arguments={})),
        LLMStreamChunk(finish=True),
    ]
    orch = DesignerChatOrchestrator(_FakeLLM(chunks), _adapter())
    events = await _collect(
        orch,
        messages=[{"role": "user", "content": "x"}],
        current_ast=minimal_ast,
        session_id=None,
        user_token="t",
    )
    tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(tool_results) == 1
    result = tool_results[0].result
    # Structurally valid (load_workflow_from_dict accepted it).
    assert result["valid"] is True
    # Quality advice surfaced separately — must not affect the valid flag.
    assert isinstance(result["advice"], list)
    messages = [a["message"] for a in result["advice"]]
    assert any("system_prompt" in m for m in messages)
    assert any("update_block" in m for m in messages)
    # New: ``critique`` field is present (may be a fallback verdict since
    # the FakeLLM does not emit a critique tool-call when re-streamed for
    # the critic; either way it must be a dict-or-None, never absent).
    assert "critique" in result


@_LEGACY_LOOP_SKIP
async def test_validate_tool_surfaces_critique_field(monkeypatch) -> None:
    """When the AST is structurally valid, the validate tool returns a
    ``critique`` field with the LLM-as-judge verdict. We monkeypatch the
    critic helper to return a known result so the test is hermetic."""
    minimal_ast = {
        "id": "wf",
        "name": "wf",
        "version": 1,
        "tools": [],
        "pools": [],
        "sources": [],
        "models": {},
        "required_inputs": ["query"],
        "output_keys": ["output"],
        "token_budget": 0,
        "timeout_seconds": 1800,
        "root": {
            "id": "root",
            "type": "agent",
            "label": "root",
            "config": {"subtype": "researcher"},
            "children": [],
        },
    }

    expected_critique = {
        "verdict": "needs_revision",
        "summary": "The root agent has no specialization for the user's intent.",
        "agent_findings": [
            {
                "node_path": "root",
                "label": "root",
                "severity": "needs_revision",
                "finding": "no topic-specific guidance",
                "suggested_action": "call update_block with task-specific system_prompt",
            }
        ],
        "coverage_gaps": [],
        "output_gaps": [],
    }

    async def _fake_critique(ast, intent, llm):  # noqa: ARG001 — protocol shape
        return expected_critique

    import deep_research.agent_designer.orchestrator as orch_module

    monkeypatch.setattr(orch_module, "_critique_ast", _fake_critique)

    chunks = [
        LLMStreamChunk(tool_call=LLMToolCall(id="t1", name="validate", arguments={})),
        LLMStreamChunk(finish=True),
    ]
    orch = DesignerChatOrchestrator(_FakeLLM(chunks), _adapter())
    events = await _collect(
        orch,
        messages=[
            {"role": "user", "content": "investment research on NVDA"},
        ],
        current_ast=minimal_ast,
        session_id=None,
        user_token="t",
    )
    tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(tool_results) == 1
    result = tool_results[0].result
    assert result["valid"] is True
    # The critique is propagated verbatim from the helper.
    assert result["critique"] == expected_critique
    # Critique does NOT affect the structural valid flag.
    assert result["valid"] is True


@_LEGACY_LOOP_SKIP
async def test_multi_turn_carries_ast_forward() -> None:
    """Two tool-calls in one stream: propose then add_block; ast carries forward."""
    chunks = [
        LLMStreamChunk(
            tool_call=LLMToolCall(id="t1", name="propose_workflow", arguments={"intent": "x"})
        ),
        LLMStreamChunk(
            tool_call=LLMToolCall(
                id="t2",
                name="add_block",
                arguments={
                    "parent_path": "root",
                    "node_type": "agent",
                    "label": "child",
                    "config": {"subtype": "researcher"},
                },
            )
        ),
        LLMStreamChunk(finish=True),
    ]
    orch = DesignerChatOrchestrator(_FakeLLM(chunks), _adapter())
    events = await _collect(
        orch,
        messages=[{"role": "user", "content": "x"}],
        current_ast=None,
        session_id=None,
        user_token="t",
    )
    mutation_evs = [e for e in events if isinstance(e, MutationProposedEvent)]
    assert len(mutation_evs) == 2
    # Second mutation's old_ast must be the first's new_ast
    assert mutation_evs[1].old_ast == mutation_evs[0].new_ast


@_LEGACY_LOOP_SKIP
async def test_stateless_invariant() -> None:
    """Two separate run_turn calls with identical inputs produce identical event-type sequences."""

    def _chunks() -> list[LLMStreamChunk]:
        return [
            LLMStreamChunk(
                tool_call=LLMToolCall(id="t1", name="propose_workflow", arguments={"intent": "x"})
            ),
            LLMStreamChunk(finish=True),
        ]

    orch1 = DesignerChatOrchestrator(_FakeLLM(_chunks()), _adapter())
    events1 = await _collect(
        orch1,
        messages=[{"role": "user", "content": "x"}],
        current_ast=None,
        session_id="A",
        user_token="t",
    )
    orch2 = DesignerChatOrchestrator(_FakeLLM(_chunks()), _adapter())
    events2 = await _collect(
        orch2,
        messages=[{"role": "user", "content": "x"}],
        current_ast=None,
        session_id="B",
        user_token="t",
    )
    assert [type(e).__name__ for e in events1] == [type(e).__name__ for e in events2]


@_LEGACY_LOOP_SKIP
async def test_mutation_path_error_yields_error_event() -> None:
    """A bad path in delete_block yields ErrorEvent, not an unhandled exception."""
    chunks = [
        LLMStreamChunk(
            tool_call=LLMToolCall(
                id="t1", name="delete_block", arguments={"path": "root.children.99"}
            )
        ),
        LLMStreamChunk(finish=True),
    ]
    orch = DesignerChatOrchestrator(_FakeLLM(chunks), _adapter())
    events = await _collect(
        orch,
        messages=[{"role": "user", "content": "x"}],
        current_ast={
            "root": {"id": "r", "type": "sequence", "label": "r", "config": {}, "children": []}
        },
        session_id=None,
        user_token="t",
    )
    assert any(isinstance(e, ErrorEvent) for e in events)


@_LEGACY_LOOP_SKIP
async def test_validation_errors_field_is_always_list() -> None:
    """MutationProposedEvent.validation_errors is always a list (empty or populated)."""
    chunks = [
        LLMStreamChunk(
            tool_call=LLMToolCall(id="t1", name="propose_workflow", arguments={"intent": "x"})
        ),
        LLMStreamChunk(finish=True),
    ]
    orch = DesignerChatOrchestrator(_FakeLLM(chunks), _adapter())
    events = await _collect(
        orch,
        messages=[{"role": "user", "content": "x"}],
        current_ast=None,
        session_id=None,
        user_token="t",
    )
    mp = [e for e in events if isinstance(e, MutationProposedEvent)][0]
    assert isinstance(mp.validation_errors, list)
    # Either summary is present (valid AST) or validation_errors is populated
    assert mp.summary is not None or len(mp.validation_errors) > 0


@_LEGACY_LOOP_SKIP
async def test_done_event_always_emitted_at_end() -> None:
    """DoneEvent is always the final event, even when the stream is immediately finished."""
    chunks = [LLMStreamChunk(finish=True)]
    orch = DesignerChatOrchestrator(_FakeLLM(chunks), _adapter())
    events = await _collect(
        orch,
        messages=[{"role": "user", "content": "x"}],
        current_ast=None,
        session_id=None,
        user_token="t",
    )
    assert isinstance(events[-1], DoneEvent)


@_LEGACY_LOOP_SKIP
async def test_mutation_without_ast_yields_error() -> None:
    """Calling a mutation tool (not propose_workflow) with no current_ast yields ErrorEvent."""
    chunks = [
        LLMStreamChunk(
            tool_call=LLMToolCall(
                id="t1",
                name="add_block",
                arguments={
                    "parent_path": "root",
                    "node_type": "agent",
                    "label": "x",
                    "config": {},
                },
            )
        ),
        LLMStreamChunk(finish=True),
    ]
    orch = DesignerChatOrchestrator(_FakeLLM(chunks), _adapter())
    events = await _collect(
        orch,
        messages=[{"role": "user", "content": "x"}],
        current_ast=None,
        session_id=None,
        user_token="t",
    )
    errors = [e for e in events if isinstance(e, ErrorEvent)]
    assert len(errors) == 1
    assert "propose_workflow" in errors[0].message


@_LEGACY_LOOP_SKIP
async def test_discover_sources_returns_tool_result() -> None:
    """discover_sources yields a ToolResultEvent with a resources list."""
    chunks = [
        LLMStreamChunk(
            tool_call=LLMToolCall(id="t1", name="discover_sources", arguments={"kinds": None})
        ),
        LLMStreamChunk(finish=True),
    ]
    orch = DesignerChatOrchestrator(_FakeLLM(chunks), _adapter())
    events = await _collect(
        orch,
        messages=[{"role": "user", "content": "x"}],
        current_ast=None,
        session_id=None,
        user_token="t",
    )
    tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
    assert len(tool_results) == 1
    assert "resources" in tool_results[0].result
    assert isinstance(tool_results[0].result["resources"], list)


@_LEGACY_LOOP_SKIP
async def test_list_modes_returns_configured_modes_and_source_kinds() -> None:
    """list_modes exposes live model tiers plus query/depth/source modes."""
    chunks = [
        LLMStreamChunk(tool_call=LLMToolCall(id="t1", name="list_modes", arguments={})),
        LLMStreamChunk(finish=True),
    ]
    orch = DesignerChatOrchestrator(_FakeLLM(chunks), _adapter())

    events = await _collect(
        orch,
        messages=[{"role": "user", "content": "x"}],
        current_ast=None,
        session_id=None,
        user_token="t",
    )

    result = [e for e in events if isinstance(e, ToolResultEvent)][0].result
    assert "bulk_analysis" in result["model_tiers"]
    assert "fast" in result["model_tiers"]
    assert "deep_research" in result["query_modes"]
    assert "auto" in result["research_depths"]
    assert any(item["kind"] == "serving_endpoint" for item in result["source_kinds"])


@_LEGACY_LOOP_SKIP
async def test_read_only_tool_result_continues_until_workflow_proposed() -> None:
    """list_modes must be fed back to the LLM so build requests can still mutate."""
    intent = "Build an investment research assistant with earnings and valuation lanes"
    llm = _FakeLLM(
        [
            [
                LLMStreamChunk(
                    tool_call=LLMToolCall(id="modes", name="list_modes", arguments={})
                ),
                LLMStreamChunk(finish=True),
            ],
            [
                LLMStreamChunk(
                    tool_call=LLMToolCall(
                        id="proposal",
                        name="propose_workflow",
                        arguments={"intent": intent},
                    )
                ),
                LLMStreamChunk(finish=True),
            ],
        ]
    )
    orch = DesignerChatOrchestrator(llm, _adapter())

    events = await _collect(
        orch,
        messages=[{"role": "user", "content": intent}],
        current_ast=None,
        session_id=None,
        user_token="t",
    )

    assert len(llm.calls) == 2
    assert any(isinstance(e, ToolResultEvent) and e.tool_name == "list_modes" for e in events)
    assert any(isinstance(e, MutationProposedEvent) for e in events)
    second_round_messages = llm.calls[1][0]
    assert second_round_messages[-2]["role"] == "assistant"
    assert second_round_messages[-2]["tool_calls"][0]["function"]["name"] == "list_modes"
    assert second_round_messages[-1]["role"] == "tool"
    assert second_round_messages[-1]["tool_call_id"] == "modes"
    assert isinstance(events[-1], DoneEvent)


@_LEGACY_LOOP_SKIP
async def test_multiple_read_only_tool_rounds_continue_until_workflow_proposed() -> None:
    """Discovery and mode inspection can happen in separate rounds before mutation."""
    intent = "Create a detailed research workflow for investment analysis"
    llm = _FakeLLM(
        [
            [
                LLMStreamChunk(
                    tool_call=LLMToolCall(
                        id="sources",
                        name="discover_sources",
                        arguments={"kinds": None},
                    )
                ),
                LLMStreamChunk(finish=True),
            ],
            [
                LLMStreamChunk(
                    tool_call=LLMToolCall(id="modes", name="list_modes", arguments={})
                ),
                LLMStreamChunk(finish=True),
            ],
            [
                LLMStreamChunk(
                    tool_call=LLMToolCall(
                        id="proposal",
                        name="propose_workflow",
                        arguments={"intent": intent},
                    )
                ),
                LLMStreamChunk(finish=True),
            ],
        ]
    )
    orch = DesignerChatOrchestrator(llm, _adapter())

    events = await _collect(
        orch,
        messages=[{"role": "user", "content": intent}],
        current_ast=None,
        session_id=None,
        user_token="t",
    )

    assert len(llm.calls) == 3
    tool_result_names = [e.tool_name for e in events if isinstance(e, ToolResultEvent)]
    assert tool_result_names == ["discover_sources", "list_modes"]
    assert any(isinstance(e, MutationProposedEvent) for e in events)
    third_round_messages = llm.calls[2][0]
    assert third_round_messages[-1]["tool_call_id"] == "modes"


@_LEGACY_LOOP_SKIP
async def test_build_request_without_mutation_emits_blocking_error() -> None:
    """A create/build request must not finish successfully after read-only tools only."""
    llm = _FakeLLM(
        [
            [
                LLMStreamChunk(
                    tool_call=LLMToolCall(id="modes", name="list_modes", arguments={})
                ),
                LLMStreamChunk(finish=True),
            ],
            [
                LLMStreamChunk(content="I can help with that."),
                LLMStreamChunk(finish=True),
            ],
        ]
    )
    orch = DesignerChatOrchestrator(llm, _adapter())

    events = await _collect(
        orch,
        messages=[
            {
                "role": "user",
                "content": "Build an investment research assistant with detailed research lanes",
            }
        ],
        current_ast=None,
        session_id=None,
        user_token="t",
    )

    errors = [e for e in events if isinstance(e, ErrorEvent)]
    assert len(llm.calls) == 2
    assert errors
    assert "did not propose a workflow change" in errors[-1].message
    assert isinstance(events[-1], DoneEvent)


@_LEGACY_LOOP_SKIP
async def test_read_only_tool_continuation_is_bounded() -> None:
    """The backend must not loop forever if the LLM keeps inspecting tools."""
    rounds = [
        [
            LLMStreamChunk(tool_call=LLMToolCall(id=f"modes-{idx}", name="list_modes", arguments={})),
            LLMStreamChunk(finish=True),
        ]
        for idx in range(MAX_DESIGNER_TOOL_ROUNDS + 1)
    ]
    llm = _FakeLLM(rounds)
    orch = DesignerChatOrchestrator(llm, _adapter())

    events = await _collect(
        orch,
        messages=[{"role": "user", "content": "Create a research workflow"}],
        current_ast=None,
        session_id=None,
        user_token="t",
    )

    errors = [e for e in events if isinstance(e, ErrorEvent)]
    assert len(llm.calls) == MAX_DESIGNER_TOOL_ROUNDS
    assert errors
    assert "multiple tool rounds" in errors[-1].message
    assert isinstance(events[-1], DoneEvent)


@_LEGACY_LOOP_SKIP
async def test_tool_call_event_contains_args() -> None:
    """ToolCallEvent carries the raw arguments dict from the LLM tool call."""
    intent = "build a web researcher"
    chunks = [
        LLMStreamChunk(
            tool_call=LLMToolCall(id="t1", name="propose_workflow", arguments={"intent": intent})
        ),
        LLMStreamChunk(finish=True),
    ]
    orch = DesignerChatOrchestrator(_FakeLLM(chunks), _adapter())
    events = await _collect(
        orch,
        messages=[{"role": "user", "content": "x"}],
        current_ast=None,
        session_id=None,
        user_token="t",
    )
    tc_events = [e for e in events if isinstance(e, ToolCallEvent)]
    assert len(tc_events) == 1
    assert tc_events[0].tool_name == "propose_workflow"
    assert tc_events[0].args == {"intent": intent}
    assert tc_events[0].tool_call_id == "t1"
