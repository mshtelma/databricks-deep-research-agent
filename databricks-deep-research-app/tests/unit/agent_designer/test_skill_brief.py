"""Tests for skill -> workflow brief: deterministic signature + fail-soft summarizer."""

from __future__ import annotations

from typing import Any

from deep_research.agent_designer.blueprint import build_blueprint
from deep_research.agent_designer.skill_brief import (
    BriefStep,
    SkillSummary,
    SkillWorkflowBrief,
    brief_to_task_signature,
    render_skill_brief,
    summarize_skills_to_brief,
)
from deep_research.agent_designer.task_signature import TaskSignature, select_topology

# --- determinism core: brief_to_task_signature ----------------------------


def _sig(brief: SkillWorkflowBrief) -> TaskSignature:
    # Must produce a COMPLETE, valid TaskSignature accepted without the classifier.
    return TaskSignature.load_from_storage(brief_to_task_signature(brief))


def test_parallel_independent_steps_select_parallel_lanes() -> None:
    brief = SkillWorkflowBrief(
        composition="parallel",
        steps=[
            BriefStep(description="Research competitors", parallelizable=True),
            BriefStep(description="Research market size", parallelizable=True),
            BriefStep(description="Research regulations", parallelizable=True),
        ],
    )
    sig = _sig(brief)
    assert select_topology(sig) == "parallel_lanes"
    assert sig.independent_workstreams_count == 3
    assert len(sig.lane_descriptions) == 3  # invariant: == max(iwc, 1)


def test_sequential_dependent_steps_select_plan_and_execute() -> None:
    brief = SkillWorkflowBrief(
        composition="sequential",
        steps=[
            BriefStep(description="Gather sources"),
            BriefStep(description="Analyze", depends_on=[0]),
            BriefStep(description="Summarize", depends_on=[1]),
        ],
    )
    sig = _sig(brief)
    assert select_topology(sig) == "plan_and_execute"
    assert sig.step_dependencies_present is True
    assert len(sig.lane_descriptions) == 1


def test_iterative_sets_iteration_required() -> None:
    brief = SkillWorkflowBrief(
        composition="iterative",
        steps=[BriefStep(description="Draft"), BriefStep(description="Critique + refine", depends_on=[0])],
    )
    sig = _sig(brief)
    assert sig.iteration_required is True
    assert select_topology(sig) == "plan_and_execute"


def test_single_step_selects_single_agent() -> None:
    brief = SkillWorkflowBrief(composition="sequential", steps=[BriefStep(description="Look up a fact")])
    sig = _sig(brief)
    assert select_topology(sig) == "single_agent"
    assert len(sig.lane_descriptions) == 1


def test_corpus_tool_marks_corpus_asset() -> None:
    brief = SkillWorkflowBrief(
        composition="sequential",
        steps=[BriefStep(description="Search the index", suggested_tool_kind="vector_search")],
    )
    d = brief_to_task_signature(brief)
    assert d["asset_signature"] == "corpus_plus_web"


def test_lane_descriptions_invariant_holds_for_all_shapes() -> None:
    for brief in [
        SkillWorkflowBrief(steps=[]),
        SkillWorkflowBrief(composition="parallel", steps=[BriefStep(description="a", parallelizable=True)]),
    ]:
        d = brief_to_task_signature(brief)
        assert len(d["lane_descriptions"]) == max(d["independent_workstreams_count"], 1)
        TaskSignature.load_from_storage(d)  # validates


# --- summarizer fail-soft --------------------------------------------------


class _Skill:
    def __init__(self, name: str, body: str, scripts: dict | None = None) -> None:
        self.name = name
        self.description = f"{name} description"
        self.body = body
        self.scripts = scripts or {}


class _Store:
    def __init__(self, skills: dict[str, Any]) -> None:
        self._skills = skills

    async def get_skill(self, name: str) -> Any:
        return self._skills.get(name)


class _Resp:
    def __init__(self, structured: Any) -> None:
        self.structured = structured
        self.content = None


class _LLM:
    def __init__(self, structured: Any) -> None:
        self._structured = structured
        self.calls = 0

    async def complete(self, **_kwargs: Any) -> _Resp:
        self.calls += 1
        return _Resp(self._structured)


class _RaisingLLM:
    async def complete(self, **_kwargs: Any) -> Any:
        raise RuntimeError("llm down")


async def test_missing_skill_degrades_to_minimal_brief() -> None:
    store = _Store({})
    brief = await summarize_skills_to_brief(["ghost"], skill_store=store, llm=_LLM(None))
    assert brief.steps  # minimal brief still has a step
    assert any("could not load" in n for n in brief.notes)


async def test_llm_failure_degrades_to_description_brief() -> None:
    store = _Store({"s": _Skill("s", "secret body lines")})
    brief = await summarize_skills_to_brief(["s"], skill_store=store, llm=_RaisingLLM())
    assert brief.skills and brief.skills[0].name == "s"
    assert brief.steps
    assert any("unavailable" in n for n in brief.notes)


async def test_happy_path_overrides_skills_from_metadata() -> None:
    store = _Store({"s": _Skill("s", "body", scripts={"x.py": "code"})})
    llm_brief = SkillWorkflowBrief(
        composition="parallel",
        steps=[BriefStep(description="a", parallelizable=True), BriefStep(description="b", parallelizable=True)],
        skills=[SkillSummary(name="HALLUCINATED")],  # must be overridden
    )
    brief = await summarize_skills_to_brief(["s"], skill_store=store, llm=_LLM(llm_brief))
    assert [s.name for s in brief.skills] == ["s"]  # ground-truth, not hallucinated
    assert brief.skills[0].has_scripts is True
    assert len(brief.steps) == 2


# --- render is privacy-safe ------------------------------------------------


def test_render_never_contains_raw_body() -> None:
    brief = SkillWorkflowBrief(
        skills=[SkillSummary(name="s")],
        steps=[BriefStep(description="Do the thing", suggested_tool_kind="web_search", runs_script=True)],
    )
    rendered = render_skill_brief(brief)
    assert "Do the thing" in rendered
    assert "[tool: web_search]" in rendered
    assert "script" in rendered.lower()


def test_render_empty_when_no_steps() -> None:
    assert render_skill_brief(SkillWorkflowBrief(steps=[])) == ""


# --- end-to-end: brief -> signature -> build_blueprint (Codex-required) ----


def _count_nodes(node: Any) -> int:
    if not isinstance(node, dict):
        return 0
    total = 1 if node.get("type") else 0
    for child in node.get("children") or []:
        total += _count_nodes(child)
    cfg = node.get("config")
    if isinstance(cfg, dict) and isinstance(cfg.get("body"), dict):
        total += _count_nodes(cfg["body"])
    return total


def test_parallel_brief_builds_multi_node_blueprint() -> None:
    """A 3-independent-step skill compiles to a multi-node blueprint (not one agent)."""
    brief = SkillWorkflowBrief(
        composition="parallel",
        steps=[
            BriefStep(description="Research A", parallelizable=True),
            BriefStep(description="Research B", parallelizable=True),
            BriefStep(description="Research C", parallelizable=True),
        ],
    )
    ast = build_blueprint(brief_to_task_signature(brief), intent="Compile the skill")
    assert isinstance(ast, dict) and isinstance(ast.get("root"), dict)
    assert _count_nodes(ast["root"]) > 1  # deterministic multi-node structure


def test_sequential_brief_builds_blueprint() -> None:
    brief = SkillWorkflowBrief(
        composition="sequential",
        steps=[BriefStep(description="A"), BriefStep(description="B", depends_on=[0])],
    )
    ast = build_blueprint(brief_to_task_signature(brief), intent="Compile the skill")
    assert isinstance(ast, dict) and isinstance(ast.get("root"), dict)
    assert _count_nodes(ast["root"]) > 1
