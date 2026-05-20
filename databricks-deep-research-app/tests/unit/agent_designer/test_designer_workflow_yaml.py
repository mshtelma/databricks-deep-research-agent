"""Validate designer_workflow.yaml loads correctly through the framework."""

from pathlib import Path

from databricks_deep_research.templates.renderer import SafeTemplateRenderer
from databricks_deep_research.workflow.loader import load_workflow

_YAML = (
    Path(__file__).parent.parent.parent.parent
    / "src/deep_research/agent_designer/designer_workflow.yaml"
)


def _walk_nodes(node):
    yield node
    for child in node.children or []:
        yield from _walk_nodes(child)


def test_designer_workflow_loads():
    wf = load_workflow(str(_YAML))
    assert wf.name == "Designer Architect+Critic Loop"
    assert wf.root.type == "loop"


def test_designer_workflow_uses_real_endpoint_identifiers():
    wf = load_workflow(str(_YAML))
    models = wf.models or {}
    assert "complex" in models
    assert "critic" in models
    complex_eps = models["complex"]["endpoints"]
    critic_eps = models["critic"]["endpoints"]
    # Codex iter-2 fix #1: real serving-endpoint identifiers (not aliases).
    assert "databricks-claude-opus-4-7" in complex_eps
    assert "databricks-gpt-5-5" in critic_eps
    # Aliases must NOT appear.
    for ep in complex_eps + critic_eps:
        assert ep not in ("opus", "sonnet", "gpt5", "gpt5mini", "gpt5nano")


def test_loop_until_uses_state_condition_syntax():
    wf = load_workflow(str(_YAML))
    until = wf.root.config["until"]
    assert until["type"] == "state"
    assert until["key"] == "critic_approved.critic_approved"
    assert until["operator"] == "eq"
    assert until["value"] is True


def test_loop_has_room_for_gate_feedback_iteration():
    wf = load_workflow(str(_YAML))
    assert wf.root.config["max_iterations"] >= 4


def test_loop_body_contains_architect_gate_critic():
    wf = load_workflow(str(_YAML))
    body = wf.root.children[0]
    assert body.type == "sequence"
    node_ids = [c.id for c in body.children]
    # Expected sequence: architect → parse_architect_ast → structural_gate → gate_router → extract_critic_approved
    assert "architect" in node_ids
    assert "parse_architect_ast" in node_ids
    assert "structural_gate" in node_ids
    assert "gate_router" in node_ids
    assert "extract_critic_approved" in node_ids


def test_architect_uses_complex_tier():
    wf = load_workflow(str(_YAML))
    body = wf.root.children[0]
    architect = next(c for c in body.children if c.id == "architect")
    assert architect.config["model_tier"] == "complex"
    assert "{gate_result}" in architect.config["user_prompt_template"]
    assert "{critic_verdict}" in architect.config["user_prompt_template"]


def test_critic_uses_critic_tier():
    wf = load_workflow(str(_YAML))
    body = wf.root.children[0]
    gate_router = next(c for c in body.children if c.id == "gate_router")
    # critic is at child_index 1 (pass branch)
    critic = gate_router.children[1]
    assert critic.id == "critic"
    assert critic.config["model_tier"] == "critic"


def test_agent_prompt_templates_are_safe_renderer_compatible():
    wf = load_workflow(str(_YAML))
    renderer = SafeTemplateRenderer()

    for node in _walk_nodes(wf.root):
        for field in ("system_prompt", "user_prompt_template"):
            prompt = (node.config or {}).get(field)
            if prompt:
                renderer.extract_variables(prompt)
