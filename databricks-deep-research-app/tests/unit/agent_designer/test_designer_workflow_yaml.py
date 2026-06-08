"""Validate designer_workflow.yaml loads correctly through the framework."""

import asyncio
import json
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


def _signature_loop_body(wf):
    """Plan v2.1 generic-robustness — root is sequence → signature_loop →
    signature_iter (sequence) → [classifier, build_blueprint, designer_loop,
    evaluate_signature_loop]. Return the inner sequence's children list so
    tests can navigate the old-style "siblings under root.children" shape.
    """
    if wf.root.type == "sequence":
        signature_loop = next(
            (c for c in wf.root.children if c.id == "signature_loop"),
            None,
        )
        if signature_loop is None:
            # Legacy shape before signature_loop landed.
            return list(wf.root.children)
        # signature_loop.children[0] is the signature_iter sequence.
        signature_iter = signature_loop.children[0]
        return list(signature_iter.children)
    return list(getattr(wf.root, "children", []) or [])


def _designer_loop(wf):
    """Find the inner architect/critic loop. The signature_loop's body is a
    sequence containing classifier → build_blueprint → designer_loop →
    evaluate_signature_loop; we want designer_loop.
    """
    if wf.root.type == "loop":
        return wf.root  # back-compat for pre-signature-loop YAML
    body = _signature_loop_body(wf)
    return next(c for c in body if c.type == "loop")


def test_designer_workflow_loads():
    wf = load_workflow(str(_YAML))
    assert wf.name == "Designer Architect+Critic Loop"
    # PR3-B Layer 1: root is now a sequence wrapping classifier → loop.
    assert wf.root.type == "sequence"
    assert _designer_loop(wf).type == "loop"


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
    until = _designer_loop(wf).config["until"]
    assert until["type"] == "state"
    assert until["key"] == "critic_approved.critic_approved"
    assert until["operator"] == "eq"
    assert until["value"] is True


def test_loop_has_room_for_gate_feedback_iteration():
    wf = load_workflow(str(_YAML))
    assert _designer_loop(wf).config["max_iterations"] >= 4


def test_loop_body_contains_architect_gate_critic():
    wf = load_workflow(str(_YAML))
    body = _designer_loop(wf).children[0]
    assert body.type == "sequence"
    node_ids = [c.id for c in body.children]
    # Expected sequence: architect → parse_architect_ast → structural_gate → gate_router → extract_critic_approved
    assert "architect" in node_ids
    assert "parse_architect_ast" in node_ids
    assert "structural_gate" in node_ids
    assert "gate_router" in node_ids
    assert "extract_critic_approved" in node_ids


def test_architect_guidance_requires_schema_backed_conditionals():
    text = _YAML.read_text()
    assert "Do not route static lanes through a planner or conditional" in text
    assert "declared schema-backed discriminator" in text
    assert "current_step.lane" in text


def test_architect_uses_complex_tier():
    wf = load_workflow(str(_YAML))
    body = _designer_loop(wf).children[0]
    architect = next(c for c in body.children if c.id == "architect")
    assert architect.config["model_tier"] == "complex"
    assert "{gate_result}" in architect.config["user_prompt_template"]
    assert "{critic_verdict}" in architect.config["user_prompt_template"]
    assert "{designer_assets}" in architect.config["user_prompt_template"]


def test_architect_has_asset_and_discovery_tools():
    wf = load_workflow(str(_YAML))
    body = _designer_loop(wf).children[0]
    architect = next(c for c in body.children if c.id == "architect")
    tools = set(architect.config["tools"])
    assert {
        "discover_sources",
        "list_tool_kinds",
        "inspect_assets",
        "recommend_tools_for_assets",
    }.issubset(tools)


def test_structural_gate_receives_designer_assets():
    wf = load_workflow(str(_YAML))
    body = _designer_loop(wf).children[0]
    structural_gate = next(c for c in body.children if c.id == "structural_gate")
    assert structural_gate.config["input_mapping"]["assets"] == "designer_assets"


def test_critic_uses_critic_tier():
    wf = load_workflow(str(_YAML))
    body = _designer_loop(wf).children[0]
    gate_router = next(c for c in body.children if c.id == "gate_router")
    # critic is at child_index 1 (pass branch)
    critic = gate_router.children[1]
    assert critic.id == "critic"
    assert critic.config["model_tier"] == "critic"


# ---------------------------------------------------------------------------
# PR3-B Layer 1: classifier + persist_task_signature wiring
# ---------------------------------------------------------------------------


def test_root_is_sequence_with_classifier_and_loop():
    wf = load_workflow(str(_YAML))
    assert wf.root.type == "sequence"
    body = _signature_loop_body(wf)
    child_ids = [c.id for c in body]
    assert child_ids[0] == "classifier"
    assert "designer_loop" in child_ids


def test_build_blueprint_node_sits_between_classifier_and_designer_loop():
    """Plan v2.1 PR-2 — deterministic blueprint builder is wired between
    the classifier (which produces task_signature) and the designer_loop
    (which contains the architect). The node is flag-gated at execute
    time — when DESIGNER_DETERMINISTIC_BLUEPRINT is OFF the tool returns
    a no-op payload and the legacy architect path runs unaffected.
    """
    wf = load_workflow(str(_YAML))
    body = _signature_loop_body(wf)
    child_ids = [c.id for c in body]
    assert "build_blueprint" in child_ids, (
        "build_blueprint node must exist between classifier and designer_loop"
    )
    i_classifier = child_ids.index("classifier")
    i_blueprint = child_ids.index("build_blueprint")
    i_loop = child_ids.index("designer_loop")
    assert i_classifier < i_blueprint < i_loop, (
        f"Wrong ordering: classifier={i_classifier} "
        f"build_blueprint={i_blueprint} designer_loop={i_loop}; "
        f"build_blueprint must run after classifier and before designer_loop"
    )

    build_blueprint = next(c for c in body if c.id == "build_blueprint")
    assert build_blueprint.type == "tool"
    assert build_blueprint.config["ref"]["name"] == "build_blueprint"
    assert build_blueprint.config["output_key"] == "initial_blueprint"
    input_mapping = build_blueprint.config["input_mapping"]
    assert input_mapping["task_signature"] == "task_signature"
    assert input_mapping["intent"] == "user_intent"
    assert input_mapping["assets"] == "designer_assets"
    assert input_mapping["resolved_tool_contract"] == "resolved_tool_contract"


def test_all_tool_nodes_use_builtin_ref_type():
    """Regression guard for framework ToolNodeConfig.ref.type validation."""

    wf = load_workflow(str(_YAML))
    tool_nodes = [node for node in _walk_nodes(wf.root) if node.type == "tool"]
    assert tool_nodes
    for node in tool_nodes:
        ref = node.config.get("ref") or {}
        assert ref.get("type") == "builtin", (
            f"tool node {node.id!r} must declare config.ref.type=builtin"
        )
        assert ref.get("name"), f"tool node {node.id!r} must declare config.ref.name"


def test_build_blueprint_tool_registered_in_builtin_tools():
    """Sanity check that the YAML's tool ref resolves via the builtin
    designer-tool registry (parallel to the
    test_architect_tools_resolve_via_builtin_designer_tools pattern).
    """
    from deep_research.agent_designer.framework_tools import (
        builtin_designer_tools,
    )

    names = {t.definition.name for t in builtin_designer_tools()}
    assert "build_blueprint" in names
    assert "request_signature_revision" in names


def test_classifier_uses_simple_tier_and_emit_signature_tool():
    wf = load_workflow(str(_YAML))
    body = _signature_loop_body(wf)
    classifier = next(c for c in body if c.id == "classifier")
    assert classifier.type == "agent"
    assert classifier.config["model_tier"] == "simple"
    assert classifier.config["tools"] == ["emit_task_signature"]
    assert classifier.config["max_tool_calls"] == 1
    # The classifier must NOT set ``output_key: task_signature``: doing so
    # would overwrite the structured tool payload (written via the
    # ``emit_task_signature`` tool's ``signature_setter``) with the agent's
    # free-form reasoning prose, which would break the downstream
    # ``build_blueprint`` step (it requires a dict, not text).
    assert classifier.config.get("output_key") != "task_signature"


def test_simple_tier_endpoint_declared():
    wf = load_workflow(str(_YAML))
    models = wf.models or {}
    assert "simple" in models
    assert models["simple"]["endpoints"]  # non-empty


def test_architect_tools_include_select_topology():
    wf = load_workflow(str(_YAML))
    body = _designer_loop(wf).children[0]
    architect = next(c for c in body.children if c.id == "architect")
    assert "select_topology" in architect.config["tools"]


def test_architect_user_prompt_includes_task_signature_slot():
    wf = load_workflow(str(_YAML))
    body = _designer_loop(wf).children[0]
    architect = next(c for c in body.children if c.id == "architect")
    assert "{task_signature}" in architect.config["user_prompt_template"]


def test_agent_prompt_templates_are_safe_renderer_compatible():
    """Every prompt in the designer workflow must pass the runtime renderer.

    The framework's harness calls ``renderer.render(prompt, vars)`` at runtime,
    which runs ``_validate`` and rejects forbidden brace patterns (e.g. JSON
    examples with ``...`` inside braces — they must be ``{{...}}``-escaped).
    ``extract_variables`` alone does NOT run that check, so this test exercises
    the full ``render`` path with a fake variable dict to catch the same class
    of bug that produced ``TemplateSecurityError`` at scaffold-and-run time.
    """
    wf = load_workflow(str(_YAML))
    renderer = SafeTemplateRenderer()

    for node in _walk_nodes(wf.root):
        for field in ("system_prompt", "user_prompt_template"):
            prompt = (node.config or {}).get(field)
            if not prompt:
                continue
            referenced = renderer.extract_variables(prompt)
            fake_vars = dict.fromkeys(referenced, "")
            renderer.render(prompt, fake_vars)


def test_architect_tools_resolve_via_builtin_designer_tools():
    """Every tool listed in the architect's YAML must resolve via the
    builtin_designer_tools() registry. Catches drift like the
    list_tool_kinds gap that produced 'TOOL_NOT_FOUND' warnings in
    the failing scaffold-and-run."""
    import yaml

    from deep_research.agent_designer.framework_tools import builtin_designer_tools

    workflow_yaml = _YAML.read_text()
    parsed = yaml.safe_load(workflow_yaml)

    # Walk to the architect agent node and collect its tools
    def find_architect(node):
        if isinstance(node, dict):
            if node.get("id") == "architect":
                return node
            for child in node.get("children", []) or []:
                found = find_architect(child)
                if found is not None:
                    return found
            for nested_key in ("body", "planner", "evaluator"):
                nested = (node.get("config") or {}).get(nested_key)
                if nested is not None:
                    found = find_architect(nested)
                    if found is not None:
                        return found
        return None

    architect_node = find_architect(parsed.get("root", parsed))
    assert architect_node is not None, "architect node not found in designer_workflow.yaml"
    declared_tools = architect_node.get("config", {}).get("tools") or []
    assert declared_tools, "architect has no tools declared"

    registered_names = {t.definition.name for t in builtin_designer_tools()}
    missing = set(declared_tools) - registered_names
    assert not missing, (
        f"Architect tools declared in YAML but not registered: {sorted(missing)}"
    )


def test_list_tool_kinds_returns_all_kinds():
    from deep_research.agent_designer.framework_tools import ListToolKindsTool

    tool = ListToolKindsTool()
    result = asyncio.run(tool.execute({}, _context=None))  # type: ignore[arg-type]
    payload = json.loads(result.content)
    assert "kinds" in payload
    assert "vector_search" in payload["kinds"]
    assert "web_search" in payload["kinds"]
    assert "table_search" in payload["kinds"]
    assert payload["count"] == len(payload["kinds"])


def test_architect_tools_exclude_structural_mutation_apis():
    """Plan v2.1 final — the architect's binding is read-only for structure.

    Asset→tool wiring is deterministic in ``build_blueprint`` (see
    ``blueprint._build_asset_tool_plan``); pool/tool/node shape is part of
    the structural fingerprint. The architect MUST NOT call these tools —
    any such call either writes to the cached AST and is discarded by
    ``parse_architect_ast`` (patch mode reads only the immutable blueprint
    + ``node_patches``), or drifts the structural fingerprint and is
    rejected at parse time.

    Keeping them off the architect's binding is the prevention; this test
    is the regression guard.
    """
    wf = load_workflow(str(_YAML))
    body = _designer_loop(wf).children[0]
    architect = next(c for c in body.children if c.id == "architect")
    bound = set(architect.config["tools"])
    forbidden = {
        "declare_tool",
        "remove_tool",
        "bind_tool_to_block",
        "update_pool",
        "add_block",
        "delete_block",
        "move_block",
        "propose_workflow",
    }
    leaked = bound & forbidden
    assert not leaked, (
        f"architect's tool binding contains structural mutation APIs "
        f"{sorted(leaked)}; these must be removed to prevent "
        f"silent-discard / structural-drift bugs"
    )


def test_architect_user_prompt_template_includes_lane_keys_variable():
    """Plan v2.1 generic-robustness — the architect needs to READ the
    content-derived lane_keys map to address ``node_patches`` by stable
    key (preferred per plan M7). The YAML user_prompt_template must
    include ``{lane_keys}`` so the renderer pulls
    ``state.lane_keys`` into the architect's context."""
    wf = load_workflow(str(_YAML))
    body = _designer_loop(wf).children[0]
    architect = next(c for c in body.children if c.id == "architect")
    template = architect.config.get("user_prompt_template") or ""
    assert "{lane_keys}" in template, (
        "architect user_prompt_template must reference {lane_keys} so the "
        "renderer surfaces the content-derived lane-key map at runtime"
    )


def test_architect_user_prompt_uses_contract_and_compact_ast_context():
    """Architect hot-loop prompt should not inline full blueprint/current AST blobs."""

    wf = load_workflow(str(_YAML))
    body = _designer_loop(wf).children[0]
    architect = next(c for c in body.children if c.id == "architect")
    template = architect.config.get("user_prompt_template") or ""
    system_prompt = architect.config.get("system_prompt") or ""
    assert "{resolved_tool_contract_summary}" in template
    assert "{initial_blueprint}" not in template
    assert "{current_ast}" not in template
    assert "{initial_blueprint}" not in system_prompt
    assert "{current_ast}" not in system_prompt
    assert "inspect_ast_summary" in template


def test_critic_uses_compact_current_ast_summary():
    """Critic prompt must not inline the full current_ast JSON blob."""

    wf = load_workflow(str(_YAML))
    body = _designer_loop(wf).children[0]
    gate_router = next(c for c in body.children if c.id == "gate_router")
    critic = gate_router.children[1]
    template = critic.config.get("user_prompt_template") or ""
    assert "{current_ast_summary}" in template
    assert "{current_ast}" not in template


def test_architect_step_2_documents_placeholder_pending_contract():
    """The architect system_prompt must teach the model that the FINAL
    ``node_patches`` JSON is the load-bearing artifact (not live
    ``update_block`` calls) and that ``placeholder_pending`` is the gate
    signal it must clear."""
    wf = load_workflow(str(_YAML))
    body = _designer_loop(wf).children[0]
    architect = next(c for c in body.children if c.id == "architect")
    system_prompt = architect.config.get("system_prompt") or ""
    assert "placeholder_pending" in system_prompt, (
        "architect system_prompt must mention placeholder_pending so the "
        "model knows the validator will reject unspecialized lanes"
    )
    assert "ADVISORY ONLY" in system_prompt or "advisory only" in system_prompt, (
        "architect system_prompt must clarify update_block is advisory; "
        "only final node_patches reach the immutable blueprint"
    )


def test_signature_loop_wraps_classifier_and_designer_loop():
    """Plan v2.1 generic-robustness — the root sequence's child is now a
    ``signature_loop`` that wraps classifier + build_blueprint +
    designer_loop. This is the loop-boundary restructure that makes
    ``request_signature_revision`` an actual control-flow mechanism (the
    architect can escalate; the classifier re-runs with the hint)."""
    wf = load_workflow(str(_YAML))
    assert wf.root.type == "sequence"
    signature_loop = next(
        c for c in wf.root.children if c.id == "signature_loop"
    )
    assert signature_loop.type == "loop"
    # K=2 revisions → max_iterations=3 (1 initial + 2 revisions)
    assert signature_loop.config["max_iterations"] >= 3
    # until-condition reads the flat boolean state key emitted by
    # ``evaluate_signature_loop``.
    until = signature_loop.config["until"]
    assert until["type"] == "state"
    assert until["key"] == "signature_loop_done.signature_loop_done"
    assert until["operator"] == "eq"
    assert until["value"] is True


def test_signature_loop_body_contains_all_expected_nodes():
    """The signature_loop body must contain classifier, build_blueprint,
    designer_loop, AND evaluate_signature_loop in that order."""
    wf = load_workflow(str(_YAML))
    body = _signature_loop_body(wf)
    child_ids = [c.id for c in body]
    assert "classifier" in child_ids
    assert "build_blueprint" in child_ids
    assert "designer_loop" in child_ids
    assert "evaluate_signature_loop" in child_ids
    i_class = child_ids.index("classifier")
    i_build = child_ids.index("build_blueprint")
    i_loop = child_ids.index("designer_loop")
    i_eval = child_ids.index("evaluate_signature_loop")
    assert i_class < i_build < i_loop < i_eval


def test_classifier_user_prompt_includes_revision_request_variable():
    """The classifier must surface ``{revision_request}`` so the LLM sees the
    architect's prior rejection (when set) and can re-emit a corrected
    TaskSignature."""
    wf = load_workflow(str(_YAML))
    body = _signature_loop_body(wf)
    classifier = next(c for c in body if c.id == "classifier")
    template = classifier.config.get("user_prompt_template") or ""
    assert "{revision_request}" in template, (
        "classifier user_prompt_template must reference {revision_request} "
        "so the LLM sees the architect's escape-valve hint on re-runs"
    )


def test_evaluate_signature_loop_node_wired_correctly():
    """The evaluate_signature_loop tool node must read all three signals
    (critic_approved, revision_request, revision_count) and emit
    ``signature_loop_done`` to drive the outer loop's until-clause."""
    wf = load_workflow(str(_YAML))
    body = _signature_loop_body(wf)
    evaluator = next(c for c in body if c.id == "evaluate_signature_loop")
    assert evaluator.type == "tool"
    assert evaluator.config["ref"]["name"] == "evaluate_signature_loop"
    input_mapping = evaluator.config["input_mapping"]
    assert input_mapping["critic_approved"] == "critic_approved"
    assert input_mapping["revision_request"] == "revision_request"
    assert input_mapping["revision_count"] == "revision_count"
    assert evaluator.config["output_key"] == "signature_loop_done"


def test_evaluate_signature_loop_tool_registered():
    """Sanity: the new tool must be registered in builtin_designer_tools()."""
    from deep_research.agent_designer.framework_tools import (
        builtin_designer_tools,
    )

    names = {t.definition.name for t in builtin_designer_tools()}
    assert "evaluate_signature_loop" in names


def test_architect_prompt_does_not_instruct_tool_bindings_emit():
    """Plan v2.1 generic-robustness — the architect must NOT be instructed
    to emit ``tool_bindings`` (or any other unknown top-level key). The
    parser rejects these explicitly; allowing the prompt to teach them
    creates documentation drift the model will follow.

    The string ``tool_bindings`` may appear in the prompt as part of a
    NEGATIVE instruction (e.g., "do NOT include tool_bindings" or
    "rejects tool_bindings"). The check below scans for any positive
    instruction telling the model to PRODUCE a tool_bindings block.
    """
    wf = load_workflow(str(_YAML))
    body = _designer_loop(wf).children[0]
    architect = next(c for c in body.children if c.id == "architect")
    full_prompt = (architect.config.get("system_prompt") or "") + "\n" + (
        architect.config.get("user_prompt_template") or ""
    )
    # Forbidden phrasings: anything that frames tool_bindings as a thing
    # the architect SHOULD emit. The accepted phrasings frame it as
    # rejected / forbidden / never-implemented.
    forbidden_phrasings = [
        '"tool_bindings": {<',  # JSON example block
        'tool_bindings`` patch',
        'emit ``tool_bindings``',
        'emit `tool_bindings`',
        'including ``tool_bindings``',
        'including `tool_bindings`',
    ]
    leaked = [phrase for phrase in forbidden_phrasings if phrase in full_prompt]
    assert not leaked, (
        f"architect prompt still instructs tool_bindings emission: "
        f"{leaked}"
    )


def test_architect_keeps_prompt_customization_tools():
    """Positive guard: removing the structural mutators must NOT take out
    the prompt-customization surface the architect actually relies on."""
    wf = load_workflow(str(_YAML))
    body = _designer_loop(wf).children[0]
    architect = next(c for c in body.children if c.id == "architect")
    bound = set(architect.config["tools"])
    required = {
        "update_block",  # system_prompt / user_prompt_template patches
        "set_model_tier",
        "inspect_ast_summary",
        "inspect_assets",
        "recommend_tools_for_assets",
        "request_signature_revision",
    }
    missing = required - bound
    assert not missing, (
        f"architect missing required prompt-customization tools: "
        f"{sorted(missing)}"
    )
