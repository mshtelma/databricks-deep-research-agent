"""Semantic validation for AgentV2 workflow definitions.

The structural Pydantic validator (``load_workflow_from_dict``) catches
malformed JSON / wrong node shapes, but it does not check the
*semantic* invariants we care about:

- Every tool an agent references in ``config.tools`` must be declared in
  the top-level ``definition.tools`` list.
- Built-in tool configurations must satisfy the required-fields schema
  from ``tool_kinds_payload``.

Until W10, these checks lived inside ``api/v1/agent_designer.py`` and
were only invoked by the explicit ``/agent-designer/validate`` route —
which the frontend page calls before save but the ``agents-v2`` CRUD
endpoints did not. That meant any client (e.g., a chat-driven mutation,
a CLI script, or anyone who bypasses the page) could persist an AST
that fails at runtime.

Extracting the helper here lets both surfaces use the same checker:
the validate route and ``CreateAgentV2Request`` / ``UpdateAgentV2Request``
schema validators.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from databricks_deep_research.tools.code_executor import (
    ALLOWED_MODULES,
    DATA_LIBS,
    SkillScriptPolicyError,
    validate_script_source,
)

from deep_research.agent_designer.registry import tool_kinds_payload
from deep_research.core.app_config import get_app_config


@dataclass(frozen=True)
class SemanticValidationError:
    """One semantic violation surfaced by ``semantic_validation_errors``.

    Mirrors the API-layer ``ValidationErrorItem`` shape so the existing
    ``/agent-designer/validate`` response stays unchanged after the
    extraction — the API endpoint adapts dataclasses to its Pydantic
    model. Schema validators use the dataclass directly when raising
    ``ValueError`` from a Pydantic ``field_validator``.

    Plan v2.1 M10 adds ``severity`` to support severity-graded test
    assertions and the eventual PR-4 deletion of
    ``detect_topology_mismatch``. Defaults to ``"blocking"`` so existing
    advice records (which test_scaffold_and_run asserts via
    ``assert not advice``) keep their blocking semantics until the
    test contract is updated in US-03/PR-3.
    """

    message: str
    path: str | None = None
    line: int | None = None
    kind: str = "validation"  # one of "syntax" | "schema" | "validation"
    severity: str = "blocking"  # one of "blocking" | "warning" | "info"


def semantic_validation_errors(
    definition: dict[str, Any],
) -> list[SemanticValidationError]:
    """Return all semantic violations for the given workflow definition.

    Same checks as the previous ``_semantic_validation_errors`` helper in
    ``api/v1/agent_designer.py``. Recurses into ``plan_and_execute``
    nested agents (planner / evaluator / body) and standard
    composite children.
    """
    errors: list[SemanticValidationError] = []
    builtin_tool_schemas = {
        item["kind"]: item.get("config_schema", {})
        for item in tool_kinds_payload()
        if isinstance(item, dict)
    }
    tools = definition.get("tools", []) or []
    declared_tool_names: set[str] = set()
    # name -> declared signature params (uc_function introspected on save,
    # python_function authored). Drives required-param coverage on tool nodes.
    declared_tool_params: dict[str, list[dict[str, Any]]] = {}
    for idx, tool in enumerate(tools):
        if not isinstance(tool, dict):
            errors.append(
                SemanticValidationError(
                    message="Tool declaration must be an object",
                    path=f"tools[{idx}]",
                )
            )
            continue
        name = tool.get("name")
        kind = tool.get("kind")
        if isinstance(name, str) and name:
            declared_tool_names.add(name)
            if kind in ("uc_function", "python_function"):
                raw_params = (tool.get("config") or {}).get("params")
                if isinstance(raw_params, list):
                    declared_tool_params[name] = [
                        p for p in raw_params if isinstance(p, dict)
                    ]
        config = tool.get("config", {})
        if config is None:
            config = {}
        if not isinstance(config, dict):
            errors.append(
                SemanticValidationError(
                    message=f"Tool '{name or idx}' config must be an object",
                    path=f"tools[{idx}].config",
                )
            )
            continue
        schema = builtin_tool_schemas.get(kind)
        if not isinstance(schema, dict):
            continue
        required = schema.get("required", [])
        if not isinstance(required, list):
            continue
        for field in required:
            if not isinstance(field, str):
                continue
            value = config.get(field)
            if value is None or (isinstance(value, str) and not value.strip()):
                errors.append(
                    SemanticValidationError(
                        message=f"Tool '{name or idx}' requires config.{field}",
                        path=f"tools[{idx}].config.{field}",
                    )
                )
        # Generic enum validation: any config key whose schema property declares
        # a non-empty ``enum`` must hold a member value when set. Catches a
        # typo'd ``provider`` / ``model_family`` at design time (the framework's
        # runtime ValueError is the backstop). Lenient by design: absent or blank
        # values are skipped — they inherit the workspace default downstream.
        properties = schema.get("properties")
        if isinstance(properties, dict):
            for prop_name, prop_schema in properties.items():
                if not isinstance(prop_schema, dict):
                    continue
                allowed = prop_schema.get("enum")
                if not isinstance(allowed, list) or not allowed:
                    continue
                value = config.get(prop_name)
                if value is None or (isinstance(value, str) and not value.strip()):
                    continue
                if value not in allowed:
                    errors.append(
                        SemanticValidationError(
                            message=(
                                f"Tool '{name or idx}' config.{prop_name} must be "
                                f"one of {allowed}; got {value!r}."
                            ),
                            path=f"tools[{idx}].config.{prop_name}",
                            kind="schema",
                        )
                    )

        # Cross-field guard: an explicit ``model_family`` that contradicts an
        # explicit ``model`` endpoint is a guaranteed runtime failure — e.g.
        # family=openai on a Gemini endpoint drives the OpenAI Responses API
        # onto a Gemini serving endpoint => HTTP 400 => every search returns
        # zero results. Block the save loudly. Fires only when BOTH are set and
        # the endpoint's family is detectable AND differs; family-only,
        # endpoint-only, and custom/undetectable-endpoint configs all pass.
        if kind in ("web_search", "web_research"):
            model = config.get("model")
            family = config.get("model_family")
            if (
                isinstance(model, str)
                and model.strip()
                and isinstance(family, str)
                and family.strip()
            ):
                detected = get_app_config().search.databricks.family_for_endpoint(model)
                if detected is not None and detected != family:
                    errors.append(
                        SemanticValidationError(
                            message=(
                                f"Tool '{name or idx}' config.model_family "
                                f"'{family}' contradicts endpoint '{model}' "
                                f"(detected family '{detected}'). Set "
                                f"model_family to '{detected}', clear it to "
                                f"auto-detect, or choose a '{family}' endpoint."
                            ),
                            path=f"tools[{idx}].config.model_family",
                            kind="schema",
                        )
                    )

        # registered: the key must exist in the operator catalog — a save that
        # can only fail at run time is a trap.
        if kind == "registered":
            key_value = config.get("key")
            if isinstance(key_value, str) and key_value.strip():
                from deep_research.agent.tools.registered_catalog import (
                    registered_tool_keys,
                )

                available = registered_tool_keys()
                if key_value not in available:
                    errors.append(
                        SemanticValidationError(
                            message=(
                                f"Tool '{name or idx}' config.key {key_value!r} "
                                f"is not in the registered catalog. "
                                f"Available: {available or '(none configured)'}"
                            ),
                            path=f"tools[{idx}].config.key",
                            kind="schema",
                        )
                    )

        # python_function: fail the save on policy-violating code (same check
        # the runtime factory applies) and on gated backends when the operator
        # trust switch is off — a save that can only fail at run time is a trap.
        if kind == "python_function":
            code_value = config.get("code")
            if isinstance(code_value, str) and code_value.strip():
                extra = frozenset(
                    m
                    for m in (config.get("extra_allowed_modules") or [])
                    if isinstance(m, str)
                )
                try:
                    validate_script_source(
                        code_value,
                        allowed_modules=ALLOWED_MODULES | (extra & DATA_LIBS),
                    )
                except SkillScriptPolicyError as exc:
                    errors.append(
                        SemanticValidationError(
                            message=f"Tool '{name or idx}' config.code: {exc}",
                            path=f"tools[{idx}].config.code",
                            kind="schema",
                        )
                    )
            if not get_app_config().execution.allow_inprocess_python_function:
                if config.get("backend") == "restricted":
                    errors.append(
                        SemanticValidationError(
                            message=(
                                f"Tool '{name or idx}': backend 'restricted' is "
                                "disabled on this workspace (in-process execution "
                                "is not a hard boundary); use 'subprocess' or ask "
                                "the operator to enable "
                                "execution.allow_inprocess_python_function."
                            ),
                            path=f"tools[{idx}].config.backend",
                            kind="schema",
                        )
                    )
                if config.get("data_lib_mode") == "live":
                    errors.append(
                        SemanticValidationError(
                            message=(
                                f"Tool '{name or idx}': data_lib_mode 'live' "
                                "requires the operator trust switch; use the "
                                "default 'facade' mode."
                            ),
                            path=f"tools[{idx}].config.data_lib_mode",
                            kind="schema",
                        )
                    )

        # uc_function is invoked via SQL: the FQN is backtick-quoted per part
        # and requires a strict catalog.schema.function of [A-Za-z0-9_] (the
        # UCFunctionTool applies the same regex at runtime — hyphenated catalog
        # names are unsupported in v1).
        if kind == "uc_function":
            fqn = config.get("function")
            fqn_str = fqn.strip() if isinstance(fqn, str) else ""
            if fqn_str and not re.fullmatch(
                r"[A-Za-z0-9_]+\.[A-Za-z0-9_]+\.[A-Za-z0-9_]+", fqn_str
            ):
                errors.append(
                    SemanticValidationError(
                        message=(
                            f"Tool '{name or idx}' config.function must be a fully "
                            f"qualified catalog.schema.function (letters, digits, "
                            f"underscores); got {fqn_str!r}."
                        ),
                        path=f"tools[{idx}].config.function",
                        kind="schema",
                    )
                )

    # Valid MCP server names: the workflow's top-level ``mcp_servers`` PLUS any
    # ``kind == 'mcp'`` tool cards not yet lifted (so the rule holds pre- and
    # post-normalization). Agents bind to these by name via ``config.mcp_servers``
    # (B2) — separate from the declared-tools rule, since discovered MCP tool
    # names are not statically known at author time.
    mcp_server_names: set[str] = set()
    # Tool-node refs of type 'mcp' resolve against runtime-discovered tool
    # names. Allow lists make those names statically known; a server WITHOUT
    # an allow list exposes an unknowable set, so ref validation must stay
    # permissive when one exists.
    mcp_allowed_tool_names: set[str] = set()
    has_open_mcp_server = False

    def _collect_mcp_allow(server_cfg: dict[str, Any]) -> None:
        nonlocal has_open_mcp_server
        allow = server_cfg.get("allow")
        prefix = server_cfg.get("name_prefix")
        prefix = prefix if isinstance(prefix, str) else ""
        if isinstance(allow, list) and allow:
            for entry in allow:
                if isinstance(entry, str) and entry:
                    mcp_allowed_tool_names.add(entry)
                    if prefix:
                        mcp_allowed_tool_names.add(f"{prefix}{entry}")
        else:
            has_open_mcp_server = True

    for server in definition.get("mcp_servers", []) or []:
        if isinstance(server, dict):
            if isinstance(server.get("name"), str):
                mcp_server_names.add(server["name"])
            _collect_mcp_allow(server)
    for tool in tools:
        if isinstance(tool, dict) and tool.get("kind") == "mcp":
            cfg = tool.get("config")
            cfg = cfg if isinstance(cfg, dict) else {}
            mcp_name = cfg.get("name") or tool.get("name")
            if isinstance(mcp_name, str) and mcp_name:
                mcp_server_names.add(mcp_name)
            _collect_mcp_allow(cfg)

    def validate_agent_tools(config: dict[str, Any], path: str) -> None:
        raw_tools = config.get("tools", [])
        if not isinstance(raw_tools, list):
            return
        for tool_name in raw_tools:
            if isinstance(tool_name, str) and tool_name not in declared_tool_names:
                errors.append(
                    SemanticValidationError(
                        message=(
                            f"Agent references undeclared tool '{tool_name}'"
                        ),
                        path=f"{path}.config.tools",
                    )
                )

    def validate_agent_mcp_servers(config: dict[str, Any], path: str) -> None:
        refs = config.get("mcp_servers", [])
        if not isinstance(refs, list):
            return
        for ref in refs:
            if isinstance(ref, str) and ref and ref not in mcp_server_names:
                errors.append(
                    SemanticValidationError(
                        message=(
                            f"Agent references undeclared MCP server '{ref}'; "
                            "declare it in the workflow's mcp_servers list."
                        ),
                        path=f"{path}.config.mcp_servers",
                    )
                )

    def validate_tool_node_ref(config: dict[str, Any], path: str) -> None:
        ref = config.get("ref")
        if not isinstance(ref, dict):
            return  # shape errors are ToolNodeConfig's (pydantic) job
        ref_name = ref.get("name")
        if not isinstance(ref_name, str) or not ref_name:
            return
        ref_type = ref.get("type", "builtin")
        if ref_name in declared_tool_names or ref_name in mcp_allowed_tool_names:
            return
        if ref_type == "mcp" and has_open_mcp_server:
            return  # tools of allow-less servers are only discoverable at runtime
        errors.append(
            SemanticValidationError(
                message=(
                    f"Tool node references unknown tool '{ref_name}' "
                    f"(type '{ref_type}'); declare it in the workflow's tools "
                    "or expose it via an mcp_servers allow list."
                ),
                path=f"{path}.config.ref",
            )
        )

    def validate_tool_node_params(config: dict[str, Any], path: str) -> None:
        """Required declared params must be bound via input_mapping or
        input_literals — an unbound required arg fails the SQL/sandbox call at
        runtime, so catch it at save time (tool-UX plan Phase 2)."""
        ref = config.get("ref")
        if not isinstance(ref, dict):
            return
        ref_name = ref.get("name")
        if not isinstance(ref_name, str) or ref_name not in declared_tool_params:
            return
        params = declared_tool_params[ref_name]
        if not params:
            return  # unknown/uninstrospected signature — nothing to enforce
        mapping = config.get("input_mapping")
        literals = config.get("input_literals")
        bound: set[str] = set()
        if isinstance(mapping, dict):
            bound |= set(mapping)
        if isinstance(literals, dict):
            bound |= set(literals)
        missing = [
            str(p.get("name"))
            for p in params
            if p.get("name") and p.get("required", True) and p.get("name") not in bound
        ]
        if missing:
            errors.append(
                SemanticValidationError(
                    message=(
                        f"Tool node calls '{ref_name}' without binding required "
                        f"parameter(s) {missing}; map them from workflow state "
                        "(input_mapping) or set literals (input_literals)."
                    ),
                    path=f"{path}.config.input_mapping",
                )
            )

    def walk(node: Any, path: str) -> None:
        if not isinstance(node, dict):
            return
        config = node.get("config", {})
        if not isinstance(config, dict):
            config = {}
        if node.get("type") == "agent":
            validate_agent_tools(config, path)
            validate_agent_mcp_servers(config, path)
        if node.get("type") == "tool":
            validate_tool_node_ref(config, path)
            validate_tool_node_params(config, path)
        if node.get("type") == "plan_and_execute":
            for nested_key in ("planner", "evaluator"):
                nested = config.get(nested_key)
                if isinstance(nested, dict):
                    validate_agent_tools(nested, f"{path}.config.{nested_key}")
                    validate_agent_mcp_servers(nested, f"{path}.config.{nested_key}")
            body = config.get("body")
            if isinstance(body, dict):
                walk(body, f"{path}.config.body")
        for child_idx, child in enumerate(node.get("children", []) or []):
            walk(child, f"{path}.children[{child_idx}]")

    walk(definition.get("root"), "root")
    # NOTE: ``detect_unspecialized_agents`` is intentionally NOT merged here.
    # CRUD save paths (``CreateAgentV2Request`` / ``UpdateAgentV2Request``)
    # reject the request when ``semantic_validation_errors`` returns
    # anything, so this entry point must stay limited to STRUCTURAL invariants
    # (undeclared tool refs, builtin tool config-schema violations). Per-agent
    # quality defects (shallow system_prompt, missing tool bindings, default
    # model_tier on synthesizers) are ADVICE to the Designer LLM during chat,
    # not hard CRUD errors — surface them via ``detect_unspecialized_agents``
    # directly from the validate-tool path / chat orchestrator, never from
    # the schema validators.
    return errors


# Heuristic thresholds for the per-agent property checks. These are
# deliberately conservative — false positives surface as advice for the LLM,
# never as hard failures, so the cost of over-flagging is small.
_MIN_SYSTEM_PROMPT_CHARS = 200
# Designers can author valid lane prompts directly, and different topologies
# should not be forced through one builder wording. The negative signal we
# reject is the legacy generic researcher prompt opening.
_DEFAULT_METHOD_OPENING_MARKER = (
    "You are the Researcher agent for a deep research system."
)
_DEFAULT_MODEL_TIER = "analytical"

# Markers for detecting that a researcher node is still on the generic
# ``RESEARCHER_USER_PROMPT`` builtin (from
# ``databricks_deep_research.agents.prompts.researcher``) instead of a
# designer-/planner-authored investigation brief. Either marker is a near-
# certain sign of generic fallback.
_GENERIC_USER_PROMPT_MARKERS = (
    "Execute the following research step:",
    "{step_title}",
)
_MIN_USER_PROMPT_TEMPLATE_CHARS = 250
_SUBQUESTION_HEADING_MARKERS = ("sub-questions", "subquestions", "sub questions")
_OUTPUT_SECTION_HEADING_MARKERS = (
    "required output",
    "output structure",
    "output sections",
)
# The lane-prompt builder emits one of two heading variants depending on
# the workflow's evidence path: "Search strategy" for web-flavoured lanes
# and "Retrieval strategy" for corpus-only lanes (see
# ``workflow_builder._CORPUS_RETRIEVAL_STRATEGY_BLOCK``). The validator
# accepts either so the corpus path does not trigger a spurious "missing
# block" finding when the prompt was generated correctly.
_SEARCH_STRATEGY_HEADING_MARKERS = ("search strategy", "retrieval strategy")
_UNKNOWNS_HANDLING_MARKERS = (
    "data unavailable",
    "definition of done",
    "do not improvise",
)


def _user_prompt_template_is_generic_default(template: str) -> bool:
    """Detect whether the researcher node is still on the generic builtin."""
    if not template:
        return True
    return any(marker in template for marker in _GENERIC_USER_PROMPT_MARKERS)


def _count_numbered_items_under_heading(
    template: str, heading_markers: tuple[str, ...]
) -> int:
    """Count numbered list items appearing after the first matching heading.

    Stops counting at the next blank-line-separated block. Heading match is
    case-insensitive and substring-based so designer prose variations like
    "### Sub-questions you MUST address" or "## Sub-questions" both match.
    """
    if not template:
        return 0
    lower_lines = template.splitlines()
    heading_idx: int | None = None
    for idx, raw_line in enumerate(lower_lines):
        line_lower = raw_line.lower()
        if any(marker in line_lower for marker in heading_markers):
            heading_idx = idx
            break
    if heading_idx is None:
        return 0
    count = 0
    blank_streak = 0
    for raw_line in lower_lines[heading_idx + 1 :]:
        stripped = raw_line.strip()
        if not stripped:
            blank_streak += 1
            if blank_streak >= 2 and count > 0:
                break
            continue
        blank_streak = 0
        if stripped.startswith("#"):
            # Next heading reached.
            break
        if (
            len(stripped) > 2
            and stripped[0].isdigit()
            and (stripped[1] == "." or (len(stripped) > 2 and stripped[1].isdigit() and stripped[2] == "."))
        ):
            count += 1
    return count


def _has_heading(template: str, heading_markers: tuple[str, ...]) -> bool:
    if not template:
        return False
    lower = template.lower()
    return any(marker in lower for marker in heading_markers)


def _has_marker(template: str, markers: tuple[str, ...]) -> bool:
    if not template:
        return False
    lower = template.lower()
    return any(marker in lower for marker in markers)


def _count_output_section_bullets(template: str) -> int:
    """Count Markdown bullet lines under the output-structure heading."""
    if not template:
        return 0
    lower_lines = template.splitlines()
    heading_idx: int | None = None
    for idx, raw_line in enumerate(lower_lines):
        line_lower = raw_line.lower()
        if any(marker in line_lower for marker in _OUTPUT_SECTION_HEADING_MARKERS):
            heading_idx = idx
            break
    if heading_idx is None:
        return 0
    count = 0
    blank_streak = 0
    for raw_line in lower_lines[heading_idx + 1 :]:
        stripped = raw_line.strip()
        if not stripped:
            blank_streak += 1
            if blank_streak >= 2 and count > 0:
                break
            continue
        blank_streak = 0
        if stripped.startswith("#"):
            break
        if stripped.startswith(("-", "*", "+")):
            count += 1
    return count


def _agent_role(node: dict[str, Any], config: dict[str, Any]) -> str:
    """Return a role label used in validate-time messages and the heuristics
    that decide whether a property needs specialization.
    """
    subtype = str(config.get("subtype", "")).strip().lower()
    if subtype:
        return subtype
    node_id = str(node.get("id", "")).lower()
    if "synthesizer" in node_id or "synth" in node_id:
        return "synthesizer"
    if "researcher" in node_id or "lane" in node_id:
        return "researcher"
    if "planner" in node_id:
        return "planner"
    if "coordinator" in node_id:
        return "coordinator"
    if "reflector" in node_id or "critic" in node_id:
        return "reflector"
    return "agent"


def _is_lane_researcher(node: dict[str, Any], config: dict[str, Any]) -> bool:
    role = _agent_role(node, config)
    if role != "researcher":
        return False
    node_id = str(node.get("id", "")).lower()
    return "lane" in node_id or "researcher" in node_id


def _collect_agent_paths(
    definition: dict[str, Any],
) -> list[tuple[dict[str, Any], dict[str, Any], str]]:
    """Yield ``(node, config, path)`` for every agent node in the workflow,
    recursing into ``plan_and_execute`` planner/evaluator/body and into
    composite children.
    """
    collected: list[tuple[dict[str, Any], dict[str, Any], str]] = []

    def walk(node: Any, path: str) -> None:
        if not isinstance(node, dict):
            return
        config = node.get("config") or {}
        if not isinstance(config, dict):
            config = {}
        if node.get("type") == "agent":
            collected.append((node, config, path))
        if node.get("type") == "plan_and_execute":
            for nested_key in ("planner", "evaluator"):
                nested = config.get(nested_key)
                if isinstance(nested, dict):
                    nested_path = f"{path}.config.{nested_key}"
                    # planner/evaluator are agent-shaped configs without the
                    # outer "type"=="agent" wrapper; treat them as agents.
                    collected.append((nested, nested, nested_path))
            body = config.get("body")
            if isinstance(body, dict):
                walk(body, f"{path}.config.body")
        for idx, child in enumerate(node.get("children", []) or []):
            walk(child, f"{path}.children[{idx}]")

    walk(definition.get("root"), "root")
    return collected


def detect_unspecialized_agents(
    definition: dict[str, Any],
) -> list[SemanticValidationError]:
    """Surface per-agent specialization gaps so the Designer LLM can fix them.

    Each defect is emitted as a ``SemanticValidationError`` with ``kind`` set
    to ``"validation"`` (same shape as the existing tool-binding checks). The
    message names the suggested follow-up tool call (``update_block`` /
    ``bind_tool_to_block`` / ``set_model_tier``) so the LLM has a clear next
    action when it sees the error.

    Checks performed:
      1. system_prompt missing, empty, or under ``_MIN_SYSTEM_PROMPT_CHARS``
         characters (the builtin scaffold alone exceeds this; a shorter value
         means the LLM stripped or replaced it without filling in real content).
      2. Lane researcher's system_prompt lacks the ``## Lane Specialization``
         block — i.e. the LLM did not populate ``research_lanes[].system_prompt``
         at propose_workflow time and did not patch it via update_block after.
      3. Lane researcher has no tools bound when the workflow declares any
         retrieval tools at the top level (web_search, web_research, web_crawl,
         vector_search, genie, knowledge_assistant, file_search, Delta/table
         retrieval tools).
      4. Synthesizer agent left on the default ``analytical`` model_tier when
         the workflow has 2+ lane producers feeding into it — multi-source
         reconciliation benefits from the ``complex`` tier.
    """
    errors: list[SemanticValidationError] = []
    agents = _collect_agent_paths(definition)

    # Top-level retrieval tools available for binding.
    retrieval_kinds = {
        "web_search",
        "web_research",
        "web_crawl",
        "vector_search",
        "genie",
        "knowledge_assistant",
        "file_search",
        "table_discovery",
        "table_search",
        "table_read",
        "table_neighbors",
        "table_load",
        "table_aggregate",
    }
    declared_tools = definition.get("tools", []) or []
    available_retrieval_tools: list[str] = []
    for tool in declared_tools:
        if not isinstance(tool, dict):
            continue
        kind = tool.get("kind")
        name = tool.get("name")
        if isinstance(kind, str) and kind in retrieval_kinds and isinstance(name, str):
            available_retrieval_tools.append(name)

    lane_researcher_count = sum(
        1 for node, config, _ in agents if _is_lane_researcher(node, config)
    )

    for node, config, path in agents:
        label = node.get("label") or node.get("id") or "agent"
        role = _agent_role(node, config)
        is_lane = _is_lane_researcher(node, config)

        # Plan v2.1 generic-robustness — Check 0: deterministic blueprint
        # placeholder lifecycle list at the AST top level
        # (``definition["placeholder_pending_nodes"]``). Every researcher
        # node the blueprint builder emits is stamped there at scaffold
        # time (see ``blueprint._stamp_placeholder_pending``); the list is
        # pruned by ``framework_tools._apply_architect_patches`` ONLY when
        # the architect's final ``node_patches`` JSON delivers a non-empty
        # ``system_prompt`` or ``user_prompt_template`` for that node id.
        # If the node id still appears in the list, the architect shipped
        # a placeholder prompt — block the workflow and direct the
        # architect at the correct fix.
        node_id = str(node.get("id") or "")
        pending_list = definition.get("placeholder_pending_nodes")
        if (
            is_lane
            and node_id
            and isinstance(pending_list, list)
            and node_id in pending_list
        ):
            # severity=warning so the gate surfaces the issue as
            # critic-feedback without blocking the workflow. The runtime
            # ReAct loop can compensate for thin lane prompts in the short
            # term while the architect's prompt is engineered to satisfy
            # the lifecycle contract. Flipping back to blocking is a
            # follow-up once we have telemetry confirming the architect
            # reliably customizes every lane.
            errors.append(
                SemanticValidationError(
                    message=(
                        f"Researcher '{label}' has no architect-authored "
                        "prompt — the deterministic blueprint placeholder "
                        "is still in place. Emit `node_patches: "
                        "{<lane_key>: {system_prompt: <80-300 words of "
                        "use-case-specific guidance>, user_prompt_template: "
                        "<lane-focused investigation brief>}}` in your final "
                        "fenced JSON block. Live `update_block` calls during "
                        "the ReAct loop are advisory only; only the final "
                        "JSON's node_patches reaches the immutable blueprint."
                    ),
                    path=f"{path}.config",
                    kind="placeholder_pending",
                    severity="warning",
                )
            )

        system_prompt = config.get("system_prompt") or ""
        if not isinstance(system_prompt, str):
            system_prompt = str(system_prompt or "")

        # Check 1: system_prompt missing / too short.
        if not system_prompt.strip():
            errors.append(
                SemanticValidationError(
                    message=(
                        f"Agent '{label}' has an empty system_prompt — call "
                        "update_block on this node with a specialized "
                        "system_prompt (80-300 words: what to investigate, "
                        "what to cite, what to flag, what NOT to do)."
                    ),
                    path=f"{path}.config.system_prompt",
                )
            )
        elif len(system_prompt) < _MIN_SYSTEM_PROMPT_CHARS:
            errors.append(
                SemanticValidationError(
                    message=(
                        f"Agent '{label}' has a short system_prompt "
                        f"({len(system_prompt)} chars) — likely missing real "
                        "specialization. Call update_block with 80-300 words "
                        "of task-specific researcher guidance (what to "
                        "investigate, what to cite, what to flag, what NOT "
                        "to do)."
                    ),
                    path=f"{path}.config.system_prompt",
                )
            )

        # Check 2: lane researcher still on the default scaffold. The
        # builder's specialized preamble is only one valid prompt shape; direct
        # designer-authored prompts are valid as long as they are not the
        # builtin generic researcher scaffold.
        elif is_lane and _DEFAULT_METHOD_OPENING_MARKER in system_prompt:
            errors.append(
                SemanticValidationError(
                    message=(
                        f"Lane researcher '{label}' is still on the generic "
                        "researcher prompt — no task-specific specialization "
                        "has landed on this agent. Either re-issue "
                        "propose_workflow with research_lanes[].system_prompt "
                        "populated for this lane, or call update_block here "
                        "with 80-300 words of task-specific guidance (what to "
                        "investigate, what to cite, what to flag, what NOT "
                        "to do)."
                    ),
                    path=f"{path}.config.system_prompt",
                )
            )

        # Check 3: lane researcher with no tools bound.
        if is_lane and available_retrieval_tools:
            bound_tools = config.get("tools") or []
            if isinstance(bound_tools, list) and not bound_tools:
                errors.append(
                    SemanticValidationError(
                        message=(
                            f"Lane researcher '{label}' has no tools bound "
                            f"(available top-level retrieval tools: "
                            f"{', '.join(available_retrieval_tools)}). Call "
                            "bind_tool_to_block to bind the appropriate "
                            "retrieval, table, or compute tools for this "
                            "research lane."
                        ),
                        path=f"{path}.config.tools",
                    )
                )

        # Check 4: synthesizer left on default model_tier with 2+ producers.
        if role == "synthesizer" and lane_researcher_count >= 2:
            tier = str(config.get("model_tier") or "").strip().lower()
            if not tier or tier == _DEFAULT_MODEL_TIER:
                errors.append(
                    SemanticValidationError(
                        message=(
                            f"Synthesizer '{label}' is on the default "
                            f"'{tier or _DEFAULT_MODEL_TIER}' model_tier but "
                            f"the workflow has {lane_researcher_count} lane "
                            "producers feeding it. Multi-source reconciliation "
                            "benefits from the 'complex' tier — call "
                            "set_model_tier on this node."
                        ),
                        path=f"{path}.config.model_tier",
                    )
                )

        # Check 5: researcher nodes must carry a designer-authored
        # user_prompt_template. Without it, the lane receives the generic
        # builtin (which references {step_title}/{step_description} and
        # offers no concrete sub-questions). Lanes then emit planning text
        # ("Let me search for...") into findings — the failure mode the
        # live planning-leak traces exposed. Applies to lane researchers and any other
        # researcher subtype that isn't a planner/reflector/synthesizer.
        if role == "researcher":
            template = config.get("user_prompt_template") or ""
            if not isinstance(template, str):
                template = str(template or "")
            template_stripped = template.strip()
            if not template_stripped:
                errors.append(
                    SemanticValidationError(
                        message=(
                            f"Researcher '{label}' has no user_prompt_template "
                            "— the lane will receive only the raw query as its "
                            "user message and will emit planning text into "
                            "findings. Re-issue propose_workflow with "
                            "research_lanes[].user_prompt_template populated "
                            "(see the contract in the designer system prompt), "
                            "or call update_block on this node with a concrete "
                            "investigation brief: restate {query}, 5 sub-"
                            "questions, 3 output sections, search strategy, "
                            "definition of done."
                        ),
                        path=f"{path}.config.user_prompt_template",
                    )
                )
            elif _user_prompt_template_is_generic_default(template_stripped):
                errors.append(
                    SemanticValidationError(
                        message=(
                            f"Researcher '{label}' is still on the generic "
                            "RESEARCHER_USER_PROMPT default — no designer-"
                            "authored investigation brief has landed. The "
                            "lane will dump planning text into findings "
                            "because it has no concrete sub-questions to "
                            "answer. Re-issue propose_workflow with "
                            "research_lanes[].user_prompt_template populated, "
                            "or call update_block on this node with the brief."
                        ),
                        path=f"{path}.config.user_prompt_template",
                    )
                )
            elif len(template_stripped) < _MIN_USER_PROMPT_TEMPLATE_CHARS:
                errors.append(
                    SemanticValidationError(
                        message=(
                            f"Researcher '{label}' has a short "
                            f"user_prompt_template ({len(template_stripped)} "
                            "chars) — likely missing the structural contract. "
                            "Expand to ≥250 chars covering: 5 sub-questions, "
                            "3 output sections, search strategy, definition "
                            "of done."
                        ),
                        path=f"{path}.config.user_prompt_template",
                    )
                )
            else:
                # Structural contract checks: each missing element emits its
                # own targeted feedback so the LLM can fix them individually.
                if "{query}" not in template_stripped and "query" not in template_stripped.lower()[:300]:
                    errors.append(
                        SemanticValidationError(
                            message=(
                                f"Researcher '{label}' user_prompt_template "
                                "does not restate the user's query in its "
                                "opening — add a 'You are investigating: "
                                "**{query}**' line referencing the {query} "
                                "template variable so the runtime fills it."
                            ),
                            path=f"{path}.config.user_prompt_template",
                        )
                    )
                subquestion_count = _count_numbered_items_under_heading(
                    template_stripped, _SUBQUESTION_HEADING_MARKERS
                )
                if subquestion_count < 5:
                    errors.append(
                        SemanticValidationError(
                            message=(
                                f"Researcher '{label}' user_prompt_template "
                                f"has {subquestion_count} numbered sub-"
                                "questions; the contract requires exactly 5. "
                                "Each sub-question must reference a specific "
                                "noun from the user's query, be answerable "
                                "through the workflow's available evidence "
                                "tools, and end with a question mark."
                            ),
                            path=f"{path}.config.user_prompt_template",
                        )
                    )
                output_bullet_count = _count_output_section_bullets(template_stripped)
                if not _has_heading(template_stripped, _OUTPUT_SECTION_HEADING_MARKERS):
                    errors.append(
                        SemanticValidationError(
                            message=(
                                f"Researcher '{label}' user_prompt_template "
                                "is missing a 'Required output structure' "
                                "block. Add 3 bullet sections with 2-3 "
                                "sentences of evidence guidance each."
                            ),
                            path=f"{path}.config.user_prompt_template",
                        )
                    )
                elif output_bullet_count < 3:
                    errors.append(
                        SemanticValidationError(
                            message=(
                                f"Researcher '{label}' output structure has "
                                f"{output_bullet_count} bullets; the contract "
                                "requires exactly 3 output sections."
                            ),
                            path=f"{path}.config.user_prompt_template",
                        )
                    )
                if not _has_heading(template_stripped, _SEARCH_STRATEGY_HEADING_MARKERS):
                    errors.append(
                        SemanticValidationError(
                            message=(
                                f"Researcher '{label}' user_prompt_template "
                                "is missing a 'Search strategy' block. Add "
                                "≥2 bullets covering query focus, primary "
                                "sources, and refinement strategy."
                            ),
                            path=f"{path}.config.user_prompt_template",
                        )
                    )
                if not _has_marker(template_stripped, _UNKNOWNS_HANDLING_MARKERS):
                    errors.append(
                        SemanticValidationError(
                            message=(
                                f"Researcher '{label}' user_prompt_template "
                                "is missing an unknowns-handling clause "
                                "(e.g., 'Data unavailable — do not improvise'). "
                                "Add a Definition of Done block specifying "
                                "how to mark unanswerable sub-questions."
                            ),
                            path=f"{path}.config.user_prompt_template",
                        )
                    )

    return errors


def _declared_pool_names(definition: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for pool in definition.get("pools", []) or []:
        if isinstance(pool, dict) and isinstance(pool.get("name"), str):
            names.add(pool["name"])
    return names


def _config_pool_names(config: dict[str, Any], key: str) -> set[str]:
    names: set[str] = set()
    for item in config.get(key, []) or []:
        if isinstance(item, dict) and isinstance(item.get("pool"), str):
            names.add(item["pool"])
    return names


def detect_grounded_research_contract(
    definition: dict[str, Any],
) -> list[SemanticValidationError]:
    """Flag grounded synthesizers that are not wired to evidence pools.

    This check is topology-agnostic. ``parallel_lanes`` and
    ``plan_and_execute`` are both valid, but grounded synthesis requires the
    same runtime contract in either shape: researchers write observations and
    citeable sources, and the synthesizer reads both pools.
    """
    if not isinstance(definition, dict):
        return []
    errors: list[SemanticValidationError] = []
    agents = _collect_agent_paths(definition)
    pool_names = _declared_pool_names(definition)

    researchers = [
        (config, path)
        for _node, config, path in agents
        if _agent_role(_node, config) == "researcher"
    ]
    has_observation_writer = any(
        "observations" in _config_pool_names(config, "pool_writes")
        for config, _path in researchers
    )
    has_source_writer = any(
        "sources" in _config_pool_names(config, "pool_writes")
        for config, _path in researchers
    )

    for node, config, path in agents:
        if _agent_role(node, config) != "synthesizer":
            continue
        grounding_mode = str(config.get("grounding_mode") or "none").lower()
        if grounding_mode == "none":
            continue
        injected = _config_pool_names(config, "pool_inject")
        if "sources" not in pool_names:
            errors.append(
                SemanticValidationError(
                    message=(
                        "Grounded synthesizer requires a top-level 'sources' "
                        "pool so citeable source records can be collected."
                    ),
                    path=f"{path}.config.pool_inject",
                    kind="grounding_contract",
                )
            )
        if "observations" not in pool_names:
            errors.append(
                SemanticValidationError(
                    message=(
                        "Grounded synthesizer requires a top-level "
                        "'observations' pool so factual findings are separated "
                        "from planner/control text."
                    ),
                    path=f"{path}.config.pool_inject",
                    kind="grounding_contract",
                )
            )
        if "sources" not in injected or "observations" not in injected:
            errors.append(
                SemanticValidationError(
                    message=(
                        "Grounded synthesizer must pool_inject both "
                        "'observations' and 'sources'. Planner, evaluator, and "
                        "reflection outputs are not citeable evidence."
                    ),
                    path=f"{path}.config.pool_inject",
                    kind="grounding_contract",
                )
            )
        if not has_source_writer or not has_observation_writer:
            errors.append(
                SemanticValidationError(
                    message=(
                        "Grounded workflow needs at least one researcher that "
                        "pool_writes both observations and sources before the "
                        "synthesizer runs."
                    ),
                    path=path,
                    kind="grounding_contract",
                )
            )
    return errors


def detect_topology_mismatch(
    definition: dict[str, Any],
) -> list[SemanticValidationError]:
    """Advise on ``plan_and_execute`` workflows that may fit ``parallel_lanes``.

    Background: ``plan_and_execute + lane_router`` depends on the planner
    LLM reliably stamping ``current_step.lane`` on each step it emits, then
    the conditional router matching that string against the lane ids. In
    practice the planner often emits steps without a valid lane field — the
    router falls through to the cross-lane fallback, and every specialized
    lane researcher is bypassed. Live traces have shown this happen for
    every step in real workflows.

    For workflows whose lanes are independent and each visited once,
    ``parallel_lanes`` can be simpler and more reliable. This finding is
    advisory only; workflows that genuinely need sequential planning,
    adaptive execution, or reflection-driven replanning can remain
    ``plan_and_execute`` as long as they satisfy the evidence contract.

    Heuristic: a ``plan_and_execute`` node with ≥4 lane researchers in its
    lane_router children is a near-certain parallel_lanes candidate. Updated
    from ``≥3`` in PR3-B: signature-driven pipelined_retrieve_read_compute
    workflows legitimately use 3 lanes (retrieve → read → compute) inside
    plan_and_execute.
    """
    errors: list[SemanticValidationError] = []

    def walk(node: Any, path: str) -> None:
        if not isinstance(node, dict):
            return
        if node.get("type") == "plan_and_execute":
            config = node.get("config") or {}
            body = config.get("body") if isinstance(config, dict) else None
            lane_router = None
            if isinstance(body, dict):
                # Body is typically sequence([lane_router, reflector]).
                for body_child in body.get("children", []) or []:
                    if (
                        isinstance(body_child, dict)
                        and body_child.get("type") == "conditional"
                    ):
                        lane_router = body_child
                        break
            if isinstance(lane_router, dict):
                router_children = lane_router.get("children", []) or []
                # Exclude the cross-lane fallback (the last child).
                lane_count = max(0, len(router_children) - 1)
                if lane_count >= 4:
                    errors.append(
                        SemanticValidationError(
                            message=(
                                f"Workflow uses plan_and_execute with "
                                f"{lane_count} lanes routed via a conditional. "
                                "This pattern depends on the planner stamping "
                                "current_step.lane on every step. If these "
                                "lanes are independent and each should run "
                                "once, consider topology='parallel_lanes'. "
                                "If the workflow needs adaptive planning or "
                                "reflection-driven replanning, keep "
                                "plan_and_execute and verify that every "
                                "research branch writes observations and "
                                "sources for grounded synthesis."
                            ),
                            path=path,
                        )
                    )
        config = node.get("config") or {}
        if isinstance(config, dict):
            body = config.get("body")
            if isinstance(body, dict):
                walk(body, f"{path}.config.body")
        for idx, child in enumerate(node.get("children", []) or []):
            walk(child, f"{path}.children[{idx}]")

    walk(definition.get("root"), "root")
    return errors


_NOUN_RE = re.compile(r"\b[a-z]{5,}\b", re.IGNORECASE)
_STOPWORDS = frozenset({
    "agent", "with", "that", "this", "will", "from", "into", "than", "over", "also", "when", "such",
    "where", "which", "what", "your", "their", "then", "they", "there", "these", "those", "each",
    "every", "some", "most", "more", "less", "very", "just", "make", "made", "take", "took", "give",
    "gave", "find", "look", "tool", "tools", "step", "steps", "work", "works", "worked",
    "user", "query", "queries", "prompt", "prompts", "report", "reports", "reportable", "produce",
    "produces", "generated", "input", "output", "outputs", "provided", "provide", "provides",
    "analysis", "analyze", "analyzes", "analyzing", "research", "researcher", "researchers",
    "researching", "describe", "described", "describes", "using", "based", "follow", "followed",
    "follows", "include", "includes", "included", "return", "returns", "returned", "section",
    "sections", "field", "fields", "required", "optional", "example", "examples", "details",
    "detailed", "approach", "approaches",
    # Generic framework / scaffolding words present in every agent's boilerplate.
    # These must not count as domain-specific vocabulary.
    "deep", "build", "system", "create", "concise", "dense", "role", "core",
    "principle", "claim", "supported", "evidence", "simple", "language", "brief",
    "assistant", "response", "request", "original", "workflow", "designed", "task",
    "must", "always", "never", "only", "avoid", "ensure", "important", "note",
    "above", "below", "first", "second", "third", "fourth", "fifth", "sixth",
    "following", "given", "please", "other", "both", "either",
    "before", "after", "during", "while", "once", "until", "unless", "whether",
    # Common adjectives / generic nouns that appear in both descriptions and generic boilerplate.
    "comprehensive", "multi", "covers", "cover", "covering", "company",
    "accepts", "ticker", "available", "information", "document", "documents", "name",
    "general", "specific", "multiple", "single", "various", "additional", "current",
    "previous", "recent", "different", "similar", "relevant", "complete", "entire",
    "overall", "summary", "summaries", "content", "context", "further",
    "gathered", "collected", "together", "structure", "format", "schema", "pattern",
})


def _strip_designer_goal(text: str) -> str:
    """Remove the appended '## Designer Goal' block from an agent system_prompt.

    The workflow builder appends a verbatim copy of the workflow description
    and design brief to every agent's system_prompt so the agent knows the
    overall goal. This means even a generic, unspecialized synthesizer or
    reflector will contain all the domain vocabulary from the description.
    Stripping this block before the coverage check ensures we evaluate only
    the agent's own authored content.
    """
    idx = text.find("## Designer Goal")
    if idx >= 0:
        return text[:idx]
    return text


def _extract_nouns(text: str, min_count: int = 3) -> list[str]:
    """Extract candidate noun-like tokens for use-case-specific vocabulary
    coverage checks. Lower-cases, deduplicates preserving order, drops stopwords,
    requires >=4 chars. Returns up to 10 distinct tokens."""
    if not text:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for match in _NOUN_RE.findall(text or ""):
        token = match.lower()
        if token in _STOPWORDS or token in seen:
            continue
        seen.add(token)
        out.append(token)
        if len(out) >= 10:
            break
    return out if len(out) >= min_count else out  # still return what we have


def _agents_by_subtype(node: Any, subtype: str) -> list[tuple[str, dict[str, Any]]]:
    """Walk an AST dict-form, return list of (path, node_dict) for agents
    matching the given subtype. Handles plan_and_execute config.body/evaluator
    and conditional children."""
    found: list[tuple[str, dict[str, Any]]] = []

    def walk(n: Any, path: str) -> None:
        if not isinstance(n, dict):
            return
        cfg = n.get("config") or {}
        if n.get("type") == "agent" and isinstance(cfg, dict) and cfg.get("subtype") == subtype:
            found.append((path, n))
        if isinstance(cfg, dict):
            body = cfg.get("body")
            if isinstance(body, dict):
                walk(body, f"{path}.config.body")
            evaluator = cfg.get("evaluator")
            if isinstance(evaluator, dict):
                # evaluator IS an agent-config shape (subtype directly on it, no outer "type")
                eval_subtype = evaluator.get("subtype") or (evaluator.get("config") or {}).get("subtype")
                if eval_subtype == subtype:
                    # Wrap so caller sees consistent {type: agent, config: {...}}
                    inner_cfg = evaluator if "subtype" in evaluator else (evaluator.get("config") or {})
                    fake: dict[str, Any] = {"type": "agent", "config": inner_cfg}
                    found.append((f"{path}.config.evaluator", fake))
        for i, child in enumerate(n.get("children") or []):
            walk(child, f"{path}.children[{i}]")

    root_node = node.get("root") if isinstance(node, dict) and "root" in node else node
    walk(root_node, "$.root")
    return found


def _domain_anchor(definition: dict[str, Any]) -> str:
    """Return the text used to extract required-domain nouns. Prefers
    workflow description; falls back to first agent's user_prompt; finally
    to empty string."""
    desc = definition.get("description") or ""
    if isinstance(desc, str) and desc.strip():
        return desc

    def first_user_prompt(n: Any) -> str:
        if not isinstance(n, dict):
            return ""
        cfg = n.get("config") or {}
        if isinstance(cfg, dict):
            up = cfg.get("user_prompt_template") or cfg.get("user_prompt") or ""
            if isinstance(up, str) and up.strip():
                return up[:200]
            body = cfg.get("body")
            if isinstance(body, dict):
                r = first_user_prompt(body)
                if r:
                    return r
        for child in (n.get("children") or []):
            r = first_user_prompt(child)
            if r:
                return r
        return ""

    return first_user_prompt(definition.get("root") or definition)


def _contract_required_terms(definition: dict[str, Any]) -> list[str]:
    """Return compact prompt terms from a resolved tool contract, if present."""

    terms = definition.get("required_prompt_terms")
    if not isinstance(terms, list) or len(terms) < 2:
        summary = definition.get("resolved_tool_contract_summary")
        if isinstance(summary, dict):
            terms = summary.get("required_terms")
    if not isinstance(terms, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in terms:
        if not isinstance(item, str):
            continue
        cleaned = item.strip().casefold()
        if len(cleaned) < 3 or cleaned in seen:
            continue
        out.append(cleaned)
        seen.add(cleaned)
        if len(out) >= 12:
            break
    return out if len(out) >= 2 else []


def _required_domain_terms(definition: dict[str, Any]) -> list[str]:
    """Prefer contract terms, then fall back to legacy domain-anchor nouns."""

    contract_terms = _contract_required_terms(definition)
    if contract_terms:
        return contract_terms
    return _extract_nouns(_domain_anchor(definition))


def _coverage_failure(prompt_text: str, nouns: list[str], min_matches: int = 2) -> bool:
    """Return True when prompt_text contains FEWER than min_matches of nouns."""
    if not nouns:
        return False
    if not prompt_text:
        return True
    lc = prompt_text.lower()
    matches = sum(1 for n in nouns if n in lc)
    return matches < min_matches


def detect_generic_synthesizer_prompt(
    definition: dict[str, Any],
) -> list[SemanticValidationError]:
    """Flag a synthesizer agent whose system_prompt + user_prompt_template
    do not reference at least 2 nouns from the workflow's description.
    kind='unspecialized_synthesizer'."""
    if not isinstance(definition, dict):
        return []
    nouns = _required_domain_terms(definition)
    if len(nouns) < 2:
        return []  # not enough vocabulary to judge; skip silently
    errors: list[SemanticValidationError] = []
    for path, node in _agents_by_subtype(definition, "synthesizer"):
        cfg = node.get("config") or {}
        # Strip the appended ## Designer Goal block before the coverage check —
        # that block contains the full workflow description verbatim, so even a
        # completely generic synthesizer would pass without stripping.
        core_sp = _strip_designer_goal(cfg.get("system_prompt", "") or "")
        combined = (
            f"{core_sp}\n"
            f"{cfg.get('user_prompt_template', '') or cfg.get('user_prompt', '') or ''}"
        )
        if _coverage_failure(combined, nouns, min_matches=2):
            errors.append(SemanticValidationError(
                message=(
                    f"Synthesizer prompt references fewer than 2 of the workflow's "
                    f"required-domain terms: {nouns[:6]}. Specialize system_prompt + "
                    f"user_prompt_template to reference the required outputs."
                ),
                path=path,
                kind="unspecialized_synthesizer",
            ))
    return errors


def detect_generic_reflector_prompt(
    definition: dict[str, Any],
) -> list[SemanticValidationError]:
    """Flag a reflector (incl. plan_and_execute evaluator) whose prompts
    don't reference the workflow's required-domain terms.
    kind='unspecialized_reflector'."""
    if not isinstance(definition, dict):
        return []
    nouns = _required_domain_terms(definition)
    if len(nouns) < 2:
        return []
    errors: list[SemanticValidationError] = []
    for path, node in _agents_by_subtype(definition, "reflector"):
        cfg = node.get("config") or {}
        # Strip the appended ## Designer Goal block before the coverage check.
        core_sp = _strip_designer_goal(cfg.get("system_prompt", "") or "")
        combined = (
            f"{core_sp}\n"
            f"{cfg.get('user_prompt_template', '') or cfg.get('user_prompt', '') or ''}"
        )
        if _coverage_failure(combined, nouns, min_matches=2):
            errors.append(SemanticValidationError(
                message=(
                    f"Reflector prompt references fewer than 2 of the workflow's "
                    f"required-domain terms: {nouns[:6]}. Add a coverage checklist "
                    f"referencing the required outputs."
                ),
                path=path,
                kind="unspecialized_reflector",
            ))
    return errors


def prompt_term_coverage_errors(
    definition: dict[str, Any],
) -> list[SemanticValidationError]:
    """Blocking save-gate coverage: the report-producing synthesizer must reference the
    workflow's required-domain terms, so a saved agent actually covers every requested
    topic. Reuses :func:`detect_generic_synthesizer_prompt`, which already strips the
    appended ``## Designer Goal`` block before matching (so a generic synthesizer cannot
    pass on the goal text alone). Terms are domain-derived (``required_prompt_terms`` /
    description nouns), NEVER hardcoded. Returns ``[]`` when there is no synthesizer or
    too few terms to judge, so it never false-blocks (e.g. legacy/single-agent ASTs)."""
    # Re-tag as a distinct, force-overridable "coverage" kind (vs the always-blocking
    # structural errors) so the UI can offer a "Save draft anyway" path.
    # SemanticValidationError is frozen — build fresh instances rather than mutating.
    return [
        SemanticValidationError(
            message=err.message,
            path=err.path,
            line=err.line,
            kind="coverage",
            severity=err.severity,
        )
        for err in detect_generic_synthesizer_prompt(definition)
    ]


_CONTRACT_REQUIRED_TOOL_KINDS = {
    "vector_search",
    "table_search",
    "table_read",
    "table_load",
    "compute",
}


def _contract_summary(definition: dict[str, Any]) -> dict[str, Any]:
    summary = definition.get("resolved_tool_contract_summary")
    return summary if isinstance(summary, dict) else {}


def _declared_tool_names_by_kind(definition: dict[str, Any]) -> dict[str, set[str]]:
    by_kind: dict[str, set[str]] = {}
    for tool in definition.get("tools", []) or []:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        kind = tool.get("kind")
        if isinstance(name, str) and isinstance(kind, str):
            by_kind.setdefault(kind, set()).add(name)
    return by_kind


def _bound_tool_names(definition: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for _node, config, _path in _collect_agent_paths(definition):
        tools = config.get("tools") if isinstance(config, dict) else None
        if isinstance(tools, list):
            names.update(item for item in tools if isinstance(item, str))
    return names


def detect_tool_contract_violations(
    definition: dict[str, Any],
) -> list[SemanticValidationError]:
    """Validate prompt-derived resolved tool contract invariants.

    This detector activates only when a resolved contract summary is present.
    It is intentionally non-executable: it checks declared/bound tool kinds
    against the contract, but it does not generate or mutate tool config.
    """

    if not isinstance(definition, dict):
        return []
    summary = _contract_summary(definition)
    if not summary.get("available"):
        return []

    errors: list[SemanticValidationError] = []
    declared_by_kind = _declared_tool_names_by_kind(definition)
    declared_kinds = set(declared_by_kind)
    bound_names = _bound_tool_names(definition)
    bound_kinds = {
        kind
        for kind, names in declared_by_kind.items()
        if any(name in bound_names for name in names)
    }

    forbidden = set(summary.get("forbidden_tool_kinds") or [])
    if forbidden:
        forbidden_declared = sorted(forbidden & declared_kinds)
        if forbidden_declared:
            errors.append(
                SemanticValidationError(
                    message=(
                        "Resolved tool contract forbids public "
                        f"web tools, but these kinds are declared: "
                        f"{forbidden_declared}."
                    ),
                    path="tools",
                    kind="tool_contract",
                )
            )
        forbidden_bound = sorted(forbidden & bound_kinds)
        if forbidden_bound:
            errors.append(
                SemanticValidationError(
                    message=(
                        "Resolved tool contract forbids public "
                        f"web tools, but these kinds are node-bound: "
                        f"{forbidden_bound}."
                    ),
                    path="root",
                    kind="tool_contract",
                )
            )

    ready_kinds = set(summary.get("ready_tool_kinds") or [])
    required_kinds = ready_kinds & _CONTRACT_REQUIRED_TOOL_KINDS
    missing_declared = sorted(required_kinds - declared_kinds)
    if missing_declared:
        errors.append(
            SemanticValidationError(
                message=(
                    "Resolved tool contract reports ready required tool "
                    f"kinds that are not declared: {missing_declared}."
                ),
                path="tools",
                kind="tool_contract",
            )
        )
    missing_bound = sorted(required_kinds - bound_kinds)
    if missing_bound:
        errors.append(
            SemanticValidationError(
                message=(
                    "Resolved tool contract reports ready required tool "
                    f"kinds that are declared but not node-bound: "
                    f"{missing_bound}."
                ),
                path="root",
                kind="tool_contract",
            )
        )
    return errors


def detect_unspecialized_fallback_researcher(
    definition: dict[str, Any],
) -> list[SemanticValidationError]:
    """Walk plan_and_execute conditional lane routers; flag when the LAST
    branch (the fallback/else) is a generic researcher (no '## Lane
    Specialization' substring and starts with the framework's generic
    builtin researcher header).
    kind='unspecialized_fallback_researcher'."""
    if not isinstance(definition, dict):
        return []
    errors: list[SemanticValidationError] = []
    generic_header = "You are the Researcher agent for a deep research system"

    def walk(n: Any, path: str) -> None:
        if not isinstance(n, dict):
            return
        # Detect conditional nodes inside plan_and_execute bodies
        if n.get("type") == "conditional":
            children = n.get("children") or []
            if children:
                fallback = children[-1]
                if isinstance(fallback, dict):
                    fcfg = fallback.get("config") or {}
                    sp = (fcfg.get("system_prompt") or "") if isinstance(fcfg, dict) else ""
                    if (
                        isinstance(sp, str)
                        and sp.startswith(generic_header)
                        and "## Lane Specialization" not in sp
                    ):
                        errors.append(SemanticValidationError(
                            message=(
                                "Conditional fallback researcher uses the generic builtin prompt. "
                                "Steps that fall through the lane router will produce unspecialized "
                                "findings. Specialize this branch."
                            ),
                            path=f"{path}.children[-1]",
                            kind="unspecialized_fallback_researcher",
                        ))
        cfg = n.get("config") or {}
        if isinstance(cfg, dict):
            body = cfg.get("body")
            if isinstance(body, dict):
                walk(body, f"{path}.config.body")
            evaluator = cfg.get("evaluator")
            if isinstance(evaluator, dict):
                walk(evaluator, f"{path}.config.evaluator")
        for i, child in enumerate(n.get("children") or []):
            walk(child, f"{path}.children[{i}]")

    root = definition.get("root") or definition
    walk(root, "$.root")
    return errors


__all__ = [
    "SemanticValidationError",
    "semantic_validation_errors",
    "detect_unspecialized_agents",
    "detect_grounded_research_contract",
    "detect_topology_mismatch",
    "detect_generic_synthesizer_prompt",
    "detect_generic_reflector_prompt",
    "detect_tool_contract_violations",
    "detect_unspecialized_fallback_researcher",
]
