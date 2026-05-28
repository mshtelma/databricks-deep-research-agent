"""Structural gate tool for Designer workflows.

Aggregates deterministic structural detectors into a single
``ResearchTool`` implementation so a workflow YAML's ``type: tool`` node
can call it as ``structural_gate``. Returns a ``{status, failures}``
payload that the loop's conditional branch reads via state.

See US-08 of the harmonization plan: this is the safety net that runs
AFTER the architect's first scaffold and BEFORE the critic agent — if
any detector fires, the architect re-runs with the gate's failures
spliced into its user prompt.
"""

from __future__ import annotations

import json
from typing import Any

from databricks_deep_research.tools.protocol import (
    ToolContext,
    ToolDefinition,
    ToolKind,
    ToolResult,
)

from deep_research.agent_designer.assets import detect_asset_contract
from deep_research.agent_designer.semantic_validation import (
    SemanticValidationError,
    detect_generic_reflector_prompt,
    detect_generic_synthesizer_prompt,
    detect_grounded_research_contract,
    detect_tool_contract_violations,
    detect_unspecialized_agents,
    detect_unspecialized_fallback_researcher,
)

_RUNTIME_TOOL_KINDS: frozenset[str] = frozenset(
    kind.value for kind in ToolKind if kind is not ToolKind.custom
)
_EVIDENCE_TOOL_KINDS: frozenset[str] = frozenset(
    kind
    for kind in _RUNTIME_TOOL_KINDS
    if kind not in {"compute", "compute_namespace"}
)


def _coerce_ast(raw: Any) -> dict[str, Any]:
    """Accept either a dict or a JSON string and return a dict."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return {}
        try:
            parsed = json.loads(stripped)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _declared_runtime_tools(ast: dict[str, Any]) -> dict[str, str]:
    declared: dict[str, str] = {}
    for tool in ast.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        kind = tool.get("kind")
        if isinstance(name, str) and isinstance(kind, str) and kind in _RUNTIME_TOOL_KINDS:
            declared[name] = kind
    return declared


def _declared_tool_kinds(ast: dict[str, Any]) -> dict[str, str]:
    declared: dict[str, str] = {}
    for tool in ast.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        kind = tool.get("kind")
        if isinstance(name, str) and isinstance(kind, str):
            declared[name] = kind
    return declared


def _declared_tool_names(ast: dict[str, Any]) -> set[str]:
    return {
        tool["name"]
        for tool in ast.get("tools") or []
        if isinstance(tool, dict) and isinstance(tool.get("name"), str)
    }


def _agent_records(ast: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any], str]]:
    records: list[tuple[dict[str, Any], dict[str, Any], str]] = []

    def walk(node: Any, path: str) -> None:
        if not isinstance(node, dict):
            return
        config = node.get("config")
        if not isinstance(config, dict):
            config = {}
        if node.get("type") == "agent":
            records.append((node, config, path))
        if node.get("type") == "plan_and_execute":
            for nested_key in ("planner", "evaluator"):
                nested = config.get(nested_key)
                if isinstance(nested, dict):
                    records.append(
                        (
                            {"id": f"{node.get('id', path)}-{nested_key}", "label": nested_key},
                            nested,
                            f"{path}.config.{nested_key}",
                        )
                    )
            body = config.get("body")
            if isinstance(body, dict):
                walk(body, f"{path}.config.body")
        for idx, child in enumerate(node.get("children") or []):
            walk(child, f"{path}.children[{idx}]")

    walk(ast.get("root"), "root")
    return records


def detect_tool_access_contract(ast: dict[str, Any]) -> list[SemanticValidationError]:
    """Validate generic tool wiring without choosing which tools are appropriate."""

    declared = _declared_runtime_tools(ast)
    declared_kinds = _declared_tool_kinds(ast)
    declared_names = _declared_tool_names(ast)
    bound_names: set[str] = set()
    errors: list[SemanticValidationError] = []

    for name, kind in declared_kinds.items():
        if kind not in _RUNTIME_TOOL_KINDS:
            errors.append(
                SemanticValidationError(
                    message=(
                        f"Tool '{name}' declares unsupported runtime kind "
                        f"'{kind}'. Choose one of the framework tool kinds "
                        "listed by list_tool_kinds, or remove this stale/"
                        "legacy declaration."
                    ),
                    path="tools",
                )
            )

    for node, config, path in _agent_records(ast):
        tools = config.get("tools") or []
        tool_names = [item for item in tools if isinstance(item, str)]
        bound_names.update(tool_names)

        subtype = str(config.get("subtype") or "").casefold()
        bound_evidence_tools = [
            name
            for name in tool_names
            if declared.get(name) in _EVIDENCE_TOOL_KINDS
        ]
        if subtype == "researcher" and not bound_evidence_tools:
            label = node.get("label") or node.get("id") or path
            errors.append(
                SemanticValidationError(
                    message=(
                        f"Researcher '{label}' has no bound executable evidence tools. "
                        "Declare an appropriate runtime tool, then call "
                        "bind_tool_to_block so this agent can gather evidence."
                    ),
                    path=f"{path}.config.tools",
                )
            )

        for name in tool_names:
            if name not in declared_names:
                errors.append(
                    SemanticValidationError(
                        message=(
                            f"Agent '{node.get('label') or node.get('id') or path}' "
                            f"binds undeclared runtime tool '{name}'. Declare it "
                            "first or remove the stale binding."
                        ),
                        path=f"{path}.config.tools",
                    )
                )

    for name, kind in declared.items():
        if name not in bound_names:
            errors.append(
                SemanticValidationError(
                    message=(
                        f"Runtime tool '{name}' ({kind}) is declared but not "
                        "bound to any agent. Bind it to the agent that needs it "
                        "or remove the stale declaration."
                    ),
                    path="tools",
                )
            )

    return errors


class StructuralGateTool:
    """Run structural detectors against an AST and emit a gate verdict."""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="structural_gate",
            description=(
                "Run deterministic structural detectors against a "
                "Designer workflow AST. Returns {status: pass|fail, failures: "
                "[{path, kind, message, suggested_action}]}."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "ast": {
                        "description": (
                            "The workflow AST (dict or JSON string). "
                            "Typically read from state.current_ast via "
                            "input_mapping."
                        ),
                    },
                    "brief": {
                        "description": (
                            "Optional design brief (dict or JSON string). "
                            "Currently unused by the detectors but accepted "
                            "for forward-compat."
                        ),
                    },
                    "assets": {
                        "description": (
                            "Optional Designer asset context. Required assets "
                            "must be referenced by declared and node-bound "
                            "workflow tools."
                        ),
                    },
                },
                "required": ["ast"],
            },
            source_type="builtin",
            source_kind="builtin",
        )

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if "ast" not in arguments:
            raise ValueError("structural_gate requires 'ast' argument")
        return arguments

    async def execute(
        self,
        arguments: dict[str, Any],
        _context: ToolContext,
    ) -> ToolResult:
        ast = _coerce_ast(arguments.get("ast"))
        defects = [
            *detect_unspecialized_agents(ast),
            *detect_grounded_research_contract(ast),
            *detect_generic_synthesizer_prompt(ast),
            *detect_generic_reflector_prompt(ast),
            *detect_tool_contract_violations(ast),
            *detect_unspecialized_fallback_researcher(ast),
            *detect_tool_access_contract(ast),
            *detect_asset_contract(ast, arguments.get("assets")),
        ]
        failures = [
            {
                "path": defect.path or "",
                "kind": defect.kind,
                "message": defect.message,
                "severity": defect.severity,
                "suggested_action": "",
            }
            for defect in defects
        ]
        # Plan v2.1 M10 — severity-graded gating. Only ``blocking`` defects
        # fail the gate; ``warning`` and ``info`` are surfaced but do not
        # route the architect back. This lets advisory detectors (like the
        # placeholder_pending lifecycle check) ship observability signals
        # without breaking workflows when the architect's prompt isn't yet
        # tuned to satisfy them.
        blocking_failures = [
            f for f in failures if f.get("severity", "blocking") == "blocking"
        ]
        result: dict[str, Any] = {
            "status": "pass" if not blocking_failures else "fail",
            "failures": failures,
        }
        return ToolResult(content=json.dumps(result), data=result)


__all__ = ["StructuralGateTool"]
