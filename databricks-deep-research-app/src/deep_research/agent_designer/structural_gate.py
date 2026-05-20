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
    ToolResult,
)

from deep_research.agent_designer.semantic_validation import (
    detect_generic_reflector_prompt,
    detect_generic_synthesizer_prompt,
    detect_grounded_research_contract,
    detect_unspecialized_agents,
    detect_unspecialized_fallback_researcher,
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
            *detect_unspecialized_fallback_researcher(ast),
        ]
        failures = [
            {
                "path": defect.path or "",
                "kind": defect.kind,
                "message": defect.message,
                "suggested_action": "",
            }
            for defect in defects
        ]
        result: dict[str, Any] = {
            "status": "pass" if not failures else "fail",
            "failures": failures,
        }
        return ToolResult(content=json.dumps(result), data=result)


__all__ = ["StructuralGateTool"]
