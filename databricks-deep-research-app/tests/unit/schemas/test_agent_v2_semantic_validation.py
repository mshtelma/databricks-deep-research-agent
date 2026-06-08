"""W10: ``agents-v2`` CRUD requests must run semantic validation at write
time, not only via the separate ``POST /agent-designer/validate`` route.

Previously, an undeclared-tool reference inside an agent ``config.tools``
list would pass ``CreateAgentV2Request._validate_definition`` (which only
ran the structural ``load_workflow_from_dict`` check) and persist
successfully — then fail at runtime when the agent tried to invoke the
missing tool. Any caller that bypassed the Designer page (chat-assistant
patches, CLI scripts, direct API access) hit this trap.

These tests pin the new behavior: semantic violations are 422s at the
schema layer, identical for both create and update payloads.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from deep_research.schemas.agent_v2 import (
    CreateAgentV2Request,
    UpdateAgentV2Request,
)


def _minimal_valid_definition() -> dict[str, object]:
    """A workflow that round-trips through both structural + semantic checks.

    Field set chosen to match the existing valid-AST fixture in
    ``tests/unit/agent_designer/test_mutations.py:_minimal_ast``.
    """
    return {
        "id": "test-wf",
        "name": "test-wf",
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


def _definition_with_undeclared_tool_ref() -> dict[str, object]:
    """Structurally valid AST where an agent references a tool that is
    NOT in the top-level ``definition.tools`` list. Pre-W10 this passed
    ``CreateAgentV2Request._validate_definition``.
    """
    defn = _minimal_valid_definition()
    # Inject the undeclared tool reference into the root agent's config.
    root = defn["root"]
    assert isinstance(root, dict)
    config = root["config"]
    assert isinstance(config, dict)
    config["tools"] = ["nonexistent_tool"]
    return defn


class TestCreateAgentV2RequestSemanticValidation:
    def test_minimal_valid_definition_accepted(self) -> None:
        req = CreateAgentV2Request(
            name="Test", definition=_minimal_valid_definition()
        )
        assert req.definition is not None

    def test_undeclared_tool_ref_rejected_at_create(self) -> None:
        """W10: bypass through CreateAgentV2Request must NOT persist a
        broken AST. Even though the structural loader accepts it, the
        semantic check fires and 422s the request.
        """
        with pytest.raises(ValidationError) as exc_info:
            CreateAgentV2Request(
                name="Test",
                definition=_definition_with_undeclared_tool_ref(),
            )
        error_msg = str(exc_info.value)
        assert "semantic validation failed" in error_msg.lower()
        assert "nonexistent_tool" in error_msg


class TestUpdateAgentV2RequestSemanticValidation:
    def test_partial_update_without_definition_accepted(self) -> None:
        # Update with no `definition` should not run validation at all.
        req = UpdateAgentV2Request(name="Renamed")
        assert req.definition is None

    def test_undeclared_tool_ref_rejected_at_update(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            UpdateAgentV2Request(
                definition=_definition_with_undeclared_tool_ref(),
            )
        error_msg = str(exc_info.value)
        assert "semantic validation failed" in error_msg.lower()
        assert "nonexistent_tool" in error_msg

    def test_valid_definition_update_accepted(self) -> None:
        req = UpdateAgentV2Request(definition=_minimal_valid_definition())
        assert req.definition is not None
