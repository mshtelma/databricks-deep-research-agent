"""Unit tests for agent_designer.critic_types."""

import pytest

from deep_research.agent_designer.critic_types import (
    CriticDirective,
    CriticVerdict,
    WorkflowAST,
)


def test_critic_verdict_approve_empty_directives() -> None:
    """CriticVerdict with approve=True and empty directives list succeeds."""
    verdict = CriticVerdict.model_validate({"approve": True, "directives": []})
    assert verdict.approve is True
    assert verdict.directives == []


def test_critic_verdict_with_directive() -> None:
    """CriticVerdict with a directive populates nested model correctly."""
    verdict = CriticVerdict.model_validate(
        {
            "approve": False,
            "directives": [
                {"node_path": "$.root", "issue": "x", "suggested_action": "y"}
            ],
        }
    )
    assert verdict.approve is False
    assert len(verdict.directives) == 1
    assert verdict.directives[0].node_path == "$.root"


def test_critic_verdict_missing_directives_defaults_to_empty() -> None:
    """CriticVerdict without directives key uses default empty list."""
    verdict = CriticVerdict.model_validate({"approve": True})
    assert verdict.directives == []


def test_critic_verdict_null_directives_coerced_to_empty() -> None:
    """CriticVerdict with directives=None is coerced to empty list by validator."""
    verdict = CriticVerdict.model_validate({"approve": True, "directives": None})
    assert verdict.directives == []


def test_workflow_ast_roundtrip() -> None:
    """WorkflowAST validates and round-trips a free-form dict."""
    data = {"root": {"type": "agent"}, "tools": []}
    ast = WorkflowAST.model_validate(data)
    assert ast.model_dump() == data


def test_critic_directive_extra_field_raises() -> None:
    """CriticDirective with an extra field raises ValidationError (extra='forbid')."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        CriticDirective.model_validate(
            {
                "node_path": "x",
                "issue": "y",
                "suggested_action": "z",
                "extra_field": "foo",
            }
        )
