"""Verify stream_workflow_via_framework accepts workflow_def directly and
that stream_research_via_framework is a thin alias."""

import inspect


def test_back_compat_alias_exists() -> None:
    from deep_research.agent.framework_orchestrator import (
        stream_research_via_framework,
        stream_workflow_via_framework,
    )
    assert callable(stream_research_via_framework)
    assert callable(stream_workflow_via_framework)
    # They are not the same function (alias is wrapper, not identity).
    assert stream_research_via_framework is not stream_workflow_via_framework


def test_generalized_signature_has_workflow_def_kwarg() -> None:
    from deep_research.agent.framework_orchestrator import stream_workflow_via_framework
    sig = inspect.signature(stream_workflow_via_framework)
    assert "workflow_def" in sig.parameters
    assert "extra_state" in sig.parameters
