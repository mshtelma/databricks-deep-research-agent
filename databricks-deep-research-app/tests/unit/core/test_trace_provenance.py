"""Tests for the shared dr.* MLflow provenance tag helper.

The helper is the single source of truth for the tag schema and is called
by all three surfaces (designer chat, main chat, deployed shell-app). Tests
verify the input-shape invariants without requiring an actual MLflow run.
"""
from __future__ import annotations

from unittest.mock import patch

from deep_research.core.trace_provenance import set_trace_provenance


def test_set_trace_provenance_prefixes_all_keys_with_dr_namespace() -> None:
    with patch("deep_research.core.trace_provenance.mlflow") as mock_mlflow:
        set_trace_provenance(surface="main-chat", agent_v2_id="abc-123")
    calls = mock_mlflow.set_trace_tag.call_args_list
    keys = [c.args[0] for c in calls]
    assert keys == ["dr.surface", "dr.agent_v2_id"]


def test_set_trace_provenance_drops_none_values() -> None:
    with patch("deep_research.core.trace_provenance.mlflow") as mock_mlflow:
        set_trace_provenance(surface="designer-chat", agent_v2_id=None, session_id="s1")
    keys = [c.args[0] for c in mock_mlflow.set_trace_tag.call_args_list]
    assert "dr.agent_v2_id" not in keys
    assert "dr.surface" in keys
    assert "dr.session_id" in keys


def test_set_trace_provenance_drops_empty_strings() -> None:
    with patch("deep_research.core.trace_provenance.mlflow") as mock_mlflow:
        set_trace_provenance(surface="shell-app", app_name="", workflow_id="wf-1")
    keys = [c.args[0] for c in mock_mlflow.set_trace_tag.call_args_list]
    assert "dr.app_name" not in keys
    assert "dr.workflow_id" in keys


def test_set_trace_provenance_bounds_value_length() -> None:
    long_value = "x" * 5000
    with patch("deep_research.core.trace_provenance.mlflow") as mock_mlflow:
        set_trace_provenance(query_preview=long_value)
    values = [c.args[1] for c in mock_mlflow.set_trace_tag.call_args_list]
    assert all(len(v) <= 512 for v in values)


def test_set_trace_provenance_no_args_does_nothing() -> None:
    with patch("deep_research.core.trace_provenance.mlflow") as mock_mlflow:
        set_trace_provenance()
    mock_mlflow.set_trace_tag.assert_not_called()


def test_set_trace_provenance_all_empty_args_does_nothing() -> None:
    with patch("deep_research.core.trace_provenance.mlflow") as mock_mlflow:
        set_trace_provenance(agent_v2_id=None, app_name="", session_id=None)
    mock_mlflow.set_trace_tag.assert_not_called()


def test_set_trace_provenance_stringifies_non_string_values() -> None:
    with patch("deep_research.core.trace_provenance.mlflow") as mock_mlflow:
        set_trace_provenance(workspace_id=12345, agent_v2_id=object())
    values = [c.args[1] for c in mock_mlflow.set_trace_tag.call_args_list]
    assert "12345" in values
    # object() stringification non-empty
    assert any("object" in v for v in values)


def test_set_trace_provenance_no_op_when_mlflow_missing() -> None:
    with patch("deep_research.core.trace_provenance.mlflow", None):
        # Should not raise.
        set_trace_provenance(surface="main-chat", agent_v2_id="abc")


def test_set_trace_provenance_swallows_tag_call_exceptions() -> None:
    with patch("deep_research.core.trace_provenance.mlflow") as mock_mlflow:
        mock_mlflow.set_trace_tag.side_effect = RuntimeError("no active trace")
        # Should not raise — tagging failures are non-fatal.
        set_trace_provenance(surface="shell-app")
