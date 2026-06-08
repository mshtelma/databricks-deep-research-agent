"""US-DF4: the dataflow check is wired into ``validate_workflow``.

In lint mode (default) a dangling read is logged as ``DATAFLOW_LINT`` and does NOT
raise; in strict mode it becomes a validation error.
"""
from __future__ import annotations

import logging

import pytest

from databricks_deep_research.errors import WorkflowValidationError
from databricks_deep_research.workflow.definition import (
    NodeType,
    WorkflowDefinition,
    WorkflowNode,
)
from databricks_deep_research.workflow.validation import validate_workflow


def _dangling_workflow() -> WorkflowDefinition:
    root = WorkflowNode(
        id="root",
        type=NodeType.sequence,
        label="root",
        children=[
            WorkflowNode(
                id="researcher",
                type=NodeType.agent,
                label="researcher",
                config={
                    "subtype": "researcher",
                    "input_keys": ["query", "ghost_key"],
                    "output_key": "findings",
                },
            ),
        ],
    )
    return WorkflowDefinition(
        id="t", name="t", root=root, required_inputs=["query"], output_keys=["findings"]
    )


def test_lint_mode_logs_and_does_not_raise(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DATAFLOW_CHECK_STRICT", raising=False)
    with caplog.at_level(logging.WARNING):
        result = validate_workflow(_dangling_workflow())  # must NOT raise in lint mode
    assert result == []
    assert any(
        "DATAFLOW_LINT" in record.message and "ghost_key" in record.message
        for record in caplog.records
    )


def test_strict_mode_raises_on_dangling_read(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATAFLOW_CHECK_STRICT", "true")
    with pytest.raises(WorkflowValidationError):
        validate_workflow(_dangling_workflow())
