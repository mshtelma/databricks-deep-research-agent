"""Drift guard + behavior tests for the typed terminal-status contract.

The single shared fixture
``databricks-deep-research-app/contracts/run_status_contract.json`` is the one
source of truth pinned to BOTH backend and frontend. This module asserts the
backend half: the JSON enum equals ``get_args(RunStatus)``. A frontend vitest
asserts the other half (the TS label map keys). Either side drifting fails CI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import get_args

import pytest

from databricks_deep_research.events.status_contract import (
    RunStatus,
    make_status_kwargs,
)
from databricks_deep_research.events.types import NodeCompletedEvent, NodeErrorEvent

# Monorepo root is parents[4] of this file
# (tests/unit/events/<file> -> tests/unit/events -> tests/unit -> tests ->
#  databricks-deep-research -> <monorepo root>).
_REPO_ROOT = Path(__file__).resolve().parents[4]
_CONTRACT_PATH = (
    _REPO_ROOT / "databricks-deep-research-app" / "contracts" / "run_status_contract.json"
)


def _load_contract() -> dict[str, object]:
    with open(_CONTRACT_PATH, encoding="utf-8") as f:
        data: dict[str, object] = json.load(f)
    return data


def test_contract_file_exists() -> None:
    assert _CONTRACT_PATH.exists(), f"missing shared status contract: {_CONTRACT_PATH}"


def test_contract_statuses_match_enum() -> None:
    """The drift guard: JSON enum == RunStatus Literal members (set equality)."""
    contract = _load_contract()
    statuses = contract["statuses"]
    assert isinstance(statuses, list)
    assert set(statuses) == set(get_args(RunStatus))


def test_contract_labels_cover_every_status() -> None:
    """Every status has a human label (frontend renders from this map)."""
    contract = _load_contract()
    statuses = contract["statuses"]
    labels = contract["labels"]
    assert isinstance(statuses, list)
    assert isinstance(labels, dict)
    assert set(labels.keys()) == set(statuses)


def test_make_status_kwargs_valid() -> None:
    assert make_status_kwargs("completed") == {"status": "completed"}
    assert make_status_kwargs("failed", error="boom") == {
        "status": "failed",
        "error": "boom",
    }
    # Falsy error is omitted.
    assert make_status_kwargs("failed", error=None) == {"status": "failed"}
    assert make_status_kwargs("failed", error="") == {"status": "failed"}


def test_make_status_kwargs_out_of_enum_raises() -> None:
    with pytest.raises(ValueError, match="out-of-enum status"):
        make_status_kwargs("bogus")  # type: ignore[arg-type]


def test_node_events_default_status_additive() -> None:
    """Event fields are additive: defaults preserve old semantics."""
    completed = NodeCompletedEvent(node_id="n1", timestamp="t", duration_ms=1.0)
    assert completed.status == "completed"

    errored = NodeErrorEvent(node_id="n1", timestamp="t", error_message="x")
    assert errored.status == "failed"


def test_node_events_accept_status_kwargs() -> None:
    """make_status_kwargs spreads cleanly into the terminal events."""
    completed = NodeCompletedEvent(
        node_id="n1", timestamp="t", duration_ms=1.0, **make_status_kwargs("completed")
    )
    assert completed.status == "completed"

    # will_retry path stamps a non-terminal "running".
    retrying = NodeErrorEvent(
        node_id="n1",
        timestamp="t",
        error_message="x",
        will_retry=True,
        **make_status_kwargs("running"),
    )
    assert retrying.status == "running"
