"""DeltaCheckpointer unit tests — sequence numbering + event capture.

The Delta DDL/insert path is exercised in env-gated integration tests
(``tests/integration/test_delta_checkpointer.py``); here we focus on the
sync, in-memory state machine.
"""

from __future__ import annotations

import pytest

from databricks_deep_research.api.checkpoint import DeltaCheckpointer


def test_seq_increments_per_thread() -> None:
    cp = DeltaCheckpointer(workspace_client=object())
    assert cp._next_seq("t1") == 1
    assert cp._next_seq("t1") == 2
    assert cp._next_seq("t1") == 3
    # Different thread starts at 1 again.
    assert cp._next_seq("t2") == 1


def test_warehouse_id_falls_back_to_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABRICKS_WAREHOUSE_ID", "wh-abc")
    cp = DeltaCheckpointer(workspace_client=object())
    assert cp._warehouse_id() == "wh-abc"


def test_default_table_name() -> None:
    cp = DeltaCheckpointer(workspace_client=object())
    assert cp._table == "main.ai.agent_runs"


def test_custom_table_name() -> None:
    cp = DeltaCheckpointer(table="custom.cat.runs", workspace_client=object())
    assert cp._table == "custom.cat.runs"


def test_synchronous_default() -> None:
    cp = DeltaCheckpointer(workspace_client=object())
    assert cp._synchronous is True
