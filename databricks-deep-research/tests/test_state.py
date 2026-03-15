"""Tests for WorkflowState append-only log with O(1) lookup."""

from __future__ import annotations

from databricks_deep_research.workflow.state import StateEntry, WorkflowState


class TestStateEntry:
    def test_frozen(self) -> None:
        entry = StateEntry(node_id="n1", key="k", value="v", timestamp="t")
        assert entry.node_id == "n1"
        # Frozen — assignment raises
        try:
            entry.node_id = "n2"  # type: ignore[misc]
            raise AssertionError("Should have raised")
        except AttributeError:
            pass


class TestWorkflowState:
    def test_append_and_get(self) -> None:
        state = WorkflowState(query="test")
        state.append("node1", "key1", "value1")
        assert state.get("key1") == "value1"

    def test_get_returns_latest(self) -> None:
        state = WorkflowState()
        state.append("n1", "key", "first")
        state.append("n2", "key", "second")
        state.append("n3", "key", "third")
        assert state.get("key") == "third"

    def test_get_missing_key_returns_none(self) -> None:
        state = WorkflowState()
        assert state.get("nonexistent") is None

    def test_get_all(self) -> None:
        state = WorkflowState()
        state.append("n1", "obs", "a")
        state.append("n2", "obs", "b")
        state.append("n3", "other", "c")
        state.append("n4", "obs", "d")
        assert state.get_all("obs") == ["a", "b", "d"]
        assert state.get_all("other") == ["c"]
        assert state.get_all("missing") == []

    def test_latest_index_o1(self) -> None:
        """Verify _latest_index enables O(1) lookup."""
        state = WorkflowState()
        for i in range(100):
            state.append(f"n{i}", "counter", i)
        # O(1) — not scanning all 100 entries
        assert state._latest_index["counter"] == 99
        assert state.get("counter") == 99

    def test_get_nested_dict(self) -> None:
        state = WorkflowState()
        state.append("coord", "coordination", {"complexity": "deep", "depth": 5})
        assert state.get_nested("coordination.complexity") == "deep"
        assert state.get_nested("coordination.depth") == 5

    def test_get_nested_object(self) -> None:
        class Obj:
            def __init__(self) -> None:
                self.value = 42
        state = WorkflowState()
        state.append("n1", "result", Obj())
        assert state.get_nested("result.value") == 42

    def test_get_nested_missing(self) -> None:
        state = WorkflowState()
        assert state.get_nested("missing.path") is None
        state.append("n1", "data", {"a": 1})
        assert state.get_nested("data.b") is None

    def test_to_dict_from_dict_roundtrip(self) -> None:
        state = WorkflowState(
            query="test query",
            model_overrides={"simple": "gpt-4o-mini"},
            user_token="tok",
            domain_filter="example.com",
        )
        state.append("n1", "key1", "val1")
        state.append("n2", "key2", {"nested": True})
        state.append("n3", "key1", "val1_updated")

        data = state.to_dict()
        restored = WorkflowState.from_dict(data)

        assert restored.query == "test query"
        assert restored.model_overrides == {"simple": "gpt-4o-mini"}
        assert restored.user_token == "tok"
        assert restored.domain_filter == "example.com"
        assert restored.get("key1") == "val1_updated"
        assert restored.get("key2") == {"nested": True}
        assert len(restored.log) == 3
        # Index rebuilt correctly
        assert restored._latest_index["key1"] == 2
        assert restored._latest_index["key2"] == 1

    def test_to_dict_excludes_runtime(self) -> None:
        state = WorkflowState()
        data = state.to_dict()
        assert "_lock" not in data
        assert "_latest_index" not in data
        assert "pools" not in data
        assert "enterprise_tools" not in data

    def test_is_cancelled_default_false(self) -> None:
        state = WorkflowState()
        assert state.is_cancelled is False

    def test_is_cancelled_roundtrip(self) -> None:
        state = WorkflowState(is_cancelled=True)
        restored = WorkflowState.from_dict(state.to_dict())
        assert restored.is_cancelled is True

    def test_log_timestamps_are_iso(self) -> None:
        state = WorkflowState()
        state.append("n1", "k", "v")
        ts = state.log[0].timestamp
        assert "T" in ts  # ISO 8601 format
        assert "+" in ts or "Z" in ts or ts.endswith("+00:00")

    def test_multiple_keys_independent(self) -> None:
        state = WorkflowState()
        state.append("n1", "a", 1)
        state.append("n2", "b", 2)
        state.append("n3", "a", 10)
        assert state.get("a") == 10
        assert state.get("b") == 2
