"""Unit tests for the pure surface_state merge helper.

Tests every semantic the docstring promises:
- new agent entry accepted
- data_model replaced on re-patch
- action_runs newest-updated_at-wins
- stale updated_at ignored
- idempotent replay (same session_id + same status)
- multiple agents are independent
"""

from deep_research.services.storage.surface_state import merge_surface_state


class TestMergeSurfaceStateNewAgent:
    """A new agent entry that is absent from the existing state is accepted."""

    def test_new_agent_entry_accepted(self) -> None:
        existing: dict = {}
        patch = {
            "agent-1": {
                "data_model": {"foo": "bar"},
                "surface_etag": "etag-1",
            }
        }
        result = merge_surface_state(existing, patch)
        assert result["agent-1"]["data_model"] == {"foo": "bar"}
        assert result["agent-1"]["surface_etag"] == "etag-1"

    def test_new_agent_with_action_runs(self) -> None:
        existing: dict = {}
        patch = {
            "agent-1": {
                "action_runs": {
                    "run-report": {
                        "session_id": "s1",
                        "message_id": "m1",
                        "status": "completed",
                        "updated_at": "2026-07-01T10:00:00Z",
                    }
                }
            }
        }
        result = merge_surface_state(existing, patch)
        assert result["agent-1"]["action_runs"]["run-report"]["status"] == "completed"


class TestMergeSurfaceStateDataModel:
    """data_model is always replaced by the incoming value (shallow merge)."""

    def test_data_model_replaced(self) -> None:
        existing = {
            "agent-1": {
                "data_model": {"old_key": "old_val"},
                "surface_etag": "old-etag",
            }
        }
        patch = {
            "agent-1": {
                "data_model": {"new_key": "new_val"},
            }
        }
        result = merge_surface_state(existing, patch)
        assert result["agent-1"]["data_model"] == {"new_key": "new_val"}
        # surface_etag from existing is preserved because patch doesn't include it
        assert result["agent-1"]["surface_etag"] == "old-etag"

    def test_surface_etag_replaced(self) -> None:
        existing = {"a": {"surface_etag": "old"}}
        patch = {"a": {"surface_etag": "new"}}
        result = merge_surface_state(existing, patch)
        assert result["a"]["surface_etag"] == "new"


class TestMergeSurfaceStateActionRunsNewestWins:
    """action_runs uses newest-updated_at-wins semantics."""

    def test_newer_updated_at_replaces_older(self) -> None:
        existing = {
            "agent-1": {
                "action_runs": {
                    "run-report": {
                        "session_id": "s1",
                        "message_id": "m1",
                        "status": "running",
                        "updated_at": "2026-07-01T10:00:00Z",
                    }
                }
            }
        }
        patch = {
            "agent-1": {
                "action_runs": {
                    "run-report": {
                        "session_id": "s1",
                        "message_id": "m1",
                        "status": "completed",
                        "updated_at": "2026-07-01T10:05:00Z",
                    }
                }
            }
        }
        result = merge_surface_state(existing, patch)
        assert result["agent-1"]["action_runs"]["run-report"]["status"] == "completed"

    def test_same_updated_at_replaces(self) -> None:
        """Equal timestamp: incoming wins (>=)."""
        ts = "2026-07-01T10:00:00Z"
        existing = {"a": {"action_runs": {"run": {"status": "running", "updated_at": ts}}}}
        patch = {"a": {"action_runs": {"run": {"status": "completed", "updated_at": ts}}}}
        result = merge_surface_state(existing, patch)
        assert result["a"]["action_runs"]["run"]["status"] == "completed"


class TestMergeSurfaceStateStaleIgnored:
    """An incoming action entry with a stale updated_at must NOT overwrite."""

    def test_stale_updated_at_is_ignored(self) -> None:
        existing = {
            "agent-1": {
                "action_runs": {
                    "run-report": {
                        "session_id": "s1",
                        "status": "completed",
                        "updated_at": "2026-07-01T10:05:00Z",
                    }
                }
            }
        }
        patch = {
            "agent-1": {
                "action_runs": {
                    "run-report": {
                        "session_id": "s2",
                        "status": "running",
                        "updated_at": "2026-07-01T09:00:00Z",  # older
                    }
                }
            }
        }
        result = merge_surface_state(existing, patch)
        # Must keep existing completed entry
        run = result["agent-1"]["action_runs"]["run-report"]
        assert run["status"] == "completed"
        assert run["session_id"] == "s1"


class TestMergeSurfaceStateIdempotentReplay:
    """Same session_id + same status is treated as idempotent replay."""

    def test_same_session_same_status_is_no_op(self) -> None:
        stored = {
            "session_id": "s1",
            "message_id": "m1",
            "status": "completed",
            "updated_at": "2026-07-01T10:05:00Z",
        }
        existing = {"a": {"action_runs": {"run": stored}}}
        # Replay: same session_id, same status, but older timestamp
        patch = {
            "a": {
                "action_runs": {
                    "run": {
                        "session_id": "s1",
                        "message_id": "m1",
                        "status": "completed",
                        "updated_at": "2026-07-01T09:00:00Z",
                    }
                }
            }
        }
        result = merge_surface_state(existing, patch)
        # Idempotent replay accepted (same session_id + status wins regardless of ts)
        run = result["a"]["action_runs"]["run"]
        assert run["status"] == "completed"
        assert run["session_id"] == "s1"

    def test_different_session_id_requires_newer_ts(self) -> None:
        existing = {
            "a": {
                "action_runs": {
                    "run": {
                        "session_id": "s1",
                        "status": "completed",
                        "updated_at": "2026-07-01T10:05:00Z",
                    }
                }
            }
        }
        patch = {
            "a": {
                "action_runs": {
                    "run": {
                        "session_id": "s2",  # different session
                        "status": "running",
                        "updated_at": "2026-07-01T10:06:00Z",  # newer ts
                    }
                }
            }
        }
        result = merge_surface_state(existing, patch)
        run = result["a"]["action_runs"]["run"]
        assert run["session_id"] == "s2"
        assert run["status"] == "running"


class TestMergeSurfaceStateMultipleAgents:
    """Multiple agents in one patch are merged independently."""

    def test_multiple_agents_independent(self) -> None:
        existing = {
            "agent-1": {"data_model": {"x": 1}},
        }
        patch = {
            "agent-1": {"data_model": {"x": 2}},
            "agent-2": {"data_model": {"y": 99}},
        }
        result = merge_surface_state(existing, patch)
        assert result["agent-1"]["data_model"] == {"x": 2}
        assert result["agent-2"]["data_model"] == {"y": 99}

    def test_untouched_agent_preserved(self) -> None:
        existing = {
            "agent-1": {"surface_etag": "etag-a"},
            "agent-2": {"surface_etag": "etag-b"},
        }
        patch = {"agent-1": {"surface_etag": "etag-a-new"}}
        result = merge_surface_state(existing, patch)
        assert result["agent-1"]["surface_etag"] == "etag-a-new"
        assert result["agent-2"]["surface_etag"] == "etag-b"

    def test_existing_not_mutated(self) -> None:
        """merge_surface_state must never mutate its inputs."""
        existing = {"a": {"data_model": {"v": 1}}}
        patch = {"a": {"data_model": {"v": 2}}}
        result = merge_surface_state(existing, patch)
        # existing unchanged
        assert existing["a"]["data_model"]["v"] == 1
        assert result["a"]["data_model"]["v"] == 2
