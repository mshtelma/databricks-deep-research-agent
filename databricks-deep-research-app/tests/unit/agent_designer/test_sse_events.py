"""Unit tests for the SSE event types in sse_events.py (US-05 / W5a refactor)."""

from __future__ import annotations

from deep_research.agent_designer.sse_events import MutationProposedEvent, ProgressEvent


class TestMutationProposedEvent:
    def test_type_discriminator(self) -> None:
        event = MutationProposedEvent(
            tool_name="propose_workflow",
            tool_call_id="t1",
            old_ast=None,
            new_ast={},
            validation_errors=[],
            summary=None,
        )
        assert event.type == "mutation_proposed"

    def test_model_dump_excludes_type(self) -> None:
        event = MutationProposedEvent(
            tool_name="propose_workflow",
            tool_call_id="t1",
            old_ast=None,
            new_ast={"nodes": []},
            validation_errors=[],
            summary={"node_count": 0, "tool_count": 0, "source_count": 0},
        )
        dumped = event.model_dump(exclude={"type"})
        assert "type" not in dumped
        assert dumped == {
            "tool_name": "propose_workflow",
            "tool_call_id": "t1",
            "old_ast": None,
            "new_ast": {"nodes": []},
            "validation_errors": [],
            "summary": {"node_count": 0, "tool_count": 0, "source_count": 0},
            # Layer 2 auto-repair — default empty when no fixes were applied.
            "normalization_fixes": [],
        }


class TestProgressEvent:
    def test_type_discriminator(self) -> None:
        assert ProgressEvent(label="Workflow Architect (Opus)").type == "progress"

    def test_iteration_total_default_none(self) -> None:
        event = ProgressEvent(label="Working")
        assert event.iteration is None
        assert event.total is None

    def test_model_dump_excludes_type(self) -> None:
        event = ProgressEvent(label="Refining", iteration=2, total=4)
        dumped = event.model_dump(exclude={"type"})
        assert "type" not in dumped
        assert dumped == {"label": "Refining", "iteration": 2, "total": 4}


class TestReExportIntegrity:
    def test_orchestrator_reexport_is_same_class(self) -> None:
        from deep_research.agent_designer.orchestrator import (
            MutationProposedEvent as A,
        )
        from deep_research.agent_designer.sse_events import (
            MutationProposedEvent as B,
        )

        assert A is B
