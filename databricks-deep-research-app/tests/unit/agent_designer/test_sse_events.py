"""Unit tests for the SSE event types in sse_events.py (US-05 / W5a refactor)."""

from __future__ import annotations

from deep_research.agent_designer.sse_events import (
    ArchitectSynopsisEvent,
    CriticReviewEvent,
    MutationProposedEvent,
    ProgressEvent,
)


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


class TestArchitectSynopsisEvent:
    def test_type_discriminator_and_defaults(self) -> None:
        event = ArchitectSynopsisEvent(
            headline="Built a parallel lanes workflow · 2 lanes",
            topology="parallel_lanes",
        )
        assert event.type == "architect_synopsis"
        assert event.change_kind == "created"
        assert event.lanes == []
        assert event.pipeline == []
        assert event.warnings == []

    def test_model_dump_excludes_type(self) -> None:
        event = ArchitectSynopsisEvent(
            headline="Updated the workflow (parallel lanes)",
            topology="parallel_lanes",
            change_kind="edited",
            lanes=[{"label": "Market", "tools": ["web_search"]}],
            pipeline=["Coordinator", "Synthesizer"],
            tools=["web_search"],
            outputs=["Summary"],
            warnings=["1 lane still uses a default prompt."],
        )
        dumped = event.model_dump(exclude={"type"})
        assert "type" not in dumped
        assert dumped["change_kind"] == "edited"
        assert dumped["lanes"] == [{"label": "Market", "tools": ["web_search"]}]
        assert dumped["pipeline"] == ["Coordinator", "Synthesizer"]


class TestCriticReviewEvent:
    def test_type_and_defaults(self) -> None:
        event = CriticReviewEvent(verdict="pass", summary="Looks good.")
        assert event.type == "critic_review"
        assert event.agent_findings == []
        assert event.coverage_gaps == []
        assert event.output_gaps == []

    def test_constructs_from_critique_result_dump(self) -> None:
        # The validate handler does CriticReviewEvent(**critique.model_dump());
        # this guards against field drift between CritiqueResult and the event.
        from deep_research.agent_designer.workflow_critic import (
            AgentFinding,
            CritiqueResult,
        )

        critique = CritiqueResult(
            verdict="needs_revision",
            summary="The pricing lane is shallow.",
            agent_findings=[
                AgentFinding(
                    node_path="root.children[1]",
                    label="Pricing Researcher",
                    severity="needs_revision",
                    finding="Does not cover competitor pricing.",
                    suggested_action="update_block to add competitor pricing.",
                )
            ],
        )
        event = CriticReviewEvent(**critique.model_dump())
        assert event.verdict == "needs_revision"
        assert event.summary == "The pricing lane is shallow."
        assert event.agent_findings[0]["label"] == "Pricing Researcher"


class TestReExportIntegrity:
    def test_orchestrator_reexport_is_same_class(self) -> None:
        from deep_research.agent_designer.orchestrator import (
            MutationProposedEvent as A,
        )
        from deep_research.agent_designer.sse_events import (
            MutationProposedEvent as B,
        )

        assert A is B
