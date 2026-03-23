"""Tests for JSONL result store."""

import json
from pathlib import Path

import pytest

from benchmarks.core.result_store import ResultStore
from benchmarks.core.types import QuestionResult


def _make_result(uid: str = "Q1", status: str = "success") -> QuestionResult:
    return QuestionResult(
        uid=uid,
        question="Test question?",
        expected_answer="42",
        predicted_answer="42" if status == "success" else None,
        raw_output="some output",
        status=status,  # type: ignore[arg-type]
        wall_time_seconds=1.5,
        num_sources=3,
    )


class TestResultStore:
    def test_append_and_load(self, tmp_path: Path) -> None:
        store = ResultStore(tmp_path / "results.jsonl")
        r1 = _make_result("Q1")
        r2 = _make_result("Q2")
        store.append(r1)
        store.append(r2)

        loaded = store.load_all()
        assert len(loaded) == 2
        assert loaded[0].uid == "Q1"
        assert loaded[1].uid == "Q2"

    def test_completed_uids(self, tmp_path: Path) -> None:
        store = ResultStore(tmp_path / "results.jsonl")
        store.append(_make_result("Q1"))
        store.append(_make_result("Q2"))
        store.append(_make_result("Q3"))

        assert store.completed_uids() == {"Q1", "Q2", "Q3"}

    def test_resume_skips_completed(self, tmp_path: Path) -> None:
        store = ResultStore(tmp_path / "results.jsonl")
        store.append(_make_result("Q1"))
        assert "Q1" in store.completed_uids()
        assert "Q2" not in store.completed_uids()

    def test_empty_file(self, tmp_path: Path) -> None:
        store = ResultStore(tmp_path / "results.jsonl")
        assert store.completed_uids() == set()
        assert store.load_all() == []

    def test_truncated_line_tolerated(self, tmp_path: Path) -> None:
        path = tmp_path / "results.jsonl"
        # Write a valid line + truncated line
        r = _make_result("Q1")
        valid_line = json.dumps({"uid": "Q1", "question": "q", "expected_answer": "a",
                                  "predicted_answer": "a", "raw_output": "o",
                                  "status": "success", "wall_time_seconds": 1.0,
                                  "num_sources": 0})
        path.write_text(valid_line + "\n" + '{"uid":"Q2"')

        store = ResultStore(path)
        # Q1 should be loaded, Q2 skipped (truncated)
        assert store.completed_uids() == {"Q1"}

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        store = ResultStore(tmp_path / "deep" / "nested" / "results.jsonl")
        store.append(_make_result("Q1"))
        assert store.completed_uids() == {"Q1"}

    def test_preserves_metadata(self, tmp_path: Path) -> None:
        store = ResultStore(tmp_path / "results.jsonl")
        r = _make_result("Q1")
        r.metadata = {"difficulty": "hard", "requires_external": True}
        store.append(r)

        loaded = store.load_all()
        assert loaded[0].metadata["difficulty"] == "hard"
        assert loaded[0].metadata["requires_external"] is True
