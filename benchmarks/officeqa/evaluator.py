"""OfficeQA evaluation — wraps official reward.py score_answer()."""

from __future__ import annotations

import logging
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from benchmarks.core.types import QuestionResult

logger = logging.getLogger(__name__)


@dataclass
class EvaluationReport:
    """Evaluation results at multiple tolerance levels."""

    total: int
    answered: int
    errors: int
    timeouts: int
    no_answers: int
    scores_by_tolerance: dict[float, list[float]]
    per_question: list[dict[str, Any]]
    wall_times: list[float]
    model: str = ""
    timestamp: str = ""
    rate_limited: int = 0

    def accuracy_at(self, tolerance: float) -> float:
        scores = self.scores_by_tolerance.get(tolerance, [])
        return sum(scores) / len(scores) if scores else 0.0

    def format_report(self) -> str:
        lines: list[str] = []
        ts = self.timestamp or datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        lines.append(f"OfficeQA — {self.model or 'Unknown'} — {ts}")
        lines.append("=" * 50)
        rl_part = f" | Rate Limited: {self.rate_limited}" if self.rate_limited else ""
        lines.append(
            f"Total: {self.total} | Answered: {self.answered} | "
            f"No Answer: {self.no_answers} | Errors: {self.errors} | "
            f"Timeouts: {self.timeouts}{rl_part}"
        )
        lines.append("")
        lines.append("Accuracy (all questions):")
        for tol, scores in sorted(self.scores_by_tolerance.items()):
            correct = sum(1 for s in scores if s > 0)
            pct = (sum(scores) / len(scores) * 100) if scores else 0
            label = "Exact" if tol == 0.0 else f"Fuzzy ({tol*100:.1f}%)"
            lines.append(f"  {label:>15s}:  {pct:5.1f}%  ({correct}/{len(scores)})")

        # By difficulty breakdown
        difficulties: dict[str, list[dict[str, Any]]] = {}
        for pq in self.per_question:
            diff = pq.get("difficulty", "unknown")
            difficulties.setdefault(diff, []).append(pq)

        if len(difficulties) > 1:
            lines.append("")
            lines.append("By Difficulty:")
            for diff, items in sorted(difficulties.items()):
                if not diff:
                    continue
                accs = []
                for tol in sorted(self.scores_by_tolerance.keys()):
                    scores = [it["scores"][tol] for it in items if tol in it.get("scores", {})]
                    pct = (sum(scores) / len(scores) * 100) if scores else 0
                    accs.append(f"{pct:.1f}%")
                lines.append(f"  {diff} ({len(items)}):  {' / '.join(accs)}")

        # Treasury-only (exclude external-source questions)
        treasury_only = [
            pq for pq in self.per_question if not pq.get("requires_external")
        ]
        if len(treasury_only) < self.total:
            lines.append("")
            lines.append(f"Treasury-Only (excl. external-source, {len(treasury_only)} questions):")
            accs = []
            for tol in sorted(self.scores_by_tolerance.keys()):
                scores = [
                    it["scores"][tol]
                    for it in treasury_only
                    if tol in it.get("scores", {})
                ]
                pct = (sum(scores) / len(scores) * 100) if scores else 0
                accs.append(f"{pct:.1f}%")
            lines.append(f"  {' / '.join(accs)}")

        # Timing
        if self.wall_times:
            lines.append("")
            avg = statistics.mean(self.wall_times)
            med = statistics.median(self.wall_times)
            p95 = sorted(self.wall_times)[int(len(self.wall_times) * 0.95)] if len(self.wall_times) > 1 else avg
            lines.append(f"Timing: avg {avg:.1f}s, median {med:.1f}s, p95 {p95:.1f}s")

        lines.append(
            f"Failures: no_answer={self.no_answers}, error={self.errors}, timeout={self.timeouts}"
        )
        return "\n".join(lines)


def _import_reward(officeqa_repo_path: Path) -> Callable[..., float]:
    """Import score_answer from the official reward.py."""
    reward_path = officeqa_repo_path / "reward.py"
    if not reward_path.exists():
        raise FileNotFoundError(
            f"reward.py not found at {reward_path}. "
            "Ensure OfficeQA repo is cloned."
        )

    # Insert repo path so reward.py's imports work
    repo_str = str(officeqa_repo_path)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    from reward import score_answer  # type: ignore[import-untyped]

    return score_answer  # type: ignore[no-any-return]


class OfficeQAEvaluator:
    """Score benchmark results using the official OfficeQA reward function."""

    def __init__(self, officeqa_repo_path: Path) -> None:
        self._score_fn = _import_reward(officeqa_repo_path)
        self._repo_path = officeqa_repo_path

    def evaluate(
        self,
        results: list[QuestionResult],
        tolerances: list[float] | None = None,
        model: str = "",
    ) -> EvaluationReport:
        """Score all results at each tolerance level.

        Rate-limited questions (status ``"rate_limited"``) are excluded from
        scoring so they don't deflate accuracy.
        """
        if tolerances is None:
            tolerances = [0.0, 0.01, 0.05]

        scorable = [r for r in results if r.status != "rate_limited"]
        rate_limited_count = len(results) - len(scorable)

        scores_by_tolerance: dict[float, list[float]] = {}
        per_question: list[dict[str, Any]] = []

        for r in scorable:
            q_scores: dict[float, float] = {}
            for tol in tolerances:
                if r.predicted_answer is None:
                    score = 0.0
                else:
                    try:
                        score = float(
                            self._score_fn(r.expected_answer, r.predicted_answer, tol)
                        )
                    except Exception as exc:
                        logger.warning(
                            "EVAL_SCORE_ERROR uid=%s error=%s", r.uid, exc
                        )
                        score = 0.0
                scores_by_tolerance.setdefault(tol, []).append(score)
                q_scores[tol] = score

            per_question.append(
                {
                    "uid": r.uid,
                    "question": r.question[:80],
                    "expected": r.expected_answer,
                    "predicted": r.predicted_answer,
                    "status": r.status,
                    "scores": q_scores,
                    "wall_time": r.wall_time_seconds,
                    "difficulty": r.metadata.get("difficulty", ""),
                    "requires_external": r.metadata.get("requires_external", False),
                }
            )

        return EvaluationReport(
            total=len(results),
            answered=sum(1 for r in scorable if r.predicted_answer is not None),
            errors=sum(1 for r in scorable if r.status == "error"),
            timeouts=sum(1 for r in scorable if r.status == "timeout"),
            no_answers=sum(1 for r in scorable if r.status == "no_answer"),
            rate_limited=rate_limited_count,
            scores_by_tolerance=scores_by_tolerance,
            per_question=per_question,
            wall_times=[r.wall_time_seconds for r in scorable],
            model=model,
            timestamp=datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        )
