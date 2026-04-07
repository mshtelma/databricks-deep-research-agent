"""JSONL-based atomic append + resume for benchmark results."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from pathlib import Path

from benchmarks.core.types import QuestionResult

logger = logging.getLogger(__name__)


_TERMINAL_STATUSES = frozenset({"success", "no_answer", "timeout", "error"})


class ResultStore:
    """Append-only JSONL storage with resume support.

    Each line is a self-contained JSON object representing one QuestionResult.
    Single-writer assumed (BenchmarkRunner serialises writes via asyncio.Lock).

    On resume, only questions with terminal status are skipped.
    ``rate_limited`` questions are automatically retried.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def completed_uids(
        self, *, exclude_statuses: frozenset[str] = frozenset()
    ) -> set[str]:
        """Return UIDs with terminal status.

        UIDs whose *only* terminal entries match ``exclude_statuses`` are
        excluded from the completed set, making them eligible for retry.
        Rate-limited UIDs are always retryable (not in _TERMINAL_STATUSES).
        """
        effective_terminal = _TERMINAL_STATUSES - exclude_statuses
        uids: set[str] = set()
        if not self._path.exists():
            return uids
        with open(self._path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if data.get("status") in effective_terminal:
                        uids.add(data["uid"])
                except (json.JSONDecodeError, KeyError):
                    logger.warning(
                        "RESULT_STORE_SKIP_LINE path=%s line=%d reason=malformed",
                        self._path,
                        line_num,
                    )
        return uids

    def append(self, result: QuestionResult) -> None:
        """Atomic append: write line + fsync."""
        data = asdict(result)
        line = json.dumps(data, ensure_ascii=False, default=str)
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())

    def load_all(self) -> list[QuestionResult]:
        """Load results, last-write-wins per UID for retried questions."""
        by_uid: dict[str, QuestionResult] = {}
        if not self._path.exists():
            return []
        with open(self._path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    by_uid[data["uid"]] = QuestionResult(**data)
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    logger.warning(
                        "RESULT_STORE_SKIP_LINE path=%s line=%d error=%s",
                        self._path,
                        line_num,
                        exc,
                    )
        return list(by_uid.values())
