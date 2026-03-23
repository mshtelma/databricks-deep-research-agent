"""JSONL-based atomic append + resume for benchmark results."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from pathlib import Path

from benchmarks.core.types import QuestionResult

logger = logging.getLogger(__name__)


class ResultStore:
    """Append-only JSONL storage with resume support.

    Each line is a self-contained JSON object representing one QuestionResult.
    Single-writer assumed (BenchmarkRunner serialises writes via asyncio.Lock).
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def completed_uids(self) -> set[str]:
        """Read all UIDs already in the file. Tolerates truncated last line."""
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
        """Load all results, skip malformed lines with warning."""
        results: list[QuestionResult] = []
        if not self._path.exists():
            return results
        with open(self._path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    results.append(QuestionResult(**data))
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    logger.warning(
                        "RESULT_STORE_SKIP_LINE path=%s line=%d error=%s",
                        self._path,
                        line_num,
                        exc,
                    )
        return results
