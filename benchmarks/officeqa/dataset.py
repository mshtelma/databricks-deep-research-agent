"""OfficeQA dataset loading — CSV to BenchmarkQuestion mapping."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from benchmarks.core.answer_extract import AnswerExtractor, XMLTagExtractor
from benchmarks.core.types import BenchmarkQuestion

logger = logging.getLogger(__name__)


class OfficeQADataset:
    """Load questions from OfficeQA CSV and flag external-source questions."""

    def __init__(
        self,
        repo_path: Path,
        dataset_name: str = "officeqa_pro",
    ) -> None:
        self._repo_path = repo_path
        self._csv_path = repo_path / f"{dataset_name}.csv"

    def load_questions(self) -> list[BenchmarkQuestion]:
        """Load questions from CSV. Flag external-source questions."""
        if not self._csv_path.exists():
            raise FileNotFoundError(f"Dataset CSV not found: {self._csv_path}")

        df = pd.read_csv(self._csv_path)
        logger.info("DATASET_LOADED csv=%s rows=%d", self._csv_path.name, len(df))

        ingested_files = self._get_ingested_filenames()
        questions: list[BenchmarkQuestion] = []

        for _, row in df.iterrows():
            source_files_raw = str(row.get("source_files", ""))
            source_files = [
                f.strip() for f in source_files_raw.split("\n") if f.strip()
            ]
            requires_external = bool(
                ingested_files
                and not all(f in ingested_files for f in source_files if f)
            )

            questions.append(
                BenchmarkQuestion(
                    uid=str(row["uid"]),
                    question=str(row["question"]),
                    expected_answer=str(row["answer"]),
                    metadata={
                        "difficulty": str(row.get("difficulty", "")),
                        "source_files": source_files_raw,
                        "source_docs": str(row.get("source_docs", "")),
                        "requires_external": requires_external,
                    },
                )
            )

        logger.info(
            "DATASET_QUESTIONS total=%d external=%d",
            len(questions),
            sum(1 for q in questions if q.metadata.get("requires_external")),
        )
        return questions

    def answer_extractor(self) -> AnswerExtractor:
        return XMLTagExtractor(tag="FINAL_ANSWER")

    def _get_ingested_filenames(self) -> set[str]:
        """Scan the transformed text directory for available filenames."""
        transformed_dir = (
            self._repo_path / "treasury_bulletins_parsed" / "transformed"
        )
        if not transformed_dir.exists():
            return set()
        return {f.name for f in transformed_dir.glob("*.txt")}
