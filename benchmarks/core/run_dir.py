"""Results directory setup with resume and legacy format support.

Used by both the CLI runner (benchmarks/run.py) and Databricks notebook
(benchmarks/notebooks/run_officeqa_v8_hybrid.py) to standardize run
directory creation, resume discovery, and path resolution.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def setup_run_dir(
    results_dir: str | Path,
    *,
    resume: bool = True,
) -> tuple[Path, Path]:
    """Create timestamped run directory with optional resume.

    If *resume* is True, looks for the latest existing run and returns it.
    If none is found (or resume is False), creates a fresh timestamped
    directory.

    Returns
    -------
    (run_dir, results_path)
        *run_dir* is the directory for this run.
        *results_path* is the JSONL file inside it.
    """
    base = Path(results_dir)
    base.mkdir(parents=True, exist_ok=True)

    if resume:
        # Current format: run-YYYYMMDD-HHMMSS/results.jsonl
        existing = sorted(base.glob("run-*/results.jsonl"))
        if existing:
            results_path = existing[-1]
            logger.info("RESUME from %s", results_path)
            return results_path.parent, results_path

        # Legacy format: run-YYYYMMDD-HHMMSS.jsonl (flat)
        legacy = sorted(base.glob("run-*.jsonl"))
        if legacy:
            logger.info("RESUME from legacy %s", legacy[-1])
            return legacy[-1].parent, legacy[-1]

    # Fresh run
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = base / f"run-{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, run_dir / "results.jsonl"


def resolve_results_path(results_arg: str | Path) -> Path:
    """Resolve user-supplied path to a concrete results JSONL file.

    Handles four cases:

    1. Direct file path
    2. Run directory containing ``results.jsonl``
    3. Results directory with ``run-*/results.jsonl`` subdirectories
    4. Legacy ``run-*.jsonl`` flat files

    Raises
    ------
    FileNotFoundError
        If no results file can be found at the given path.
    """
    p = Path(results_arg)

    if p.is_file():
        return p

    if p.is_dir():
        # Run directory with results.jsonl directly inside
        direct = p / "results.jsonl"
        if direct.exists():
            return direct

        # Results directory with run-*/results.jsonl subdirectories
        run_dirs = sorted(p.glob("run-*/results.jsonl"))
        if run_dirs:
            return run_dirs[-1]

        # Legacy flat files
        flat = sorted(p.glob("run-*.jsonl"))
        if flat:
            return flat[-1]

    raise FileNotFoundError(f"No results found at {p}")
