#!/usr/bin/env python3
"""CLI: Evaluate benchmark results using official scoring.

Usage:
    uv run benchmarks/evaluate.py officeqa --results results/officeqa/run-20260315-120000/
    uv run benchmarks/evaluate.py officeqa --results results/officeqa/  # latest run dir
    uv run benchmarks/evaluate.py officeqa --results results/officeqa/ --tolerances 0.0,0.01,0.05,0.10
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import databricks_deep_research._fips_compat  # noqa: F401, E402  # FIPS md5 patch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

BENCHMARKS_DIR = Path(__file__).resolve().parent


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    root = BENCHMARKS_DIR.parent
    for name in (".env.officeqa", ".env", ".env.test"):
        candidate = root / name
        if candidate.exists():
            load_dotenv(candidate, override=False)


_load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate benchmark results")
    parser.add_argument("benchmark", help="Benchmark name (e.g., officeqa)")
    parser.add_argument("--results", type=str, required=True, help="Path to results JSONL or directory")
    parser.add_argument("--tolerances", type=str, default="0.0,0.01,0.05", help="Comma-separated tolerances")
    parser.add_argument("--model", type=str, default="", help="Model name for report")
    parser.add_argument("--output", type=str, default=None, help="Save report to file")
    parser.add_argument(
        "--uids", type=str, default=None,
        help="Comma-separated UID substrings to filter results (e.g., 0029,0030,0057)",
    )
    return parser.parse_args()


def _resolve_results_path(results_arg: str) -> Path:
    """Resolve to a JSONL file — handles run dirs, results dirs, and legacy flat files."""
    from benchmarks.core.run_dir import resolve_results_path

    try:
        return resolve_results_path(results_arg)
    except FileNotFoundError as exc:
        print(str(exc))
        sys.exit(1)


def evaluate_officeqa(args: argparse.Namespace) -> None:
    from benchmarks.core.config_loader import load_config
    from benchmarks.core.result_store import ResultStore
    from benchmarks.officeqa.evaluator import OfficeQAEvaluator

    # Load config
    config_path = BENCHMARKS_DIR / "officeqa" / "config.yaml"
    config = load_config(config_path)

    # Load results
    results_path = _resolve_results_path(args.results)
    logger.info("EVALUATE_LOAD results=%s", results_path)
    store = ResultStore(results_path)
    results = store.load_all()

    if args.uids:
        from benchmarks.core.uid_filter import filter_by_uid_fragments, parse_uid_fragments

        fragments = parse_uid_fragments(args.uids)
        results = filter_by_uid_fragments(results, fragments, lambda r: r.uid)
        logger.info("FILTERED to %d results by UID substring match", len(results))

    if not results:
        print(f"No results found in {results_path}")
        sys.exit(1)

    logger.info("EVALUATE_RESULTS count=%d", len(results))

    # Parse tolerances
    tolerances = [float(t) for t in args.tolerances.split(",")]

    # Evaluate (uses bundled reward.py)
    evaluator = OfficeQAEvaluator()
    model_name = args.model or config.get("model", "")
    report = evaluator.evaluate(results, tolerances=tolerances, model=model_name)

    # Print report
    report_text = report.format_report()
    print(f"\n{report_text}")

    # Save if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report_text, encoding="utf-8")
        print(f"\nReport saved to {output_path}")

    # Write standard evaluation artifacts
    from benchmarks.core.mlflow_utils import (
        log_evaluation_to_mlflow,
        write_evaluation_artifacts,
    )

    eval_json, eval_txt = write_evaluation_artifacts(report, results_path.parent)
    print(f"Details saved to {eval_json}")

    # Log to active MLflow run if one exists (no-ops otherwise)
    log_evaluation_to_mlflow(
        report,
        artifact_paths=[results_path, eval_json, eval_txt],
    )


def main() -> None:
    args = parse_args()

    if args.benchmark == "officeqa":
        evaluate_officeqa(args)
    else:
        print(f"Unknown benchmark: {args.benchmark}")
        print("Available: officeqa")
        sys.exit(1)


if __name__ == "__main__":
    main()
