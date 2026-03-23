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
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
    for name in (".env.officeqa", ".env.ais", ".env", ".env.test"):
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
    return parser.parse_args()


def _resolve_results_path(results_arg: str) -> Path:
    """Resolve to a JSONL file — handles run dirs, results dirs, and legacy flat files."""
    p = Path(results_arg)
    if p.is_file():
        return p
    if p.is_dir():
        # Run directory containing results.jsonl directly
        direct = p / "results.jsonl"
        if direct.exists():
            return direct
        # Results directory with run-*/results.jsonl subdirectories
        run_dirs = sorted(p.glob("run-*/results.jsonl"))
        if run_dirs:
            return run_dirs[-1]
        # Legacy: flat run-*.jsonl files
        files = sorted(p.glob("run-*.jsonl"))
        if files:
            return files[-1]
        print(f"No results found in {p}")
        sys.exit(1)
    print(f"Results path not found: {p}")
    sys.exit(1)


def evaluate_officeqa(args: argparse.Namespace) -> None:
    from benchmarks.core.config_loader import load_config
    from benchmarks.core.result_store import ResultStore
    from benchmarks.officeqa.evaluator import OfficeQAEvaluator

    # Load config for repo path
    config_path = BENCHMARKS_DIR / "officeqa" / "config.yaml"
    config = load_config(config_path)
    repo_path = Path(config["repo"]["local_path"])

    if not repo_path.exists():
        print(f"OfficeQA repo not found at {repo_path}.")
        print("Run: uv run benchmarks/ingest.py officeqa")
        sys.exit(1)

    # Load results
    results_path = _resolve_results_path(args.results)
    logger.info("EVALUATE_LOAD results=%s", results_path)
    store = ResultStore(results_path)
    results = store.load_all()

    if not results:
        print(f"No results found in {results_path}")
        sys.exit(1)

    logger.info("EVALUATE_RESULTS count=%d", len(results))

    # Parse tolerances
    tolerances = [float(t) for t in args.tolerances.split(",")]

    # Evaluate
    evaluator = OfficeQAEvaluator(repo_path)
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

    # Also save per-question details as JSON (in same directory as results)
    details_path = results_path.parent / "results.eval.json"
    details = {
        "summary": {
            "total": report.total,
            "answered": report.answered,
            "errors": report.errors,
            "timeouts": report.timeouts,
            "accuracy": {
                str(tol): report.accuracy_at(tol) for tol in tolerances
            },
        },
        "per_question": report.per_question,
    }
    details_path.write_text(json.dumps(details, indent=2, default=str), encoding="utf-8")
    print(f"Details saved to {details_path}")


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
