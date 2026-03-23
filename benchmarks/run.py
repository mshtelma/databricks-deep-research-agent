#!/usr/bin/env python3
"""CLI: Run benchmark questions through the framework.

Usage:
    uv run benchmarks/run.py officeqa
    uv run benchmarks/run.py officeqa --concurrency 5 --timeout 600
    uv run benchmarks/run.py officeqa --model databricks-claude-sonnet-4-5
    uv run benchmarks/run.py officeqa --workflow workflow-sonnet.yaml
    uv run benchmarks/run.py officeqa --limit 10  # run only first 10 questions
    uv run benchmarks/run.py officeqa --trace-output  # collect MLflow traces
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
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
    parser = argparse.ArgumentParser(description="Run a benchmark")
    parser.add_argument("benchmark",  help="Benchmark name (e.g., officeqa)")
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=None, help="Timeout per question (seconds)")
    parser.add_argument("--model", type=str, default=None, help="Override model endpoint")
    parser.add_argument("--workflow", type=str, default=None, help="Workflow YAML filename")
    parser.add_argument("--limit", type=int, default=None, help="Max questions to run")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh, ignore prior results")
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument(
        "--trace-output", nargs="?", const="auto", default=None,
        help="Collect MLflow traces (default: in run dir; or specify custom dir)",
    )
    return parser.parse_args()


async def run_officeqa(args: argparse.Namespace) -> None:
    import yaml

    from databricks_deep_research import FrameworkLLMClient, WorkflowRunner, load_workflow_from_dict
    from databricks_deep_research.tools.factory import ToolFactoryContext

    from benchmarks.core.config_loader import _interpolate_recursive, load_config
    from benchmarks.core.runner import BenchmarkRunner
    from benchmarks.core.types import RunConfig
    from benchmarks.officeqa.dataset import OfficeQADataset

    # Load config
    config_path = BENCHMARKS_DIR / "officeqa" / "config.yaml"
    config = load_config(config_path)

    # Build RunConfig with CLI overrides
    run_cfg = config.get("run", {})
    run_config = RunConfig(
        concurrency=args.concurrency or run_cfg.get("concurrency", 3),
        timeout_per_question=args.timeout or run_cfg.get("timeout_per_question", 300),
        results_dir=args.results_dir or run_cfg.get("results_dir", "results/officeqa"),
        resume=not args.no_resume,
    )

    # Load workflow (with env var interpolation for ${VAR:-default} patterns)
    workflow_file = args.workflow or "workflow.yaml"
    workflow_path = BENCHMARKS_DIR / "officeqa" / workflow_file
    workflow_def = load_workflow_from_dict(
        _interpolate_recursive(yaml.safe_load(workflow_path.read_text()))
    )

    # Create LLM client
    model = args.model or config.get("model", "databricks-claude-opus-4-6")
    llm_client = FrameworkLLMClient.from_databricks(model=model)

    # Create tool factory
    factory = ToolFactoryContext.from_defaults()

    # Load dataset
    repo_path = Path(config["repo"]["local_path"])
    if not repo_path.exists():
        logger.error("OfficeQA repo not found at %s. Run: uv run benchmarks/ingest.py officeqa", repo_path)
        sys.exit(1)

    dataset = OfficeQADataset(repo_path)
    questions = dataset.load_questions()
    extractor = dataset.answer_extractor()

    if args.limit:
        questions = questions[: args.limit]
        logger.info("LIMITED to first %d questions", args.limit)

    # Per-run directory
    results_dir = Path(run_config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = results_dir / f"run-{timestamp}"

    # If resuming, find latest existing run directory (or legacy flat file)
    if run_config.resume:
        existing_dirs = sorted(results_dir.glob("run-*/results.jsonl"))
        existing_flat = sorted(results_dir.glob("run-*.jsonl"))
        if existing_dirs:
            results_path = existing_dirs[-1]
            run_dir = results_path.parent
            logger.info("RESUME from %s", results_path)
        elif existing_flat:
            results_path = existing_flat[-1]
            run_dir = results_path.parent  # stays in results_dir
            logger.info("RESUME from legacy %s", results_path)
        else:
            run_dir.mkdir(parents=True, exist_ok=True)
            results_path = run_dir / "results.jsonl"
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        results_path = run_dir / "results.jsonl"

    logger.info(
        "BENCHMARK_START benchmark=officeqa questions=%d concurrency=%d timeout=%d model=%s results=%s",
        len(questions),
        run_config.concurrency,
        run_config.timeout_per_question,
        model,
        results_path,
    )

    # Trace collection setup
    sys.path.insert(0, str(BENCHMARKS_DIR.parent / "databricks-deep-research"))
    collector = None
    if args.trace_output is not None:
        try:
            from tests.trace_collector import TraceCollector

            trace_dir = str(run_dir / "traces") if args.trace_output == "auto" else args.trace_output
            collector = TraceCollector(output_dir=trace_dir)
            if collector.setup_mlflow():
                collector.start()
                logger.info("TRACE_COLLECTION enabled, output=%s", trace_dir)
            else:
                logger.warning("MLflow setup failed — trace collection disabled")
                collector = None
        except ImportError:
            logger.warning("TraceCollector not found — trace collection disabled")

    # Run
    runner = BenchmarkRunner(llm_client, factory, run_config)
    t0 = time.monotonic()
    results = await runner.run(questions, workflow_def, extractor, results_path)
    elapsed = time.monotonic() - t0

    await llm_client.aclose()

    # Summary
    success = sum(1 for r in results if r.status == "success")
    errors = sum(1 for r in results if r.status == "error")
    timeouts = sum(1 for r in results if r.status == "timeout")
    no_answer = sum(1 for r in results if r.status == "no_answer")

    print(f"\n{'=' * 50}")
    print(f"Benchmark complete in {elapsed:.0f}s")
    print(f"Results: {results_path}")
    print(f"Success: {success} | No Answer: {no_answer} | Error: {errors} | Timeout: {timeouts}")
    print(f"\nRun evaluation: uv run benchmarks/evaluate.py officeqa --results {run_dir}")

    # Collect traces
    if collector:
        report = collector.collect()
        if report:
            collector.print_terminal_summary(report)


async def main() -> None:
    args = parse_args()

    if args.benchmark == "officeqa":
        await run_officeqa(args)
    else:
        print(f"Unknown benchmark: {args.benchmark}")
        print("Available: officeqa")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
