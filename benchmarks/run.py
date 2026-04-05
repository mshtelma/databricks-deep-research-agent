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
    parser = argparse.ArgumentParser(description="Run a benchmark")
    parser.add_argument("benchmark",  help="Benchmark name (e.g., officeqa)")
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=None, help="Timeout per question (seconds)")
    parser.add_argument("--model", type=str, default=None, help="Override model endpoint")
    parser.add_argument("--workflow", type=str, default=None, help="Workflow YAML filename")
    parser.add_argument("--limit", type=int, default=None, help="Max questions to run")
    parser.add_argument(
        "--uids", type=str, default=None,
        help="Comma-separated UID substrings to filter questions (e.g., 0029,0030,0057)",
    )
    parser.add_argument("--no-resume", action="store_true", help="Start fresh, ignore prior results")
    parser.add_argument(
        "--retry-status", type=str, default=None,
        help="Comma-separated statuses to retry on resume (e.g., timeout,error)",
    )
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument(
        "--trace-output", nargs="?", const="auto", default=None,
        help="Collect MLflow traces (default: in run dir; or specify custom dir)",
    )
    return parser.parse_args()


async def run_officeqa(args: argparse.Namespace) -> None:
    import os

    import yaml

    from databricks_deep_research import FrameworkLLMClient, WorkflowRunner, load_workflow_from_dict
    from databricks_deep_research.tools.factory import ToolFactoryContext

    from benchmarks.core.config_loader import _interpolate_recursive, load_config
    from benchmarks.core.runner import BenchmarkRunner
    from benchmarks.core.types import RunConfig
    from benchmarks.core.workspace_pool import WorkspacePool
    from benchmarks.officeqa.dataset import OfficeQADataset

    # Load config
    config_path = BENCHMARKS_DIR / "officeqa" / "config.yaml"
    config = load_config(config_path)

    # Parse and validate --retry-status
    _KNOWN_STATUSES = {"success", "error", "timeout", "no_answer", "rate_limited"}
    retry_statuses: frozenset[str] = frozenset()
    if args.retry_status:
        raw = frozenset(s.strip() for s in args.retry_status.split(",") if s.strip())
        unknown = raw - _KNOWN_STATUSES
        if unknown:
            logger.error(
                "Unknown status(es): %s. Valid: %s",
                ", ".join(sorted(unknown)),
                ", ".join(sorted(_KNOWN_STATUSES)),
            )
            sys.exit(1)
        retry_statuses = raw

    if args.retry_status and args.no_resume:
        logger.warning("--retry-status has no effect with --no-resume (all questions run fresh)")

    # Load workflow (with env var interpolation for ${VAR:-default} patterns)
    workflow_file = args.workflow or "workflow.yaml"
    workflow_path = BENCHMARKS_DIR / "officeqa" / workflow_file
    workflow_def = load_workflow_from_dict(
        _interpolate_recursive(yaml.safe_load(workflow_path.read_text()))
    )

    # Resolve model endpoint
    model = args.model or config.get("model", "databricks-claude-opus-4-6")

    # Workspace pool: distribute LLM calls across multiple workspaces.
    # Primary source: config.yaml (direct value or ${VAR} interpolation).
    profiles_str = config.get("workspace_profiles", "")
    profiles = [p.strip() for p in profiles_str.split(",") if p.strip()]

    if profiles:
        pool = WorkspacePool.from_profiles(profiles, model=model)
    else:
        llm_client = FrameworkLLMClient.from_databricks(model=model)
        pool = WorkspacePool.single(llm_client)

    # Create tool factory (after pool creation so restored env vars are visible)
    factory = ToolFactoryContext.from_defaults()

    # Build RunConfig — auto-adjust concurrency to profile count when not
    # explicitly set via --concurrency.
    run_cfg = config.get("run", {})
    if profiles and args.concurrency is None:
        effective_concurrency = len(profiles)
    else:
        effective_concurrency = args.concurrency or run_cfg.get("concurrency", 3)

    run_config = RunConfig(
        concurrency=effective_concurrency,
        timeout_per_question=args.timeout or run_cfg.get("timeout_per_question", 300),
        results_dir=args.results_dir or run_cfg.get("results_dir", "results/officeqa"),
        resume=not args.no_resume,
        retry_statuses=retry_statuses,
    )

    # Load dataset (uses bundled CSV by default)
    dataset = OfficeQADataset()
    questions = dataset.load_questions()
    extractor = dataset.answer_extractor()

    if args.uids:
        from benchmarks.core.uid_filter import filter_by_uid_fragments, parse_uid_fragments

        fragments = parse_uid_fragments(args.uids)
        questions = filter_by_uid_fragments(questions, fragments, lambda q: q.uid)
        logger.info("FILTERED to %d questions by UID substring match", len(questions))

    if args.limit:
        questions = questions[: args.limit]
        logger.info("LIMITED to first %d questions", args.limit)

    # Per-run directory (with resume)
    from benchmarks.core.run_dir import setup_run_dir

    run_dir, results_path = setup_run_dir(
        run_config.results_dir, resume=run_config.resume
    )

    logger.info(
        "BENCHMARK_START benchmark=officeqa questions=%d concurrency=%d timeout=%d model=%s results=%s",
        len(questions),
        run_config.concurrency,
        run_config.timeout_per_question,
        model,
        results_path,
    )
    if profiles:
        logger.info(
            "WORKSPACE_MODE=multi profiles=%s count=%d",
            ",".join(profiles),
            len(profiles),
        )
    else:
        logger.info(
            "WORKSPACE_MODE=single (set OFFICEQA_WORKSPACE_PROFILES in .env.officeqa to enable multi-workspace)",
        )
    if retry_statuses:
        logger.info("BENCHMARK_RETRY_STATUSES statuses=%s", ",".join(sorted(retry_statuses)))

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

    # Enable MLflow tracing so framework trace_span calls are recorded.
    from databricks_deep_research.tracing import setup_mlflow_tracing, shutdown_mlflow_tracing

    tracing_ok = setup_mlflow_tracing()
    if tracing_ok:
        logger.info("MLFLOW_TRACING enabled")
    else:
        logger.warning("MLFLOW_TRACING disabled (spans will not be recorded)")

    # Run — wrapped in an MLflow run so traces, params, metrics, and artifacts
    # are all associated with a single run.
    from benchmarks.core.mlflow_utils import benchmark_mlflow_run

    run_name = f"officeqa-{workflow_file.replace('.yaml', '')}-{model}"
    with benchmark_mlflow_run(
        run_name=run_name,
        params={
            "model": model,
            "workflow": workflow_file,
            "workflow_id": workflow_def.id,
            "workflow_version": getattr(workflow_def, "version", ""),
            "workflow_name": workflow_def.name,
            "tool_count": len(workflow_def.tools),
            "concurrency": run_config.concurrency,
            "timeout": run_config.timeout_per_question,
            "limit": str(args.limit or "all"),
            "uids": args.uids or "all",
            "total_questions": len(questions),
            "resume": not args.no_resume,
            "workspace_profiles": ",".join(profiles) if profiles else "single",
            "workspace_count": pool.size,
        },
        artifact_paths=[workflow_path],
    ):
        runner = BenchmarkRunner(pool, factory, run_config)
        t0 = time.monotonic()
        results = await runner.run(questions, workflow_def, extractor, results_path)
        elapsed = time.monotonic() - t0

        await pool.aclose()

        # Summary
        success = sum(1 for r in results if r.status == "success")
        errors = sum(1 for r in results if r.status == "error")
        timeouts = sum(1 for r in results if r.status == "timeout")
        no_answer = sum(1 for r in results if r.status == "no_answer")

        print(f"\n{'=' * 50}")
        print(f"Benchmark complete in {elapsed:.0f}s")
        print(f"Results: {results_path}")
        print(f"Success: {success} | No Answer: {no_answer} | Error: {errors} | Timeout: {timeouts}")

        # Log raw results immediately — survives evaluation failures
        try:
            import mlflow

            if mlflow.active_run():
                mlflow.log_artifact(str(results_path))
        except Exception:
            pass

        # Evaluate + log metrics to the SAME run
        try:
            from benchmarks.core.mlflow_utils import (
                log_evaluation_to_mlflow,
                write_evaluation_artifacts,
            )
            from benchmarks.officeqa.evaluator import OfficeQAEvaluator

            evaluator = OfficeQAEvaluator()
            report = evaluator.evaluate(
                results, tolerances=[0.0, 0.01, 0.05], model=model,
            )

            eval_json, eval_txt = write_evaluation_artifacts(
                report, results_path.parent,
            )
            log_evaluation_to_mlflow(
                report,
                elapsed_seconds=elapsed,
                artifact_paths=[eval_json, eval_txt],
            )

            print(report.format_report())
            if mlflow.active_run():
                print(f"\nMLflow run: {mlflow.active_run().info.run_id}")
        except Exception as exc:
            logger.warning("BENCHMARK_EVAL_LOG_FAILED error=%s", exc)

    # Flush tracing (after MLflow run ends so all spans are associated)
    shutdown_mlflow_tracing()

    print(f"\nRun evaluation: uv run benchmarks/evaluate.py officeqa --results {run_dir}")

    # Collect traces (after run ends — collector searches by experiment + timestamp)
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
