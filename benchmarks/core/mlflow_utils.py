"""MLflow run management and metric logging for benchmark sessions."""

from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator

if TYPE_CHECKING:
    from benchmarks.officeqa.evaluator import EvaluationReport

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def benchmark_mlflow_run(
    run_name: str,
    params: dict[str, Any] | None = None,
    artifact_paths: list[Path | str] | None = None,
) -> Generator[None, None, None]:
    """Wrap a benchmark session in an MLflow run.

    All ``trace_span`` calls inside this context auto-associate with
    the run via MLflow's ``contextvars`` mechanism.  Callers should
    log metrics and additional artifacts before exiting the context.

    Usage (CLI — single async function)::

        with benchmark_mlflow_run("officeqa-v9", params={...}, artifact_paths=[yaml_path]):
            results = await runner.run(...)
            mlflow.log_metrics({...})

    Usage (notebook — explicit enter/exit across cells)::

        ctx = benchmark_mlflow_run("officeqa-v9", params={...}, artifact_paths=[yaml_path])
        ctx.__enter__()
        # ... benchmark cells ...
        mlflow.log_metrics({...})
        ctx.__exit__(None, None, None)

    If MLflow is not installed or ``start_run`` fails, yields without
    tracking (benchmark still runs, just not tracked).
    """
    try:
        import mlflow
    except ImportError:
        logger.info("BENCHMARK_RUN mlflow_not_available — running without tracking")
        yield
        return

    try:
        mlflow.start_run(run_name=run_name)
    except Exception as exc:
        logger.warning("BENCHMARK_RUN mlflow_start_failed error=%s", exc)
        yield
        return

    try:
        # Log hyperparameters
        if params:
            safe_params = {k: str(v) for k, v in params.items()}
            mlflow.log_params(safe_params)

        # Log artifacts (workflow YAML, config files, etc.)
        if artifact_paths:
            for path in artifact_paths:
                try:
                    mlflow.log_artifact(str(path))
                except Exception as exc:
                    logger.warning(
                        "BENCHMARK_RUN artifact_log_failed path=%s error=%s",
                        path, exc,
                    )

        run_id = mlflow.active_run().info.run_id
        logger.info("BENCHMARK_MLFLOW_RUN run_id=%s run_name=%s", run_id, run_name)

        yield

    finally:
        try:
            mlflow.end_run()
        except Exception:
            pass


def write_evaluation_artifacts(
    report: EvaluationReport,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write evaluation JSON and text report to *output_dir*.

    Returns ``(eval_json_path, eval_txt_path)``.
    """
    tolerances = sorted(report.scores_by_tolerance.keys())

    details = {
        "summary": {
            "total": report.total,
            "answered": report.answered,
            "errors": report.errors,
            "timeouts": report.timeouts,
            "no_answers": report.no_answers,
            "rate_limited": report.rate_limited,
            "accuracy": {
                str(tol): report.accuracy_at(tol) for tol in tolerances
            },
        },
        "per_question": report.per_question,
    }

    eval_json = output_dir / "results.eval.json"
    eval_json.write_text(
        json.dumps(details, indent=2, default=str), encoding="utf-8",
    )

    eval_txt = output_dir / "results.eval.txt"
    eval_txt.write_text(report.format_report(), encoding="utf-8")

    return eval_json, eval_txt


def log_evaluation_to_mlflow(
    report: EvaluationReport,
    elapsed_seconds: float | None = None,
    artifact_paths: list[Path | str] | None = None,
) -> None:
    """Log evaluation metrics and artifacts to the **active** MLflow run.

    No-ops if MLflow is not installed or no run is active.
    Never creates its own run.
    """
    try:
        import mlflow
    except ImportError:
        return

    if not mlflow.active_run():
        logger.info("LOG_EVAL_SKIP no active MLflow run")
        return

    # Log metrics — independent of artifact upload
    try:
        import statistics

        tolerances = sorted(report.scores_by_tolerance.keys())
        metrics: dict[str, float] = {}

        for tol in tolerances:
            label = "exact" if tol == 0.0 else f"fuzzy_{tol}"
            metrics[f"accuracy_{label}"] = report.accuracy_at(tol)

        metrics["total"] = report.total
        metrics["answered"] = report.answered
        metrics["errors"] = report.errors
        metrics["timeouts"] = report.timeouts
        metrics["no_answers"] = report.no_answers
        metrics["rate_limited"] = report.rate_limited

        if elapsed_seconds is not None:
            metrics["elapsed_seconds"] = elapsed_seconds

        if report.wall_times:
            metrics["avg_wall_time"] = statistics.mean(report.wall_times)
            metrics["median_wall_time"] = statistics.median(report.wall_times)

        mlflow.log_metrics(metrics)

    except Exception as exc:
        logger.warning("LOG_EVAL_METRICS_FAILED error=%s", exc)

    # Log artifacts — independent of metric success
    if artifact_paths:
        for path in artifact_paths:
            try:
                mlflow.log_artifact(str(path))
            except Exception as exc:
                logger.warning(
                    "EVAL_ARTIFACT_FAILED path=%s error=%s", path, exc,
                )
