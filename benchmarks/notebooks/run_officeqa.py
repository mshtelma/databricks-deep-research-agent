# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # OfficeQA Benchmark Runner
# MAGIC
# MAGIC Runs OfficeQA benchmark questions through a configurable workflow YAML
# MAGIC (default: **v8-hybrid**) and evaluates accuracy against ground truth.
# MAGIC
# MAGIC ### Prerequisites
# MAGIC - **Vector search index** `{catalog}.{schema}.treasury_chunks_vs_index` exists
# MAGIC - **Delta table** `{catalog}.{schema}.treasury_chunks` is populated
# MAGIC - **SQL warehouse** is running (needed for delta_read / delta_grep tools)
# MAGIC - **Model endpoint** (e.g. `databricks-claude-opus-4-6`) is available
# MAGIC - (Optional) **BRAVE_API_KEY** in a Databricks secret scope — only needed for
# MAGIC   workflows that use web search tools; the v8-hybrid workflow does **not**.
# MAGIC
# MAGIC ### How to Run
# MAGIC 1. Fill in the **warehouse_id** widget (required)
# MAGIC 2. Adjust other widgets as needed (model, concurrency, limit, etc.)
# MAGIC 3. Run All

# COMMAND ----------

# MAGIC %pip install nest_asyncio
# MAGIC %pip install ../../databricks-deep-research
dbutils.library.restartPython()

# COMMAND ----------

# -- Widgets ----------------------------------------------------------------
# Create all configurable parameters as Databricks widgets.
# Widget values persist across cluster restarts and re-runs.

dbutils.widgets.text("warehouse_id", "f45852ca675f5dcb", "SQL Warehouse ID (required)")
dbutils.widgets.text("catalog", "main", "Unity Catalog Name")
dbutils.widgets.text("schema", "officeqa_benchmark", "Schema Name")
dbutils.widgets.text("vs_endpoint", "dbdemos_vs_endpoint", "Vector Search Endpoint")
dbutils.widgets.text("model", "databricks-claude-opus-4-6", "Model Endpoint")
dbutils.widgets.text("workflow_file", "workflow-v9-hybrid.yaml", "Workflow YAML filename")
dbutils.widgets.text("concurrency", "1", "Parallel questions")
dbutils.widgets.text("timeout", "3600", "Timeout per question (seconds)")
dbutils.widgets.text("limit", "", "Max questions (empty = all)")
dbutils.widgets.text("uids", "", "UID filter (comma-separated, empty = all)")
dbutils.widgets.dropdown("resume", "no", ["yes", "no"], "Resume from prior run")
dbutils.widgets.text("retry_statuses", "timeout", "Retry statuses on resume (comma-separated, empty = none)")
dbutils.widgets.text("brave_secret_scope", "deep-research-secrets", "Secret scope for BRAVE_API_KEY (optional)")
dbutils.widgets.text("brave_secret_key", "BRAVE_API_KEY", "Secret key name")
dbutils.widgets.text("results_path", "/Volumes/main/officeqa_benchmark/results", "Results directory (empty = UC volume)")

# COMMAND ----------

# -- Read & validate configuration ------------------------------------------
import os
import sys
from pathlib import Path

warehouse_id = dbutils.widgets.get("warehouse_id").strip()
catalog = dbutils.widgets.get("catalog").strip()
schema = dbutils.widgets.get("schema").strip()
vs_endpoint = dbutils.widgets.get("vs_endpoint").strip()
model = dbutils.widgets.get("model").strip()
workflow_file = dbutils.widgets.get("workflow_file").strip()
concurrency = int(dbutils.widgets.get("concurrency").strip() or "1")
timeout = int(dbutils.widgets.get("timeout").strip() or "1800")
limit_str = dbutils.widgets.get("limit").strip()
limit = int(limit_str) if limit_str else None
uids = dbutils.widgets.get("uids").strip() or None
resume = dbutils.widgets.get("resume") == "yes"
retry_statuses_raw = dbutils.widgets.get("retry_statuses").strip()
retry_statuses: frozenset[str] = frozenset(
    s.strip() for s in retry_statuses_raw.split(",") if s.strip()
)
brave_scope = dbutils.widgets.get("brave_secret_scope").strip()
brave_key_name = dbutils.widgets.get("brave_secret_key").strip()
results_base = dbutils.widgets.get("results_path").strip()
if not results_base:
    results_base = f"/Volumes/{catalog}/{schema}/results"

assert warehouse_id, (
    "warehouse_id widget is required — set it to your SQL warehouse ID"
)

# Detect repo root — walk up from CWD until we find benchmarks/core/.
# Works for both Git Folders (CWD=repo root) and databricks-sync (CWD=notebook dir).
_cwd = Path(os.getcwd())
REPO_ROOT = None
for _d in [_cwd, *_cwd.parents]:
    if (_d / "benchmarks" / "core").is_dir():
        REPO_ROOT = str(_d)
        break
assert REPO_ROOT is not None, (
    f"Could not locate repo root (benchmarks/core/ not found above {_cwd}). "
    "Ensure this notebook is run from a Databricks Git Folder or synced workspace."
)
BENCHMARKS_DIR = os.path.join(REPO_ROOT, "benchmarks")

print(f"Repo root:      {REPO_ROOT}")
print(f"Model:          {model}")
print(f"Workflow:        {workflow_file}")
print(f"Warehouse:       {warehouse_id}")
print(f"Catalog/Schema:  {catalog}.{schema}")
print(f"Concurrency:     {concurrency}, Timeout: {timeout}s")
print(f"Limit:           {limit or 'all'}, UIDs: {uids or 'all'}")
print(f"Resume:          {resume}")
print(f"Retry statuses:  {', '.join(sorted(retry_statuses)) or 'none'}")

# COMMAND ----------

# -- Environment variables + secrets ----------------------------------------
# The workflow YAML uses ${OFFICEQA_WAREHOUSE_ID} etc. which are resolved
# by _interpolate_recursive() reading os.environ at YAML parse time.
# These MUST be set before Cell 9 loads the workflow.

os.environ["OFFICEQA_CATALOG"] = catalog
os.environ["OFFICEQA_SCHEMA"] = schema
os.environ["OFFICEQA_WAREHOUSE_ID"] = warehouse_id
os.environ["OFFICEQA_VS_ENDPOINT"] = vs_endpoint

if brave_scope:
    try:
        brave_api_key = dbutils.secrets.get(scope=brave_scope, key=brave_key_name)
        os.environ["BRAVE_API_KEY"] = brave_api_key
        print(f"BRAVE_API_KEY loaded from scope={brave_scope}, key={brave_key_name}")
    except Exception as e:
        print(f"WARNING: Could not load BRAVE_API_KEY: {e}")
        print("Web search tools will not be available (not needed for v8-hybrid)")
else:
    print("No brave_secret_scope set — skipping BRAVE_API_KEY (not needed for v8-hybrid)")

# COMMAND ----------

# -- Logging, async patching, imports ---------------------------------------
import asyncio
import logging
import time

import nest_asyncio

nest_asyncio.apply()

# Add repo root to sys.path so benchmarks.* modules resolve
sys.path.insert(0, REPO_ROOT)

# FIPS compatibility patch (must import before any framework use)
import databricks_deep_research._fips_compat  # noqa: F401

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("officeqa_notebook")

import yaml

from databricks_deep_research import FrameworkLLMClient, load_workflow_from_dict
from databricks_deep_research.tools.factory import ToolFactoryContext

from benchmarks.core.config_loader import _interpolate_recursive
from benchmarks.core.run_dir import setup_run_dir
from benchmarks.core.runner import BenchmarkRunner
from benchmarks.core.types import RunConfig
from benchmarks.core.workspace_pool import WorkspacePool
from benchmarks.officeqa.dataset import OfficeQADataset

print("All imports successful")

# COMMAND ----------

# -- Enable MLflow tracing --------------------------------------------------
# Ensures all framework spans (workflow, node, agent, tool) are persisted
# to MLflow so traces can be inspected after the notebook run completes.
from databricks_deep_research.tracing import setup_mlflow_tracing, shutdown_mlflow_tracing

tracing_ok = setup_mlflow_tracing()
print(f"MLflow tracing: {'enabled' if tracing_ok else 'disabled (spans will not be recorded)'}")

# COMMAND ----------

# -- Set up trace collection (record start time BEFORE benchmark) -----------
from datetime import datetime, UTC

import mlflow

_trace_collector = None
_benchmark_start_time = datetime.now(tz=UTC)

# Resolve experiment ID for trace collection
_experiment_id = None
try:
    notebook_path = (
        dbutils.notebook.entry_point.getDbutils()
        .notebook().getContext().notebookPath().get()
    )
    exp = mlflow.get_experiment_by_name(notebook_path)
    if exp:
        _experiment_id = exp.experiment_id
        print(f"Experiment: {exp.name} (id={exp.experiment_id})")
except Exception as e:
    print(f"WARNING: Could not determine experiment ID: {e}")

try:
    sys.path.insert(0, os.path.join(REPO_ROOT, "databricks-deep-research"))
    from tests.trace_collector import TraceCollector

    _trace_collector = TraceCollector(output_dir=str(Path(results_base) / "traces"))
    _trace_collector._start_time = _benchmark_start_time
    if _experiment_id:
        _trace_collector._experiment_id = _experiment_id
        print("Trace collection ready")
    else:
        _trace_collector = None
except ImportError as e:
    print(f"TraceCollector not available ({e}) — traces will remain in MLflow only")

# COMMAND ----------

# -- Load workflow definition -----------------------------------------------
workflow_path = Path(BENCHMARKS_DIR) / "officeqa" / workflow_file
assert workflow_path.exists(), f"Workflow not found: {workflow_path}"

raw_yaml = yaml.safe_load(workflow_path.read_text())
interpolated = _interpolate_recursive(raw_yaml)
workflow_def = load_workflow_from_dict(interpolated)

print(f"Workflow: {workflow_def.id} ({workflow_def.name})")
print(f"Tools:    {len(workflow_def.tools)}")
print(f"Timeout:  {workflow_def.timeout_seconds}s")

# COMMAND ----------

# -- Create LLM client & tool factory --------------------------------------
# Inside Databricks notebooks, WorkspaceClient() auto-detects auth from
# the notebook execution context — no token or profile needed.

llm_client = FrameworkLLMClient.from_databricks(model=model)
pool = WorkspacePool.single(llm_client)
factory = ToolFactoryContext.from_defaults()

print(f"LLM client ready (model={model})")
print(f"Tool factory ready (search_client={'yes' if factory.search_client else 'no'})")

# COMMAND ----------

# -- Load dataset & apply filters -------------------------------------------
dataset = OfficeQADataset()
questions = dataset.load_questions()
extractor = dataset.answer_extractor()
total_count = len(questions)

if uids:
    from benchmarks.core.uid_filter import filter_by_uid_fragments, parse_uid_fragments

    fragments = parse_uid_fragments(uids)
    questions = filter_by_uid_fragments(questions, fragments, lambda q: q.uid)
    logger.info("FILTERED to %d questions by UID substring match", len(questions))

if limit:
    questions = questions[:limit]
    logger.info("LIMITED to first %d questions", len(questions))

print(f"Questions to run: {len(questions)} (dataset total: {total_count})")

# COMMAND ----------

# -- Configure run & results directory --------------------------------------
run_config = RunConfig(
    concurrency=concurrency,
    timeout_per_question=timeout,
    results_dir=results_base,
    resume=resume,
    retry_statuses=retry_statuses,
)

run_dir, results_path = setup_run_dir(results_base, resume=resume)

print(f"Run directory:  {run_dir}")
print(f"Results file:   {results_path}")

# COMMAND ----------

# -- Start MLflow benchmark run --------------------------------------------
# Creates a single MLflow run that encompasses the entire benchmark session.
# Traces, params, metrics, and artifacts all associate with this run.
from benchmarks.core.mlflow_utils import benchmark_mlflow_run

# End any stale run from a previous interrupted execution
if mlflow.active_run():
    print(f"Ending stale run: {mlflow.active_run().info.run_id}")
    mlflow.end_run()

_run_name = f"officeqa-{workflow_file.replace('.yaml', '')}-{model}"
_mlflow_run_ctx = benchmark_mlflow_run(
    run_name=_run_name,
    params={
        "model": model,
        "workflow": workflow_file,
        "workflow_id": workflow_def.id,
        "workflow_version": str(getattr(workflow_def, "version", "")),
        "workflow_name": workflow_def.name,
        "tool_count": len(workflow_def.tools),
        #"warehouse_id": warehouse_id,
        "catalog": catalog,
        "schema": schema,
        #"vs_endpoint": vs_endpoint,
        "concurrency": concurrency,
        "timeout": timeout,
        "limit": str(limit or "all"),
        "uids": uids or "all",
        "total_questions": len(questions),
        "resume": resume,
    },
    artifact_paths=[workflow_path],
)
_mlflow_run_ctx.__enter__()
print(f"MLflow run: {mlflow.active_run().info.run_id}")
print(f"All traces will associate with this run.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Benchmark Execution
# MAGIC Running questions through the workflow. Progress is logged below.

# COMMAND ----------

# -- Execute benchmark -------------------------------------------------------

async def _run_benchmark():
    runner = BenchmarkRunner(pool, factory, run_config)
    t0 = time.monotonic()
    results = await runner.run(questions, workflow_def, extractor, results_path)
    elapsed = time.monotonic() - t0
    await pool.aclose()
    return results, elapsed


# Use run_until_complete (not asyncio.run) to preserve the current context —
# including MLflow's active run, so traces auto-associate with the benchmark run.
# nest_asyncio.apply() (Cell 6) makes re-entrant event loop usage safe.
_loop = asyncio.get_event_loop()
results, elapsed = _loop.run_until_complete(_run_benchmark())

# Summary
success = sum(1 for r in results if r.status == "success")
errors = sum(1 for r in results if r.status == "error")
timeouts = sum(1 for r in results if r.status == "timeout")
no_answer = sum(1 for r in results if r.status == "no_answer")
rate_limited = sum(1 for r in results if getattr(r, "status", "") == "rate_limited")

print(f"\n{'=' * 60}")
print(f"Benchmark complete in {elapsed:.0f}s")
print(f"Results: {results_path}")
print(f"Success: {success} | No Answer: {no_answer} | Error: {errors} | Timeout: {timeouts} | Rate Limited: {rate_limited}")

# Log raw results immediately — survives evaluation failures
if mlflow.active_run():
    mlflow.log_artifact(str(results_path))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluation
# MAGIC Scoring results against ground truth using the official OfficeQA reward function.

# COMMAND ----------

# -- Evaluate + log metrics to the active benchmark run --------------------
from benchmarks.officeqa.evaluator import OfficeQAEvaluator
from benchmarks.core.mlflow_utils import (
    write_evaluation_artifacts,
    log_evaluation_to_mlflow,
)

evaluator = OfficeQAEvaluator()
tolerances = [0.0, 0.01, 0.05]
report = evaluator.evaluate(results, tolerances=tolerances, model=model)

# Print report
print(report.format_report())

# Write + log evaluation artifacts and metrics to the active benchmark run
eval_json, eval_txt = write_evaluation_artifacts(report, results_path.parent)
log_evaluation_to_mlflow(
    report,
    elapsed_seconds=elapsed,
    artifact_paths=[eval_json, eval_txt],
)

print(f"\nEvaluation saved to: {eval_json}")
print(f"Report saved to: {eval_txt}")
print(f"Metrics + artifacts logged to run: {mlflow.active_run().info.run_id}")

# COMMAND ----------

# -- Per-question results (interactive table) --------------------------------
import pandas as pd

df = pd.DataFrame(report.per_question)
for tol in tolerances:
    label = "exact" if tol == 0.0 else f"fuzzy_{tol}"
    df[label] = df["scores"].apply(lambda s, t=tol: s.get(t, 0.0))
df = df.drop(columns=["scores"])
display(df.sort_values("uid"))

# COMMAND ----------

# -- Error analysis ----------------------------------------------------------
errors_df = df[df["status"].isin(["error", "timeout", "no_answer", "rate_limited"])]
if not errors_df.empty:
    print(f"Failed questions ({len(errors_df)}):")
    display(errors_df)
else:
    print("All questions answered successfully")

# COMMAND ----------

# -- End MLflow run + collect traces -----------------------------------------
# End the benchmark run (flushes trace data), then download traces.
_mlflow_run_ctx.__exit__(None, None, None)
print("MLflow run closed")

if _trace_collector and _trace_collector._experiment_id:
    _trace_collector._output_dir = str(run_dir / "traces")
    trace_report = _trace_collector.collect()  # flushes tracing + downloads
    if trace_report:
        _trace_collector.print_terminal_summary(trace_report)
        print(f"\nTrace files saved to: {run_dir / 'traces'}")
    else:
        print("No traces collected")
else:
    shutdown_mlflow_tracing()
    print("MLflow traces flushed (no trace collection configured)")
