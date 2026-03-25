"""BenchmarkRunner — concurrent workflow execution engine."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from openai import RateLimitError

from databricks_deep_research import (
    FrameworkLLMClient,
    WorkflowDefinition,
    WorkflowRunner,
)
from databricks_deep_research.tools.factory import ToolFactoryContext

from benchmarks.core.answer_extract import AnswerExtractor
from benchmarks.core.result_store import ResultStore
from benchmarks.core.types import BenchmarkQuestion, QuestionResult, RunConfig

logger = logging.getLogger(__name__)


def _log_progress(
    counter: dict[str, int], uid: str, result: QuestionResult
) -> None:
    done = counter["done"]
    total = counter["total"]
    pct = (done / total * 100) if total else 0
    status = result.status
    answer_preview = (result.predicted_answer or "")[:60]
    logger.info(
        "BENCHMARK_PROGRESS %d/%d (%.0f%%) uid=%s status=%s time=%.1fs answer=%s",
        done,
        total,
        pct,
        uid,
        status,
        result.wall_time_seconds,
        answer_preview,
    )


class BenchmarkRunner:
    """Concurrent workflow execution engine for benchmarks.

    Creates a fresh WorkflowRunner per question (required because last_result
    is instance-level state). Shares FrameworkLLMClient and ToolFactoryContext
    across all questions.
    """

    def __init__(
        self,
        llm_client: FrameworkLLMClient,
        factory_context: ToolFactoryContext,
        run_config: RunConfig,
    ) -> None:
        self._client = llm_client
        self._factory = factory_context
        self._config = run_config

    async def run(
        self,
        questions: list[BenchmarkQuestion],
        workflow_definition: WorkflowDefinition,
        extractor: AnswerExtractor,
        results_path: Path,
    ) -> list[QuestionResult]:
        """Run all questions through the workflow with concurrency control."""
        store = ResultStore(results_path)
        completed = store.completed_uids() if self._config.resume else set()
        pending = [q for q in questions if q.uid not in completed]

        logger.info(
            "BENCHMARK total=%d pending=%d skipped=%d",
            len(questions),
            len(pending),
            len(completed),
        )

        if not pending:
            logger.info("BENCHMARK_COMPLETE all questions already answered")
            return store.load_all()

        semaphore = asyncio.Semaphore(self._config.concurrency)
        write_lock = asyncio.Lock()
        counter = {"done": len(completed), "total": len(questions)}

        async def run_one(q: BenchmarkQuestion) -> QuestionResult:
            async with semaphore:
                t0 = time.monotonic()
                try:
                    runner = WorkflowRunner(
                        llm_client=self._client,
                        factory_context=self._factory,
                    )
                    result = await asyncio.wait_for(
                        runner.run(workflow_definition, query=q.question),
                        timeout=self._config.timeout_per_question,
                    )
                    predicted = extractor.extract(result.output)
                    if predicted is None and result.output:
                        logger.warning(
                            "EXTRACTOR_NO_MATCH uid=%s output_len=%d first_200=%s",
                            q.uid,
                            len(result.output),
                            result.output[:200],
                        )
                    status = "success" if predicted else "no_answer"
                    qr = QuestionResult(
                        uid=q.uid,
                        question=q.question,
                        expected_answer=q.expected_answer,
                        predicted_answer=predicted,
                        raw_output=result.output,
                        status=status,
                        wall_time_seconds=time.monotonic() - t0,
                        num_sources=len(result.sources),
                        metadata=q.metadata,
                    )
                except asyncio.TimeoutError:
                    qr = QuestionResult(
                        uid=q.uid,
                        question=q.question,
                        expected_answer=q.expected_answer,
                        predicted_answer=None,
                        raw_output="",
                        status="timeout",
                        wall_time_seconds=time.monotonic() - t0,
                        num_sources=0,
                        error_message=f"Timed out after {self._config.timeout_per_question}s",
                        metadata=q.metadata,
                    )
                except RateLimitError as exc:
                    qr = QuestionResult(
                        uid=q.uid,
                        question=q.question,
                        expected_answer=q.expected_answer,
                        predicted_answer=None,
                        raw_output="",
                        status="rate_limited",
                        wall_time_seconds=time.monotonic() - t0,
                        num_sources=0,
                        error_message=str(exc)[:500],
                        metadata=q.metadata,
                    )
                except Exception as exc:
                    # Safety net: check if a wrapped exception is really a rate limit
                    is_rate_limit = "429" in str(exc) or "rate_limit" in str(exc).lower()
                    qr = QuestionResult(
                        uid=q.uid,
                        question=q.question,
                        expected_answer=q.expected_answer,
                        predicted_answer=None,
                        raw_output="",
                        status="rate_limited" if is_rate_limit else "error",
                        wall_time_seconds=time.monotonic() - t0,
                        num_sources=0,
                        error_message=str(exc)[:500],
                        metadata=q.metadata,
                    )

                async with write_lock:
                    store.append(qr)
                    counter["done"] += 1
                    _log_progress(counter, q.uid, qr)
                return qr

        tasks = [run_one(q) for q in pending]
        # gather won't raise because run_one catches all exceptions
        await asyncio.gather(*tasks, return_exceptions=False)
        return store.load_all()
