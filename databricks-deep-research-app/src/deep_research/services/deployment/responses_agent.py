"""DeepResearchResponsesAgent — Mosaic AI Agent Framework PythonModel wrapper.

Plan reference: agent-designer-deployment.md Section D.1.

This is the Mode 3 (MLflow agent serving) entry point. ``MlflowAgentTranslator``
calls ``mlflow.pyfunc.log_model`` with the path to THIS module (not an
instance). At serve time, MLflow imports the module, instantiates the class,
and calls ``load_context`` with the model artifacts. Then ``predict`` /
``predict_stream`` are invoked per request.

Lives in the app (not the framework) because serving requires
``mlflow>=3.1.3`` (ResponsesAgent + agents.deploy) while the framework's
``[tracing]`` extra pins ``mlflow>=2.10`` — a version-ceiling conflict.

``WorkflowRunner.from_databricks()`` takes no arguments — auth resolves
from the workspace environment. The workflow definition is passed at
``run()`` / ``stream()`` time, NOT at runner construction.
"""
# ruff: noqa: ARG002 -- predict/predict_stream signatures are dictated by
# mlflow.pyfunc.PythonModel; context+params are passed by the serving runtime
# even when this implementation does not consume them.
from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, cast

import mlflow.pyfunc
from databricks_deep_research import WorkflowRunner
from databricks_deep_research.workflow.loader import load_workflow_from_dict

if TYPE_CHECKING:
    from databricks_deep_research.workflow.definition import WorkflowDefinition

_PythonModel: Any = mlflow.pyfunc.PythonModel
_PythonModelContext: Any = mlflow.pyfunc.PythonModelContext


class DeepResearchResponsesAgent(_PythonModel):  # type: ignore[misc]
    """Wrap ``WorkflowRunner`` for Mosaic AI Agent Framework serving.

    The runner + workflow definition are loaded lazily in ``load_context``
    because the MLflow model context (with artifact paths) is only available
    there, not at ``__init__`` time.
    """

    def __init__(self) -> None:
        super().__init__()
        self._runner: WorkflowRunner | None = None
        self._definition: WorkflowDefinition | None = None

    def load_context(self, context: _PythonModelContext) -> None:
        """Load the workflow definition + construct the runner.

        ``context.artifacts['workflow_definition']`` is a path to a JSON file
        written by ``MlflowAgentTranslator.translate()``.
        ``WorkflowRunner.from_databricks()`` resolves auth from the serving
        environment (DATABRICKS_HOST etc., injected by Databricks).
        """
        definition_path = context.artifacts["workflow_definition"]
        with open(definition_path, encoding="utf-8") as f:
            definition_dict: dict[str, Any] = json.load(f)
        self._definition = load_workflow_from_dict(definition_dict)
        self._runner = WorkflowRunner.from_databricks()

    @staticmethod
    def _extract_query(model_input: dict[str, Any]) -> str:
        """Pull the user query from either ``input.text`` or ``input.messages[-1].content``."""
        input_section = model_input.get("input", {})
        text = input_section.get("text")
        if isinstance(text, str) and text:
            return text
        messages = input_section.get("messages", [])
        if isinstance(messages, list) and messages:
            last = messages[-1]
            if isinstance(last, dict):
                content = last.get("content", "")
                if isinstance(content, str):
                    return content
        return ""

    def predict(
        self,
        context: _PythonModelContext,
        model_input: dict[str, Any],
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Synchronous prediction — runs the full workflow."""
        if self._runner is None or self._definition is None:
            raise RuntimeError("load_context must be called before predict")

        query = self._extract_query(model_input)
        result = asyncio.run(
            self._runner.run(
                workflow=self._definition,
                query=query,
                strict_tool_resolution=True,
            )
        )

        return {
            "output": {
                "text": result.output,
                "sources": [
                    {"url": s.url, "title": s.title}
                    for s in result.sources[:10]
                ],
            },
        }

    async def predict_stream(
        self,
        context: _PythonModelContext,
        model_input: dict[str, Any],
        params: dict[str, Any] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Streaming prediction — Responses-API-compatible delta events.

        MLflow's Agent Server frames each yielded dict as
        ``data: <json>\\n\\n`` SSE. The framework ``StreamEvent`` is a
        discriminated union; only events whose ``event_type ==
        "agent_stream_chunk"`` carry chunk text.
        """
        if self._runner is None or self._definition is None:
            raise RuntimeError("load_context must be called before predict_stream")

        query = self._extract_query(model_input)
        async for event in self._runner.stream(
            workflow=self._definition,
            query=query,
            strict_tool_resolution=True,
        ):
            if event.event_type == "agent_stream_chunk":
                chunk_event = cast(Any, event)
                yield {
                    "type": "response.output_text.delta",
                    "delta": chunk_event.chunk,
                }

        yield {"type": "response.output_item.done"}
