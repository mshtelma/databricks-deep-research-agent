from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from databricks_deep_research.events.types import StreamEvent
from databricks_deep_research.tools.factory import ToolFactory, ToolFactoryContext
from databricks_deep_research.tools.protocol import ResearchTool
from databricks_deep_research.tools.registry import ToolRegistry
from databricks_deep_research.workflow.definition import WorkflowDefinition
from databricks_deep_research.workflow.runtime_core.models import RuntimeState


@dataclass
class WorkflowRunRequest:
    definition: WorkflowDefinition
    query: str = ""
    inputs: dict[str, Any] = field(default_factory=dict)
    enterprise_tools: list[ResearchTool] | None = None
    tool_registry: ToolRegistry | None = None
    tool_factories: list[ToolFactory] | None = None
    factory_context: ToolFactoryContext | None = None
    strict_tool_resolution: bool = False


@dataclass
class WorkflowRunResult:
    runtime_state: RuntimeState
    events: list[StreamEvent] = field(default_factory=list)

    @property
    def artifacts(self) -> dict[str, Any]:
        return self.runtime_state.artifacts

    @property
    def output(self) -> str:
        synthesis = self.runtime_state.capabilities.synthesis
        if synthesis is not None and synthesis.report_artifact_id:
            artifact = self.runtime_state.artifacts.get(synthesis.report_artifact_id)
            if artifact is not None and artifact.payload is not None:
                return str(artifact.payload)
        return ""

    @property
    def sources(self) -> list[dict[str, Any]]:
        evidence = self.runtime_state.capabilities.evidence
        if evidence is None:
            return []
        return [source.model_dump(mode="json") for source in evidence.sources]
