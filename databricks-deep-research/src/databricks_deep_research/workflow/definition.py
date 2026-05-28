"""Workflow definition models: the static, serialisable description of a workflow.

These models are the *schema* layer -- they describe **what** a workflow looks
like (nodes, pools, budgets) without any runtime behaviour.  Loading from /
saving to YAML is handled by :pymod:`databricks_deep_research.workflow.loader`.
"""

from __future__ import annotations

import uuid as _uuid
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from databricks_deep_research.tools.catalog_types import ProbeSample

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class NodeType(StrEnum):
    """Discriminator for the eight supported workflow node kinds.

    Leaf types (``agent``, ``tool``, ``subworkflow``) must have no children.
    Composite types (``sequence``, ``parallel``, ``loop``, ``conditional``,
    ``plan_and_execute``) carry child nodes.
    """

    agent = "agent"
    tool = "tool"
    sequence = "sequence"
    parallel = "parallel"
    loop = "loop"
    conditional = "conditional"
    subworkflow = "subworkflow"
    plan_and_execute = "plan_and_execute"


# ---------------------------------------------------------------------------
# Support models
# ---------------------------------------------------------------------------


class ServicePrincipalRunAs(BaseModel):
    """V1.5: run a workflow as a Databricks Service Principal instead of the calling user."""

    model_config = ConfigDict(extra="forbid")
    service_principal_id: str

    @field_validator("service_principal_id")
    @classmethod
    def _validate_uuid(cls, v: str) -> str:
        try:
            _uuid.UUID(v)
        except ValueError as exc:
            raise ValueError(
                f"service_principal_id must be a valid UUID, got {v!r}"
            ) from exc
        return v


class ErrorConfig(BaseModel):
    """Per-node error-handling policy.

    * ``fail``  -- propagate the exception (default).
    * ``skip``  -- emit a :class:`NodeSkippedEvent` and continue.
    * ``retry`` -- retry up to *max_retries* with exponential back-off.
    """

    model_config = ConfigDict(extra="forbid")

    on_error: str = "fail"
    max_retries: int = 2
    retry_delay_seconds: float = 1.0


# ---------------------------------------------------------------------------
# Workflow node (recursive tree)
# ---------------------------------------------------------------------------


class WorkflowNode(BaseModel):
    """A single node in the workflow tree.

    ``config`` is a free-form dict at this layer; concrete per-type config
    models (``AgentNodeConfig``, ``LoopNodeConfig``, ...) are validated by
    the executor or loader when the node type is known.

    ``children`` enables the recursive tree structure -- leaf node types
    (``agent``, ``tool``, ``subworkflow``) must leave this empty while
    composite types must populate it according to their structural rules.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    type: NodeType
    label: str
    config: dict[str, Any] = {}
    children: list[WorkflowNode] = []
    error_handling: ErrorConfig | None = None
    budget_seconds: float | None = None


# ---------------------------------------------------------------------------
# Tool declarations
# ---------------------------------------------------------------------------


class ToolDeclaration(BaseModel):
    """Declaration of a tool in the workflow's top-level ``tools:`` section.

    Tools are declared with a ``kind`` (matching :class:`ToolKind` or a custom
    string) and kind-specific ``config``.  Agent nodes reference tools by
    ``name`` (a plain string) rather than by ``{type, name}`` dicts.

    Example YAML::

        tools:
          - name: earnings_index
            kind: vector_search
            config:
              index_name: prod_catalog.finance.earnings_idx
              num_results: 10
            description: "Quarterly earnings filings"
    """

    model_config = ConfigDict(extra="forbid")

    name: str  # unique tool name, referenced in agent configs
    kind: str  # ToolKind value or custom string
    config: dict[str, Any] = {}  # kind-specific configuration
    description: str = ""  # human-readable, injected into tool definition
    probe: ProbeSample | None = None  # optional SafeProbe sample, sanitized/truncated


# ---------------------------------------------------------------------------
# Source declarations
# ---------------------------------------------------------------------------


class SourceDefinition(BaseModel):
    """Declaration of a data source available to the workflow.

    Used by the planner to generate source-aware research plans with
    appropriate query strategies per source kind. NOT a tool declaration --
    tools are registered separately via ToolRef/ToolRegistry.
    """

    model_config = ConfigDict(extra="forbid")

    name: str  # unique ID matching tool name
    kind: str  # SourceKind value
    endpoint: str = ""  # index name, genie space ID, etc.
    description: str = ""  # human-readable, for planner context
    query_strategy: dict[str, Any] = {}  # kind-specific config
    metadata: dict[str, Any] = {}  # extra (columns, schema, etc.)


# ---------------------------------------------------------------------------
# Top-level workflow definition
# ---------------------------------------------------------------------------


class WorkflowDefinition(BaseModel):
    """Top-level, serialisable description of a complete workflow.

    Instances are typically created by :meth:`from_yaml` (implemented in
    ``workflow.loader``) or built programmatically in tests.

    Parameters
    ----------
    id:
        Unique workflow identifier.
    name:
        Human-readable name shown in UIs and logs.
    description:
        Optional prose description of the workflow's purpose.
    version:
        Schema version for forward-compatible migration.
    root:
        The root :class:`WorkflowNode` of the execution tree.
    pools:
        Pool declarations (validated later via ``PoolConfig``).
    required_inputs:
        State keys that **must** be present before execution begins.
    output_keys:
        State keys the workflow is expected to produce.
    token_budget:
        Maximum total tokens across all LLM calls.  ``0`` means unlimited.
    timeout_seconds:
        Hard wall-clock timeout for the entire workflow execution.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    description: str = ""
    schema_version: int = 1
    version: int = 1
    root: WorkflowNode
    tools: list[ToolDeclaration] = []
    pools: list[dict[str, Any]] = []
    sources: list[SourceDefinition] = []
    models: dict[str, Any] = {}
    required_inputs: list[str] = ["query"]
    output_keys: list[str] = ["output"]
    token_budget: int = 0
    timeout_seconds: int = 1800
    run_as: Literal["caller"] | ServicePrincipalRunAs = Field(default="caller")

    @model_validator(mode="before")
    @classmethod
    def _coerce_run_as(cls, data: Any) -> Any:
        if isinstance(data, dict):
            ra = data.get("run_as")
            if ra is None:
                data["run_as"] = "caller"
            elif isinstance(ra, str) and ra != "caller":
                raise ValueError(
                    f"run_as must be 'caller' or a ServicePrincipalRunAs object, got {ra!r}"
                )
        return data

    # -- Serialisation stubs (real implementation lives in loader.py) -------

    @classmethod
    def from_yaml(cls, path: str | Path) -> WorkflowDefinition:
        """Deserialise a workflow definition from a YAML file.

        .. note::
           This is a stub.  The full implementation -- including schema
           validation, node-type-specific config parsing, and pool resolution
           -- lives in :pymod:`databricks_deep_research.workflow.loader`.

        Raises
        ------
        NotImplementedError
            Always, until ``loader.py`` is wired up.
        """
        raise NotImplementedError(
            "from_yaml is not yet implemented -- see workflow.loader"
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkflowDefinition:
        """Build a workflow definition from a plain dictionary.

        .. note::
           This is a stub.  The full implementation -- including schema
           validation, node-type-specific config parsing, and pool resolution
           -- lives in :pymod:`databricks_deep_research.workflow.loader`.

        Raises
        ------
        NotImplementedError
            Always, until ``loader.py`` is wired up.
        """
        raise NotImplementedError(
            "from_dict is not yet implemented -- see workflow.loader"
        )

    def to_yaml(self, path: str | Path) -> None:
        """Serialise this workflow definition to a YAML file.

        .. note::
           This is a stub.  The full implementation lives in
           :pymod:`databricks_deep_research.workflow.loader`.

        Raises
        ------
        NotImplementedError
            Always, until ``loader.py`` is wired up.
        """
        raise NotImplementedError(
            "to_yaml is not yet implemented -- see workflow.loader"
        )
