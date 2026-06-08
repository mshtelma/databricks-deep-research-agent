"""Pydantic models used by the designer workflow's critic agent and
the architect's structured AST output. These are output_model classes
assigned programmatically to AgentConfig nodes by the route shim
before the workflow runs (see W5c)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, RootModel, field_validator

# PR3-C: severity field on CriticDirective. ``blocking`` directives keep the
# loop from terminating; ``advisory`` directives are polish suggestions the
# scaffolder_specializer may or may not act on. ``extract_critic_approved``
# treats all-advisory verdicts as approved.
DirectiveSeverity = Literal["blocking", "advisory"]


class CriticDirective(BaseModel):
    """One revision directive emitted by the designer critic agent."""

    model_config = ConfigDict(extra="forbid")

    node_path: str = ""           # JSON-pointer-like address (e.g. "$.root.children[2]")
    issue: str = ""               # one-line problem statement
    suggested_action: str = ""    # one-line fix instruction
    # PR3-C additions. Optional with defaults so existing tests creating
    # CriticDirective without these fields continue to pass.
    severity: DirectiveSeverity = "blocking"
    tool_hint: str | None = None  # mutation tool the auditor recommends


class CriticVerdict(BaseModel):
    """Structured output emitted by the designer critic. The loop's stop
    condition reads `approve`; the architect's next iteration reads
    `directives` to know what to revise."""

    model_config = ConfigDict(extra="forbid")

    approve: bool = False
    directives: list[CriticDirective] = Field(default_factory=list)

    @field_validator("directives", mode="before")
    @classmethod
    def _coerce_directives(cls, value: Any) -> list[Any]:
        # Tolerate missing/null directives — older LLMs may omit the field
        # when approving. Treat as empty list.
        if value is None:
            return []
        if not isinstance(value, list):
            return []
        return value


class WorkflowAST(RootModel[dict[str, Any]]):
    """Permissive structured-output type for the architect agent. The
    framework's structured-output path needs a Pydantic class; the AST
    itself is a free-form dict (validated downstream by
    load_workflow_from_dict). This wrapper exists so the harness can pass
    a class to the LLM client without imposing a rigid schema."""
