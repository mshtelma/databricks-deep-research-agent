"""Catalog metadata types — :class:`CatalogCard`, :class:`SafeProbe`, :class:`ProbeSample`.

These types let a tool factory declare per-kind affordance metadata that the
catalog renderer (:mod:`databricks_deep_research.tools.catalog_renderer`)
turns into a system-prompt block. Factories own the metadata so adding a new
kind forces declaring a card in the same place — no second registry to
forget.

Cards are deliberately affordance-only (``summary`` + ``input_prose`` +
``output_prose``). Topology guidance ("if X is generic, try Y") encodes
assumptions and biases the LLM toward the card-author's mental model rather
than the user's actual data, so it is excluded by design.

Cards reference tool *kinds* and asset *kinds* (e.g., "a structured table",
"an embedded text corpus"), never asset *names*. The framework runs a
stop-list test (``test_cards_contain_no_corpus_specific_strings``) to keep
corpus and customer names out of card prose.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from databricks_deep_research.tools.protocol import ToolContext

_SUMMARY_MAX_CHARS = 120


@dataclass(frozen=True)
class CatalogCard:
    """Affordance-only metadata for a single tool kind.

    Attributes
    ----------
    summary:
        One-line affordance statement, ≤120 chars. Used in summary-only
        rendering when the catalog exceeds the configured tool count
        threshold or when shedding under the character budget.
    input_prose:
        Natural-language description of the tool's input shape. The renderer
        escapes braces in untrusted blocks (probe samples, user-edited
        prose) at output time — card prose itself MUST NOT contain ``{`` or
        ``}`` characters.
    output_prose:
        Natural-language description of the tool's output shape. Same brace
        constraint as ``input_prose``.
    """

    summary: str
    input_prose: str
    output_prose: str

    def __post_init__(self) -> None:
        if not isinstance(self.summary, str) or not self.summary.strip():
            raise ValueError("CatalogCard.summary must be a non-empty string")
        if len(self.summary) > _SUMMARY_MAX_CHARS:
            raise ValueError(
                f"CatalogCard.summary must be ≤{_SUMMARY_MAX_CHARS} chars; "
                f"got {len(self.summary)}"
            )
        if not isinstance(self.input_prose, str):
            raise ValueError("CatalogCard.input_prose must be a string")
        if not isinstance(self.output_prose, str):
            raise ValueError("CatalogCard.output_prose must be a string")
        if "{" in self.summary or "}" in self.summary:
            raise ValueError(
                "CatalogCard.summary must not contain literal '{' or '}' "
                "(would collide with template substitution)"
            )
        if "{" in self.input_prose or "}" in self.input_prose:
            raise ValueError(
                "CatalogCard.input_prose must not contain literal '{' or '}'"
            )
        if "{" in self.output_prose or "}" in self.output_prose:
            raise ValueError(
                "CatalogCard.output_prose must not contain literal '{' or '}'"
            )


class ProbeSample(BaseModel):
    """A captured sample from running a tool's :class:`SafeProbe`.

    Attributes
    ----------
    sample_input:
        The arguments passed to the tool's ``execute`` (or equivalent) for
        the probe call.  Keys are stable; values are the actual probe input.
    sample_output:
        Truncated and sanitized stringified output.  The orchestrator
        truncates to ``ProbeConfig.max_output_chars`` and runs a sanitizer
        that scrubs PII patterns (emails, SSN-like, JWT-like, AWS-key-like).
    probed_at:
        UTC timestamp at which the probe was captured.  Stamped by the
        orchestrator, not the factory's :meth:`SafeProbe.run`.
    status:
        ``"ok"`` — probe ran and produced output.
        ``"error"`` — probe raised; ``reason`` carries the message.
        ``"skipped"`` — factory has no :class:`SafeProbe` for this kind, or
        an upstream gate (timeout, missing config) prevented the run.
    reason:
        Optional human-readable error message or skip reason.
    """

    model_config = ConfigDict(extra="forbid")

    sample_input: dict[str, Any] = Field(default_factory=dict)
    sample_output: str = ""
    probed_at: datetime
    status: Literal["ok", "error", "skipped"]
    reason: str | None = None


@runtime_checkable
class SafeProbe(Protocol):
    """Tool-kind-specific safe probe, declared by the factory itself.

    Each factory chooses which kinds have a SafeProbe.  Kinds without one
    (mutating tools, tools requiring user-specific arguments the framework
    cannot synthesize safely) leave their entry as ``None`` in
    :attr:`CatalogProvider.safe_probes`, and the orchestrator records
    ``status="skipped"`` for them.

    There is intentionally no generic ``SELECT 1`` / introspection fallback:
    the factory writer chooses what is safe for that kind, not the
    orchestrator.
    """

    async def run(
        self,
        *,
        config: dict[str, Any],
        ctx: ToolContext,
        user_query: str | None,
    ) -> ProbeSample:
        """Run a safe sample call against the tool's bound config.

        Implementations MUST honor ``ctx.read_only`` and MUST NOT mutate
        any external state.  The orchestrator wraps the call in
        :func:`asyncio.wait_for` and a concurrency semaphore.
        """
        ...


@runtime_checkable
class CatalogProvider(Protocol):
    """Structural protocol for objects that supply per-kind catalog metadata.

    All :class:`databricks_deep_research.tools.factory.ToolFactory`
    implementations satisfy this Protocol by declaring class-level
    ``catalog_cards`` and ``safe_probes`` attributes.

    Lives in this module (rather than ``factory.py``) so the renderer can
    depend on it without pulling in the factory module's heavier imports.
    """

    catalog_cards: Mapping[str, CatalogCard]
    safe_probes: Mapping[str, SafeProbe | None]


__all__ = [
    "CatalogCard",
    "CatalogProvider",
    "ProbeSample",
    "SafeProbe",
]
