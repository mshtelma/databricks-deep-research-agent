"""Pure tool-catalog renderer — turns ToolDeclaration + CatalogCard pairs
into a deterministic system-prompt block.

The renderer is deliberately pure (no I/O, no clock, no factory lookups
beyond the inputs) so save-time materialization (Designer pipeline) and
runtime resolution (harness fallback) call the *same* function with the
*same* inputs and produce the *same* output. Determinism is enforced by
sorting tools by their declared name before assembling the block.

A single :data:`REGISTRY_VERSION` string travels with every persisted
catalog (saved into ``LaneSpec.tool_catalog_registry_version``). When the
framework upgrades cards or render logic, bumping this constant forces a
re-render on the next run — see :mod:`databricks_deep_research.services.
catalog_service` (created in Phase 1).

Trust boundaries
----------------
Card prose (``CatalogCard.summary`` / ``input_prose`` / ``output_prose``)
is framework-authored: it must NOT contain literal ``{`` or ``}``
characters. The dataclass enforces this in :meth:`__post_init__`. By
contrast, probe samples (``ProbeSample.sample_input`` /
``sample_output``) carry user-touched payloads, so the renderer escapes
their braces at output time before substituting them into the prose
template. This is the only safe template-substitution boundary in the
catalog system.

Budget shedding
---------------
``CatalogConfig.max_chars`` caps the rendered block length. When the
catalog would exceed the budget, the renderer progressively sheds
detail in this fixed order:

1. Drop probe samples (oldest tools first).
2. Collapse to summary-only entries (oldest tools first).
3. Truncate the tail with a "(N tools omitted)" notice.

The ordering is deterministic so two runs with the same input always
produce the same output even at the budget boundary.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from databricks_deep_research.tools.catalog_types import (
    CatalogCard,
    CatalogProvider,
    ProbeSample,
)
from databricks_deep_research.workflow.definition import ToolDeclaration

# Bumped when render output, card schema, or shedding logic changes in a
# way that breaks parity with previously-persisted catalogs. Phase-0 baseline.
REGISTRY_VERSION: str = "1"
CATALOG_VERSION: str = "1"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, init=False)
class CatalogConfig:
    """Renderer knobs — everything that influences rendered output.

    Defaults match the prompt-budget targets in the plan; override per-call
    from :class:`CatalogService` when the surrounding system prompt is
    unusually large or small.

    Attributes
    ----------
    max_chars:
        Hard upper bound on the rendered block size (in characters).
        Enforced by progressive shedding.
    max_probe_chars:
        Hard upper bound on each probe sample slice.
    include_probes:
        When ``True`` and a probe is present (``ProbeSample.status == "ok"``),
        embed a short, brace-escaped example block under each tool.
    summary_only_above_n_tools:
        When the catalog declares more than this many tools, render
        summary-only by default (skip ``input_prose`` / ``output_prose``
        and any probe samples). Keeps large catalogs from dominating the
        prompt.
    """

    max_chars: int
    max_probe_chars: int
    include_probes: bool
    summary_only_above_n_tools: int

    def __init__(
        self,
        max_chars: int = 4000,
        max_probe_chars: int = 200,
        include_probes: bool = True,
        summary_only_above_n_tools: int = 8,
        *,
        include_probe_samples: bool | None = None,
        summary_only_threshold: int | None = None,
    ) -> None:
        if include_probe_samples is not None:
            include_probes = include_probe_samples
        if summary_only_threshold is not None:
            summary_only_above_n_tools = summary_only_threshold
        object.__setattr__(self, "max_chars", max_chars)
        object.__setattr__(self, "max_probe_chars", max_probe_chars)
        object.__setattr__(self, "include_probes", include_probes)
        object.__setattr__(
            self, "summary_only_above_n_tools", summary_only_above_n_tools
        )

    @property
    def include_probe_samples(self) -> bool:
        """Backward-compatible alias for earlier Phase-0 tests."""
        return self.include_probes

    @property
    def summary_only_threshold(self) -> int:
        """Backward-compatible alias for earlier Phase-0 tests."""
        return self.summary_only_above_n_tools


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


_BLOCK_HEADER = "## Available Tools"
_BLOCK_INTRO = (
    "The following tools are bound to this agent. Reach for them when their "
    "described affordance matches the work you need to do; ignore them when "
    "it does not. Tool names appear exactly as they must be invoked."
)


def _escape_braces(text: str) -> str:
    """Escape literal '{' / '}' so the renderer's output is safe to feed
    through a brace-substitution template downstream."""
    return text.replace("{", "{{").replace("}", "}}")


def _format_summary_only(decl: ToolDeclaration, card: CatalogCard, position: int) -> str:
    return f"- **{decl.name}** (kind: `{decl.kind}`, #{position}): {card.summary}"


def _format_probe_block(probe: ProbeSample, *, max_probe_chars: int) -> str | None:
    """Format a probe sample, escaping braces in user-touched payloads."""
    if probe.status != "ok" or not probe.sample_output:
        return None
    sample_output = probe.sample_output[:max_probe_chars]
    safe_output = _escape_braces(sample_output)
    if probe.sample_input:
        # Render input as compact key=value pairs; brace-escape values.
        rendered = ", ".join(
            f"{k}={_escape_braces(str(v))}" for k, v in probe.sample_input.items()
        )
        prefix = f"<probe sample, captured {probe.probed_at.date().isoformat()}, untrusted-data>\n"
        prefix += f"Example call: `{rendered}`\n"
    else:
        prefix = f"<probe sample, captured {probe.probed_at.date().isoformat()}, untrusted-data>\n"
    return f"{prefix}{safe_output}\n</probe sample>"


def _format_full(
    decl: ToolDeclaration,
    card: CatalogCard,
    probe: ProbeSample | None,
    *,
    include_probe_samples: bool,
    max_probe_chars: int,
) -> str:
    parts: list[str] = []
    parts.append(f"### `{decl.name}` (kind: `{decl.kind}`)")
    parts.append(card.summary)
    parts.append(f"**Input:** {card.input_prose}")
    parts.append(f"**Output:** {card.output_prose}")
    if include_probe_samples and probe is not None:
        block = _format_probe_block(probe, max_probe_chars=max_probe_chars)
        if block:
            parts.append(block)
    return "\n\n".join(parts)


def _format_unknown(decl: ToolDeclaration, position: int) -> str:
    return (
        f"- **{decl.name}** (kind: `{decl.kind}`, #{position}): "
        "(no catalog metadata)"
    )


def _assemble(
    *,
    full_entries: list[str],
    summary_entries: list[str],
    omitted_count: int,
) -> str:
    parts: list[str] = [
        _BLOCK_HEADER,
        f"(catalog-version: {CATALOG_VERSION}, registry-version: {REGISTRY_VERSION})",
        _BLOCK_INTRO,
    ]
    if full_entries:
        parts.append("\n\n".join(full_entries))
    if summary_entries:
        parts.append("\n".join(summary_entries))
    if omitted_count > 0:
        parts.append(f"_({omitted_count} additional tool(s) omitted to stay within prompt budget.)_")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CatalogRenderResult:
    """Outcome of a render pass.

    Attributes
    ----------
    text:
        The rendered tool-catalog block, ready to splice into a system
        prompt. Always brace-safe — feeding it through a downstream
        ``str.format``-style substitution will not corrupt the contents.
    rendered_tool_count:
        Number of tools that received any presence (summary or full) in
        the output. Useful for telemetry / regression assertions.
    omitted_count:
        Number of tools that had to be dropped entirely under the
        ``max_chars`` budget.
    used_summary_only:
        ``True`` when the renderer collapsed to summary-only mode,
        either because tool count exceeded the threshold or because
        budget shedding forced it.
    """

    text: str
    rendered_tool_count: int
    omitted_count: int
    used_summary_only: bool
    registry_version: str = field(default=REGISTRY_VERSION)


def render_tool_catalog(
    declarations: Sequence[ToolDeclaration] | Iterable[ToolDeclaration],
    catalog_cards_by_kind: Mapping[str, CatalogCard] | None = None,
    *,
    factory_lookup: Callable[[str], CatalogProvider | None] | None = None,
    config: CatalogConfig | None = None,
    probe_samples_by_name: Mapping[str, ProbeSample] | None = None,
) -> CatalogRenderResult:
    """Render the tool-catalog block for a set of tool declarations.

    The function is pure — given the same inputs, it always returns the
    same output. Tools are emitted in a deterministic order (sorted by
    declared name) regardless of the iteration order of ``declarations``.

    Parameters
    ----------
    declarations:
        The tools to describe. Declarations whose ``kind`` is not present
        in ``catalog_cards_by_kind`` are silently skipped (they are not
        catalog-aware tools — e.g., legacy ``decorated`` callables).
    catalog_cards_by_kind:
        Per-kind affordance metadata, typically the union of
        ``factory.catalog_cards`` mappings collected by the catalog
        service. Kept for existing tests and services.
    factory_lookup:
        Plan-aligned lookup path. When provided, the renderer asks the
        factory that supports each declaration's kind for its card; unknown
        kinds emit a sentinel line instead of failing the whole catalog.
    config:
        Render knobs. Defaults to :class:`CatalogConfig` defaults.
    probe_samples_by_name:
        Optional probe samples keyed by ``ToolDeclaration.name``. Only
        entries with ``status == "ok"`` and non-empty ``sample_output``
        are considered; the renderer also requires
        ``config.include_probe_samples=True`` to embed them.

    Returns
    -------
    CatalogRenderResult
        The rendered text, plus accounting fields useful for telemetry.
    """
    cfg = config or CatalogConfig()
    probes = probe_samples_by_name or {}

    decls = list(declarations)
    cards = catalog_cards_by_kind or {}
    paired: list[tuple[int, ToolDeclaration, CatalogCard, ProbeSample | None]] = []
    unknown_entries: list[str] = []
    for position, decl in enumerate(decls, start=1):
        card: CatalogCard | None = cards.get(decl.kind)
        if card is None and factory_lookup is not None:
            factory = factory_lookup(decl.kind)
            if factory is not None:
                card = factory.catalog_cards.get(decl.kind)
        if card is None:
            if factory_lookup is not None:
                logger.warning("TOOL_CATALOG_KIND_UNKNOWN kind=%s", decl.kind)
                unknown_entries.append(_format_unknown(decl, position))
            continue
        probe = decl.probe or probes.get(decl.name)
        paired.append((position, decl, card, probe))

    # Deterministic ordering by tool name.
    paired.sort(key=lambda item: (item[1].name, item[0]))
    unknown_entries.sort()

    if not paired and not unknown_entries:
        return CatalogRenderResult(
            text="(no tools wired)" if not decls else "",
            rendered_tool_count=0,
            omitted_count=0,
            used_summary_only=False,
        )

    # Decide initial render mode by tool count.
    use_summary_only = len(paired) > cfg.summary_only_above_n_tools

    # First pass — render at chosen verbosity.
    if use_summary_only:
        full_entries: list[str] = []
        summary_entries: list[str] = [
            _format_summary_only(d, c, pos) for pos, d, c, _ in paired
        ] + list(unknown_entries)
    else:
        full_entries = [
            _format_full(
                d,
                c,
                p,
                include_probe_samples=cfg.include_probes,
                max_probe_chars=cfg.max_probe_chars,
            )
            for _, d, c, p in paired
        ]
        summary_entries = list(unknown_entries)

    rendered = _assemble(
        full_entries=full_entries,
        summary_entries=summary_entries,
        omitted_count=0,
    )

    # Progressive shedding to respect max_chars.
    omitted = 0
    while len(rendered) > cfg.max_chars and (full_entries or summary_entries):
        if full_entries:
            # Step 1: collapse oldest full entry to summary form.
            pos, decl, card, _ = paired[len(paired) - len(full_entries)]
            full_entries.pop(0)
            summary_entries.append(_format_summary_only(decl, card, pos))
            use_summary_only = True
        elif summary_entries:
            # Step 2: drop oldest summary entry entirely.
            summary_entries.pop(0)
            omitted += 1
            logger.warning(
                "TOOL_CATALOG_BUDGET_SHED max_chars=%d overflow=%d",
                cfg.max_chars,
                max(len(rendered) - cfg.max_chars, 0),
            )
        rendered = _assemble(
            full_entries=full_entries,
            summary_entries=summary_entries,
            omitted_count=omitted,
        )

    if len(rendered) > cfg.max_chars:
        note = "\n\n> note: catalog truncated"
        rendered = (rendered[: max(cfg.max_chars - len(note), 0)] + note)[: cfg.max_chars]

    rendered_tool_count = len(full_entries) + len(summary_entries)
    return CatalogRenderResult(
        text=rendered,
        rendered_tool_count=rendered_tool_count,
        omitted_count=omitted,
        used_summary_only=use_summary_only,
    )


__all__ = [
    "REGISTRY_VERSION",
    "CatalogConfig",
    "CatalogRenderResult",
    "render_tool_catalog",
]
