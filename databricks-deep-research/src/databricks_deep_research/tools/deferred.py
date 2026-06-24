"""RAG-over-tools — deferred tool catalog (spec §5.5, Tier-3).

When an agent is wired to many tools (e.g. a large MCP server surfaced by
§4.3), listing every tool's full JSON Schema in the base prompt is wasteful and
crowds out reasoning tokens. This module mirrors *this harness's own*
``ToolSearch`` pattern: the base catalog lists each deferred tool's NAME and a
one-line description only; the full schema is fetched on demand via the
``tool_search`` builtin (see :mod:`tools.builtins.tool_search`). Once fetched, a
tool is **promoted** — its full schema is exposed to the LLM on subsequent
turns.

Two hard invariants:

* **Fail-closed** (mirrors DeerFlow ``tool_search.py:184``): a tool that
  survives policy/step filtering but is NOT registered as deferred AND has not
  been promoted must never reach the LLM with a stubbed-out schema. The
  :class:`DeferredToolRegistry` exposes :meth:`schema_status_for` so the caller
  can reject any tool whose schema would be silently missing.
* **Optional-with-default**: deferral only engages above a catalog-size
  threshold or when explicitly enabled. Below the threshold the registry is
  never constructed and the catalog is byte-identical to today.

The registry is intentionally transport-agnostic: it operates purely over
:class:`~databricks_deep_research.tools.protocol.ToolDefinition` objects, so it
works for MCP tools, dynamic skills, or any other ``ResearchTool``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from databricks_deep_research.tools.protocol import ToolDefinition

logger = logging.getLogger(__name__)

__all__ = [
    "DeferredToolRegistry",
    "SchemaStatus",
    "ToolSchemaMatch",
    "first_line",
]

# Metadata key a tool can set to opt INTO deferral regardless of catalog size.
# MCP/dynamic toolsets stamp this so they are deferred as soon as the registry
# engages. Absence of the key means "defer me only by the size threshold".
DEFERRABLE_METADATA_KEY = "deferrable"


class SchemaStatus:
    """Whether a tool's full schema is currently visible to the LLM.

    Plain string constants (not an enum) to keep the call sites terse and the
    values trivially loggable.
    """

    #: Tool is not deferred — its full schema is always in the catalog.
    EAGER = "eager"
    #: Tool is deferred and not yet fetched — only name + one-liner are listed.
    DEFERRED = "deferred"
    #: Tool was deferred and has been fetched (promoted) — full schema visible.
    PROMOTED = "promoted"


def first_line(text: str, *, max_chars: int = 200) -> str:
    """Return a compact one-line summary of *text* for the deferred catalog.

    Collapses to the first non-empty line and hard-caps the length so a verbose
    tool description cannot defeat the point of deferral.
    """
    for raw in text.splitlines():
        line = raw.strip()
        if line:
            return line[:max_chars]
    return text.strip()[:max_chars]


@dataclass(frozen=True)
class ToolSchemaMatch:
    """One matched deferred tool returned by :meth:`DeferredToolRegistry.match`."""

    name: str
    description: str
    parameters: dict[str, object]


class DeferredToolRegistry:
    """Tracks which tools are deferred and which have been promoted.

    Construction takes the FULL set of tools the loop intends to expose. Tools
    are classified once: a tool is *deferred* when it either stamps the
    ``deferrable`` metadata flag OR the registry is told to defer everything
    eligible by size. Everything else is *eager* (full schema always listed).

    The registry holds the deferred tools' definitions so :meth:`match` can
    return their full schemas on demand; :meth:`promote` flips a deferred tool to
    promoted (schema now visible). The registry is the single source of truth for
    :meth:`schema_status_for`, which the catalog builder uses to (a) emit a stub
    for an un-promoted deferred tool and (b) FAIL CLOSED on any tool that is
    neither eager nor promoted-or-deferred-registered.
    """

    def __init__(
        self,
        definitions: list[ToolDefinition],
        *,
        deferred_names: set[str],
    ) -> None:
        """Build the registry.

        Args:
            definitions: Every tool definition the loop intends to expose
                (including the always-eager ``tool_search`` tool itself).
            deferred_names: The subset of names to defer. The caller computes
                this from the size threshold / explicit flag; a name not in this
                set is treated as eager.
        """
        self._defs: dict[str, ToolDefinition] = {d.name: d for d in definitions}
        # Only names that are both present and asked-to-be-deferred are deferred.
        self._deferred: set[str] = {
            n for n in deferred_names if n in self._defs
        }
        self._promoted: set[str] = set()

    # -- classification ------------------------------------------------------

    def register_eager(self, definition: ToolDefinition) -> None:
        """Register *definition* as an always-eager tool (never deferred).

        Used to register the ``tool_search`` tool itself so it passes the
        fail-closed catalog builder. Idempotent; does NOT add the name to the
        deferred set.
        """
        self._defs[definition.name] = definition

    def is_deferred(self, name: str) -> bool:
        """True if *name* is a deferred tool that has NOT yet been promoted."""
        return name in self._deferred and name not in self._promoted

    def is_known(self, name: str) -> bool:
        """True if *name* has a registered definition (deferred or eager)."""
        return name in self._defs

    def schema_status_for(self, name: str) -> str:
        """Return the :class:`SchemaStatus` for *name*.

        Raises:
            KeyError: If *name* is unknown to the registry. The catalog builder
                treats this as the fail-closed signal (an un-schema'd tool must
                never reach the LLM).
        """
        if name not in self._defs:
            raise KeyError(name)
        if name not in self._deferred:
            return SchemaStatus.EAGER
        if name in self._promoted:
            return SchemaStatus.PROMOTED
        return SchemaStatus.DEFERRED

    def deferred_names(self) -> list[str]:
        """Sorted names of tools that are deferred and not yet promoted."""
        return sorted(n for n in self._deferred if n not in self._promoted)

    def promoted_count(self) -> int:
        """Number of deferred tools fetched/promoted so far (monotonic)."""
        return len(self._promoted)

    # -- catalog rendering ---------------------------------------------------

    def stub_definition(self, name: str) -> ToolDefinition:
        """Return a NAME + one-line-description-only definition for the catalog.

        The parameters schema is emptied to an open object so the full schema
        stays out of the prompt until promotion. The description carries a
        compact one-liner plus a hint to fetch via ``tool_search`` — this is the
        only thing the LLM sees for a deferred tool.
        """
        defn = self._defs[name]
        one_liner = first_line(defn.description)
        return ToolDefinition(
            name=defn.name,
            description=(
                f"{one_liner} [deferred — call tool_search with this name to "
                "load its full schema before use]"
            ),
            parameters={"type": "object", "properties": {}},
            source_type=defn.source_type,
            source_kind=defn.source_kind,
            metadata=defn.metadata,
        )

    # -- on-demand fetch + promotion ----------------------------------------

    def match(
        self,
        *,
        names: list[str] | None = None,
        query: str = "",
        limit: int = 25,
    ) -> list[ToolSchemaMatch]:
        """Return full schemas for deferred tools matching *names* or *query*.

        Mirrors this harness's ``ToolSearch`` query forms:

        * ``names`` — exact selection (the ``select:a,b,c`` form). Unknown or
          non-deferred names are skipped silently (the caller already has eager
          schemas for non-deferred names).
        * ``query`` — keyword ranking over the deferred tools' name +
          description when no explicit names are given.

        This is a pure lookup: it does NOT promote. The caller promotes the
        returned names so a failed/empty match never widens the catalog.
        """
        candidates = self.deferred_names()
        selected: list[str]
        if names:
            wanted = [n.strip() for n in names if n.strip()]
            selected = [n for n in wanted if n in set(candidates)]
        elif query.strip():
            selected = self._rank_by_query(candidates, query)[:limit]
        else:
            selected = candidates[:limit]

        matches: list[ToolSchemaMatch] = []
        for name in selected:
            defn = self._defs[name]
            matches.append(
                ToolSchemaMatch(
                    name=defn.name,
                    description=defn.description,
                    parameters=dict(defn.parameters),
                )
            )
        return matches

    def promote(self, names: list[str]) -> list[str]:
        """Mark *names* as promoted (full schema visible from now on).

        Only names that are currently deferred are promoted; returns the names
        actually newly promoted (so the caller can rebuild the catalog and
        record the promotion exactly once). Idempotent.
        """
        newly: list[str] = []
        for name in names:
            if name in self._deferred and name not in self._promoted:
                self._promoted.add(name)
                newly.append(name)
        if newly:
            logger.info("DEFERRED_TOOL_PROMOTED names=%s", newly)
        return newly

    def _rank_by_query(self, candidates: list[str], query: str) -> list[str]:
        """Rank *candidates* by simple token-overlap against *query*.

        Deliberately dependency-free (no embeddings): the deferred catalog is
        small relative to a corpus, and an exact ``select:`` is the primary path.
        Ties keep the deterministic sorted order from :meth:`deferred_names`.
        """
        terms = {t for t in _tokenize(query) if t}
        if not terms:
            return candidates
        scored: list[tuple[int, str]] = []
        for name in candidates:
            defn = self._defs[name]
            haystack = _tokenize(f"{defn.name} {defn.description}")
            score = sum(1 for t in terms if t in haystack)
            if score > 0:
                scored.append((score, name))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [name for _, name in scored]


def _tokenize(text: str) -> set[str]:
    """Lowercase alphanumeric token set used for keyword matching."""
    return {
        token
        for token in "".join(
            ch.lower() if ch.isalnum() else " " for ch in text
        ).split()
        if token
    }
