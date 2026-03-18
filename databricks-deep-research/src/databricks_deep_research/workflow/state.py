"""Workflow state module: append-only log with O(1) latest-value lookup.

WorkflowState is the single mutable state object threaded through every node
in a workflow execution.  It stores an append-only log of ``StateEntry``
records and maintains an internal index so that the *latest* value for any
key can be retrieved in O(1) time.

Design decisions
----------------
* ``StateEntry`` is **frozen** — entries are immutable once created.
* ``WorkflowState`` is **mutable** — nodes append entries during execution.
* An ``asyncio.Lock`` guards the log so concurrent node writes are safe.
* Pool / tool types use ``Any`` to avoid circular imports with modules that
  depend on this one.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from databricks_deep_research.workflow.runtime_core import TypedRuntimeStateStore
from databricks_deep_research.workflow.runtime_core.selectors import resolve_input_key

logger = logging.getLogger(__name__)

_RUNTIME_DERIVED_KEYS = frozenset(
    {
        "claims",
        "verification_summary",
        "analysis_summary",
        "verification_details",
    }
)


# ---------------------------------------------------------------------------
# Immutable log entry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StateEntry:
    """A single immutable record in the workflow state log."""

    node_id: str
    key: str
    value: Any
    timestamp: str  # ISO 8601


# ---------------------------------------------------------------------------
# Mutable workflow state
# ---------------------------------------------------------------------------


@dataclass
class WorkflowState:
    """Mutable state object shared across all nodes in a workflow run.

    The primary data structure is an **append-only log** of ``StateEntry``
    records.  A secondary ``_latest_index`` dict maps each key to the index
    of its most recent entry so that ``get()`` runs in O(1).

    Attributes
    ----------
    query:
        The original user query that initiated the workflow.
    log:
        Append-only list of ``StateEntry`` objects.
    pools:
        Named pool states (``str`` → ``PoolState``).  Typed as ``Any`` to
        avoid circular imports with the pool module.
    model_overrides:
        Per-tier model overrides supplied by the caller (tier → endpoint).
    enterprise_tools:
        Loaded enterprise tool instances (typed as ``Any`` to avoid a
        circular dependency on the tool module).
    user_token:
        Optional OBO token for enterprise tool calls.
    domain_filter:
        Optional domain restriction for web searches.
    is_cancelled:
        Flag that nodes should check to abort early.
    """

    query: str = ""
    log: list[StateEntry] = field(default_factory=list)
    pools: dict[str, Any] = field(default_factory=dict)
    model_overrides: dict[str, str] = field(default_factory=dict)
    enterprise_tools: list[Any] = field(default_factory=list)
    user_token: str | None = None
    domain_filter: str | None = None
    is_cancelled: bool = False
    runtime_store: TypedRuntimeStateStore | None = None

    # -- internal bookkeeping (not serialised) ------------------------------
    _latest_index: dict[str, int] = field(default_factory=dict, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    # -- mutators -----------------------------------------------------------

    def append(self, node_id: str, key: str, value: Any) -> None:
        """Append a new entry to the log and update the latest-value index.

        Parameters
        ----------
        node_id:
            Identifier of the node that produced this entry.
        key:
            Lookup key (e.g. ``"coordination"``, ``"plan"``).
        value:
            Arbitrary payload — typically a Pydantic model or primitive.
        """
        timestamp = datetime.now(tz=UTC).isoformat()
        entry = StateEntry(
            node_id=node_id,
            key=key,
            value=value,
            timestamp=timestamp,
        )
        idx = len(self.log)
        self.log.append(entry)
        self._latest_index[key] = idx
        if self.runtime_store is not None:
            self.runtime_store.set_artifact(key, value)
        logger.debug(
            "STATE_APPEND node_id=%s key=%s idx=%d",
            node_id,
            key,
            idx,
        )

    # -- accessors ----------------------------------------------------------

    def get(self, key: str) -> Any | None:
        """Return the latest value for *key*, or ``None`` if absent.

        Uses ``_latest_index`` for O(1) lookup.

        For runtime-backed derived keys that may not be present in the append-only
        log, fall back to the runtime selector registry.
        """
        idx = self._latest_index.get(key)
        if idx is not None:
            return self.log[idx].value

        if key in _RUNTIME_DERIVED_KEYS:
            resolved = resolve_input_key(self, key)
            if resolved is not None:
                return resolved

        return None


    def runtime_state(self):
        """Return a typed runtime snapshot when available."""
        if self.runtime_store is None:
            return None
        return self.runtime_store.snapshot()

    def get_all(self, key: str) -> list[Any]:
        """Return **all** values ever appended under *key* (oldest first)."""
        return [entry.value for entry in self.log if entry.key == key]

    def extract_output(self, key: str) -> str | None:
        """Extract readable text from the given output key.

        Handles Pydantic models, dicts, and plain strings.  Tries common
        text field names in priority order, falling back to ``str(value)``.
        """
        value = self.get(key)
        if value is None:
            return None
        if isinstance(value, str):
            return value
        _TEXT_FIELDS = ("report", "direct_response", "summary", "findings", "observation")
        for attr in _TEXT_FIELDS:
            text = getattr(value, attr, None)
            if isinstance(text, str) and text:
                return text
        if isinstance(value, dict):
            for k in _TEXT_FIELDS:
                text = value.get(k)
                if isinstance(text, str) and text:
                    return text
        return str(value)

    def get_nested(self, dot_path: str) -> Any | None:
        """Resolve a dot-separated path against the latest entry.

        Examples
        --------
        >>> state.append("coord", "coordination", obj_with_complexity)
        >>> state.get_nested("coordination.complexity")
        'deep'

        The first segment is used as the log key; subsequent segments are
        resolved via attribute access (``getattr``) with a fallback to item
        access (``[]``).
        """
        parts = dot_path.split(".")
        if not parts:
            return None

        root_key = parts[0]
        current: Any = self.get(root_key)
        if current is None:
            return None

        for part in parts[1:]:
            # Try attribute access first (Pydantic / dataclass), then mapping
            attr = getattr(current, part, _SENTINEL)
            if attr is not _SENTINEL:
                current = attr
                continue

            try:
                current = current[part]
            except (KeyError, TypeError, IndexError):
                return None

        return current

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise state for checkpointing / persistence.

        Only deterministic, JSON-safe fields are included.  Internal
        bookkeeping (``_latest_index``, ``_lock``) and runtime-only
        references (``enterprise_tools``, ``pools``) are excluded.
        """
        return {
            "query": self.query,
            "model_overrides": dict(self.model_overrides),
            "user_token": self.user_token,
            "domain_filter": self.domain_filter,
            "is_cancelled": self.is_cancelled,
            "log": [
                {
                    "node_id": entry.node_id,
                    "key": entry.key,
                    "value": entry.value,
                    "timestamp": entry.timestamp,
                }
                for entry in self.log
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkflowState:
        """Reconstruct a ``WorkflowState`` from a ``to_dict()`` payload.

        The ``_latest_index`` is rebuilt by replaying the log entries in
        order.
        """
        state = cls(
            query=data.get("query", ""),
            model_overrides=dict(data.get("model_overrides", {})),
            user_token=data.get("user_token"),
            domain_filter=data.get("domain_filter"),
            is_cancelled=data.get("is_cancelled", False),
        )

        for raw_entry in data.get("log", []):
            entry = StateEntry(
                node_id=raw_entry["node_id"],
                key=raw_entry["key"],
                value=raw_entry["value"],
                timestamp=raw_entry["timestamp"],
            )
            idx = len(state.log)
            state.log.append(entry)
            state._latest_index[entry.key] = idx

        return state


# ---------------------------------------------------------------------------
# Internal sentinel for getattr fallback
# ---------------------------------------------------------------------------

_SENTINEL: object = object()
