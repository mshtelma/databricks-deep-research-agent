"""Re-export of surface slot compilation (moved to the framework).

The canonical implementation now lives in
``databricks_deep_research.surface.output_schema`` so the framework generation
core and the standalone shell-app share ONE copy. This module preserves the app
import path used by validation, the orchestrator, and the restructure endpoint.
"""

from databricks_deep_research.surface.output_schema import (
    CELL_MAX,
    ITEM_MAX,
    LIST_CAP,
    METRICS_CAP,
    ROWS_CAP,
    SOURCE_REFS_CAP,
    CollectedSlots,
    ColumnSpec,
    ResolvedBinding,
    SlotIssue,
    SlotSpec,
    WireModelBase,
    build_output_model,
    build_slot_wire_model,
    collect_output_slots,
    pointer_under_target,
    resolve_binding_for_run,
    slot_docs,
    split_slot_pointer,
    wire_slot_docs,
)

__all__ = [
    "CELL_MAX",
    "ITEM_MAX",
    "LIST_CAP",
    "METRICS_CAP",
    "ROWS_CAP",
    "SOURCE_REFS_CAP",
    "CollectedSlots",
    "ColumnSpec",
    "ResolvedBinding",
    "SlotIssue",
    "SlotSpec",
    "WireModelBase",
    "build_output_model",
    "build_slot_wire_model",
    "collect_output_slots",
    "pointer_under_target",
    "resolve_binding_for_run",
    "slot_docs",
    "split_slot_pointer",
    "wire_slot_docs",
]
