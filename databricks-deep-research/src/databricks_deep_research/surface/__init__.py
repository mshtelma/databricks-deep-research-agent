"""Declarative agent-surface schema + structured-output generation core.

The framework-side, app-independent half of the surface feature: the schema
model, slot compilation, evidence assembly, the per-slot wire generator, and
the schema-aware prompt-injection helper. The app layers authoring
(catalog/validation/scaffold) and persistence on top of these; the standalone
shell-app uses them directly to render the same structured dashboards.
"""

from databricks_deep_research.surface.contract import (
    RESEARCH_CONTRACT_MARKER,
    STRUCTURED_CONTRACT_MARKER,
    build_contracts,
    inject_structured_output_contract,
    visit_agent_configs,
)
from databricks_deep_research.surface.evidence import (
    EvidenceItem,
    build_evidence,
    build_legend,
    render_evidence_block,
)
from databricks_deep_research.surface.generation import (
    SlotOutcome,
    SlotWireResults,
    StructuredCompletionClient,
    apply_slot_guards,
    build_envelope_v2,
    build_pending_envelope,
    build_structured_envelope,
    run_slot_wires,
)
from databricks_deep_research.surface.output_schema import (
    CollectedSlots,
    ColumnSpec,
    ResolvedBinding,
    SlotIssue,
    SlotSpec,
    build_output_model,
    build_slot_wire_model,
    collect_output_slots,
    resolve_binding_for_run,
    slot_docs,
    split_slot_pointer,
    wire_slot_docs,
)
from databricks_deep_research.surface.pointers import (
    INPUT_COMPONENTS,
    OUTPUT_COMPONENTS,
    OUTPUT_POINTER_PROPS,
)
from databricks_deep_research.surface.schema import (
    IDENTIFIER_PATTERN,
    POINTER_PATTERN,
    SURFACE_VERSION,
    ActionBinding,
    DynamicValue,
    OutputTarget,
    PathRef,
    RunOptions,
    Surface,
    SurfaceComponent,
    SurfaceLayout,
    SurfaceRuntimeControls,
    SurfaceSectionLayout,
    is_valid_identifier,
    is_valid_pointer,
    resolve_pointer,
)

__all__ = [
    # schema
    "IDENTIFIER_PATTERN",
    "POINTER_PATTERN",
    "SURFACE_VERSION",
    "ActionBinding",
    "DynamicValue",
    "OutputTarget",
    "PathRef",
    "RunOptions",
    "Surface",
    "SurfaceComponent",
    "SurfaceLayout",
    "SurfaceRuntimeControls",
    "SurfaceSectionLayout",
    "is_valid_identifier",
    "is_valid_pointer",
    "resolve_pointer",
    # pointers / component families
    "INPUT_COMPONENTS",
    "OUTPUT_COMPONENTS",
    "OUTPUT_POINTER_PROPS",
    # slot compilation
    "CollectedSlots",
    "ColumnSpec",
    "ResolvedBinding",
    "SlotIssue",
    "SlotSpec",
    "build_output_model",
    "build_slot_wire_model",
    "collect_output_slots",
    "resolve_binding_for_run",
    "slot_docs",
    "split_slot_pointer",
    "wire_slot_docs",
    # evidence
    "EvidenceItem",
    "build_evidence",
    "build_legend",
    "render_evidence_block",
    # generation
    "SlotOutcome",
    "SlotWireResults",
    "StructuredCompletionClient",
    "apply_slot_guards",
    "build_envelope_v2",
    "build_pending_envelope",
    "build_structured_envelope",
    "run_slot_wires",
    # contract (Part A)
    "RESEARCH_CONTRACT_MARKER",
    "STRUCTURED_CONTRACT_MARKER",
    "build_contracts",
    "inject_structured_output_contract",
    "visit_agent_configs",
]
