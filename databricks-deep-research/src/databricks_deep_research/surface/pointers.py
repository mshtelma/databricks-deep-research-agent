"""Surface component-family constants shared by the structuring pass.

The single source of truth for which components are inputs, which are
structured-output slots, and which prop carries each output component's
slot pointer. The app's authoring catalog
(``deep_research.surface.catalog``) re-exports these so the designer and the
framework generation core never drift.
"""

from __future__ import annotations

# Inputs whose ``value`` prop two-way binds to the data model.
INPUT_COMPONENTS: frozenset[str] = frozenset(
    {"TextField", "TextArea", "Select", "Checkbox"}
)

# Components whose pointer prop names a structured-output SLOT the model
# fills after a run (List may also read a static array — it participates in
# slot collection only when its pointer sits under a binding output target).
OUTPUT_COMPONENTS: frozenset[str] = frozenset(
    {"Table", "MetricGrid", "KeyFindings", "Chart", "List"}
)

# Pointer-bearing prop per output component (slot collection reads these).
OUTPUT_POINTER_PROPS: dict[str, str] = {
    "Table": "source",
    "MetricGrid": "source",
    "KeyFindings": "source",
    "Chart": "source",
    "List": "items",
}

__all__ = [
    "INPUT_COMPONENTS",
    "OUTPUT_COMPONENTS",
    "OUTPUT_POINTER_PROPS",
]
