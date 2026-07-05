"""Re-export of the tolerant wire-model helpers (moved to the framework).

The canonical implementation now lives in
``databricks_deep_research.surface.tolerant`` so the framework generation core
and the standalone shell-app share ONE copy. This module preserves the app
import path.
"""

from databricks_deep_research.surface.tolerant import (
    TolerantWireBase,
    WireValidationError,
    coerce_citation_ref,
    json_repair_structured,
    unwrap_placeholder_envelope,
    validate_lenient,
)

__all__ = [
    "TolerantWireBase",
    "WireValidationError",
    "coerce_citation_ref",
    "json_repair_structured",
    "unwrap_placeholder_envelope",
    "validate_lenient",
]
