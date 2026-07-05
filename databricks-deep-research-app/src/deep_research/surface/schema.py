"""Re-export of the surface schema (moved to the framework).

The canonical schema now lives in ``databricks_deep_research.surface.schema``
so the framework generation core and the standalone shell-app share ONE
definition. This module preserves the app import path; the app's authoring,
validation, and persistence layers build on top of it.
"""

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
]
