"""Declarative agent UI surfaces (A2UI-shaped, app-owned).

Public API for the surface feature: schema models, the fixed component
catalog, deterministic validation, and the default scaffold. The surface
lives at ``definition["surface"]`` on agents_v2 workflow definitions and is
opaque to the framework engine.
"""

from deep_research.surface.catalog import (
    CATALOG,
    CONTAINER_COMPONENTS,
    INPUT_COMPONENTS,
    ComponentSpec,
    PropSpec,
    catalog_reference,
    component_names,
    component_props_json_schema,
    component_spec,
)
from deep_research.surface.scaffold import scaffold_surface_from_workflow
from deep_research.surface.schema import (
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
from deep_research.surface.validation import (
    MAX_COMPONENTS,
    MAX_SURFACE_BYTES,
    RESERVED_INPUT_KEYS,
    SurfaceValidationError,
    has_blocking,
    validate_surface,
)

__all__ = [
    "CATALOG",
    "CONTAINER_COMPONENTS",
    "IDENTIFIER_PATTERN",
    "INPUT_COMPONENTS",
    "MAX_COMPONENTS",
    "MAX_SURFACE_BYTES",
    "POINTER_PATTERN",
    "RESERVED_INPUT_KEYS",
    "SURFACE_VERSION",
    "ActionBinding",
    "ComponentSpec",
    "DynamicValue",
    "OutputTarget",
    "PathRef",
    "PropSpec",
    "RunOptions",
    "Surface",
    "SurfaceComponent",
    "SurfaceLayout",
    "SurfaceRuntimeControls",
    "SurfaceSectionLayout",
    "SurfaceValidationError",
    "catalog_reference",
    "component_names",
    "component_props_json_schema",
    "component_spec",
    "has_blocking",
    "is_valid_identifier",
    "is_valid_pointer",
    "resolve_pointer",
    "scaffold_surface_from_workflow",
    "validate_surface",
]
