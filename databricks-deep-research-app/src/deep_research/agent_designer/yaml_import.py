"""YAML import and re-validation for agent designer workflows.

The single public function :func:`parse_and_validate_yaml` is the SOLE entry
point for any externally-supplied YAML.  It enforces:

1. Size limit (default 256 KiB, overridable via ``AGENT_DESIGNER_YAML_MAX_BYTES``).
2. Safe parsing — only ``yaml.safe_load`` is ever used; ``yaml.load`` is
   never called so that !!python/object gadgets cannot reach the evaluator.
3. Top-level mapping check — the parsed value must be a dict.
4. Registry-version pinning — when present, the ``registry_version`` field must
   match :data:`~deep_research.agent_designer.registry.REGISTRY_VERSION`; an
   absent (or null) field is accepted and treated as the current version so raw
   framework YAML and legacy pre-envelope exports import cleanly.
5. Canonical AST re-validation via
   :func:`databricks_deep_research.load_workflow_from_dict` — every imported
   AST is passed through the same validator that the /validate endpoint uses.
6. Designer-metadata carriage — the framework loader deliberately ignores
   unknown top-level keys, so the validated projection is re-hydrated with the
   whitelisted, schema-validated designer metadata from the document
   (:func:`yaml_metadata.carry_designer_metadata`). Invalid or inconsistent
   metadata never rejects the import; it is dropped/pruned/recomputed with a
   structured warning on :class:`YamlImportResult` (never-silent).

Any violation raises :class:`YamlImportError` with a machine-readable
``error_kind`` and a human-readable ``message``.
"""

from __future__ import annotations

import os
from typing import Any

import yaml
from databricks_deep_research import load_workflow_from_dict
from pydantic import BaseModel, Field

from deep_research.agent_designer.registry import REGISTRY_VERSION
from deep_research.agent_designer.yaml_metadata import (
    ImportMetadataWarning,
    carry_designer_metadata,
)

__all__ = [
    "ImportMetadataWarning",
    "YamlImportError",
    "YamlImportResult",
    "parse_and_validate_yaml",
]


class YamlImportResult(BaseModel):
    """Validated definition plus structured metadata warnings (never-silent).

    ``definition`` is the framework loader's healed projection with the
    whitelisted designer metadata re-attached; ``warnings`` lists every
    metadata key that was dropped, pruned, or recomputed on the way in.
    """

    definition: dict[str, Any]
    warnings: list[ImportMetadataWarning] = Field(default_factory=list)


class YamlImportError(Exception):
    """Raised when an imported YAML payload fails any validation step.

    Attributes:
        error_kind: Machine-readable error category.  One of:
            ``"too_large"``, ``"schema_error"``, ``"unsafe"``,
            ``"registry_version_mismatch"``.
        message: Human-readable description of the failure.
        path: Optional dot-path pointing to the offending field in the AST,
            or ``None`` when the error is at the document level.
    """

    def __init__(
        self,
        error_kind: str,
        message: str,
        path: str | None = None,
    ) -> None:
        self.error_kind = error_kind
        self.message = message
        self.path = path
        super().__init__(f"{error_kind}: {message}")


def parse_and_validate_yaml(body: bytes) -> YamlImportResult:
    """Parse a raw YAML payload and validate the resulting workflow AST.

    Enforces the full import acceptance criteria in order:

    1. Size check against ``AGENT_DESIGNER_YAML_MAX_BYTES`` (default 256 KiB).
    2. Safe YAML parsing via ``yaml.safe_load`` — never ``yaml.load``.
    3. Top-level mapping assertion.
    4. ``registry_version`` extraction — absent/null is accepted as the current
       version; a present-but-different value is rejected.
    5. AST re-validation via :func:`load_workflow_from_dict`.

    Args:
        body: Raw request bytes — the YAML document as UTF-8.

    Returns:
        A :class:`YamlImportResult` whose ``definition`` is the validated
        :class:`~databricks_deep_research.workflow.definition.WorkflowDefinition`
        projection (``model_dump()``, without the ``registry_version`` field)
        re-hydrated with the document's whitelisted designer metadata, and
        whose ``warnings`` lists every metadata key dropped, pruned, or
        recomputed during carriage.

    Raises:
        YamlImportError: On any validation failure.  Inspect
            :attr:`YamlImportError.error_kind` to distinguish categories.
    """
    max_bytes: int = int(
        os.environ.get("AGENT_DESIGNER_YAML_MAX_BYTES", str(256 * 1024))
    )
    if len(body) > max_bytes:
        raise YamlImportError(
            "too_large",
            f"YAML body exceeds {max_bytes} bytes",
        )

    try:
        # MUST be safe_load — yaml.load is never called here (defense in depth
        # against !!python/object/apply and similar gadgets).
        parsed: object = yaml.safe_load(body)
    except yaml.YAMLError as exc:
        raise YamlImportError("schema_error", f"YAML parse error: {exc}") from exc

    if not isinstance(parsed, dict):
        raise YamlImportError(
            "schema_error",
            "YAML body must be a mapping at top level",
        )

    # Registry-version handling (extract before passing to the AST loader):
    #   • absent / null      → accept, treat as the current registry version, so
    #     raw framework YAML and legacy pre-envelope exports import without edits.
    #   • present & equal    → accept.
    #   • present & different → reject with an actionable message.
    received_version: object = parsed.pop("registry_version", None)
    if received_version is not None and received_version != REGISTRY_VERSION:
        raise YamlImportError(
            "registry_version_mismatch",
            f"document was built for registry_version {received_version!r}, but "
            f"this workspace requires {REGISTRY_VERSION!r}. Re-export the agent "
            f"from this workspace, or remove the registry_version line to import "
            f"it as a raw framework workflow.",
        )

    # Canonical AST re-validation — MUST happen before any persistence.
    try:
        workflow = load_workflow_from_dict(parsed)
    except Exception as exc:
        raise YamlImportError(
            "schema_error",
            f"AST validation failed: {exc}",
        ) from exc

    # The loader's projection drops app-level metadata (unknown top-level keys
    # are ignored by design). Re-attach the whitelisted designer metadata from
    # the parsed document — validated per key, never blind passthrough — and
    # surface anything dropped/pruned/recomputed as structured warnings.
    # mode="json": a plain model_dump() leaves StrEnum members (NodeType) in the
    # dict, which yaml.safe_dump cannot represent — re-EXPORTING an imported
    # definition would crash. JSON mode matches the wire shape the response
    # serializer produces anyway, making import a true fixed point.
    definition: dict[str, Any] = workflow.model_dump(mode="json")
    warnings = carry_designer_metadata(parsed, definition)
    return YamlImportResult(definition=definition, warnings=warnings)
