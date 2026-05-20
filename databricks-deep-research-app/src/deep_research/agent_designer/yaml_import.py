"""YAML import and re-validation for agent designer workflows.

The single public function :func:`parse_and_validate_yaml` is the SOLE entry
point for any externally-supplied YAML.  It enforces:

1. Size limit (default 256 KiB, overridable via ``AGENT_DESIGNER_YAML_MAX_BYTES``).
2. Safe parsing — only ``yaml.safe_load`` is ever used; ``yaml.load`` is
   never called so that !!python/object gadgets cannot reach the evaluator.
3. Top-level mapping check — the parsed value must be a dict.
4. Registry-version pinning — the ``registry_version`` field must match
   :data:`~deep_research.agent_designer.registry.REGISTRY_VERSION`.
5. Canonical AST re-validation via
   :func:`databricks_deep_research.load_workflow_from_dict` — every imported
   AST is passed through the same validator that the /validate endpoint uses.

Any violation raises :class:`YamlImportError` with a machine-readable
``error_kind`` and a human-readable ``message``.
"""

from __future__ import annotations

import os
from typing import Any

import yaml
from databricks_deep_research import load_workflow_from_dict

from deep_research.agent_designer.registry import REGISTRY_VERSION

__all__ = ["YamlImportError", "parse_and_validate_yaml"]


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


def parse_and_validate_yaml(body: bytes) -> dict[str, Any]:
    """Parse a raw YAML payload and validate the resulting workflow AST.

    Enforces the full import acceptance criteria in order:

    1. Size check against ``AGENT_DESIGNER_YAML_MAX_BYTES`` (default 256 KiB).
    2. Safe YAML parsing via ``yaml.safe_load`` — never ``yaml.load``.
    3. Top-level mapping assertion.
    4. ``registry_version`` field extraction and version-pinning check.
    5. AST re-validation via :func:`load_workflow_from_dict`.

    Args:
        body: Raw request bytes — the YAML document as UTF-8.

    Returns:
        A plain dict representation of the validated
        :class:`~databricks_deep_research.workflow.definition.WorkflowDefinition`
        (produced by ``WorkflowDefinition.model_dump()``), without the
        ``registry_version`` field.

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

    # Extract and validate registry version before passing to AST loader.
    received_version: object = parsed.pop("registry_version", None)
    if received_version != REGISTRY_VERSION:
        raise YamlImportError(
            "registry_version_mismatch",
            f"expected {REGISTRY_VERSION}, received {received_version}",
        )

    # Canonical AST re-validation — MUST happen before any persistence.
    try:
        workflow = load_workflow_from_dict(parsed)
    except Exception as exc:
        raise YamlImportError(
            "schema_error",
            f"AST validation failed: {exc}",
        ) from exc

    return workflow.model_dump()
