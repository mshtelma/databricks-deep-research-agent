"""Whitelist-based carriage of designer metadata across the YAML import boundary.

The Designer stamps app-level metadata as top-level keys on the workflow
definition dict (sole writer: :func:`blueprint.build_blueprint`). The framework
loader deliberately ignores unknown top-level keys, so the import path
(:func:`yaml_import.parse_and_validate_yaml`) rebuilds the definition from the
framework projection and would silently lose that metadata. This module is the
single seam that carries it back — **whitelisted and schema-validated per key,
never blind passthrough** — with structured, never-silent warnings for anything
dropped, pruned, or recomputed.

Carriage rules (one :class:`MetadataKeySpec` per key in
:data:`DESIGNER_METADATA_KEYS`):

* A key **absent** from the document is never synthesized — raw framework YAML
  imports exactly as before, with zero designer metadata.
* **Provenance is carried verbatim** after validation (``designer_signature``
  survives byte-identically so legacy payloads keep loading through
  ``TaskSignature.load_from_storage``), cross-checked for consistency against
  the imported AST where a deterministic check exists.
* **Derived state is recomputed** (``structural_fingerprint``): the framework
  loader heals the document on import (synthesizes missing builtin web tool
  declarations, auto-populates sources), so a carried fingerprint can be
  legitimately stale — and a stale fingerprint would corrupt the architect-patch
  immutability check downstream.
* **Fail-open on metadata, fail-closed on structure**: nothing in this module
  ever rejects an import; invalid metadata degrades to today's absent-key
  behavior, loudly.

Adding a future key (e.g. the composable-topologies ``designer_plan``) is one
new ``MetadataKeySpec`` entry plus its carry function.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field

from deep_research.agent_designer.ast_introspection import topology_of_ast
from deep_research.agent_designer.blueprint import (
    compute_lane_key,
    compute_structural_fingerprint,
)
from deep_research.agent_designer.designer_types import EvidencePolicy
from deep_research.agent_designer.mutations import _collect_id_index
from deep_research.agent_designer.task_signature import (
    TaskSignature,
    select_topology,
)
from deep_research.agent_designer.topology_registry import structural_family
from deep_research.surface.validation import validate_surface

logger = logging.getLogger(__name__)

MetadataWarningCode = Literal[
    "invalid_shape",  # value failed schema validation
    "consistency_mismatch",  # cross-check against the imported AST failed
    "recomputed_divergent",  # derived value recomputed; differs from carried
    "stale_entries_pruned",  # a subset of entries was removed
]
MetadataAction = Literal["dropped", "recomputed", "pruned"]


class ImportMetadataWarning(BaseModel):
    """One structured warning about a designer-metadata key during YAML import."""

    model_config = ConfigDict(frozen=True)

    key: str
    code: MetadataWarningCode
    action: MetadataAction
    message: str
    detail: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class MetadataCarryOutcome:
    """Result of carrying ONE key. ``value`` is meaningful only when ``carried``."""

    carried: bool
    value: Any = None
    warnings: list[ImportMetadataWarning] = field(default_factory=list)


# (raw_value_from_yaml_doc, healed_definition_dump) -> outcome
CarryFn = Callable[[Any, dict[str, Any]], MetadataCarryOutcome]


@dataclass(frozen=True)
class MetadataKeySpec:
    key: str
    carry: CarryFn
    doc: str


class _ResourceSummaryItem(BaseModel):
    """One resource row of the sanitized contract summary."""

    model_config = ConfigDict(extra="forbid")

    kind: str = ""
    identity: str = ""
    usage: str = "optional"
    access_status: str = "unverified"
    capabilities: list[str] = Field(default_factory=list, max_length=8)
    domain_terms: list[str] = Field(default_factory=list, max_length=8)


class ResolvedToolContractSummaryV1(BaseModel):
    """Validator for the exporter shape produced by
    :func:`tool_contract.sanitized_resolved_tool_contract_summary`.

    The sanitizer flattens ``prompt_obligations`` into top-level keys, so the
    summary can NOT be re-validated via ``ResolvedToolContract`` — this model
    mirrors the sanitizer's output instead (including the unavailable stub
    ``{"schema": ..., "available": False}``). ``extra="forbid"`` plus the
    exporter↔validator parity unit test pin the two against drift. The
    ``max_length`` caps equal the sanitizer's truncation limits, so the sole
    legitimate writer always passes while a hand-crafted oversized summary is
    rejected (drop + warn) instead of persisted.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_: Literal["resolved_tool_contract.v1"] = Field(alias="schema")
    available: bool
    evidence_policy: EvidencePolicy | None = None
    resources_count: int = 0
    resources: list[_ResourceSummaryItem] = Field(default_factory=list, max_length=8)
    required_capabilities: list[str] = Field(default_factory=list, max_length=12)
    ready_tool_kinds: list[str] = Field(default_factory=list, max_length=12)
    required_terms: list[str] = Field(default_factory=list, max_length=12)
    synthesis_obligations: list[str] = Field(default_factory=list, max_length=8)
    planner_obligations: list[str] = Field(default_factory=list, max_length=8)
    forbidden_tool_kinds: list[str] = Field(default_factory=list, max_length=8)
    diagnostics: list[dict[str, Any]] = Field(default_factory=list, max_length=8)


def _warning(
    key: str,
    code: MetadataWarningCode,
    action: MetadataAction,
    message: str,
    detail: list[str] | None = None,
) -> ImportMetadataWarning:
    return ImportMetadataWarning(
        key=key, code=code, action=action, message=message, detail=detail or []
    )


def _carry_designer_signature(raw: Any, definition: dict[str, Any]) -> MetadataCarryOutcome:
    key = "designer_signature"
    try:
        sig = TaskSignature.load_from_storage(raw)
    except Exception:
        return MetadataCarryOutcome(
            carried=False,
            warnings=[
                _warning(
                    key,
                    "invalid_shape",
                    "dropped",
                    "designer_signature is not a valid TaskSignature; topology "
                    "edits will fall back to an explicit rebuild.",
                )
            ],
        )
    expected_family = structural_family(select_topology(sig))
    actual_topology = topology_of_ast(definition)
    if actual_topology != "unknown" and structural_family(actual_topology) != expected_family:
        return MetadataCarryOutcome(
            carried=False,
            warnings=[
                _warning(
                    key,
                    "consistency_mismatch",
                    "dropped",
                    f"designer_signature declares topology family '{expected_family}' "
                    f"but the document's structure is '{actual_topology}' (hand-edited "
                    "YAML?); dropped — topology edits will fall back to an explicit "
                    "rebuild.",
                )
            ],
        )
    # Carry the RAW dict verbatim (not re-dumped): legacy payloads survive
    # byte-identically and load_from_storage re-fills defaults at read time.
    return MetadataCarryOutcome(carried=True, value=raw)


def _carry_lane_keys(raw: Any, _definition: dict[str, Any]) -> MetadataCarryOutcome:
    key = "lane_keys"
    if not isinstance(raw, dict) or not all(
        isinstance(k, str) and k and isinstance(v, str) and v.strip()
        for k, v in raw.items()
    ):
        return MetadataCarryOutcome(
            carried=False,
            warnings=[
                _warning(
                    key,
                    "invalid_shape",
                    "dropped",
                    "lane_keys must map lane keys to non-empty lane descriptions.",
                )
            ],
        )
    kept: dict[str, str] = {}
    pruned: list[str] = []
    for lane_key, description in raw.items():
        try:
            expected = compute_lane_key(description)
        except Exception:
            pruned.append(lane_key)
            continue
        if expected == lane_key:
            kept[lane_key] = description
        else:
            pruned.append(lane_key)
    if not kept:
        return MetadataCarryOutcome(
            carried=False,
            warnings=[
                _warning(
                    key,
                    "consistency_mismatch",
                    "dropped",
                    "No lane_keys entry matches its description (keys are "
                    "content-derived); architect patch targeting degrades to "
                    "node-id addressing.",
                    detail=pruned,
                )
            ],
        )
    warnings: list[ImportMetadataWarning] = []
    if pruned:
        warnings.append(
            _warning(
                key,
                "consistency_mismatch",
                "pruned",
                f"{len(pruned)} lane_keys entr{'y' if len(pruned) == 1 else 'ies'} "
                "did not match their descriptions and were removed; architect "
                "patch targeting for those lanes degrades to node-id addressing.",
                detail=pruned,
            )
        )
    return MetadataCarryOutcome(carried=True, value=kept, warnings=warnings)


def _carry_evidence_policy(raw: Any, _definition: dict[str, Any]) -> MetadataCarryOutcome:
    key = "evidence_policy"
    if raw is None or (isinstance(raw, str) and raw in get_args(EvidencePolicy)):
        return MetadataCarryOutcome(carried=True, value=raw)
    return MetadataCarryOutcome(
        carried=False,
        warnings=[
            _warning(
                key,
                "invalid_shape",
                "dropped",
                f"evidence_policy {raw!r} is not one of "
                f"{'/'.join(get_args(EvidencePolicy))}.",
            )
        ],
    )


def _carry_required_prompt_terms(raw: Any, _definition: dict[str, Any]) -> MetadataCarryOutcome:
    key = "required_prompt_terms"
    if not isinstance(raw, list):
        return MetadataCarryOutcome(
            carried=False,
            warnings=[
                _warning(
                    key,
                    "invalid_shape",
                    "dropped",
                    "required_prompt_terms must be a list of strings.",
                )
            ],
        )
    kept = [term for term in raw if isinstance(term, str) and term.strip()]
    warnings: list[ImportMetadataWarning] = []
    if len(kept) != len(raw):
        dropped_count = len(raw) - len(kept)
        warnings.append(
            _warning(
                key,
                "stale_entries_pruned",
                "pruned",
                f"{dropped_count} non-string/empty required_prompt_terms "
                "entries were removed; the prompt-coverage gate will evaluate "
                "the remaining terms.",
                detail=[repr(term) for term in raw if not (isinstance(term, str) and term.strip())][:8],
            )
        )
    return MetadataCarryOutcome(carried=True, value=kept, warnings=warnings)


def _carry_resolved_tool_contract_summary(
    raw: Any, _definition: dict[str, Any]
) -> MetadataCarryOutcome:
    key = "resolved_tool_contract_summary"
    try:
        ResolvedToolContractSummaryV1.model_validate(raw)
    except Exception:
        return MetadataCarryOutcome(
            carried=False,
            warnings=[
                _warning(
                    key,
                    "invalid_shape",
                    "dropped",
                    "resolved_tool_contract_summary does not match schema "
                    "resolved_tool_contract.v1; contract-aware validation falls "
                    "back to description-derived terms.",
                )
            ],
        )
    # Validate-only: carry the ORIGINAL dict so the round trip is byte-lossless.
    return MetadataCarryOutcome(carried=True, value=raw)


def _carry_placeholder_pending_nodes(
    raw: Any, definition: dict[str, Any]
) -> MetadataCarryOutcome:
    key = "placeholder_pending_nodes"
    if not isinstance(raw, list) or not all(
        isinstance(item, str) and item for item in raw
    ):
        return MetadataCarryOutcome(
            carried=False,
            warnings=[
                _warning(
                    key,
                    "invalid_shape",
                    "dropped",
                    "placeholder_pending_nodes must be a list of node ids.",
                )
            ],
        )
    known_ids = set(_collect_id_index(definition))
    kept = [node_id for node_id in raw if node_id in known_ids]
    stale = [node_id for node_id in raw if node_id not in known_ids]
    warnings: list[ImportMetadataWarning] = []
    if stale:
        warnings.append(
            _warning(
                key,
                "stale_entries_pruned",
                "pruned" if kept else "dropped",
                "placeholder_pending_nodes referenced nodes that do not exist "
                "in this document; these lane-prompt lifecycle markers were "
                "removed.",
                detail=stale,
            )
        )
    if not kept:
        # blueprint only ever stamps a non-empty list; an empty one is noise.
        return MetadataCarryOutcome(carried=False, warnings=warnings)
    return MetadataCarryOutcome(carried=True, value=kept, warnings=warnings)


def _carry_surface(raw: Any, definition: dict[str, Any]) -> MetadataCarryOutcome:
    key = "surface"
    # Validate against the HEALED definition so binding coverage is checked
    # against the required_inputs that will actually run (the loader may have
    # healed the document). Carried verbatim on success (byte-lossless round
    # trip, like resolved_tool_contract_summary).
    probe = dict(definition)
    probe[key] = raw
    findings = validate_surface(probe)
    blocking = [f for f in findings if f.severity == "blocking"]
    if blocking:
        return MetadataCarryOutcome(
            carried=False,
            warnings=[
                _warning(
                    key,
                    "invalid_shape",
                    "dropped",
                    "surface failed validation against the imported workflow "
                    "and was dropped; the agent imports without a UI. Re-author "
                    "it in the Designer or fix the YAML and re-import.",
                    detail=[
                        f"{f.path or '<surface>'}: {f.message}" for f in blocking
                    ][:8],
                )
            ],
        )
    return MetadataCarryOutcome(carried=True, value=raw)


def _carry_structural_fingerprint(raw: Any, definition: dict[str, Any]) -> MetadataCarryOutcome:
    key = "structural_fingerprint"
    recomputed = compute_structural_fingerprint(definition)
    if not isinstance(raw, str):
        return MetadataCarryOutcome(
            carried=True,
            value=recomputed,
            warnings=[
                _warning(
                    key,
                    "invalid_shape",
                    "recomputed",
                    "structural_fingerprint was not a string; recomputed from "
                    "the imported structure.",
                )
            ],
        )
    if raw == recomputed:
        return MetadataCarryOutcome(carried=True, value=recomputed)
    return MetadataCarryOutcome(
        carried=True,
        value=recomputed,
        warnings=[
            _warning(
                key,
                "recomputed_divergent",
                "recomputed",
                "structural_fingerprint recomputed on import (structure changed "
                "since export: loader healing or manual YAML edits).",
            )
        ],
    )


# Table order mirrors build_blueprint's stamp order; structural_fingerprint runs
# LAST so the derived value is recomputed after every other key settles.
DESIGNER_METADATA_KEYS: tuple[MetadataKeySpec, ...] = (
    MetadataKeySpec(
        "lane_keys",
        _carry_lane_keys,
        "lane_key -> verbatim lane description; architect patch targeting",
    ),
    MetadataKeySpec(
        "evidence_policy",
        _carry_evidence_policy,
        "grounded evidence policy from the resolved tool contract",
    ),
    MetadataKeySpec(
        "required_prompt_terms",
        _carry_required_prompt_terms,
        "terms the prompt-coverage gate enforces on the report producer",
    ),
    MetadataKeySpec(
        "resolved_tool_contract_summary",
        _carry_resolved_tool_contract_summary,
        "prompt-safe summary of the resolved tool contract",
    ),
    MetadataKeySpec(
        "placeholder_pending_nodes",
        _carry_placeholder_pending_nodes,
        "lifecycle list of lanes still on the blueprint placeholder prompt",
    ),
    MetadataKeySpec(
        "surface",
        _carry_surface,
        "declarative agent UI (form + actions + result regions); validated "
        "against the imported AST",
    ),
    MetadataKeySpec(
        "designer_signature",
        _carry_designer_signature,
        "TaskSignature that produced this AST; topology-edit soundness",
    ),
    MetadataKeySpec(
        "structural_fingerprint",
        _carry_structural_fingerprint,
        "derived structural hash; ALWAYS recomputed on import",
    ),
    # FUTURE (composable topologies): add exactly one entry here —
    # MetadataKeySpec("designer_plan", _carry_designer_plan, "Structure Plan v1").
)


def carry_designer_metadata(
    source: dict[str, Any],
    definition: dict[str, Any],
) -> list[ImportMetadataWarning]:
    """Attach whitelisted designer metadata from *source* onto *definition*.

    *source* is the parsed YAML document; *definition* is the framework
    loader's healed ``model_dump()`` — mutated in place. Per key: absent in
    *source* → untouched (never synthesized); present → validated/normalized
    per its :class:`MetadataKeySpec`; valid → attached; invalid/inconsistent →
    NOT attached, warning appended. Never raises for metadata problems
    (fail-open on metadata; the caller's structural validation stays
    fail-closed). Every warning is also logged so non-endpoint callers are
    never silent.
    """
    warnings: list[ImportMetadataWarning] = []
    for spec in DESIGNER_METADATA_KEYS:
        if spec.key not in source:
            continue
        try:
            outcome = spec.carry(source.get(spec.key), definition)
        except Exception:  # noqa: BLE001 — carry fns are defensive; this is the net
            logger.exception("YAML_IMPORT_METADATA_CARRY_CRASH key=%s", spec.key)
            outcome = MetadataCarryOutcome(
                carried=False,
                warnings=[
                    _warning(
                        spec.key,
                        "invalid_shape",
                        "dropped",
                        f"{spec.key} could not be processed and was dropped.",
                    )
                ],
            )
        if outcome.carried:
            definition[spec.key] = outcome.value
        warnings.extend(outcome.warnings)
    for warning in warnings:
        logger.warning(
            "YAML_IMPORT_METADATA key=%s code=%s action=%s detail=%s",
            warning.key,
            warning.code,
            warning.action,
            warning.detail,
        )
    return warnings


__all__ = [
    "DESIGNER_METADATA_KEYS",
    "ImportMetadataWarning",
    "MetadataAction",
    "MetadataCarryOutcome",
    "MetadataKeySpec",
    "MetadataWarningCode",
    "ResolvedToolContractSummaryV1",
    "carry_designer_metadata",
]
