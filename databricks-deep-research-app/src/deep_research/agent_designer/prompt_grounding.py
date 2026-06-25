"""Deterministic prompt grounding for Agent Designer resource mentions.

The initial user prompt is evidence, not executable configuration. This module
extracts resource identities and operation requirements from prompt text,
resolves what it can through user-scoped discovery, and emits normalized
``DesignerAsset`` records plus diagnostics for the existing blueprint path.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from deep_research.agent_designer.assets import (
    DesignerAsset,
    _infer_asset_kind_from_intent,
    normalize_assets,
    recommend_tools_for_assets,
    resolve_default_table_warehouse_id,
)
from deep_research.agent_designer.discovery import (
    DesignerDiscoveryAdapter,
    DiscoveredResource,
    SourceKind,
    extract_fqn_candidates,
    match_text_to_resources,
)

logger = logging.getLogger(__name__)

AccessStatus = Literal["verified", "unverified", "inaccessible"]
ResourceKindHint = Literal[
    "vector_index",
    "delta_table",
    "genie_space",
    "knowledge_assistant",
    "serving_endpoint",
    "sql_warehouse",
    "unknown",
]
IdentifierKind = Literal[
    "uc_fqn",
    "endpoint_name",
    "source_id",
    "short_name",
    "natural_language",
]
OperationKind = Literal[
    "semantic_lookup",
    "row_search",
    "row_read",
    "table_load",
    "numeric_compute",
    "metadata_discovery",
    "open_web_research",
    "synthesis",
]
Capability = Literal[
    "vector_search",
    "table_search",
    "table_read",
    "table_load",
    "compute",
    "web_research",
]

_PROMPT_CONFIG_RE = re.compile(
    r"\b("
    r"warehouse_id|sql_warehouse_id|num_results|endpoint_url|access_token|"
    r"secret|credential|credentials|api_key|sql_query|python_code"
    r")\b\s*[:=]",
    flags=re.IGNORECASE,
)


class GroundingDiagnostic(BaseModel):
    model_config = ConfigDict(extra="forbid")

    severity: Literal["info", "warning", "error"]
    code: Literal[
        "ambiguous_resource_kind",
        "resource_not_found",
        "multiple_resource_matches",
        "missing_warehouse_id",
        "missing_table_field_roles",
        "prompt_config_ignored",
        "conflicting_operation_intent",
        "inaccessible_resource",
        "resource_unverified",
        "discovery_unavailable",
        "safe_blueprint_blocked",
    ]
    message: str
    mention_id: str | None = None
    resource_kind: str | None = None
    access_status: AccessStatus | None = None
    blocking: bool = False
    recommended_action: str | None = None


class ResourceMention(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mention_id: str
    text: str
    span: tuple[int, int]
    kind_hint: ResourceKindHint
    identifier_kind: IdentifierKind
    confidence: float
    evidence: list[str] = Field(default_factory=list)
    trust_level: Literal["prompt_text"] = "prompt_text"
    diagnostics: list[GroundingDiagnostic] = Field(default_factory=list)


class OperationIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: OperationKind
    target_mentions: list[str] = Field(default_factory=list)
    confidence: float
    evidence: list[str] = Field(default_factory=list)
    required_capabilities: list[Capability] = Field(default_factory=list)
    diagnostics: list[GroundingDiagnostic] = Field(default_factory=list)


class GroundedAsset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset: DesignerAsset
    source_mentions: list[str] = Field(default_factory=list)
    access_status: AccessStatus
    provenance: Literal[
        "ui_selected",
        "obo_discovery",
        "prompt_exact_identity",
        "prompt_discovery_match",
        "llm_grounding_merge",
    ]
    required_for_capabilities: list[str] = Field(default_factory=list)
    diagnostics: list[GroundingDiagnostic] = Field(default_factory=list)


class ToolReadiness(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool_kind: str
    asset_ref: str | None = None
    ready: bool
    blocking: bool = False
    source_mentions: list[str] = Field(default_factory=list)
    diagnostics: list[GroundingDiagnostic] = Field(default_factory=list)
    config_sources: dict[
        str,
        Literal[
            "prompt_identity",
            "discovery_metadata",
            "environment_default",
            "static_default",
            "existing_ui_asset",
        ],
    ] = Field(default_factory=dict)


class PromptGroundingResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mentions: list[ResourceMention] = Field(default_factory=list)
    operation_intents: list[OperationIntent] = Field(default_factory=list)
    grounded_assets: list[GroundedAsset] = Field(default_factory=list)
    resolved_assets: list[DesignerAsset] = Field(default_factory=list)
    unresolved_mentions: list[ResourceMention] = Field(default_factory=list)
    tool_readiness: list[ToolReadiness] = Field(default_factory=list)
    diagnostics: list[GroundingDiagnostic] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "low"
    requires_user_action: bool = False
    safe_to_build_blueprint: bool = True


def _identity(asset: DesignerAsset) -> str:
    return asset.full_name or asset.source_id or asset.name or ""


def _identity_key(kind: str, identity: str) -> tuple[str, str]:
    return (kind, identity.casefold())


def _mention_id(text: str, span: tuple[int, int], kind_hint: str) -> str:
    payload = f"{span[0]}:{span[1]}:{kind_hint}:{text}".encode()
    return hashlib.sha1(payload).hexdigest()[:12]


def _uc_fqn_spans(intent: str) -> list[tuple[str, tuple[int, int]]]:
    spans: list[tuple[str, tuple[int, int]]] = []
    seen: set[str] = set()
    for full_name in extract_fqn_candidates(intent):
        if full_name in seen:
            continue
        seen.add(full_name)
        pattern = (
            r"(?<![A-Za-z0-9_.])"
            + re.escape(full_name)
            + r"(?![A-Za-z0-9_])"
            + r"(?!\.[A-Za-z_])"
        )
        match = re.search(pattern, intent, flags=re.IGNORECASE)
        if match is None:
            continue
        spans.append((match.group(0), (match.start(), match.end())))
    return spans


def extract_resource_mentions(intent: str) -> list[ResourceMention]:
    """Extract exact resource-shaped mentions from prompt text."""

    if not intent:
        return []

    mentions: list[ResourceMention] = []
    for text, span in _uc_fqn_spans(intent):
        inferred_kind = _infer_asset_kind_from_intent(intent, text)
        kind_hint: ResourceKindHint = inferred_kind or "unknown"  # type: ignore[assignment]
        evidence = ["three_part_fqn"]
        confidence = 0.7
        if inferred_kind == "vector_index":
            evidence.append("near_vector_index_keyword")
            confidence = 0.95
        elif inferred_kind == "delta_table":
            evidence.append("near_delta_table_keyword")
            confidence = 0.95
        diagnostics: list[GroundingDiagnostic] = []
        if kind_hint == "unknown":
            diagnostics.append(
                GroundingDiagnostic(
                    severity="warning",
                    code="ambiguous_resource_kind",
                    message="Prompt named a Unity Catalog resource without enough context to infer its kind.",
                    blocking=False,
                )
            )
        mention_id = _mention_id(text, span, kind_hint)
        mentions.append(
            ResourceMention(
                mention_id=mention_id,
                text=text,
                span=span,
                kind_hint=kind_hint,
                identifier_kind="uc_fqn",
                confidence=confidence,
                evidence=evidence,
                diagnostics=[
                    diag.model_copy(update={"mention_id": mention_id})
                    for diag in diagnostics
                ],
            )
        )
    return mentions


def _has_any(text: str, terms: Sequence[str]) -> bool:
    normalized = text.casefold()
    return any(term in normalized for term in terms)


def _has_positive_web_research_intent(intent: str) -> bool:
    """Return True only when web terms are requested, not explicitly forbidden."""

    normalized = " ".join(intent.casefold().split())
    terms = ("public web", "web tools", "news", "latest", "internet")
    negators = (
        "do not use ",
        "don't use ",
        "without ",
        "no ",
        "not use ",
        "avoid ",
        "forbid ",
    )
    for term in terms:
        start = normalized.find(term)
        while start >= 0:
            prefix = normalized[max(0, start - 40) : start]
            clause_prefix = re.split(r"[.;!?]", prefix)[-1]
            if not any(negator in clause_prefix for negator in negators):
                return True
            start = normalized.find(term, start + len(term))
    return False


def infer_operation_intents(
    intent: str,
    mentions: Sequence[ResourceMention],
) -> list[OperationIntent]:
    """Infer resource operations that should drive requiredness."""

    operations: list[OperationIntent] = []
    vector_mentions = [m.mention_id for m in mentions if m.kind_hint == "vector_index"]
    table_mentions = [m.mention_id for m in mentions if m.kind_hint == "delta_table"]

    if vector_mentions:
        operations.append(
            OperationIntent(
                operation="semantic_lookup",
                target_mentions=vector_mentions,
                confidence=0.9,
                evidence=["vector_resource_mentioned"],
                required_capabilities=["vector_search"],
            )
        )

    if table_mentions and _has_any(
        intent,
        (
            "table",
            "tables",
            "row",
            "rows",
            "exact",
            "read",
            "structured",
            "delta",
            "calculate",
            "compute",
            "sum",
            "total",
            "average",
            "numeric",
            "expenditure",
        ),
    ):
        operations.append(
            OperationIntent(
                operation="row_search",
                target_mentions=table_mentions,
                confidence=0.85,
                evidence=["delta_resource_with_table_terms"],
                required_capabilities=["table_search", "table_read"],
            )
        )
        operations.append(
            OperationIntent(
                operation="table_load",
                target_mentions=table_mentions,
                confidence=0.8,
                evidence=["delta_resource_with_table_terms"],
                required_capabilities=["table_load"],
            )
        )

    if table_mentions and _has_any(
        intent,
        (
            "calculate",
            "compute",
            "sum",
            "total",
            "average",
            "ratio",
            "percentage",
            "numeric",
            "expenditure",
            "amount",
        ),
    ):
        operations.append(
            OperationIntent(
                operation="numeric_compute",
                target_mentions=table_mentions,
                confidence=0.85,
                evidence=["delta_resource_with_numeric_terms"],
                required_capabilities=["compute"],
            )
        )

    if _has_positive_web_research_intent(intent):
        operations.append(
            OperationIntent(
                operation="open_web_research",
                target_mentions=[],
                confidence=0.7,
                evidence=["web_research_terms"],
                required_capabilities=["web_research"],
            )
        )

    return operations


# DiscoveryService.discover_all runs its per-type lookups in parallel with its
# OWN budgets (Vector Search 15s, Genie 10s, Serving 10s) and no outer timeout,
# returning partial results rather than raising. The grounding wrapper must
# therefore allow MORE than that worst case (~15s) so it acts as a true
# hang-backstop, not a guillotine that cancels discovery before it can return.
_DISCOVERY_TIMEOUT_SECONDS = 20.0

# DiscoveryService can only enumerate these kinds. Delta tables are accepted as
# manual assets (never discovered), and "unknown" hints are intentionally absent
# (see _scoped_discovery_kinds — they widen discovery rather than restrict it).
_KIND_HINT_TO_SOURCE_KIND: dict[ResourceKindHint, SourceKind] = {
    "vector_index": "vector_index",
    "genie_space": "genie_space",
    "knowledge_assistant": "knowledge_assistant",
    "serving_endpoint": "serving_endpoint",
    "sql_warehouse": "sql_warehouse",
}


def _should_attempt_discovery(mentions: Sequence[ResourceMention]) -> bool:
    """Whether user-scoped discovery can verify any mentioned resource.

    Discovery enumerates vector indexes, Genie spaces, knowledge assistants /
    serving endpoints, and SQL warehouses — never Delta tables. A prompt that
    names ONLY Delta tables would otherwise pay the full discovery latency for a
    lookup that can never match, so skip it. ``unknown`` hints stay eligible.
    """
    return any(mention.kind_hint != "delta_table" for mention in mentions)


def _scoped_discovery_kinds(
    mentions: Sequence[ResourceMention],
) -> list[SourceKind] | None:
    """Restrict discovery to the mentioned kinds, or ``None`` to leave it open.

    Returns ``None`` whenever any mention's kind is ``unknown`` — a bare FQN may
    resolve to any discoverable kind, so do not narrow it. The filter is applied
    to discovery *results* only; ``discover_all`` still enumerates and caches the
    full per-user set, so scoping never poisons the shared discovery cache.
    """
    kinds: set[SourceKind] = set()
    for mention in mentions:
        if mention.kind_hint == "unknown":
            return None
        mapped = _KIND_HINT_TO_SOURCE_KIND.get(mention.kind_hint)
        if mapped is not None:
            kinds.add(mapped)
    return sorted(kinds) if kinds else None


async def _discover_resources(
    discovery: DesignerDiscoveryAdapter | None,
    *,
    user_id: str | None,
    user_token: str | None,
    timeout_seconds: float,
    kinds: list[SourceKind] | None = None,
) -> tuple[list[DiscoveredResource], GroundingDiagnostic | None]:
    if discovery is None or not (user_id or user_token):
        return (
            [],
            GroundingDiagnostic(
                severity="warning",
                code="discovery_unavailable",
                message="No discovery context was available to verify prompt-named resources.",
                blocking=False,
            ),
        )
    try:
        resources = await asyncio.wait_for(
            discovery.list_for_user(
                user_token=user_token or "",
                kinds=kinds,
                user_id=user_id or "",
            ),
            timeout=timeout_seconds,
        )
    except TimeoutError:
        logger.warning(
            "DESIGNER_DISCOVERY_TIMEOUT", extra={"timeout_seconds": timeout_seconds}
        )
        return (
            [],
            GroundingDiagnostic(
                severity="warning",
                code="discovery_unavailable",
                message=(
                    f"Discovery timed out after {timeout_seconds:.0f}s while "
                    "verifying prompt-named resources."
                ),
                blocking=False,
                recommended_action="Resource verification was skipped; confirm access before deployment.",
            ),
        )
    except Exception:
        logger.warning("DESIGNER_DISCOVERY_FAILED", exc_info=True)
        return (
            [],
            GroundingDiagnostic(
                severity="warning",
                code="discovery_unavailable",
                message="Discovery failed while verifying prompt-named resources.",
                blocking=False,
                recommended_action="Resource verification was skipped; confirm access before deployment.",
            ),
        )
    return resources, None


def _resource_identities(resource: DiscoveredResource) -> list[str]:
    values: list[str] = []
    for value in (resource.full_name, resource.source_id, resource.name):
        if isinstance(value, str) and value.strip():
            values.append(value.strip())
    return values


def _resource_lookup(
    resources: Sequence[DiscoveredResource],
) -> dict[str, DiscoveredResource]:
    lookup: dict[str, DiscoveredResource] = {}
    for resource in resources:
        for identity in _resource_identities(resource):
            lookup.setdefault(identity.casefold(), resource)
    return lookup


def _asset_from_resource(
    resource: DiscoveredResource,
    *,
    usage: str,
) -> DesignerAsset:
    payload = {
        "kind": resource.kind,
        "full_name": resource.full_name,
        "source_id": resource.source_id,
        "name": resource.name,
        "description": resource.description,
        "usage": usage,
        "metadata": resource.metadata,
    }
    return DesignerAsset.model_validate(payload)


def _resource_is_inaccessible(resource: DiscoveredResource) -> bool:
    status = str(resource.status or "").casefold()
    return status in {"inaccessible", "permission_denied", "unauthorized", "forbidden"}


def _required_capabilities_for_mention(
    mention: ResourceMention,
    operations: Sequence[OperationIntent],
) -> list[str]:
    required: list[str] = []
    for operation in operations:
        if mention.mention_id not in operation.target_mentions:
            continue
        for capability in operation.required_capabilities:
            if capability not in required:
                required.append(capability)
    return required


def _usage_for_mention(
    mention: ResourceMention,
    operations: Sequence[OperationIntent],
) -> Literal["required", "preferred"]:
    return (
        "required"
        if _required_capabilities_for_mention(mention, operations)
        else "preferred"
    )


def _with_default_warehouse(
    asset: DesignerAsset,
    default_warehouse_id: str | None,
) -> DesignerAsset:
    if asset.kind != "delta_table" or not default_warehouse_id:
        return asset
    metadata = dict(asset.metadata)
    metadata.setdefault("warehouse_id", default_warehouse_id)
    return asset.model_copy(update={"metadata": metadata})


def _merge_grounded_assets(assets: Sequence[GroundedAsset]) -> list[GroundedAsset]:
    merged: dict[tuple[str, str], GroundedAsset] = {}
    for grounded in assets:
        identity = _identity(grounded.asset)
        if not identity:
            continue
        key = _identity_key(grounded.asset.kind, identity)
        previous = merged.get(key)
        if previous is None:
            merged[key] = grounded
            continue

        usage = previous.asset.usage
        if grounded.asset.usage == "required" or previous.asset.usage == "required":
            usage = "required"
        metadata = dict(grounded.asset.metadata)
        metadata.update(previous.asset.metadata)
        asset = previous.asset.model_copy(
            update={
                "usage": usage,
                "metadata": metadata,
            }
        )
        access_status: AccessStatus = previous.access_status
        if access_status != "verified" and grounded.access_status == "verified":
            access_status = "verified"
        if grounded.access_status == "inaccessible":
            access_status = "inaccessible"
        merged[key] = GroundedAsset(
            asset=asset,
            source_mentions=list(dict.fromkeys(previous.source_mentions + grounded.source_mentions)),
            access_status=access_status,
            provenance=previous.provenance,
            required_for_capabilities=list(
                dict.fromkeys(
                    previous.required_for_capabilities
                    + grounded.required_for_capabilities
                )
            ),
            diagnostics=previous.diagnostics + grounded.diagnostics,
        )
    return list(merged.values())


def _unsupported_config_diagnostics(intent: str) -> list[GroundingDiagnostic]:
    if not _PROMPT_CONFIG_RE.search(intent):
        return []
    return [
        GroundingDiagnostic(
            severity="warning",
            code="prompt_config_ignored",
            message="Prompt-supplied executable tool configuration was ignored; only resource identities are grounded.",
            blocking=False,
            recommended_action="Provide executable config through trusted UI/environment settings, not prompt text.",
        )
    ]


def _asset_has_warehouse(asset: DesignerAsset) -> bool:
    return bool(
        isinstance(asset.metadata.get("warehouse_id"), str)
        and asset.metadata.get("warehouse_id", "").strip()
    )


def _compute_tool_readiness(
    assets: Sequence[DesignerAsset],
    *,
    intent: str,
    grounded_assets: Sequence[GroundedAsset],
) -> list[ToolReadiness]:
    reco = recommend_tools_for_assets(
        [asset.model_dump(exclude_none=True) for asset in assets],
        intent=intent,
    )
    diagnostics_by_asset: dict[str, list[GroundingDiagnostic]] = {}
    for diagnostic in reco.get("diagnostics") or []:
        if not isinstance(diagnostic, dict):
            continue
        asset_ref = str(diagnostic.get("asset") or "")
        if not asset_ref:
            continue
        diagnostics_by_asset.setdefault(asset_ref, []).append(
            GroundingDiagnostic(
                severity="error" if diagnostic.get("severity") == "error" else "warning",
                code="missing_warehouse_id"
                if "warehouse" in str(diagnostic.get("message") or "").casefold()
                else "safe_blueprint_blocked",
                message=str(diagnostic.get("message") or "Tool recommendation diagnostic."),
                blocking=diagnostic.get("severity") == "error",
            )
        )

    source_mentions_by_asset = {
        _identity(grounded.asset): grounded.source_mentions
        for grounded in grounded_assets
    }
    readiness: list[ToolReadiness] = []
    for tool in reco.get("recommended_tools") or []:
        if not isinstance(tool, dict):
            continue
        config = tool.get("config") if isinstance(tool.get("config"), dict) else {}
        asset_ref = str(tool.get("asset_ref") or "") or None
        config_sources: dict[str, Any] = {}
        for key in config:
            if key in {"index_name", "table_name"}:
                config_sources[key] = "prompt_identity"
            elif key == "warehouse_id":
                config_sources[key] = "environment_default"
            else:
                config_sources[key] = "static_default"
        readiness.append(
            ToolReadiness(
                tool_kind=str(tool.get("kind") or ""),
                asset_ref=asset_ref,
                ready=True,
                blocking=False,
                source_mentions=source_mentions_by_asset.get(asset_ref or "", []),
                diagnostics=[],
                config_sources=config_sources,
            )
        )

    for asset_ref, diagnostics in diagnostics_by_asset.items():
        readiness.append(
            ToolReadiness(
                tool_kind="table_tools",
                asset_ref=asset_ref,
                ready=False,
                blocking=any(diag.blocking for diag in diagnostics),
                source_mentions=source_mentions_by_asset.get(asset_ref, []),
                diagnostics=diagnostics,
                config_sources={},
            )
        )
    return readiness


async def ground_prompt(
    *,
    intent: str,
    existing_assets: Sequence[DesignerAsset | Mapping[str, Any]] = (),
    discovery: DesignerDiscoveryAdapter | None = None,
    user_id: str | None = None,
    user_token: str | None = None,
    default_warehouse_id: str | None = None,
    discovery_timeout_seconds: float = _DISCOVERY_TIMEOUT_SECONDS,
) -> PromptGroundingResult:
    """Ground prompt resource mentions into assets and build-safety diagnostics."""

    default_warehouse_id = default_warehouse_id or resolve_default_table_warehouse_id()
    mentions = extract_resource_mentions(intent)
    operations = infer_operation_intents(intent, mentions)
    diagnostics: list[GroundingDiagnostic] = _unsupported_config_diagnostics(intent)

    resources: list[DiscoveredResource] = []
    if mentions and _should_attempt_discovery(mentions):
        resources, discovery_diag = await _discover_resources(
            discovery,
            user_id=user_id,
            user_token=user_token,
            timeout_seconds=discovery_timeout_seconds,
            kinds=_scoped_discovery_kinds(mentions),
        )
        if discovery_diag is not None:
            diagnostics.append(discovery_diag)

    resource_lookup = _resource_lookup(resources)
    grounded_assets: list[GroundedAsset] = []
    unresolved_mentions: list[ResourceMention] = []

    for asset in normalize_assets(list(existing_assets)):
        grounded_assets.append(
            GroundedAsset(
                asset=_with_default_warehouse(asset, default_warehouse_id),
                access_status="verified",
                provenance="ui_selected",
            )
        )

    matches = match_text_to_resources(intent, resources)
    matches_by_text = {match.matched_text.casefold(): match for match in matches}

    for mention in mentions:
        required_capabilities = _required_capabilities_for_mention(mention, operations)
        usage = _usage_for_mention(mention, operations)
        diagnostics.extend(mention.diagnostics)
        matched_resource = resource_lookup.get(mention.text.casefold())
        if matched_resource is None:
            match = matches_by_text.get(mention.text.casefold())
            matched_resource = match.resource if match is not None else None

        if matched_resource is not None:
            asset = _asset_from_resource(matched_resource, usage=usage)
            asset = _with_default_warehouse(asset, default_warehouse_id)
            if _resource_is_inaccessible(matched_resource):
                diag = GroundingDiagnostic(
                    severity="error",
                    code="inaccessible_resource",
                    message="Prompt-named resource is not accessible to the current user.",
                    mention_id=mention.mention_id,
                    resource_kind=asset.kind,
                    access_status="inaccessible",
                    blocking=bool(required_capabilities),
                    recommended_action="Choose a resource the user can access or revise the prompt.",
                )
                diagnostics.append(diag)
                grounded_assets.append(
                    GroundedAsset(
                        asset=asset,
                        source_mentions=[mention.mention_id],
                        access_status="inaccessible",
                        provenance="obo_discovery",
                        required_for_capabilities=required_capabilities,
                        diagnostics=[diag],
                    )
                )
                continue
            grounded_assets.append(
                GroundedAsset(
                    asset=asset,
                    source_mentions=[mention.mention_id],
                    access_status="verified",
                    provenance="obo_discovery",
                    required_for_capabilities=required_capabilities,
                )
            )
            continue

        if mention.identifier_kind == "uc_fqn" and mention.kind_hint != "unknown":
            kind = mention.kind_hint
            mention_diagnostics = [
                GroundingDiagnostic(
                    severity="warning",
                    code="resource_unverified",
                    message="Exact prompt-named resource was accepted without discovery verification.",
                    mention_id=mention.mention_id,
                    resource_kind=kind,
                    access_status="unverified",
                    blocking=False,
                    recommended_action="Verify resource access before deployment.",
                )
            ]
            diagnostics.extend(mention_diagnostics)
            asset = DesignerAsset(
                kind=kind,  # type: ignore[arg-type]
                full_name=mention.text,
                usage=usage,
                metadata={"warehouse_id": default_warehouse_id}
                if kind == "delta_table" and default_warehouse_id
                else {},
            )
            grounded_assets.append(
                GroundedAsset(
                    asset=asset,
                    source_mentions=[mention.mention_id],
                    access_status="unverified",
                    provenance="prompt_exact_identity",
                    required_for_capabilities=required_capabilities,
                    diagnostics=mention_diagnostics,
                )
            )
            continue

        unresolved_diag = GroundingDiagnostic(
            severity="warning",
            code="resource_not_found",
            message="Prompt-named resource could not be resolved to a supported asset.",
            mention_id=mention.mention_id,
            resource_kind=mention.kind_hint,
            blocking=bool(required_capabilities),
            recommended_action="Select the resource in the UI or use an exact supported resource name.",
        )
        diagnostics.append(unresolved_diag)
        unresolved_mentions.append(
            mention.model_copy(update={"diagnostics": mention.diagnostics + [unresolved_diag]})
        )

    grounded_assets = _merge_grounded_assets(grounded_assets)
    resolved_assets = [grounded.asset for grounded in grounded_assets]
    tool_readiness = _compute_tool_readiness(
        resolved_assets,
        intent=intent,
        grounded_assets=grounded_assets,
    )

    safe_to_build = True
    for grounded in grounded_assets:
        if grounded.asset.usage != "required":
            continue
        identity = _identity(grounded.asset)
        if grounded.access_status == "inaccessible":
            safe_to_build = False
        if grounded.asset.kind == "delta_table" and not _asset_has_warehouse(grounded.asset):
            safe_to_build = False
            diag = GroundingDiagnostic(
                severity="error",
                code="missing_warehouse_id",
                message=(
                    "Required Delta table needs TABLE_TOOLS_WAREHOUSE_ID, "
                    "STORAGE_WAREHOUSE_ID, a SQL warehouse asset, or trusted metadata."
                ),
                resource_kind="delta_table",
                access_status=grounded.access_status,
                blocking=True,
                recommended_action="Configure a table tools warehouse before building this workflow.",
            )
            diagnostics.append(diag)
            grounded.diagnostics.append(diag)
        if not identity:
            safe_to_build = False

    if unresolved_mentions:
        required_unresolved = any(
            any(diag.blocking for diag in mention.diagnostics)
            for mention in unresolved_mentions
        )
        if required_unresolved:
            safe_to_build = False

    if not safe_to_build:
        diagnostics.append(
            GroundingDiagnostic(
                severity="error",
                code="safe_blueprint_blocked",
                message="Prompt grounding found required resources that cannot be safely wired into tools.",
                blocking=True,
                recommended_action="Resolve blocking diagnostics before generating the blueprint.",
            )
        )

    blocking = any(diag.blocking or diag.severity == "error" for diag in diagnostics)
    confidence: Literal["high", "medium", "low"]
    if blocking:
        confidence = "low"
    elif any(diag.code in {"resource_unverified", "discovery_unavailable"} for diag in diagnostics):
        confidence = "medium"
    else:
        confidence = "high" if resolved_assets or not mentions else "low"

    return PromptGroundingResult(
        mentions=mentions,
        operation_intents=operations,
        grounded_assets=grounded_assets,
        resolved_assets=resolved_assets,
        unresolved_mentions=unresolved_mentions,
        tool_readiness=tool_readiness,
        diagnostics=diagnostics,
        confidence=confidence,
        requires_user_action=blocking,
        safe_to_build_blueprint=safe_to_build,
    )


def sanitized_prompt_grounding_summary(
    result: PromptGroundingResult,
) -> dict[str, Any]:
    """Return the prompt-safe grounding summary stored in workflow state."""

    return {
        "schema": "prompt_grounding.v1",
        "confidence": result.confidence,
        "safe_to_build_blueprint": result.safe_to_build_blueprint,
        "resolved_assets": [
            {
                "kind": grounded.asset.kind,
                "identity": _identity(grounded.asset),
                "usage": grounded.asset.usage,
                "access_status": grounded.access_status,
                "provenance": grounded.provenance,
                "required_for_capabilities": grounded.required_for_capabilities,
            }
            for grounded in result.grounded_assets
        ],
        "required_capabilities": sorted(
            {
                capability
                for operation in result.operation_intents
                for capability in operation.required_capabilities
            }
        ),
        "diagnostics": sanitized_prompt_grounding_diagnostics(result),
    }


def _mention_preview_by_id(result: PromptGroundingResult) -> dict[str, str]:
    previews: dict[str, str] = {}
    for mention in result.mentions:
        text = mention.text
        if len(text) > 40:
            text = text[:37] + "..."
        previews[mention.mention_id] = text
    return previews


def sanitized_prompt_grounding_diagnostics(
    result: PromptGroundingResult,
) -> list[dict[str, Any]]:
    previews = _mention_preview_by_id(result)
    payload: list[dict[str, Any]] = []
    for diagnostic in result.diagnostics:
        payload.append(
            {
                "code": diagnostic.code,
                "message": diagnostic.message,
                "severity": diagnostic.severity,
                "blocking": diagnostic.blocking or diagnostic.severity == "error",
                "resource_kind": diagnostic.resource_kind,
                "access_status": diagnostic.access_status,
                "mention_preview": previews.get(diagnostic.mention_id or ""),
                "recommended_action": diagnostic.recommended_action,
            }
        )
    return payload


def prompt_grounding_sse_result(result: PromptGroundingResult) -> dict[str, Any]:
    kinds: dict[str, int] = {}
    for grounded in result.grounded_assets:
        kinds[grounded.asset.kind] = kinds.get(grounded.asset.kind, 0) + 1
    return {
        "schema": "prompt_grounding.v1",
        "mentions_count": len(result.mentions),
        "resolved_assets_count": len(result.resolved_assets),
        "resolved_resources": [
            {
                "kind": grounded.asset.kind,
                "identity": _identity(grounded.asset),
                "usage": grounded.asset.usage,
                "access_status": grounded.access_status,
                "provenance": grounded.provenance,
            }
            for grounded in result.grounded_assets
        ],
        "resource_kinds": kinds,
        "ready_tool_kinds": sorted(
            {tool.tool_kind for tool in result.tool_readiness if tool.ready}
        ),
        "safe_to_build_blueprint": result.safe_to_build_blueprint,
        "diagnostics": sanitized_prompt_grounding_diagnostics(result),
    }


__all__ = [
    "AccessStatus",
    "Capability",
    "GroundedAsset",
    "GroundingDiagnostic",
    "OperationIntent",
    "PromptGroundingResult",
    "ResourceMention",
    "ToolReadiness",
    "extract_resource_mentions",
    "ground_prompt",
    "infer_operation_intents",
    "prompt_grounding_sse_result",
    "sanitized_prompt_grounding_diagnostics",
    "sanitized_prompt_grounding_summary",
]
