"""Prompt-safe resolved tool contract for Agent Designer.

Prompt grounding resolves resource identity and executable readiness. This
module adds the compact semantic contract that downstream blueprint/prompt
generation can rely on without creating a second executable tool-config path.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from deep_research.agent_designer.assets import DesignerAsset
from deep_research.agent_designer.designer_types import (
    PromptObligationContract,
    ResolvedToolContract,
    ResourceContract,
    ResourceSemanticExtraction,
    ResourceSemanticItem,
)
from deep_research.agent_designer.prompt_grounding import (
    GroundedAsset,
    PromptGroundingResult,
)
from deep_research.services.llm.types import ModelTier

_MAX_INTENT_CHARS = 4000
_MAX_TEXT_CHARS = 220
_MAX_TERMS = 12
_MIN_ANCHOR_TERMS = 6
_MAX_RESOURCE_TERMS = 8
_MAX_OBLIGATIONS = 8
_WEB_TOOL_KINDS = ("web_search", "web_crawl", "web_research")
_CORPUS_TOOL_KINDS = {
    "vector_search",
    "table_search",
    "table_read",
    "table_load",
    "table_neighbors",
    "table_aggregate",
}
_STRUCTURED_TOOL_KINDS = {
    "table_search",
    "table_read",
    "table_load",
    "table_neighbors",
    "table_aggregate",
    "compute",
    "compute_namespace",
}
_FORBIDDEN_CONFIG_KEYS = {
    "warehouse_id",
    "sql_warehouse_id",
    "endpoint_url",
    "access_token",
    "secret",
    "credential",
    "credentials",
    "api_key",
    "sql_query",
    "python_code",
    "tool_config",
    "tools",
    "tool_declarations",
    "config",
}
_CONFIG_VALUE_RE = re.compile(
    r"\b("
    r"warehouse_id|sql_warehouse_id|endpoint_url|access_token|secret|"
    r"credential|credentials|api_key|select\s+.+\s+from|python_code|"
    r"databricks://|https?://"
    r")\b",
    flags=re.IGNORECASE | re.DOTALL,
)
_TERM_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{2,}")
_STOPWORDS = {
    "about",
    "across",
    "agent",
    "analysis",
    "answer",
    "assistant",
    "build",
    "candidate",
    "configuration",
    "concise",
    "data",
    "delta",
    "document",
    "documents",
    "evidence",
    "exact",
    "from",
    "index",
    "infer",
    "needed",
    "public",
    "question",
    "questions",
    "reads",
    "research",
    "resource",
    "resources",
    "selected",
    "specific",
    "style",
    "task",
    "text",
    "these",
    "tool",
    "tools",
    "unless",
    "with",
}
_WEB_FORBID_TERMS = (
    "public web",
    "web tools",
    "web search",
    "web_research",
    "web_search",
    "web_crawl",
    "internet",
)
_WEB_FORBID_PREFIX_RE = re.compile(
    r"\b("
    r"do\s+not\s+use|don't\s+use|must\s+not\s+use|never\s+use|"
    r"not\s+use|without|no|avoid|forbid|forbidden|disallow"
    r")\b",
    flags=re.IGNORECASE,
)
_WEB_FORBID_SUFFIX_RE = re.compile(
    r"\b("
    r"forbidden|disallowed|not\s+allowed|must\s+not\s+be\s+used|"
    r"should\s+not\s+be\s+used"
    r")\b",
    flags=re.IGNORECASE,
)


def _identity(asset: DesignerAsset) -> str:
    return asset.full_name or asset.source_id or asset.name or ""


def _coerce_grounding(raw: Any) -> PromptGroundingResult | None:
    if isinstance(raw, PromptGroundingResult):
        return raw
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):
            return None
    if isinstance(raw, Mapping):
        try:
            return PromptGroundingResult.model_validate(raw)
        except Exception:
            return None
    return None


def _coerce_semantics(raw: Any) -> ResourceSemanticExtraction | None:
    if raw is None:
        return None
    if isinstance(raw, ResourceSemanticExtraction):
        return raw
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):
            return None
    if isinstance(raw, Mapping):
        try:
            return ResourceSemanticExtraction.model_validate(raw)
        except Exception:
            return None
    return None


def _clean_text(value: Any, *, max_length: int = _MAX_TEXT_CHARS) -> str:
    cleaned = " ".join(str(value or "").split())
    if len(cleaned) <= max_length:
        return cleaned
    return cleaned[: max_length - 15].rstrip() + " ...(truncated)"


def _dedupe(values: Sequence[str], *, limit: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = _clean_text(value, max_length=_MAX_TEXT_CHARS)
        key = cleaned.casefold()
        if cleaned and key not in seen:
            out.append(cleaned)
            seen.add(key)
        if len(out) >= limit:
            break
    return out


def _term_tokens(text: str) -> list[str]:
    terms: list[str] = []
    for raw in _TERM_RE.findall(text or ""):
        for part in re.split(r"[_\W]+", raw):
            token = part.strip().casefold()
            if len(token) < 3 or token in _STOPWORDS:
                continue
            if token not in terms:
                terms.append(token)
    return terms


def _explicitly_forbids_public_web(intent: str) -> bool:
    """Return True only when the prompt itself forbids public-web use."""

    normalized = " ".join(str(intent or "").casefold().split())
    if not normalized:
        return False

    for term in _WEB_FORBID_TERMS:
        start = normalized.find(term)
        while start >= 0:
            end = start + len(term)
            prefix = normalized[max(0, start - 80) : start]
            prefix_clause = re.split(r"[.;!?]", prefix)[-1]
            if _WEB_FORBID_PREFIX_RE.search(prefix_clause):
                return True

            suffix = normalized[end : min(len(normalized), end + 80)]
            suffix_clause = re.split(r"[.;!?]", suffix)[0]
            if _WEB_FORBID_SUFFIX_RE.search(suffix_clause):
                return True

            start = normalized.find(term, end)
    return False


def _asset_identity_terms(asset: DesignerAsset) -> list[str]:
    values = [_identity(asset), asset.name or "", asset.description or ""]
    return _term_tokens(" ".join(values))


def _value_looks_executable(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key).casefold() in _FORBIDDEN_CONFIG_KEYS
            or _value_looks_executable(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_value_looks_executable(item) for item in value)
    if not isinstance(value, str):
        return False
    return bool(_CONFIG_VALUE_RE.search(value))


def _semantic_item_has_forbidden_config(item: ResourceSemanticItem) -> bool:
    extras = getattr(item, "model_extra", None) or {}
    for key, value in extras.items():
        if str(key).casefold() in _FORBIDDEN_CONFIG_KEYS:
            return True
        if _value_looks_executable(value):
            return True
    fields = [
        item.role_description,
        *item.domain_terms,
        *item.intended_operations,
    ]
    return any(_value_looks_executable(value) for value in fields)


def validate_resource_semantics(
    grounding: PromptGroundingResult,
    semantics: ResourceSemanticExtraction | Mapping[str, Any] | str | None,
) -> tuple[ResourceSemanticExtraction | None, list[dict[str, Any]]]:
    """Validate advisory semantic extraction against deterministic grounding."""

    parsed = _coerce_semantics(semantics)
    if parsed is None:
        return None, []

    valid_identities = {
        _identity(asset)
        for asset in grounding.resolved_assets
        if _identity(asset)
    }
    diagnostics: list[dict[str, Any]] = []
    resources: list[ResourceSemanticItem] = []
    for item in parsed.resources:
        identity = item.identity.strip()
        if identity not in valid_identities:
            diagnostics.append(
                {
                    "severity": "warning",
                    "code": "semantic_resource_discarded",
                    "message": "Semantic extraction named an ungrounded resource identity.",
                    "identity": identity,
                }
            )
            continue
        if _semantic_item_has_forbidden_config(item):
            diagnostics.append(
                {
                    "severity": "warning",
                    "code": "semantic_executable_config_discarded",
                    "message": "Semantic extraction carried executable-looking config and was ignored.",
                    "identity": identity,
                }
            )
            continue
        resources.append(
            ResourceSemanticItem(
                identity=identity,
                role_description=_clean_text(item.role_description),
                domain_terms=_dedupe(
                    [term.casefold() for term in item.domain_terms],
                    limit=_MAX_RESOURCE_TERMS,
                ),
                intended_operations=_dedupe(
                    item.intended_operations,
                    limit=_MAX_OBLIGATIONS,
                ),
            )
        )

    task_terms = [
        term.casefold()
        for term in parsed.task_domain_terms
        if not _value_looks_executable(term)
    ]
    obligations = [
        item
        for item in parsed.answer_obligations
        if not _value_looks_executable(item)
    ]
    validated = ResourceSemanticExtraction(
        resources=resources,
        task_domain_terms=_dedupe(task_terms, limit=_MAX_TERMS),
        answer_obligations=_dedupe(obligations, limit=_MAX_OBLIGATIONS),
    )
    return validated, diagnostics


async def extract_resource_semantics_structured(
    *,
    llm: Any,
    intent: str,
    grounding: PromptGroundingResult,
) -> tuple[ResourceSemanticExtraction | None, list[dict[str, Any]]]:
    """Run the optional small structured LLM extraction call.

    Failures are non-blocking; callers fall back to deterministic projection.
    """

    if llm is None or not callable(getattr(llm, "complete", None)):
        return None, []
    if not grounding.resolved_assets:
        return None, []

    resources = [
        {
            "identity": _identity(asset),
            "kind": asset.kind,
            "usage": asset.usage,
        }
        for asset in grounding.resolved_assets
        if _identity(asset)
    ]
    prompt = (
        "Extract advisory resource semantics for an Agent Designer workflow.\n"
        "Use only the already-grounded resource identities listed below.\n"
        "Do not invent resources. Do not emit warehouse ids, endpoint URLs, "
        "credentials, SQL, Python, tool config, or tool declarations. If unsure, "
        "leave fields empty.\n\n"
        "Return roles, domain terms, intended operations, and answer obligations.\n\n"
        f"Initial prompt:\n{intent[:_MAX_INTENT_CHARS]}\n\n"
        "Grounded resources:\n"
        f"{json.dumps(resources, ensure_ascii=True, indent=2)}\n\n"
        "Required capabilities:\n"
        + ", ".join(
            sorted(
                {
                    capability
                    for op in grounding.operation_intents
                    for capability in op.required_capabilities
                }
            )
        )
    )
    try:
        response = await llm.complete(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You extract compact, prompt-safe semantics. You never "
                        "produce executable configuration."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            tier=ModelTier.SIMPLE,
            temperature=0,
            max_tokens=900,
            structured_output=ResourceSemanticExtraction,
        )
    except Exception as exc:  # noqa: BLE001 - semantics are advisory
        return None, [
            {
                "severity": "warning",
                "code": "semantic_extraction_failed",
                "message": _clean_text(str(exc), max_length=180),
            }
        ]

    structured = getattr(response, "structured", None)
    parsed = _coerce_semantics(structured) or _coerce_semantics(
        getattr(response, "content", None)
    )
    validated, diagnostics = validate_resource_semantics(grounding, parsed)
    return validated, diagnostics


def _capabilities_by_identity(grounding: PromptGroundingResult) -> dict[str, list[str]]:
    by_identity: dict[str, list[str]] = {}
    for grounded in grounding.grounded_assets:
        identity = _identity(grounded.asset)
        if not identity:
            continue
        values = by_identity.setdefault(identity, [])
        for capability in grounded.required_for_capabilities:
            if capability not in values:
                values.append(capability)
    for readiness in grounding.tool_readiness:
        if not readiness.asset_ref:
            continue
        values = by_identity.setdefault(readiness.asset_ref, [])
        if readiness.tool_kind and readiness.tool_kind not in values:
            values.append(readiness.tool_kind)
    return by_identity


def _grounded_by_identity(grounding: PromptGroundingResult) -> dict[str, GroundedAsset]:
    return {
        _identity(item.asset): item
        for item in grounding.grounded_assets
        if _identity(item.asset)
    }


def _semantic_by_identity(
    semantics: ResourceSemanticExtraction | None,
) -> dict[str, ResourceSemanticItem]:
    if semantics is None:
        return {}
    return {item.identity: item for item in semantics.resources if item.identity}


def _required_capabilities(grounding: PromptGroundingResult) -> list[str]:
    values: list[str] = []
    for operation in grounding.operation_intents:
        for capability in operation.required_capabilities:
            if capability not in values:
                values.append(capability)
    return values


def _ready_tool_kinds(grounding: PromptGroundingResult) -> list[str]:
    kinds = {
        readiness.tool_kind
        for readiness in grounding.tool_readiness
        if readiness.ready and readiness.tool_kind
    }
    return sorted(kinds)


def _derive_evidence_policy(
    grounding: PromptGroundingResult,
    ready_tool_kinds: Sequence[str],
    required_capabilities: Sequence[str],
) -> str:
    ready = set(ready_tool_kinds)
    required = set(required_capabilities)
    has_web = bool(ready & set(_WEB_TOOL_KINDS)) or "web_research" in required
    has_vector = "vector_search" in ready or "vector_search" in required
    has_table = bool((ready | required) & _STRUCTURED_TOOL_KINDS)
    has_corpus_asset = any(
        asset.kind in {"vector_index", "delta_table"}
        for asset in grounding.resolved_assets
    )
    if has_corpus_asset and (has_vector or has_table) and has_web:
        return "corpus_plus_web"
    if has_corpus_asset and has_vector:
        return "corpus_only"
    if has_corpus_asset and has_table:
        return "structured_only"
    return "web_only"


def _fallback_terms(
    *,
    intent: str,
    grounding: PromptGroundingResult,
    required_capabilities: Sequence[str],
) -> list[str]:
    terms: list[str] = []
    terms.extend(_term_tokens(intent))
    for asset in grounding.resolved_assets:
        terms.extend(_asset_identity_terms(asset))
    for capability in required_capabilities:
        terms.extend(_term_tokens(str(capability).replace("_", " ")))
    return _dedupe(terms, limit=_MAX_TERMS)


def _contract_terms(
    *,
    intent: str,
    grounding: PromptGroundingResult,
    required_capabilities: Sequence[str],
    semantics: ResourceSemanticExtraction | None,
) -> list[str]:
    semantic_terms: list[str] = []
    if semantics is not None:
        semantic_terms.extend(semantics.task_domain_terms)
        for resource in semantics.resources:
            semantic_terms.extend(resource.domain_terms)
    semantic = _dedupe(
        [term.casefold() for term in semantic_terms if term],
        limit=_MAX_TERMS,
    )
    fallback = _fallback_terms(
        intent=intent,
        grounding=grounding,
        required_capabilities=required_capabilities,
    )
    # Deterministic resource/capability anchors (table & index name tokens,
    # "vector", "compute", ...) must survive the _MAX_TERMS cap even when the
    # advisory extraction returns a full slate of semantic phrases. Reserve a
    # slot budget for them — drawn from resource identities + capabilities,
    # NOT generic intent prose — so required_terms always names the corpus the
    # agent has to use. Without this, a verbose LLM truncated every resource
    # token out of required_terms and the contract lost its anchor.
    anchor_pool: list[str] = []
    for asset in grounding.resolved_assets:
        anchor_pool.extend(_asset_identity_terms(asset))
    for capability in required_capabilities:
        anchor_pool.extend(_term_tokens(str(capability).replace("_", " ")))
    reserved = _dedupe(anchor_pool, limit=_MIN_ANCHOR_TERMS)

    return _dedupe([*reserved, *semantic, *fallback], limit=_MAX_TERMS)


def _planner_obligations(
    evidence_policy: str,
    required_capabilities: Sequence[str],
    forbidden_tool_kinds: Sequence[str],
) -> list[str]:
    capabilities = set(required_capabilities)
    forbidden = set(forbidden_tool_kinds)
    obligations: list[str] = []
    if evidence_policy in {"corpus_only", "corpus_plus_web"}:
        obligations.append("Use the named Databricks corpus resources before synthesis.")
    if "vector_search" in capabilities:
        obligations.append("Run semantic lookup on the grounded vector index.")
    if capabilities & {"table_search", "table_read", "table_load"}:
        obligations.append("Read or load the grounded Delta tables for exact evidence.")
    if "compute" in capabilities:
        obligations.append("Run numeric calculations through the compute tool.")
    if forbidden & set(_WEB_TOOL_KINDS):
        obligations.append("Do not fall back to public web tools; the user prompt forbids them.")
    return _dedupe(obligations, limit=_MAX_OBLIGATIONS)


def _mentions_fiscal_calendar_distinction(intent: str) -> bool:
    normalized = " ".join(str(intent or "").casefold().split())
    if not normalized:
        return False
    has_fiscal = bool(re.search(r"\bfiscal(?:[-\s]+year)?\b|\bfy\b", normalized))
    has_calendar = bool(re.search(r"\bcalendar(?:[-\s]+year)?\b|\bcy\b", normalized))
    return has_fiscal and has_calendar


def _synthesis_obligations(
    *,
    intent: str,
    terms: Sequence[str],
    semantics: ResourceSemanticExtraction | None,
) -> list[str]:
    obligations: list[str] = []
    if semantics is not None:
        obligations.extend(semantics.answer_obligations)
    term_set = {term.casefold() for term in terms}
    if {"fiscal", "calendar"} <= term_set or _mentions_fiscal_calendar_distinction(intent):
        obligations.append("Preserve the fiscal/calendar-year distinction when supported by evidence.")
    return _dedupe(obligations, limit=_MAX_OBLIGATIONS)


def project_resolved_tool_contract(
    prompt_grounding: PromptGroundingResult | Mapping[str, Any] | str | None,
    *,
    intent: str,
    task_signature: Mapping[str, Any] | None = None,
    semantics: ResourceSemanticExtraction | Mapping[str, Any] | str | None = None,
) -> ResolvedToolContract | None:
    """Project deterministic grounding plus validated semantics into a contract."""

    del task_signature  # Future extension point; keep signature stable.
    grounding = _coerce_grounding(prompt_grounding)
    if grounding is None:
        return None
    validated_semantics, diagnostics = validate_resource_semantics(grounding, semantics)

    required_capabilities = _required_capabilities(grounding)
    ready_tool_kinds = _ready_tool_kinds(grounding)
    evidence_policy = _derive_evidence_policy(
        grounding,
        ready_tool_kinds,
        required_capabilities,
    )
    semantic_map = _semantic_by_identity(validated_semantics)
    grounded_map = _grounded_by_identity(grounding)
    capabilities = _capabilities_by_identity(grounding)

    resources: list[ResourceContract] = []
    for asset in grounding.resolved_assets:
        identity = _identity(asset)
        if not identity:
            continue
        grounded = grounded_map.get(identity)
        semantic = semantic_map.get(identity)
        resource_terms = (
            semantic.domain_terms if semantic is not None else _asset_identity_terms(asset)
        )
        resources.append(
            ResourceContract(
                kind=asset.kind,
                identity=identity,
                usage="required" if asset.usage == "required" else "optional",
                access_status=grounded.access_status if grounded else "unverified",
                provenance=grounded.provenance if grounded else "ui_selected",
                capabilities=_dedupe(capabilities.get(identity, []), limit=10),
                role_description=semantic.role_description if semantic else "",
                domain_terms=_dedupe(resource_terms, limit=_MAX_RESOURCE_TERMS),
                intended_operations=semantic.intended_operations if semantic else [],
            )
        )

    terms = _contract_terms(
        intent=intent,
        grounding=grounding,
        required_capabilities=required_capabilities,
        semantics=validated_semantics,
    )
    forbidden = list(_WEB_TOOL_KINDS) if _explicitly_forbids_public_web(intent) else []
    obligations = PromptObligationContract(
        required_terms=terms,
        synthesis_obligations=_synthesis_obligations(
            intent=intent,
            terms=terms,
            semantics=validated_semantics,
        ),
        planner_obligations=_planner_obligations(
            evidence_policy,
            required_capabilities,
            forbidden,
        ),
        forbidden_tool_kinds=forbidden,
    )
    contract_diagnostics = list(diagnostics)
    for diagnostic in grounding.diagnostics:
        if diagnostic.blocking or diagnostic.severity == "error":
            contract_diagnostics.append(
                {
                    "severity": diagnostic.severity,
                    "code": diagnostic.code,
                    "message": diagnostic.message,
                    "blocking": diagnostic.blocking,
                }
            )
    return ResolvedToolContract(
        evidence_policy=evidence_policy,
        resources=resources,
        required_capabilities=_dedupe(required_capabilities, limit=12),
        ready_tool_kinds=ready_tool_kinds,
        prompt_obligations=obligations,
        diagnostics=contract_diagnostics,
    )


def sanitized_resolved_tool_contract_summary(
    contract: ResolvedToolContract | Mapping[str, Any] | str | None,
) -> dict[str, Any]:
    parsed: ResolvedToolContract | None
    if isinstance(contract, ResolvedToolContract):
        parsed = contract
    elif isinstance(contract, str):
        try:
            parsed = ResolvedToolContract.model_validate_json(contract)
        except Exception:
            parsed = None
    elif isinstance(contract, Mapping):
        try:
            parsed = ResolvedToolContract.model_validate(contract)
        except Exception:
            parsed = None
    else:
        parsed = None
    if parsed is None:
        return {"schema": "resolved_tool_contract.v1", "available": False}
    return {
        "schema": parsed.schema_,
        "available": True,
        "evidence_policy": parsed.evidence_policy,
        "resources_count": len(parsed.resources),
        "resources": [
            {
                "kind": resource.kind,
                "identity": resource.identity,
                "usage": resource.usage,
                "access_status": resource.access_status,
                "capabilities": resource.capabilities[:8],
                "domain_terms": resource.domain_terms[:8],
            }
            for resource in parsed.resources[:8]
        ],
        "required_capabilities": parsed.required_capabilities[:12],
        "ready_tool_kinds": parsed.ready_tool_kinds[:12],
        "required_terms": parsed.prompt_obligations.required_terms[:12],
        "synthesis_obligations": parsed.prompt_obligations.synthesis_obligations[:8],
        "planner_obligations": parsed.prompt_obligations.planner_obligations[:8],
        "forbidden_tool_kinds": parsed.prompt_obligations.forbidden_tool_kinds[:8],
        "diagnostics": parsed.diagnostics[:8],
    }


def resolved_tool_contract_sse_result(
    contract: ResolvedToolContract | Mapping[str, Any] | str | None,
) -> dict[str, Any]:
    return sanitized_resolved_tool_contract_summary(contract)


def resource_semantics_summary(
    semantics: ResourceSemanticExtraction | Mapping[str, Any] | str | None,
) -> dict[str, Any]:
    parsed = _coerce_semantics(semantics)
    if parsed is None:
        return {"schema": "resource_semantics.v1", "available": False}
    return {
        "schema": parsed.schema_,
        "available": True,
        "resources_count": len(parsed.resources),
        "resources": [
            {
                "identity": item.identity,
                "role_description": item.role_description,
                "domain_terms": item.domain_terms[:_MAX_RESOURCE_TERMS],
                "intended_operations": item.intended_operations[:_MAX_OBLIGATIONS],
            }
            for item in parsed.resources[:8]
        ],
        "task_domain_terms": parsed.task_domain_terms[:_MAX_TERMS],
        "answer_obligations": parsed.answer_obligations[:_MAX_OBLIGATIONS],
    }


__all__ = [
    "extract_resource_semantics_structured",
    "project_resolved_tool_contract",
    "resolved_tool_contract_sse_result",
    "resource_semantics_summary",
    "sanitized_resolved_tool_contract_summary",
    "validate_resource_semantics",
]
