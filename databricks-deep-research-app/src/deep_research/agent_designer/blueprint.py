"""Plan v2.1 PR-2 — deterministic workflow blueprint builder.

This module implements the structural half of the architect/classifier
split mandated by plan v2.1: a pure-Python builder that maps an enriched
:class:`TaskSignature` to a scaffolded workflow AST. The architect later
customizes per-node prompts via ``node_patches`` (PR-3); structural
decisions (topology, lane count, pool ``dedup_key``, evaluator presence)
are deterministic functions of the classifier's output and are NOT the
architect's to make.

Three pieces live here:

* :func:`compute_lane_key` — plan v2.1 M7 stable identifier derivation.
  Lane keys are content-derived (lowercase-snake of the lane description
  plus an 8-char sha256 suffix), so prompt-preservation across signature
  revisions can match by description rather than ordinal position.
* :func:`compute_structural_fingerprint` — plan v2.1 M2 sha256 over the
  canonicalized structural fields. Any architect mutation that changes
  the fingerprint is rejected as ``structural_drift_detected`` (PR-3).
* :func:`build_blueprint` — the main entry point. Validates the task
  signature (fail-closed per plan v2.1 M11), synthesizes placeholder
  lane specs from ``sig.lane_descriptions``, delegates to the existing
  :func:`build_web_research_workflow` so the pool ``dedup_key`` is
  derived from real assets (M6 anti-hardcoding), then attaches
  ``lane_keys`` and ``structural_fingerprint`` to the resulting AST.

A feature-flag helper (:func:`is_deterministic_blueprint_enabled`) reads
``DESIGNER_DETERMINISTIC_BLUEPRINT`` from the environment. PR-2 keeps the
default OFF; PR-3 flips it ON.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from typing import Any

from deep_research.agent_designer.assets import (
    normalize_assets,
    recommend_tools_for_assets,
)
from deep_research.agent_designer.designer_types import (
    LaneSpec,
    ResolvedToolContract,
    ToolBindingSpec,
    ToolDeclarationSpec,
    ToolPlan,
    WorkflowDesignBrief,
)
from deep_research.agent_designer.task_signature import (
    SignatureError,
    TaskSignature,
    select_topology,
)
from deep_research.agent_designer.tool_contract import (
    sanitized_resolved_tool_contract_summary,
)
from deep_research.agent_designer.workflow_builder import (
    build_web_research_workflow,
)

# Asset signatures that pin the workflow to corpus/structured evidence and
# forbid any silent fallback to public-web tools. Generic; driven purely
# by the classifier's structural axis, not by intent text or per-case rules.
_CORPUS_ONLY_ASSET_SIGNATURES: frozenset[str] = frozenset(
    {"corpus_only", "structured_only"}
)
# Asset signature that explicitly mixes user-selected corpus assets with
# public-web research. Both kinds get wired into the workflow tools list.
_MIXED_ASSET_SIGNATURES: frozenset[str] = frozenset({"corpus_plus_web"})
# Primary (evidence-source) asset kinds. Their presence means the workflow has a
# corpus/structured evidence base — so a classifier ``no_assets``/``web_only``
# verdict is an under-classification that the deterministic floor corrects.
# ``sql_warehouse`` is excluded: it is a SUPPORTING asset for table tools, not a
# primary evidence source on its own.
_CORPUS_ASSET_KINDS: frozenset[str] = frozenset(
    {"vector_index", "delta_table", "genie_space", "knowledge_assistant", "serving_endpoint"}
)

logger = logging.getLogger(__name__)
# Default web tool names declared by ``_default_web_tool_decls`` in
# ``workflow_builder.py``. Re-declared here so the corpus_plus_web branch
# can extend ``tool_plan`` with the right pair without re-importing the
# private helper across module boundaries.
# Plan v2.1 generic-robustness — ``web_crawl`` removed from default web
# tool list (it produced deterministic full-workflow failures when
# parallel lanes speculatively called it before any ``web_research``
# populated the shared URL registry). ``web_research`` alone provides
# search + auto-fetch of top-K bodies in a single call.
_DEFAULT_WEB_TOOL_NAMES: tuple[str, ...] = ("web_research",)

DESIGNER_DETERMINISTIC_BLUEPRINT_ENV = "DESIGNER_DETERMINISTIC_BLUEPRINT"
"""Environment variable name for the PR-2 feature flag.

Plan v2.1: the deterministic-blueprint path is rolled out behind this
flag with no half-states — PR-1 + PR-2 land it; PR-3 flips the default;
PR-4 deletes the legacy path. PR-2 keeps the default OFF so existing
tests + the legacy architect-authored-AST path continue working.
"""

_LANE_KEY_PREFIX_MAX = 24
_LANE_KEY_HASH_LEN = 8


def is_deterministic_blueprint_enabled() -> bool:
    """Return True iff the deterministic-blueprint feature flag is enabled.

    Plan v2.1 PR-3 flipped the default ON. The flag is now opt-out:
    setting ``DESIGNER_DETERMINISTIC_BLUEPRINT`` to one of ``0``,
    ``false``, ``no``, ``off`` (case-insensitive) keeps the legacy
    architect-authored-AST path; anything else — including unset —
    enables the deterministic blueprint + patch-merge contract.

    Existing tests that need the legacy semantics call
    ``monkeypatch.setenv(DESIGNER_DETERMINISTIC_BLUEPRINT_ENV, "0")`` or
    ``monkeypatch.delenv(...); monkeypatch.setenv(..., "off")`` — see
    PR-3 acceptance criteria for the migration recipe.
    """
    raw = os.environ.get(DESIGNER_DETERMINISTIC_BLUEPRINT_ENV)
    if raw is None:
        return True  # PR-3 default-ON
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def compute_lane_key(description: str) -> str:
    """Plan v2.1 M7 stable lane identifier.

    The lane key is a content-derived handle the architect uses to target
    a specific lane with ``node_patches`` even after a signature revision
    reorders lanes. Two parts:

    * ``snake_prefix`` — lowercase-snake of the lane description,
      truncated to :data:`_LANE_KEY_PREFIX_MAX` characters. Provides a
      human-readable hint when reading logs and YAML dumps.
    * ``hash_suffix`` — first :data:`_LANE_KEY_HASH_LEN` characters of
      ``sha256(description)``. Provides collision resistance for cases
      where two lanes share the first 24 characters of their description.

    The hash is computed on the FULL description (not the truncated
    prefix) so two lanes that differ only after the truncation boundary
    still get distinct keys.
    """
    raw = str(description or "").strip()
    if not raw:
        raise SignatureError("lane description must be a non-empty string")
    prefix = raw.lower()[:_LANE_KEY_PREFIX_MAX]
    snake = re.sub(r"[^a-z0-9]+", "_", prefix).strip("_")
    if not snake:
        snake = "lane"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"{snake}_{digest[:_LANE_KEY_HASH_LEN]}"


def _canonical_node_shape(node: Any) -> dict[str, Any]:
    """Project a workflow node to its structural-only fields.

    Plan v2.1 M2: the structural fingerprint excludes anything the
    architect is allowed to mutate via prompt patches (``system_prompt``,
    ``user_prompt_template``, ``model_tier``, ``error_handling``,
    ``max_tool_calls``) and includes everything else (node identity,
    type, subtype, child shape, nested body/evaluator structure).
    """
    if not isinstance(node, dict):
        return {}
    config = node.get("config") if isinstance(node.get("config"), dict) else {}
    body = config.get("body") if isinstance(config, dict) else None
    evaluator = config.get("evaluator") if isinstance(config, dict) else None
    planner = config.get("planner") if isinstance(config, dict) else None
    return {
        "id": str(node.get("id") or ""),
        "type": str(node.get("type") or ""),
        "subtype": str(config.get("subtype") or "") if isinstance(config, dict) else "",
        "children": [
            _canonical_node_shape(child)
            for child in (node.get("children") or [])
            if isinstance(child, dict)
        ],
        "body": _canonical_node_shape(body) if isinstance(body, dict) else None,
        "evaluator_present": isinstance(evaluator, dict) and bool(evaluator),
        "planner_present": isinstance(planner, dict) and bool(planner),
        "tools_bound": sorted(
            [
                str(name)
                for name in (config.get("tools") or [])
                if isinstance(name, str)
            ]
            if isinstance(config, dict)
            else []
        ),
    }


def compute_structural_fingerprint(ast: dict[str, Any]) -> str:
    """Plan v2.1 M2 immutability fingerprint.

    sha256 over a canonicalized projection of the AST that includes
    every structural field (topology shape, node IDs, types, child
    layout, pool count + dedup keys, tool declarations) and excludes
    every field the architect is allowed to mutate via ``node_patches``.

    Architect mutations that change the fingerprint are rejected with
    ``structural_drift_detected`` at parse time (PR-3).
    """
    pools_shape: list[dict[str, Any]] = [
        {
            "name": str(p.get("name") or ""),
            "dedup_key": str(p.get("dedup_key") or ""),
            "max_items": int(p.get("max_items") or 0),
        }
        for p in (ast.get("pools") or [])
        if isinstance(p, dict)
    ]
    tools_shape: list[dict[str, Any]] = [
        {
            "name": str(t.get("name") or ""),
            "kind": str(t.get("kind") or ""),
        }
        for t in (ast.get("tools") or [])
        if isinstance(t, dict)
    ]
    structural = {
        "root": _canonical_node_shape(ast.get("root") or {}),
        "pools": sorted(pools_shape, key=lambda d: str(d["name"])),
        "tools": sorted(tools_shape, key=lambda d: str(d["name"])),
    }
    canonical = json.dumps(structural, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# Plan v2.1 generic-robustness — the placeholder prompts ship as
# functional defaults rather than "fill me in" markers. The runtime
# researcher can execute these as-is when the architect skips
# customization; the architect's job is to add domain-specific
# intelligence. The ``placeholder_pending_nodes`` list at the AST top
# level tracks which lanes still want architect customization; it
# surfaces as ``severity=warning`` advice from
# ``detect_unspecialized_agents`` so observability captures the gap
# without blocking the workflow.
_PLACEHOLDER_SYSTEM_PROMPT_TEMPLATE = (
    "You are a research agent investigating the following concern:\n"
    "\n"
    "    {lane_focus}\n"
    "\n"
    "## How to investigate\n"
    "- Use the runtime tools bound to this lane (visible in your tool\n"
    "  list) to gather source-backed evidence.\n"
    "- For web search + fetch: call ``web_research`` FIRST. It returns\n"
    "  search results AND auto-fetches the top-K page bodies in a\n"
    "  single call; the URLs it discovers populate the shared URL\n"
    "  index that downstream tools read. Cite by full URL; prefer\n"
    "  primary and high-authority sources over aggregators.\n"
    "- If ``web_crawl`` is bound (rare; only when extra URLs beyond\n"
    "  auto_fetch_top_k are needed): call it ONLY AFTER ``web_research``\n"
    "  has populated URLs. Pass a ``url_index`` from a prior\n"
    "  ``web_research`` result; never call ``web_crawl`` with\n"
    "  ``url_index=0`` before any search — the URL registry is empty\n"
    "  at that point and the call will error.\n"
    "- For corpus tools (vector_search, table_search, table_read,\n"
    "  table_neighbors, table_load, table_aggregate): cite by\n"
    "  (file_name, page_info, chunk_id) when those fields are available;\n"
    "  do NOT cite URLs.\n"
    "- For computation (compute, compute_namespace): pass numeric work\n"
    "  through the tool — never narrate sums, ratios, or aggregations\n"
    "  from prose.\n"
    "\n"
    "## Output contract\n"
    "Return JSON matching the standard researcher output schema:\n"
    "  - ``observation``: 200-500 word summary of findings, with\n"
    "    inline citations.\n"
    "  - ``findings``: list of source-backed claims, each carrying its\n"
    "    source id, snippet, and confidence.\n"
    "  - ``sources``: list of source records (URL or chunk_id, title,\n"
    "    extracted_text).\n"
    "\n"
    "If a sub-question is unanswerable from available evidence, mark\n"
    "it \"Data unavailable\" — do NOT improvise."
)

_PLACEHOLDER_USER_PROMPT_TEMPLATE = (
    "## Concern focus\n"
    "{lane_focus}\n"
    "\n"
    "## Research request\n"
    "{{ query }}\n"
    "\n"
    "Investigate the concern focus using the tools bound to this lane.\n"
    "Anchor every numeric, named-entity, or dated claim to a cited\n"
    "source. If a question is unanswerable from available evidence,\n"
    "mark it \"Data unavailable\" rather than improvising."
)


def _placeholder_system_prompt(lane_focus: str) -> str:
    return _PLACEHOLDER_SYSTEM_PROMPT_TEMPLATE.format(lane_focus=lane_focus)


def _placeholder_user_prompt_template(lane_focus: str) -> str:
    return _PLACEHOLDER_USER_PROMPT_TEMPLATE.format(lane_focus=lane_focus)


def _coerce_tool_contract(raw: Any) -> ResolvedToolContract | None:
    if raw is None:
        return None
    if isinstance(raw, ResolvedToolContract):
        return raw
    if isinstance(raw, str):
        try:
            return ResolvedToolContract.model_validate_json(raw)
        except Exception:
            return None
    if isinstance(raw, dict):
        try:
            return ResolvedToolContract.model_validate(raw)
        except Exception:
            return None
    return None


def _contract_resource_lines(contract: ResolvedToolContract | None) -> list[str]:
    if contract is None:
        return []
    lines: list[str] = []
    for resource in contract.resources[:6]:
        bits = [
            f"{resource.kind}:{resource.identity}",
            f"usage={resource.usage}",
        ]
        if resource.capabilities:
            bits.append("capabilities=" + ", ".join(resource.capabilities[:6]))
        if resource.domain_terms:
            bits.append("terms=" + ", ".join(resource.domain_terms[:6]))
        if resource.role_description:
            bits.append(resource.role_description)
        lines.append("- " + "; ".join(bits))
    return lines


def _contract_evidence_block(contract: ResolvedToolContract | None) -> str:
    if contract is None:
        return ""
    obligations = contract.prompt_obligations
    lines = [
        "## Resolved Tool Contract",
        "",
        f"Evidence policy: {contract.evidence_policy}",
    ]
    if contract.ready_tool_kinds:
        lines.append("Ready tool kinds: " + ", ".join(contract.ready_tool_kinds[:12]))
    if obligations.required_terms:
        lines.append("Required prompt terms: " + ", ".join(obligations.required_terms[:12]))
    if obligations.forbidden_tool_kinds:
        lines.append(
            "Forbidden tool kinds: "
            + ", ".join(obligations.forbidden_tool_kinds[:8])
        )
    resource_lines = _contract_resource_lines(contract)
    if resource_lines:
        lines.append("")
        lines.append("Grounded resources:")
        lines.extend(resource_lines)
    if obligations.planner_obligations:
        lines.append("")
        lines.append("Evidence steps:")
        lines.extend(f"- {item}" for item in obligations.planner_obligations[:8])
    if obligations.synthesis_obligations:
        lines.append("")
        lines.append("Answer obligations:")
        lines.extend(f"- {item}" for item in obligations.synthesis_obligations[:8])
    return "\n".join(lines)


def _contract_specialized_system_prompt(
    lane_focus: str,
    contract: ResolvedToolContract | None,
) -> str:
    if contract is None:
        return _placeholder_system_prompt(lane_focus)
    block = _contract_evidence_block(contract)
    if not block:
        return _placeholder_system_prompt(lane_focus)
    forbidden = set(contract.prompt_obligations.forbidden_tool_kinds)
    missing_evidence = (
        "Do not use forbidden tool kinds. Mark missing evidence as Data "
        "unavailable rather than improvising."
        if forbidden
        else "Mark missing evidence as Data unavailable rather than improvising."
    )
    return (
        "You are a resource-grounded research agent for this workflow. "
        "Use only the declared runtime tools and preserve the compact evidence "
        "contract below while investigating the lane focus.\n\n"
        f"{block}\n\n"
        "## Lane Focus\n"
        f"{lane_focus}\n\n"
        "Use grounded resources before synthesis. "
        f"{missing_evidence}"
    )


def _contract_specialized_user_prompt_template(
    lane_focus: str,
    contract: ResolvedToolContract | None,
) -> str:
    block = _contract_evidence_block(contract)
    if not block:
        return _placeholder_user_prompt_template(lane_focus)
    return (
        "## Resolved evidence contract\n"
        f"{block}\n\n"
        "## Concern focus\n"
        f"{lane_focus}\n\n"
        "## Research request\n"
        "{{ query }}\n\n"
        "Investigate the concern focus using the declared Databricks/corpus "
        "tools. Return source-backed findings, resource identifiers used, "
        "and explicit missing-data notes."
    )


def _derive_workflow_name(intent: str) -> str:
    """Return a short workflow_name suitable for the synthesized brief.

    The workflow_name is descriptive only; the framework does not key off
    it. Truncating to 80 characters keeps log lines and chat surfaces
    tidy without losing information.
    """
    cleaned = " ".join(str(intent or "").split())
    return cleaned[:80] if cleaned else "deterministic_blueprint"


def _validated_signature(task_signature: Any) -> TaskSignature:
    """Validate the incoming signature payload, fail-closed per plan M11.

    Three failure modes raise :class:`SignatureError`:

    * ``None`` (no signature emitted)
    * non-dict payload (wrong wire shape)
    * pydantic validation failure (missing required field, wrong type)

    The legacy fallback at ``workflow_builder.py:1262`` that silently
    swallowed exceptions and reverted to the brief topology is the
    bypass v2.1 is closing. Builder callers must propagate this error so
    the designer flow halts on classification failure instead of
    producing a wrong AST.
    """
    if task_signature is None:
        raise SignatureError("task_signature is required (got None)")
    if not isinstance(task_signature, dict):
        raise SignatureError(
            f"task_signature must be a dict, got {type(task_signature).__name__}"
        )
    try:
        return TaskSignature.load_from_storage(task_signature)
    except Exception as exc:
        raise SignatureError(
            f"task_signature payload failed validation: {exc}"
        ) from exc


def _resolve_lane_descriptions(sig: TaskSignature) -> list[str]:
    """Reconcile ``lane_descriptions`` with ``independent_workstreams_count``.

    Per plan v2.1 M6, ``lane_descriptions`` is an extractive list — the
    classifier copies phrases verbatim from the user intent. The contract
    is ``len(lane_descriptions) == max(independent_workstreams_count, 1)``,
    but the builder degrades gracefully if the classifier under-populates
    (rather than failing-closed on a survivable shape mismatch):

    * Too few descriptions: pad with ``concern_<i>`` placeholders.
    * Too many descriptions: truncate to lane_count.

    Under-population is logged via ``state.blueprint_warnings`` in
    :class:`BuildBlueprintTool` so the architect knows to call
    ``request_signature_revision`` if it sees these generic placeholders.
    """
    lane_count = max(int(sig.independent_workstreams_count), 1)
    descriptions = [
        str(desc).strip()
        for desc in (sig.lane_descriptions or [])
        if str(desc).strip()
    ]
    while len(descriptions) < lane_count:
        descriptions.append(f"concern_{len(descriptions) + 1}")
    return descriptions[:lane_count]


def _build_asset_tool_plan(
    sig: TaskSignature,
    assets: list[dict[str, Any]] | None,
    intent: str,
) -> ToolPlan | None:
    """Deterministically derive the workflow's ``tool_plan`` from assets.

    Plan v2.1 closes the long-standing gap where ``build_blueprint``
    handed an unbriefed ``WorkflowDesignBrief`` to
    ``build_web_research_workflow``: the downstream
    ``_tool_plan_declarations`` then returned ``None`` and the builder
    silently fell back to ``[web_research, web_crawl]`` regardless of
    user-selected assets. For ``asset_signature=corpus_only`` workflows
    that explicitly forbid public-web tools at runtime (e.g. OfficeQA),
    the silent fallback produced a workflow that violated the case
    contract before the runner ever started.

    This helper is the structural counterpart of ``select_topology``:
    deterministic, no LLM, generic across the five asset kinds in
    :data:`assets._EXPECTED_TOOL_KINDS_BY_ASSET_KIND`. It does not name
    any corpus/table/asset identifier — every config field is read off
    the user-supplied :class:`DesignerAsset` metadata via
    :func:`recommend_tools_for_assets`.

    Behavior, by ``sig.asset_signature``:

    * ``no_assets`` / ``web_only`` → returns ``None`` so the builder's
      :func:`_default_web_tool_decls` fallback fires.
    * ``corpus_only`` / ``structured_only`` → returns a ``ToolPlan``
      containing ONLY the recommended corpus/structured tools. No web
      defaults appended. Required by case contracts that forbid public-web
      tool kinds.
    * ``corpus_plus_web`` → returns a ``ToolPlan`` containing the
      recommended corpus tools PLUS ``web_research`` + ``web_crawl``.

    Fails closed when any ``usage="required"`` asset has a severity=error
    diagnostic from :func:`recommend_tools_for_assets` (the most common
    case is a Delta table asset missing its ``warehouse_id`` so
    ``table_*`` tools cannot be wired).
    The alternative — falling back to web defaults — would silently break
    the workflow contract; per plan v2.1 M11, prefer fail-closed.
    """
    normalized_assets = list(normalize_assets(assets or []))
    sig_value = str(sig.asset_signature)

    # Issue #2 deterministic floor (RC6): the classifier may emit ``no_assets``
    # even though corpus/structured assets ARE present — e.g. an edit where the
    # existing workflow's tools were seeded back as assets (the reported defect),
    # or UI-selected corpus assets the classifier overlooked. ``no_assets`` means
    # "the classifier saw no evidence base"; when corpus assets are in fact
    # present that is an under-classification, so honor the assets and build
    # corpus tools instead of silently rebuilding web-only. Scoped to
    # ``no_assets`` ONLY — a deliberate ``web_only`` verdict is respected even
    # when corpus assets happen to be selected. Generic: driven purely by asset
    # presence, never by intent text or domain. Recorded as a diagnostic so the
    # under-classification stays observable rather than hidden.
    if sig_value == "no_assets" and any(
        asset.kind in _CORPUS_ASSET_KINDS for asset in normalized_assets
    ):
        logger.warning(
            "BLUEPRINT_ASSET_SIGNATURE_FLOOR emitted=no_assets coerced=corpus_only "
            "corpus_assets=%d reason=classifier_underclassified_with_corpus_assets_present",
            len(normalized_assets),
        )
        sig_value = "corpus_only"

    if sig_value not in _CORPUS_ONLY_ASSET_SIGNATURES and sig_value not in _MIXED_ASSET_SIGNATURES:
        # ``no_assets`` and ``web_only`` with no corpus assets are handled by the
        # builder's default-web-tool path. No tool_plan needed.
        return None

    reco = recommend_tools_for_assets(
        [asset.model_dump(exclude_none=True) for asset in normalized_assets],
        intent=intent,
    )
    recommended_tools: list[dict[str, Any]] = list(reco.get("recommended_tools") or [])
    diagnostics: list[dict[str, str]] = list(reco.get("diagnostics") or [])

    required_identities: set[str] = {
        asset.full_name or asset.source_id or asset.name or ""
        for asset in normalized_assets
        if asset.usage == "required"
    }
    required_identities.discard("")

    blocking_diagnostics: list[str] = []
    for diagnostic in diagnostics:
        if not isinstance(diagnostic, dict):
            continue
        if diagnostic.get("severity") != "error":
            continue
        asset_ref = str(diagnostic.get("asset") or "")
        if asset_ref and asset_ref in required_identities:
            blocking_diagnostics.append(
                f"{asset_ref}: {diagnostic.get('message') or 'unspecified error'}"
            )
    if blocking_diagnostics:
        raise SignatureError(
            "required asset cannot be wired into a runtime tool: "
            + "; ".join(blocking_diagnostics)
        )

    # Required-asset failure mode: classifier said corpus_only/structured_only,
    # asset is marked required, but no tool kind could be recommended for any
    # of the required assets (e.g., unknown asset.kind, no field_roles).
    # Fall-through to web defaults would violate the contract; fail-closed.
    if sig_value in _CORPUS_ONLY_ASSET_SIGNATURES and not recommended_tools:
        if required_identities:
            raise SignatureError(
                "asset_signature="
                + sig_value
                + " requires at least one runtime tool but "
                + "recommend_tools_for_assets returned none for required "
                + f"assets {sorted(required_identities)}"
            )
        # Plan v2.2 grounding — fail-closed even without required identities.
        # The original branch returned None ("degenerate but not unsafe") and
        # let the builder fall back to ``web_research``. With intent
        # grounding upstream, reaching here means: the classifier inferred a
        # corpus-grounded signature from the user_intent text, BUT neither
        # UI-selected nor grounded assets produced a recommendable tool. The
        # honest answer is to halt and surface — silently swapping in
        # public-web tools violates the corpus-only contract regardless of
        # whether any asset was marked "required". The outer signature_loop
        # bounds retries; persistent failure exits with a clear message
        # rather than a workflow that researches the wrong evidence.
        raise SignatureError(
            "asset_signature="
            + sig_value
            + " but no corpus/structured tool could be recommended. "
            + "The intent-grounding stage did not resolve any workspace "
            + "resource named in user_intent, and no UI-selected assets "
            + "were provided. Either select an asset in the UI, name a "
            + "workspace resource the user can access, or revise the "
            + "task signature via request_signature_revision."
        )

    tool_decls: list[ToolDeclarationSpec] = []
    seen_names: set[str] = set()
    for entry in recommended_tools:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "").strip()
        kind = str(entry.get("kind") or "").strip()
        if not name or not kind or name in seen_names:
            continue
        seen_names.add(name)
        tool_decls.append(
            ToolDeclarationSpec(
                name=name,
                kind=kind,
                config=entry.get("config") if isinstance(entry.get("config"), dict) else {},
                description=str(entry.get("description") or ""),
            )
        )

    # Plan v2.1 generic-robustness — ``corpus_plus_web`` adds only
    # ``web_research`` (not ``web_crawl``) on top of the recommended
    # corpus tools. ``web_research`` returns search results AND
    # auto-fetches the top-K bodies in one call, which is sufficient
    # for the web side of mixed-evidence workflows. ``web_crawl`` was
    # producing deterministic failures (see plan F1) and is re-addable
    # via architect ``tool_plan`` customization when a workflow truly
    # needs the extra crawl capability.
    if sig_value in _MIXED_ASSET_SIGNATURES and "web_research" not in seen_names:
        tool_decls.append(
            ToolDeclarationSpec(
                name="web_research",
                kind="web_research",
                config={"total_results": 10, "auto_fetch_top_k": 5},
                description=(
                    "Search the public web and automatically fetch selected "
                    "source bodies for evidence-grounded research. Call "
                    "this FIRST for any web-evidence question — it "
                    "combines search + fetch in a single call."
                ),
            )
        )
        seen_names.add("web_research")

    if not tool_decls:
        # Defensive: every branch above either returned None or appended at
        # least one decl. Reaching here means an unsupported asset_signature
        # combined with empty recommendations; treat as no-op.
        return None

    bindings = [
        ToolBindingSpec(
            node_id="all_researchers",
            tool_names=[decl.name for decl in tool_decls],
        )
    ]
    return ToolPlan(tools=tool_decls, bindings=bindings)


def build_blueprint(
    task_signature: dict[str, Any],
    intent: str,
    assets: list[dict[str, Any]] | None = None,
    tool_contract: ResolvedToolContract | dict[str, Any] | str | None = None,
) -> dict[str, Any]:
    """Plan v2.1 M1+M6+M11 deterministic blueprint builder.

    Synthesizes a fully-scaffolded workflow AST from the classifier's
    enriched :class:`TaskSignature`. Every structural decision is
    derived deterministically:

    * Topology — :func:`select_topology` (plan M4 three-rule precedence;
      independence wins first).
    * Lane count — ``max(sig.independent_workstreams_count, 1)``.
    * Lane descriptions — verbatim from ``sig.lane_descriptions`` (the
      classifier's extractive contract); padded only if under-populated.
    * Pool ``dedup_key`` — derived from real ``assets`` via
      ``_sources_dedup_key`` inside :func:`build_web_research_workflow`
      (M6: ``asset_signature`` from the classifier is descriptive only,
      never load-bearing).
    * Evaluator/reflector presence — driven by ``iteration_required``
      via the existing topology builders.

    Architect customization happens later via ``node_patches`` (PR-3) —
    structural keys (``type``, ``subtype``, ``children``, ``body``,
    ``evaluator``, ``pools``, ``node_id``) are NOT theirs to mutate.

    Returns the AST dict with two added top-level metadata fields:

    * ``lane_keys`` — mapping ``lane_key`` → verbatim ``lane_description``,
      so the architect's patches can target lanes by content-derived
      key rather than ordinal position (plan M7 prompt-preservation).
    * ``structural_fingerprint`` — sha256 over the canonical structural
      projection for the M2 immutability check.

    Failure-closed (M11): invalid task_signature, missing required
    fields, non-dict payload, or empty intent → :class:`SignatureError`.
    """
    sig = _validated_signature(task_signature)
    intent_str = str(intent or "").strip()
    if not intent_str:
        raise SignatureError("intent must be a non-empty string")

    descriptions = _resolve_lane_descriptions(sig)
    lane_key_pairs = [(compute_lane_key(desc), desc) for desc in descriptions]
    contract = _coerce_tool_contract(tool_contract)

    lane_specs = [
        LaneSpec(
            description=desc,
            system_prompt=_contract_specialized_system_prompt(desc, contract),
            user_prompt_template=_contract_specialized_user_prompt_template(desc, contract),
        )
        for desc in descriptions
    ]

    tool_plan = _build_asset_tool_plan(sig, assets, intent_str)
    brief = WorkflowDesignBrief(
        workflow_name=_derive_workflow_name(intent_str),
        workflow_description=intent_str[:280],
        user_goal=intent_str,
        research_lanes=lane_specs,
        topology=select_topology(sig),
        tool_plan=tool_plan,
        tool_contract=contract,
    )

    ast = build_web_research_workflow(
        intent=intent_str,
        design_brief=brief,
        assets=assets,
        task_signature=task_signature,
    )

    if contract is None:
        _stamp_placeholder_pending(ast)
    ast["lane_keys"] = dict(lane_key_pairs)
    ast["evidence_policy"] = contract.evidence_policy if contract is not None else None
    ast["required_prompt_terms"] = (
        list(contract.prompt_obligations.required_terms)
        if contract is not None
        else []
    )
    ast["resolved_tool_contract_summary"] = (
        sanitized_resolved_tool_contract_summary(contract)
        if contract is not None
        else {"schema": "resolved_tool_contract.v1", "available": False}
    )
    ast["structural_fingerprint"] = compute_structural_fingerprint(ast)
    # Persist the signature that produced this AST so a later topology EDIT can
    # retrieve it (rather than re-infer it from arbitrary AST shape) and apply
    # only the requested delta. Stamped AFTER the fingerprint and excluded from
    # it (top-level metadata only, like ``lane_keys``), so it never perturbs the
    # PR-3 immutability check. See ``edit_planning.stored_signature``.
    ast["designer_signature"] = sig.model_dump(mode="json")
    return ast


# Plan v2.1 generic-robustness — placeholder lifecycle list lives in the
# AST's top-level metadata so the framework's strict ``AgentNodeConfig``
# Pydantic model does not reject it (the per-node config dict has
# ``extra="forbid"``). The list is a SET of node ids whose researcher
# prompts are still on the deterministic-blueprint placeholder. The
# semantic validator rejects an AST whose list is non-empty at the end of
# the architect loop — that is the structural pressure forcing the
# architect to emit ``node_patches`` for every lane prompt.
# ``_apply_architect_patches`` removes a node id from the list when a
# non-empty prompt patch lands. Not a string sentinel; not a domain
# marker; purely a lifecycle list.
PLACEHOLDER_PENDING_KEY = "placeholder_pending_nodes"


def _stamp_placeholder_pending(ast: dict[str, Any]) -> None:
    """Stamp the top-level ``placeholder_pending_nodes`` list.

    Walks the AST root-to-leaves; collects every agent node id with
    ``config.subtype in {"researcher", "background_researcher"}`` whose
    prompts came from the deterministic-blueprint placeholder. The list
    is the only generic signal the validator uses to decide whether the
    architect has actually customized each lane — string-matching the
    placeholder text would be both fragile and a recipe; a node-id list
    on top-level AST metadata is structural and survives Pydantic strict
    validation on individual node configs.

    Idempotent: re-stamping yields the same list (set semantics over ids).
    """
    pending: list[str] = []

    def _walk(node: Any) -> None:
        if not isinstance(node, dict):
            return
        config = node.get("config") if isinstance(node.get("config"), dict) else {}
        if (
            node.get("type") == "agent"
            and isinstance(config, dict)
            and config.get("subtype") in {"researcher", "background_researcher"}
        ):
            node_id = str(node.get("id") or "").strip()
            if node_id and node_id not in pending:
                pending.append(node_id)
        if isinstance(config, dict):
            body = config.get("body")
            if isinstance(body, dict):
                _walk(body)
            evaluator = config.get("evaluator")
            if isinstance(evaluator, dict):
                _walk(evaluator)
            planner = config.get("planner")
            if isinstance(planner, dict):
                _walk(planner)
        for child in node.get("children") or []:
            if isinstance(child, dict):
                _walk(child)

    root = ast.get("root")
    if isinstance(root, dict):
        _walk(root)
    if pending:
        ast[PLACEHOLDER_PENDING_KEY] = pending


__all__ = [
    "DESIGNER_DETERMINISTIC_BLUEPRINT_ENV",
    "PLACEHOLDER_PENDING_KEY",
    "SignatureError",
    "build_blueprint",
    "compute_lane_key",
    "compute_structural_fingerprint",
    "is_deterministic_blueprint_enabled",
]
