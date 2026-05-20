"""Designer architect/critic strategy loaded from YAML.

The Designer LLM needs an explicit product contract before it calls mutation
tools, and the deterministic workflow builder needs a compact representation of
that contract.  This module keeps both surfaces backed by the same YAML file so
the chat prompt and fallback builder profiles do not drift.

Pure data types and coercion helpers have been moved to :mod:`designer_types`.
This module re-exports them for backwards compatibility so existing importers
using ``from .designer_architect import WorkflowDesignBrief`` continue to work
until this shim is removed at the end of US-11.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .designer_types import (
    _MAX_BRIEF_ITEMS as _MAX_BRIEF_ITEMS,
)
from .designer_types import (
    _MAX_ITEM_LENGTH as _MAX_ITEM_LENGTH,
)
from .designer_types import (
    _MAX_LANE_SYSTEM_PROMPT_LENGTH as _MAX_LANE_SYSTEM_PROMPT_LENGTH,
)
from .designer_types import (
    _MAX_LANE_USER_PROMPT_TEMPLATE_LENGTH as _MAX_LANE_USER_PROMPT_TEMPLATE_LENGTH,
)

# ---------------------------------------------------------------------------
# Back-compat re-exports — types now live in designer_types.py
# PEP 484 explicit re-export form so mypy treats these as public.
# ---------------------------------------------------------------------------
from .designer_types import DomainProfile as DomainProfile
from .designer_types import GroundingKind as GroundingKind
from .designer_types import LaneSpec as LaneSpec
from .designer_types import TopologyKind as TopologyKind
from .designer_types import WorkflowDesignBrief as WorkflowDesignBrief
from .designer_types import _bounded_multiline as _bounded_multiline
from .designer_types import _clean_text as _clean_text
from .designer_types import _coerce_brief as _coerce_brief
from .designer_types import _coerce_lane_item as _coerce_lane_item
from .designer_types import _coerce_lane_list as _coerce_lane_list
from .designer_types import _compact_lane_descriptions as _compact_lane_descriptions
from .designer_types import _compact_list as _compact_list
from .designer_types import _merge_lane_lists as _merge_lane_lists
from .designer_types import _merge_lists as _merge_lists


class DesignerArchitectStrategy(BaseModel):
    """YAML-backed strategy for Designer prompt and compatibility profile."""

    model_config = ConfigDict(extra="forbid")

    version: int
    name: str
    system_prompt: str
    workflow_method: list[dict[str, str]] = Field(default_factory=list)
    domain_profiles: dict[str, DomainProfile] = Field(default_factory=dict)
    default_profile: DomainProfile


def _strategy_path() -> Path:
    return Path(__file__).with_name("designer_architect.yaml")


@lru_cache(maxsize=1)
def load_designer_architect_strategy() -> DesignerArchitectStrategy:
    """Load and validate the YAML strategy used by Designer chat and builder."""
    raw = yaml.safe_load(_strategy_path().read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("designer_architect.yaml must contain a mapping")
    return DesignerArchitectStrategy.model_validate(raw)


def designer_system_prompt() -> str:
    """Return the YAML-backed system prompt injected into Designer chat."""
    strategy = load_designer_architect_strategy()
    profile_summaries: list[str] = []
    for profile in strategy.domain_profiles.values():
        lanes = "; ".join(_compact_lane_descriptions(profile.research_lanes, limit=6))
        profile_summaries.append(f"- {profile.label}: {lanes}")

    default_lanes = "; ".join(
        _compact_lane_descriptions(strategy.default_profile.research_lanes, limit=4)
    )
    method = "\n".join(
        f"- {item.get('label', item.get('id', 'Step'))}: {item.get('instruction', '')}"
        for item in strategy.workflow_method
    )
    profiles = "\n".join(profile_summaries)
    return (
        f"{strategy.system_prompt.strip()}\n\n"
        "Workflow design method from YAML:\n"
        f"{method}\n\n"
        "Legacy compatibility profile summaries from YAML "
        "(do not use these as prompt authoring defaults):\n"
        f"{profiles}\n"
        f"- {strategy.default_profile.label}: {default_lanes}\n\n"
        "propose_workflow design_brief contract:\n"
        "- workflow_name: short, customer-facing name\n"
        "- domain: specific domain or task category\n"
        "- topology: choose parallel_lanes, plan_and_execute, or single_agent "
        "from the task structure\n"
        "- research_lanes: concrete workstreams the runtime planner must execute,"
        " each {description, specialized system_prompt, user_prompt_template}; "
        "the LLM authors these use-case prompts\n"
        "- required_outputs: sections or deliverables the final answer must contain\n"
        "- quality_gates: critic checks that make generic workflows unacceptable\n"
        "- constraints: scope, tone, freshness, source, or compliance constraints\n"
    )


def compile_workflow_design_brief(
    intent: str,
    supplied: WorkflowDesignBrief | dict[str, Any] | None = None,
) -> WorkflowDesignBrief:
    """Compile an LLM-supplied brief with compatibility defaults.

    Semantic authoring belongs to the Designer LLM. When a design_brief is
    supplied, do not backfill lanes, report outputs, quality gates, or
    constraints from the YAML default profile; missing semantic fields must
    remain visible so the structural gate / critic can ask the Designer to
    author them. The profile remains only for legacy callers that provide no
    brief at all.
    """
    has_supplied_brief = supplied is not None
    supplied_brief = _coerce_brief(supplied)
    profile = _select_domain_profile(intent)

    if supplied_brief.research_lanes:
        merged_lanes = _merge_lane_lists(supplied_brief.research_lanes, [])
    elif has_supplied_brief:
        merged_lanes = []
    else:
        merged_lanes = _merge_lane_lists([], profile.research_lanes)

    if has_supplied_brief:
        required_outputs = _merge_lists(supplied_brief.required_outputs, [])
        quality_gates = _merge_lists(supplied_brief.quality_gates, [])
        constraints = _merge_lists(supplied_brief.constraints, [])
    else:
        required_outputs = _merge_lists(
            supplied_brief.required_outputs,
            profile.required_outputs,
        )
        quality_gates = _merge_lists(
            supplied_brief.quality_gates,
            profile.quality_gates,
        )
        constraints = _merge_lists(supplied_brief.constraints, profile.constraints)

    return WorkflowDesignBrief(
        workflow_name=_clean_text(supplied_brief.workflow_name or profile.workflow_name_template),
        workflow_description=_clean_text(supplied_brief.workflow_description or intent),
        domain=_clean_text(supplied_brief.domain or profile.label),
        user_goal=_clean_text(supplied_brief.user_goal or intent),
        research_lanes=merged_lanes,
        required_outputs=required_outputs,
        quality_gates=quality_gates,
        constraints=constraints,
        # Carry the supplied brief's topology forward; default (parallel_lanes)
        # applies only when the LLM didn't pick one. Topology has no
        # profile-side default to merge with — it belongs solely to the
        # supplied brief.
        topology=supplied_brief.topology,
        # Same pattern for grounding_mode (default "reclaim"): the Designer
        # LLM controls the strictness vs latency trade-off per workflow.
        grounding_mode=supplied_brief.grounding_mode,
    )


def format_workflow_design_brief(brief: WorkflowDesignBrief) -> str:
    """Format a design brief as prompt text for generated workflow agents."""
    lane_descriptions = _compact_lane_descriptions(brief.research_lanes)
    sections: list[tuple[str, list[str]]] = [
        ("Domain", [brief.domain] if brief.domain else []),
        ("Required research lanes", lane_descriptions),
        ("Required outputs", _compact_list(brief.required_outputs)),
        ("Critic quality gates", _compact_list(brief.quality_gates)),
        ("Constraints", _compact_list(brief.constraints)),
    ]
    lines: list[str] = []
    for title, values in sections:
        if not values:
            continue
        if len(values) == 1 and title == "Domain":
            lines.append(f"{title}: {values[0]}")
            continue
        lines.append(f"{title}:")
        lines.extend(f"- {value}" for value in values)
    return "\n".join(lines)


def _default_profile() -> DomainProfile:
    """Return the YAML-backed default profile.

    Domain-specific keyword-matched profiles were removed deliberately. Their
    plain-string research_lanes coerced to LaneSpec entries with empty
    system_prompts, which produced unspecialized lane researchers — the exact
    failure mode users complained about. The Designer LLM is now the only
    authoritative source of per-lane specialization; this default is only for
    legacy callers that provide no design_brief at all.
    """
    strategy = load_designer_architect_strategy()
    return strategy.default_profile


def _select_domain_profile(intent: str) -> DomainProfile:
    """Deprecated thin shim kept for backwards compatibility.

    Always returns the default profile — domain keyword matching was removed
    (see ``_default_profile``). The ``intent`` parameter is intentionally
    unused; callers may continue passing it for clarity.
    """
    del intent  # explicitly unused
    return _default_profile()
