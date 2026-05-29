"""Runtime-injected key registry for the dataflow checker.

These keys are present in workflow STATE / prompt-template context at runtime
**without** any node ``output_key`` producing them. They are injected by the
agent harness (template variables), the plan-and-execute runner (bookkeeping
appends), or derived from the runtime store. The dataflow checker (Pass A,
``dataflow_contracts.py``) seeds with this set so it does not false-positive on
harness/runtime-supplied reads.

Maintained next to the runtime that injects them so the seed set does not rot.
This is intentionally kept **separate** from the app-side
``_RUNTIME_TEMPLATE_KEYS`` (agent_designer/workflow_builder.py): that set also
feeds Designer lane-prompt coercion, so aliasing the two would change coercion
behavior. The inventories overlap by design; reconciliation is a follow-up.

NOTE: the per-iteration ``item_state_key`` (default ``current_step``) is **not**
here — plan_and_execute injects it into the loop-body scope only, so it is bound
locally in ``dataflow_contracts._resolve_pae`` rather than seeded globally.
"""
from __future__ import annotations

from databricks_deep_research.workflow.state import _RUNTIME_DERIVED_KEYS

# Harness-injected template variables (mirror of the app's _RUNTIME_TEMPLATE_KEYS
# inventory). Auto-rendered into prompts by the harness; never a node output_key.
_HARNESS_TEMPLATE_KEYS: frozenset[str] = frozenset(
    {
        "all_observations",
        "background",
        "completed_steps",
        "compute_namespace",
        "conversation_history",
        "current_date",
        "current_iso_datetime",
        "current_timezone",
        "fallback_discovery_sources",
        "file_context",
        "iteration",
        "max_steps",
        "max_words",
        "min_steps",
        "min_words",
        "observation",
        "page_contents",
        "plan_iterations",
        "plan_summary",
        "previous_observations",
        "reflector_feedback",
        "remaining_steps",
        "replan_budget",
        "revision_block_md",
        "research_depth",
        "search_results",
        "source_quality",
        "source_topics",
        "sources_count",
        "sources_list",
        "step_description",
        "step_prompt_guidance",
        "step_title",
        "step_type",
        "steps_completed",
        "steps_executed",
        "total_steps",
        "tool_catalog",
    }
)

# plan_execute_runner bookkeeping appends (runner :209-212, :419-428).
_RUNNER_BOOKKEEPING_KEYS: frozenset[str] = frozenset(
    {
        "observed_tool_kinds",
        "missing_required_tool_kind_groups",
        "last_blocked_step",
    }
)

# Root state key (every workflow receives the user query). NOTE: this is a
# baseline; the checker additionally seeds ``definition.required_inputs`` at the
# root (which normally includes "query" and any other declared required inputs).
_ROOT_KEYS: frozenset[str] = frozenset({"query"})

RUNTIME_INJECTED_KEYS: frozenset[str] = (
    _ROOT_KEYS | _RUNTIME_DERIVED_KEYS | _HARNESS_TEMPLATE_KEYS | _RUNNER_BOOKKEEPING_KEYS
)
