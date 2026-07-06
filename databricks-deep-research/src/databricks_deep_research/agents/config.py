"""Agent node configuration models and subtype defaults."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from databricks_deep_research.workflow.conditions import Condition, ConditionBranch
from databricks_deep_research.workflow.definition import WorkflowNode

# ---------------------------------------------------------------------------
# Report tone
# ---------------------------------------------------------------------------


class Tone(StrEnum):
    """Self-describing writing tone for synthesized reports.

    The PATTERN is ported from gpt-researcher: each member's *value* carries
    its own parenthetical definition, so the enum is the glossary — there is no
    separate lookup table to keep in sync. ``directive()`` returns the value
    verbatim for injection into a generation prompt; the parenthetical tells the
    model precisely what the tone means, removing the need for the model to guess.
    """

    OBJECTIVE = "Objective (impartial and unbiased presentation of facts and findings)"
    FORMAL = "Formal (adheres to academic standards with sophisticated language and structure)"
    ANALYTICAL = "Analytical (critical evaluation and detailed examination of data and theories)"
    PERSUASIVE = "Persuasive (convincing the audience of a particular viewpoint or argument)"
    INFORMATIVE = "Informative (providing clear and comprehensive information on a topic)"
    EXPLANATORY = "Explanatory (clarifying complex concepts and processes)"
    DESCRIPTIVE = "Descriptive (detailed depiction of phenomena, experiments, or case studies)"
    CRITICAL = "Critical (judging the validity and relevance of the research and its conclusions)"
    COMPARATIVE = "Comparative (juxtaposing different theories, data, or methods to highlight differences and similarities)"
    SPECULATIVE = "Speculative (exploring hypotheses and potential implications or future directions)"
    REFLECTIVE = "Reflective (considering the research process and personal insights or experiences)"
    NARRATIVE = "Narrative (telling a story to illustrate research findings or methodologies)"
    HUMOROUS = "Humorous (light-hearted and engaging, usually to make the content more relatable)"
    OPTIMISTIC = "Optimistic (highlighting positive findings and potential benefits)"
    PESSIMISTIC = "Pessimistic (focusing on limitations, challenges, or negative outcomes)"
    SIMPLE = "Simple (written for young readers, using basic vocabulary and clear explanations)"
    CASUAL = "Casual (conversational and relaxed style for everyday reading)"

    def directive(self) -> str:
        """Return the self-describing tone directive for prompt injection."""
        return self.value

    @classmethod
    def from_name(cls, name: str | None) -> Tone | None:
        """Parse a short tone NAME (e.g. "objective") into a ``Tone``.

        Accepts the lowercase member name as exposed to API callers / the
        frontend dropdown. Returns ``None`` for an empty or unrecognized value
        so an unknown tone degrades to unchanged synthesis rather than raising.
        """
        if not name:
            return None
        key = name.strip().upper()
        return cls.__members__.get(key)

    @classmethod
    def names(cls) -> list[str]:
        """Return the lowercase member names (the API/dropdown enum values)."""
        return [member.name.lower() for member in cls]


# ---------------------------------------------------------------------------
# Pool-related configs
# ---------------------------------------------------------------------------


class PoolWriteConfig(BaseModel):
    """Describes how an agent writes items to a shared pool."""

    model_config = ConfigDict(extra="forbid")
    pool: str
    extract: str  # Jinja / dot-path expression on agent output
    transform: str | None = None  # Optional transformation template


class PromptCompactionConfig(BaseModel):
    """Controls prompt-context compaction for injected pool content."""

    model_config = ConfigDict(extra="forbid")
    mode: Literal["none", "dedupe", "summarize", "auto"] = "none"
    max_total_chars: int = 0
    target_chars: int = 0
    summary_model_tier: str = "simple"
    dedupe_key: Literal["auto", "url", "title", "text"] = "auto"


class PoolInjectConfig(BaseModel):
    """Describes how pool contents are injected into an agent prompt."""

    model_config = ConfigDict(extra="forbid")
    pool: str
    threshold: float = 0.0  # BM25 / relevance threshold
    format: Literal["text", "json", "markdown"] = "text"
    max_items: int = 20  # max items to inject
    max_item_chars: int = 0  # 0 = unlimited; >0 truncates each item
    compaction: PromptCompactionConfig | None = None


class SynthesisContextFieldConfig(BaseModel):
    """Controls how one synthesizer context field is populated.

    The compilation pipeline applies a three-tier preservation policy driven by
    the fields below:

    1. The first ``keep_full_top_k`` items are always passed through verbatim
       (bounded only by ``per_item_hard_cap`` if set).
    2. Items past ``keep_full_top_k`` are kept in full as long as the running
       total character count stays within ``total_budget_chars``.
    3. Items that would overflow the budget are handled per
       ``truncation_policy``: ``soft_tail`` trims only the overflowing tail
       item, ``compact`` runs an LLM summariser on overflow items (when an
       LLM is available), and ``hard_clip`` reverts to legacy
       per-item ``max_item_chars`` truncation.
    """

    model_config = ConfigDict(extra="forbid")
    max_items: int = 20
    max_item_chars: int = 0
    compaction: PromptCompactionConfig | None = None
    # Budget-based preservation controls (see class docstring).
    keep_full_top_k: int = 0
    total_budget_chars: int = 0
    truncation_policy: Literal["soft_tail", "compact", "hard_clip"] = "soft_tail"
    per_item_hard_cap: int = 0
    # Sources-only rendering knobs; harmless for observations.
    include_snippet: bool = True
    include_content: bool = True
    max_content_chars_top_k: int = 5000
    max_content_chars_other: int = 1500


class SynthesisContextConfig(BaseModel):
    """Controls synthesizer-specific context materialization."""

    model_config = ConfigDict(extra="forbid")
    observations: SynthesisContextFieldConfig | None = None
    sources: SynthesisContextFieldConfig | None = None
    fallback_discovery_sources: SynthesisContextFieldConfig | None = None


class ToolOutputBudgetConfig(BaseModel):
    """Budget policy for offloading large tool outputs to the compute scratchpad.

    Drives the "MemEx-first tool I/O" lever (spec §1.1): when a non-builtin
    research tool returns text larger than the threshold, the full result is
    stored as a Python object in the compute namespace and the model-visible
    content is replaced with a compact preview + a handle the model can operate
    on via ``compute`` code. All fields are optional-with-default; the feature
    is gated by ``AgentNodeConfig.tool_output_offload`` (default ``"off"``).
    """

    model_config = ConfigDict(extra="forbid")
    externalize_min_chars: int = 12000
    preview_head_chars: int = 2000
    preview_tail_chars: int = 1000
    exempt_tools: list[str] = Field(
        default_factory=lambda: ["read_file", "compute", "compute_namespace_list"]
    )
    # Per-tool threshold overrides (tool name -> externalize_min_chars).
    tool_overrides: dict[str, int] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Primary agent node config
# ---------------------------------------------------------------------------

# Default tool-call budget the ``long_horizon`` profile (spec §2.4) applies when
# a node does not set ``max_tool_calls`` explicitly.
_LONG_HORIZON_MAX_TOOL_CALLS = 100


class AgentNodeConfig(BaseModel):
    """Full configuration for a single agent (LLM) node."""

    model_config = ConfigDict(extra="forbid")

    subtype: str  # coordinator, planner, researcher, reflector, synthesizer, evaluator
    model_tier: str = "analytical"
    # Optional model FAMILY (orthogonal to model_tier): pins this node's LLM to a
    # configured family (e.g. "claude"/"llama") regardless of tier. Resolved by
    # FrameworkLLMClient against its model_families catalog; None => tier routing.
    # Used for multi-model ensembles (e.g. iterative_refinement proposers).
    model_family: str | None = None
    system_prompt: str = ""
    user_prompt_template: str = ""
    input_keys: list[str] = Field(default_factory=list)
    output_key: str = "output"
    output_mode: str = "text"  # text, json, structured
    output_format: str = "text"  # text, markdown, json
    output_schema: dict[str, Any] | None = None
    grounding_mode: Literal["none", "classical_lite", "reclaim"] | None = None
    tools: list[str | dict[str, Any]] = Field(default_factory=list)
    # Attached skill names (Feature 2.2 runtime wiring — closes the deferred
    # skills-attach item noted under ``profile`` below). Each name resolves from
    # the wired SkillStore (``ctx.extras["_skill_store"]``); a metadata-ONLY
    # section is injected into the system prompt and the agent pulls full bodies
    # on demand via ``read_skill`` (auto-attached when this is non-empty). Empty
    # (default) => byte-identical to today (no skills section, no read_skill).
    skills: list[str] = Field(default_factory=list)
    # Per-agent gate for executing skill SCRIPTS (Feature 2.2 — A2). When an
    # attached skill bundles named ``scripts``, the agent may run one via the
    # auto-attached ``run_skill_script`` tool ONLY when BOTH this flag AND the
    # global ``skills.allow_script_execution`` (surfaced via
    # ``ctx.extras["_skill_scripts_enabled"]``) are True. Default False =>
    # byte-identical to today: no ``run_skill_script`` tool is attached and skill
    # scripts are never executed. Scripts run in a hardened out-of-process sandbox
    # (rlimits + scrubbed env + SIGKILL); reading a skill BODY (``read_skill``) is
    # unaffected by this flag.
    allow_skill_scripts: bool = False
    # Attached MCP server NAMES (Feature 4.3 / B2). Each name must match a
    # workflow-level ``mcp_servers`` entry; at runtime the host injects that
    # server's discovered tools for this agent via a resolver override. Agents
    # bind to SERVERS (not individual tool names) so the declared-tools validator
    # (``validate_agent_tools``) is untouched — discovered MCP tool names are not
    # statically known at author time. Empty (default) => no MCP tools attached.
    mcp_servers: list[str] = Field(default_factory=list)
    # Static vocabulary hints merged into the source-admission query
    # profile. Intended for researcher-subtype nodes that drive tool
    # selection from their own system prompt rather than a planner step;
    # the terms should be capability-level vocabulary (e.g. "competitor
    # battle card", "vendor comparison"), not customer or competitor
    # names. Consumed by
    # databricks_deep_research.agents.source_aware.admit_tool_result
    # when the ``ADMISSION_ENFORCE_NODE_HINTS`` flag is set.
    hint_queries: list[str] = Field(default_factory=list)
    pool_writes: list[PoolWriteConfig] = Field(default_factory=list)
    pool_tools: list[str] = Field(default_factory=list)
    max_tool_calls: int | None = None
    per_tool_limits: dict[str, int] | None = None
    max_retries: int = 2
    max_result_chars: int = 4000  # 0=unlimited; >0 truncates old tool results
    compaction_strategy: Literal["truncate", "mask"] = "truncate"
    keep_intact_iterations: int = 3  # Recent tool-calling iterations to keep uncompacted
    # MemEx-first tool I/O (spec §1.1): offload large non-builtin tool results to
    # the compute scratchpad as Python objects; the model sees a preview + handle.
    # Default "off" => byte-identical behavior to today.
    tool_output_offload: Literal["off", "preview", "auto"] = "off"
    tool_output_budget: ToolOutputBudgetConfig = Field(default_factory=ToolOutputBudgetConfig)
    # MemEx code-action bridge (spec §1.4). When ``action_mode`` is ``"code"`` /
    # ``"hybrid"``, the Cell may emit Python that calls its allowlisted research
    # tools AS FUNCTIONS inside the ``compute`` sandbox and ``submit()`` a typed
    # result. ``code_action_tools`` is the per-Cell ALLOWLIST: only listed tool
    # names get a sandbox closure; ``None`` => no tools are bridged. DEFAULT
    # ``action_mode="tools"`` is byte-identical to today: no closures injected,
    # no ``submit``. SECURITY: this is the highest-scrutiny knob in the roadmap —
    # closures route through the same gating spine as JSON tool calls (HITL,
    # per-tool budget, admission, pool/source writes, tracing) and capture
    # nothing the model can reach (see ``agents/code_action.py``).
    action_mode: Literal["tools", "code", "hybrid"] = "tools"
    code_action_tools: list[str] | None = None
    # MemEx GOVERNED spawn_agent (spec §3.3). A code-action Cell may spawn a
    # DECLARED child subworkflow from sandbox Python, within Designer-declared
    # bounds. SECURITY: OPT-IN, OFF BY DEFAULT. ``spawn_agent`` is injected into
    # the sandbox ONLY when ``action_mode in ("code","hybrid")`` AND
    # ``spawnable_subagents`` is non-empty AND ``spawn_budget > 0``. With the
    # defaults below (empty dict, budget 0) NO ``spawn_agent`` is injected and the
    # path is byte-identical to a code-action Cell without spawn.
    #
    # ``spawnable_subagents`` maps a spawn NAME -> an inline WorkflowDefinition
    # dump (the DECLARED subagent the Cell may spawn). The graph/Designer fixes
    # this set; the model can only spawn these names (re-validated from this
    # framework-owned dict at call time — a model-constructed name is rejected).
    spawnable_subagents: dict[str, dict[str, Any]] = Field(default_factory=dict)
    # Max TOTAL spawns per Cell run (a hard cap). ``0`` => spawning DISABLED
    # (the default). The budget increments once per spawn ATTEMPT (a failed spawn
    # still counts) so a denied/erroring spawn cannot drive a retry-storm.
    spawn_budget: int = 0
    # Concurrency ceiling for any future parallel fan-out. v1 ``spawn_agent``
    # calls block SEQUENTIALLY from the sandbox worker thread (one at a time), so
    # this is enforced as a guard but is effectively the ceiling for a deferred
    # parallel-fan-out enhancement — see ``agents/code_action.py`` spawn closure.
    max_concurrent_spawns: int = 4
    # History compaction value-rank (spec §1.3): when True, compaction never
    # discards high-value (accepted) tool results before low-value (unaccepted)
    # ones — it ranks candidates lowest-value-first and compacts those first.
    # Default True is safe: with no per-message value metadata the compactor
    # falls back to the byte-identical §1.2 mask/truncate ladder.
    evidence_rescue: bool = True
    # Total character budget the rescue compactor targets across all old tool
    # results. ``0`` => preserve current per-message ``max_result_chars``
    # budgeting (no new budget enforcement).
    compaction_budget_chars: int = 0
    # Long-horizon Cell profile (spec §2.4): a named preset activating the
    # offload + evidence-rescue + large-budget knobs for "reason for a very long
    # time" agents. Expanded by _apply_long_horizon_profile below; explicit
    # per-field values always win. The todos / VFS-checkpoint / deep-research
    # skill-attach behaviors need runtime wiring and are deferred (see
    # .omc/progress.txt), consistent with the deferred skills loop injection.
    profile: Literal["default", "long_horizon"] = "default"
    # RAG-over-tools / deferred tools (spec §5.5, Tier-3). When the tool catalog
    # is large (e.g. a big MCP server surfaced by §4.3), listing every tool's
    # full JSON Schema crowds the prompt. With deferral engaged, deferred tools
    # are listed by NAME + a one-line description only; the model fetches full
    # schemas on demand via the auto-injected ``tool_search`` builtin.
    # Optional-with-default: deferral engages only when ``defer_tools`` is True
    # OR the exposed catalog size exceeds ``defer_tool_threshold`` (when > 0).
    # With the defaults below (False / 0) the path is byte-identical to today —
    # every tool is listed with its full schema and no ``tool_search`` exists.
    defer_tools: bool = False
    defer_tool_threshold: int = 0
    dedup_jaccard_threshold: float = 0.8  # Jaccard word overlap threshold for near-dedup (0.0-1.0)
    force_convergence: bool = False  # Gate novelty/anti-loop heuristics (convergence, nudges)
    convergence_rounds: int = 4  # Zero-novel rounds before forced convergence (requires force_convergence=True)
    # Claude's API default cap is 8192 output tokens when unset, which the
    # planner/reflector regularly bump into producing structured plans. Bump
    # the default so callers don't have to remember to set it per-agent.
    # Overridable via YAML at any node. Claude Sonnet supports up to 64000,
    # Opus up to 32000; Haiku is hard-capped at 8192 regardless.
    conversation_budget: int | None = 32000
    pool_inject: list[PoolInjectConfig] = Field(default_factory=list)
    synthesis_context: SynthesisContextConfig | None = None
    output_model: Any = None  # Pydantic model class for structured output
    # Per-run report-style knobs (prompts-over-knobs). Both default ``None`` =>
    # byte-identical generation to today; when set they are injected into the
    # synthesizer's generation instructions AFTER the hard numeric/unit citation
    # rules (never replacing them). ``tone`` is a self-describing ``Tone`` member;
    # ``output_language`` is a free-form language name (e.g. "Spanish") re-forced
    # in the drift-prone Stage-2 generation sub-call.
    tone: Tone | None = None
    output_language: str | None = None
    # Reserved-prefix namespace for framework-attached runtime capabilities
    # (approval broker, virtual filesystem, todos store, checkpointer, thread_id).
    # Keys prefixed with ``_framework_`` are reserved for framework use; user-chosen
    # keys MUST NOT use this prefix. Pydantic ``extra="forbid"`` is preserved at the
    # model level — only this explicitly declared field accepts the dict.
    extras: dict[str, Any] = Field(default_factory=dict)

    @field_validator("per_tool_limits")
    @classmethod
    def _validate_per_tool_limits(cls, v: dict[str, int] | None) -> dict[str, int] | None:
        if v is not None:
            for name, limit in v.items():
                if not isinstance(limit, int) or limit < 0:
                    raise ValueError(
                        f"per_tool_limits values must be non-negative integers, got {name}={limit!r}"
                    )
        return v

    @field_validator("spawn_budget")
    @classmethod
    def _validate_spawn_budget(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"spawn_budget must be a non-negative integer, got {v!r}")
        return v

    @field_validator("max_concurrent_spawns")
    @classmethod
    def _validate_max_concurrent_spawns(cls, v: int) -> int:
        if v < 1:
            raise ValueError(
                f"max_concurrent_spawns must be a positive integer, got {v!r}"
            )
        return v

    @model_validator(mode="after")
    def _apply_long_horizon_profile(self) -> AgentNodeConfig:
        """Expand the ``long_horizon`` profile preset (spec §2.4).

        Activates the long-horizon Cell knobs — scratchpad offload + a large
        tool-call budget — but ONLY for fields the caller did not set explicitly
        (``model_fields_set``), so an explicit per-field value always wins.
        ``evidence_rescue`` already defaults True. The todos / VFS-checkpoint /
        deep-research-skill-attach behaviors need runtime wiring and are deferred
        (see .omc/progress.txt). ``profile="default"`` is a no-op.
        """
        if self.profile == "long_horizon":
            if "tool_output_offload" not in self.model_fields_set:
                self.tool_output_offload = "auto"
            if "max_tool_calls" not in self.model_fields_set:
                self.max_tool_calls = _LONG_HORIZON_MAX_TOOL_CALLS
        return self


# ---------------------------------------------------------------------------
# Non-agent node configs
# ---------------------------------------------------------------------------


class ToolRefConfig(BaseModel):
    """Typed legacy tool reference descriptor used by tool nodes."""

    model_config = ConfigDict(extra="forbid")
    type: str
    name: str

    @model_validator(mode="before")
    @classmethod
    def _default_legacy_type(cls, value: Any) -> Any:
        if isinstance(value, dict) and "name" in value and "type" not in value:
            return {**value, "type": "builtin"}
        return value

    def get(self, key: str, default: Any = None) -> Any:
        """Small compatibility shim for older runtime code that treated refs as dicts."""
        return getattr(self, key, default)


class ToolNodeConfig(BaseModel):
    """Configuration for a pure-tool (no LLM) node."""

    model_config = ConfigDict(extra="forbid")
    ref: ToolRefConfig  # tool reference descriptor
    input_mapping: dict[str, str] = Field(default_factory=dict)
    input_literals: dict[str, Any] = Field(default_factory=dict)
    output_key: str = "tool_result"
    output_data_key: str | None = None  # where ToolResult.data (+ success/error) lands
    bind_namespace: str | None = None  # also inject the result into the compute namespace
    # False preserves pre-existing semantics (failed ToolResults are stored, not
    # raised); newly authored nodes should set True to engage error_handling.
    fail_on_error: bool = False
    enforce_output_schema: bool = False  # opt-in required-keys check of the output
    output_schema: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _no_arg_collisions(self) -> ToolNodeConfig:
        overlap = sorted(set(self.input_mapping) & set(self.input_literals))
        if overlap:
            raise ValueError(
                f"input_mapping and input_literals define the same argument(s): {overlap}"
            )
        return self


class LoopNodeConfig(BaseModel):
    """Configuration for a loop control node."""

    model_config = ConfigDict(extra="forbid")
    until: Condition  # Serialised Condition (StateCondition / LLMCondition / Composite)
    min_iterations: int = Field(
        default=1, description="Minimum loop passes before the until-condition can stop the loop."
    )
    max_iterations: int = Field(
        default=10,
        description="Hard cap on loop passes (raise for deeper iterative refinement). "
        "Scaled by the agent's effort level at runtime.",
    )


class ConditionalNodeConfig(BaseModel):
    """Configuration for a conditional branching node."""

    model_config = ConfigDict(extra="forbid")
    conditions: list[ConditionBranch]  # list of serialised ConditionBranch
    default_branch: int = 0

    @field_validator("conditions", mode="before")
    @classmethod
    def _wrap_legacy_condition_branches(cls, value: Any) -> Any:
        if not isinstance(value, list):
            return value
        wrapped: list[Any] = []
        for index, item in enumerate(value):
            if isinstance(item, dict) and "condition" not in item:
                wrapped.append({"condition": item, "child_index": index})
            else:
                wrapped.append(item)
        return wrapped


class SubworkflowNodeConfig(BaseModel):
    """Configuration for invoking a nested workflow."""

    model_config = ConfigDict(extra="forbid")
    ref: str  # workflow name or path
    params: dict[str, Any] = Field(default_factory=dict)
    input_mapping: dict[str, str] = Field(default_factory=dict)
    output_mapping: dict[str, str] = Field(default_factory=dict)
    output_key: str = "subworkflow_result"
    pool_mode: str = "inherit"  # inherit, isolate, merge
    # Recursion guard: max nesting depth for subworkflow-within-subworkflow. The
    # executor increments a per-instance counter on each descent and raises a
    # WorkflowError once this is exceeded, preventing unbounded/cyclic nesting.
    max_subworkflow_depth: int = 5
    # Self-contained embedded sub-workflow: a WorkflowDefinition dump produced by
    # api/compile.py when a SubAgent is compiled to a subworkflow node. Typed as a
    # raw dict to decouple this config from re-validating the nested definition on
    # every load; the P2 subworkflow executor parses it via
    # WorkflowDefinition.model_validate(...) once subworkflow execution lands.
    inline: dict[str, Any] | None = None


class PlanAndExecuteNodeConfig(BaseModel):
    """Configuration for the plan-and-execute meta-node."""

    model_config = ConfigDict(extra="forbid")
    planner: dict[str, Any]  # Serialised AgentNodeConfig for the planner agent
    items_path: str = "steps"  # dot-path into planner output for the iterable
    item_state_key: str = "current_step"
    body: WorkflowNode | None = None  # Serialised child node(s) to run per item
    evaluator: dict[str, Any] | None = None  # Optional evaluator agent config
    max_iterations: int = Field(
        default=10,
        description="Max research steps executed across all replan cycles (raise to dig "
        "deeper into multi-step questions). Scaled by the agent's effort level at runtime.",
    )
    min_iterations: int = Field(
        default=1, description="Minimum research steps before the evaluator may declare completion."
    )
    max_replan_cycles: int = 3
    complete_on_exhaustion: bool = True
    planner_guidance: str = ""  # Free-text guidance injected into planner prompt
    synthesis_metadata: dict[str, str] = Field(default_factory=dict)  # Key-value pairs written to state for synthesizer
    required_tool_kind_groups: list[list[str]] = Field(default_factory=list)
    # Each inner list is an OR group; every group must be observed before an
    # evaluator "complete" decision is accepted.


# ---------------------------------------------------------------------------
# Subtype defaults
#
# NOTE: ``input_keys`` values below are documentation only — at runtime,
# input_keys are auto-detected from prompt templates by the harness
# (``execute_agent()`` in ``harness.py``) using
# ``SafeTemplateRenderer.extract_variables()``.  Explicit ``input_keys``
# in YAML workflow definitions always override auto-detection.
# ---------------------------------------------------------------------------

SUBTYPE_DEFAULTS: dict[str, dict[str, Any]] = {
    "coordinator": {
        "model_tier": "simple",
        "output_key": "coordination",
        "output_format": "json",
        "input_keys": ["query"],
        "tools": [],
        "pool_writes": [],
        "pool_tools": [],
    },
    "planner": {
        "model_tier": "analytical",
        "output_key": "plan",
        "output_format": "json",
        "input_keys": ["query", "background"],
        "tools": [],
        "pool_writes": [],
        "pool_tools": [],
    },
    "researcher": {
        "model_tier": "analytical",
        "output_key": "findings",
        "output_format": "json",
        "input_keys": ["query", "current_step", "plan"],
        "tools": ["web_search", "web_crawl"],
        "pool_writes": [{"pool": "sources", "extract": "sources"}],
        "pool_tools": ["pool_search"],
    },
    "reflector": {
        "model_tier": "analytical",
        "output_key": "reflection",
        "output_format": "json",
        "input_keys": [
            "query", "plan_summary", "findings", "current_step",
            "remaining_steps", "total_steps", "steps_completed",
            "min_steps", "step_title", "iteration", "observation",
            "all_observations", "sources_count", "source_topics",
            "source_quality",
        ],
        "tools": [],
        "pool_writes": [],
        "pool_tools": [],
    },
    "synthesizer": {
        "model_tier": "complex",
        "output_key": "report",
        "output_format": "markdown",
        "input_keys": ["query", "plan"],
        "tools": [],
        "pool_writes": [],
        "pool_tools": ["pool_search"],
    },
    "evaluator": {
        "model_tier": "analytical",
        "output_key": "evaluation",
        "output_format": "json",
        "input_keys": [
            "query", "plan_summary", "findings", "current_step",
            "remaining_steps", "total_steps", "steps_completed",
            "min_steps", "step_title", "iteration", "observation",
            "all_observations", "sources_count", "source_topics",
            "source_quality",
        ],
        "tools": [],
        "pool_writes": [],
        "pool_tools": [],
    },
}
