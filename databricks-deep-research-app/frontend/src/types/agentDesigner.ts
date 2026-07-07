/**
 * Types for Agent Designer V2 — registry, agents_v2 CRUD, and SSE chat stream.
 *
 * SSE wire format produced by the backend orchestrator:
 *   event: <type>
 *   data: <json>
 *
 * AST and ValidationError are defined by US-302 in @/types/ast.
 * Import them with `import type { AST, ValidationError } from '@/types/ast'`.
 */

import type { AST, ValidationError } from '@/types/ast'

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

/**
 * Metadata for a single workflow node type, returned by GET /registry.
 * Shape derived from registry.py::node_types_payload().
 */
export interface NodeTypeSpec {
  /** Internal node type identifier (e.g. "agent", "sequence"). */
  type: string
  label: string
  icon: string
  category: string
  is_composite: boolean
  /** JSON Schema object for the node's config model (null when no config model). */
  config_schema: Record<string, unknown> | null
  default_config?: Record<string, unknown>
  summary_template?: string
  children_kind?: 'ordered_list' | 'index_paired_branches' | 'named_slots' | string
  children_slots?: Array<Record<string, unknown>>
  branches_pairing?: Record<string, unknown>
}

/**
 * Metadata for a tool kind, returned by GET /registry.
 * Shape derived from registry.py::tool_kinds_payload().
 */
export interface ToolKindSpec {
  /** ToolKind enum value (e.g. "web_search", "vector_index"). */
  kind: string
  label: string
  icon: string
  layer?: string
  config_schema?: Record<string, unknown>
  discoverable?: boolean
  discovery_path?: string | null
}

/** Source kind descriptor returned by the Designer registry. */
export interface SourceKindSpec {
  kind: string
  label: string
  source_type: string
  icon: string
}

export interface DesignerResource {
  kind: string
  source_id?: string | null
  name: string
  full_name?: string | null
  description?: string | null
  status?: string | null
  capabilities: string[]
  metadata: Record<string, unknown>
}

export interface DesignerBrowseError {
  /** 'permission' | 'not_found' | 'other' — drives the picker's message. */
  code: string
  message: string
}

export interface DesignerResourcesResponse {
  resources: DesignerResource[]
  total: number
  /** Set when a Unity Catalog browse failed (e.g. no BROWSE on the catalog). */
  error?: DesignerBrowseError | null
  /** Partial-result caveat from a catalog-wide function search (truncation). */
  warning?: string | null
}

export type DesignerAssetKind =
  | 'vector_index'
  | 'delta_table'
  | 'genie_space'
  | 'knowledge_assistant'
  | 'serving_endpoint'
  | 'sql_warehouse'

export type DesignerAssetUsage = 'required' | 'preferred' | 'available'

export interface DesignerAsset {
  kind: DesignerAssetKind
  full_name?: string | null
  source_id?: string | null
  name?: string | null
  description?: string | null
  usage?: DesignerAssetUsage
  role?: string | null
  field_roles?: Record<string, string>
  metadata?: Record<string, unknown>
}

/**
 * Agent subtype descriptor (coordinator, planner, researcher, …).
 * Shape derived from registry.py::AGENT_SUBTYPES.
 */
export interface AgentSubtype {
  id: string
  label: string
  icon: string
  default_model_tier?: string
  default_prompt_template_id?: string | null
}

/** Model tier identifier configured by the backend model catalog. */
export type ModelTier = string

/** Full registry payload returned by GET /api/v1/agent-designer/registry. */
export interface RegistryResponse {
  node_types: NodeTypeSpec[]
  agent_subtypes: AgentSubtype[]
  tool_kinds: ToolKindSpec[]
  model_tiers: string[]
  query_modes?: string[]
  research_depths?: string[]
  source_kinds?: SourceKindSpec[]
  version: string
}

// ---------------------------------------------------------------------------
// Agents V2 CRUD schemas
// ---------------------------------------------------------------------------

/** Visibility values allowed by the backend (UserSettableVisibility + system). */
export type AgentVisibility = 'private' | 'workspace' | 'system'

/** POST /api/v1/agents-v2 request body. */
export interface CreateAgentV2Request {
  name: string
  description?: string | null
  avatar_url?: string | null
  visibility?: 'private' | 'workspace'
  /** Workflow AST — arbitrary JSON object validated by the framework loader. */
  definition: Record<string, unknown>
}

/** PATCH /api/v1/agents-v2/{id} request body (partial update). */
export interface UpdateAgentV2Request {
  name?: string | null
  description?: string | null
  avatar_url?: string | null
  visibility?: 'private' | 'workspace' | null
  definition?: Record<string, unknown> | null
}

/** Full agent representation returned by create / get / update. */
export interface AgentV2Response {
  id: string
  owner_id: string
  name: string
  description: string | null
  avatar_url: string | null
  visibility: AgentVisibility
  definition: Record<string, unknown>
  schema_version: number
  etag: string
  created_at: string
  updated_at: string
  /**
   * Workflow validation result returned after a save. Present when the
   * backend validator ran; absent when skipped or not configured.
   * The shape matches {@link import('@/api/agentsV2').WorkflowValidationResult}.
   */
  validation?: import('@/api/agentsV2').WorkflowValidationResult | null
  /**
   * True when an advisory validation for the CURRENT definition is still
   * running in the background (cache miss on save). The verdict is delivered
   * asynchronously: poll GET /agents-v2/{id} until this is false, then read
   * `validation`. Distinguishes "pending" from "not applicable" (both null).
   */
  validation_pending?: boolean
}

/** Lightweight summary item returned in list responses. */
export interface AgentV2Summary {
  id: string
  name: string
  description: string | null
  visibility: AgentVisibility
  owner_id: string
  updated_at: string
  node_count: number
  /** True when the agent has an active in_app deployment. Drives chat picker. */
  in_app_active: boolean
  /**
   * The agent's authored default for NLI "verify sources" (True if its
   * synthesizer uses grounding_mode='reclaim'). The composer seeds the verify
   * toggle from this; defaults to true if the backend omits it.
   */
  default_verify_sources?: boolean
  /**
   * True when the agent carries a declarative UI (definition.surface). The
   * chat shows its Agent UI panel; the surface itself is read from the detail
   * endpoint's definition.
   */
  has_surface?: boolean
}

/** GET /api/v1/agents-v2 response. */
export interface AgentV2ListResponse {
  items: AgentV2Summary[]
  total: number
}

// ---------------------------------------------------------------------------
// Agent Designer chat types
// ---------------------------------------------------------------------------

/**
 * OpenAI-compatible tool call object embedded in assistant messages.
 */
export interface ToolCall {
  id: string
  type: 'function'
  function: {
    name: string
    arguments: string
  }
}

/**
 * One milestone in the per-turn "Designer activity" feed, accumulated from the
 * transient `progress` SSE events so the user sees what the designer did
 * (architect → critic → revisions → finalizing) instead of a single flickering
 * status line. UI-only — never sent back to the API (see ChatMessage.activity).
 */
export interface ActivityStep {
  label: string
  iteration?: number | null
  total?: number | null
}

/**
 * Chat message shape accepted by POST /api/v1/agent-designer/chat.
 * Mirrors the backend ChatMessage Pydantic model.
 */
export interface ChatMessage {
  role: 'user' | 'assistant' | 'tool'
  content: string
  tool_calls?: ToolCall[] | null
  tool_call_id?: string | null
  /**
   * UI-only label for tool result rendering. The backend ChatMessage schema
   * has extra="forbid", so API clients must strip this before POSTing chat
   * history back to /agent-designer/chat.
   */
  tool_name?: string | null
  /**
   * UI-only display artifacts attached to an assistant turn from the
   * `architect_synopsis` / `critic_review` SSE events. Like `tool_name`, the
   * backend ChatMessage schema is extra="forbid", so `chatStream`'s wire
   * whitelist strips these before POSTing chat history back — they never reach
   * the request payload (so they cannot bloat it or trip the byte cap).
   */
  synopsis?: ArchitectSynopsis | null
  review?: CriticReview | null
  /**
   * UI-only accumulated activity feed for this assistant turn (from `progress`
   * SSE events). Stripped before POSTing chat history back — like `synopsis` /
   * `review`, the wire builder in `chatStream` reconstructs each message with a
   * fixed key set, so this never reaches the request payload.
   */
  activity?: ActivityStep[] | null
}

// ---------------------------------------------------------------------------
// SSE event discriminated union
//
// Wire format (from _format_sse in agent_designer.py):
//   event: <type>
//   data: <json of model_dump(exclude={"type"})>
//
// Backend models live in orchestrator.py.
// ---------------------------------------------------------------------------

/** LLM produced a text delta. */
export interface MessageSSEEvent {
  type: 'message'
  /** Incremental text content from the LLM. */
  content: string
}

/** LLM invoked a tool. */
export interface ToolCallSSEEvent {
  type: 'tool_call'
  tool_name: string
  tool_call_id: string
  args: Record<string, unknown>
}

/**
 * One deterministic auto-repair the designer-side normalizer (Layer 2)
 * applied to the architect-emitted AST before downstream nodes saw it.
 *
 * See `.omc/plans/designer-hardening.md` (Layer 2) for the full taxonomy.
 * The frontend renders these as an amber "auto-fixed" pill on the chat
 * turn — see `NormalizationFixPill.tsx`.
 */
export interface NormalizationFix {
  /**
   * One of: subtype_rewrite | tier_rewrite | tool_kind_rewrite |
   * auto_bind_retrieval | auto_declare_pool | set_minimum_max_tool_calls.
   * UI components should fall back gracefully on unknown kinds (forward-
   * compat for future normalizer rules).
   */
  kind: string
  /** Dot-path address into the AST (e.g. `root.children.1.config.subtype`). */
  path: string
  /** Original value the architect emitted. Captured as `unknown` because it
   * can be string, list, number, etc. */
  before: unknown
  /** Rewritten value the normalizer applied. */
  after: unknown
  /** Single-sentence user-facing explanation of why the fix fired. */
  rationale: string
}

/**
 * A mutation tool produced a new AST.
 * `old_ast` is null when there was no previous AST (first propose_workflow call).
 */
export interface MutationProposedSSEEvent {
  type: 'mutation_proposed'
  /** Mutation tool that produced this AST. Older streams may omit it. */
  tool_name?: string
  tool_call_id: string
  old_ast: AST | null
  new_ast: AST
  validation_errors: ValidationError[] | string[]
  summary: { node_count: number; tool_count: number; source_count: number } | null
  /**
   * Layer 2 auto-repair records. Empty list when the architect emitted a
   * clean AST (the common case once Layer 4 prompt guardrails settle in).
   * Older backends may omit the field entirely — treat undefined as `[]`.
   */
  normalization_fixes?: NormalizationFix[]
}

/** A non-mutating tool returned a result (discover_sources, list_node_types, …). */
export interface ToolResultSSEEvent {
  type: 'tool_result'
  tool_call_id: string
  tool_name: string
  result: Record<string, unknown>
}

/** Orchestrator or LLM encountered an error. */
export interface ErrorSSEEvent {
  type: 'error'
  message: string
  tool_call_id?: string | null
}

/** Stream is complete. Always the final event. */
export interface DoneSSEEvent {
  type: 'done'
}

/**
 * Transient progress heartbeat emitted while a long designer turn streams
 * (current agent node, or architect-critic loop iteration). Rendered as a
 * transient status line and never persisted to the transcript.
 */
export interface ProgressSSEEvent {
  type: 'progress'
  label: string
  iteration?: number | null
  total?: number | null
}

/**
 * First frame of a chat turn: carries the server-assigned turn_id so the client
 * can reconnect (GET /chat/{turn_id}/events?since=N) and resume after the
 * gateway's absolute connection cap severs the stream. Not part of the buffered
 * event sequence (carries no id), so it is never replayed on resume.
 */
export interface TurnStartedSSEEvent {
  type: 'turn_started'
  turn_id: string
}

// ---------------------------------------------------------------------------
// Designer-review display payloads (additive). The backend derives these and
// emits them as their own SSE events; the frontend renders them as readable
// cards. Display-only — never re-sent on the wire (see ChatMessage above).
// ---------------------------------------------------------------------------

/** One researcher/agent lane in the workflow synopsis. */
export interface SynopsisLane {
  label: string
  tools: string[]
}

/** "What I built" payload. Mirrors the backend `ArchitectSynopsisEvent`. */
export interface ArchitectSynopsis {
  headline: string
  topology: string
  change_kind: 'created' | 'edited'
  lanes: SynopsisLane[]
  pipeline: string[]
  tools: string[]
  outputs: string[]
  warnings: string[]
  /** Present when the agent carries a declarative UI (definition.surface). */
  ui?: { components: number; actions: string[] } | null
}

export interface CriticAgentFinding {
  node_path?: string
  label?: string
  severity?: 'fail' | 'needs_revision' | 'minor'
  finding?: string
  suggested_action?: string
}

export interface CriticCoverageGap {
  aspect?: string
  rationale?: string
}

export interface CriticOutputGap {
  required_output?: string
  rationale?: string
}

/** "Designer review" payload. Mirrors the backend `CriticReviewEvent`
 *  (= workflow_critic.CritiqueResult). */
export interface CriticReview {
  verdict: 'pass' | 'needs_revision' | 'fail'
  summary: string
  agent_findings: CriticAgentFinding[]
  coverage_gaps: CriticCoverageGap[]
  output_gaps: CriticOutputGap[]
}

/**
 * Synopsis of what the designer built/changed this turn. Emitted once at
 * finalize; rendered as the "What I built" card. Display-only.
 */
export interface ArchitectSynopsisSSEEvent extends ArchitectSynopsis {
  type: 'architect_synopsis'
}

/**
 * The critic's structured verdict, surfaced as a first-class event instead of
 * buried in the `validate` tool result. Rendered as the "Designer review" card.
 */
export interface CriticReviewSSEEvent extends CriticReview {
  type: 'critic_review'
}

/** Discriminated union of all SSE events emitted by the agent-designer chat endpoint. */
export type DesignerSSEEvent =
  | TurnStartedSSEEvent
  | MessageSSEEvent
  | ToolCallSSEEvent
  | MutationProposedSSEEvent
  | ToolResultSSEEvent
  | ProgressSSEEvent
  | ArchitectSynopsisSSEEvent
  | CriticReviewSSEEvent
  | ErrorSSEEvent
  | DoneSSEEvent

// ---------------------------------------------------------------------------
// Validate endpoint
// ---------------------------------------------------------------------------

export interface ValidationErrorItem {
  message: string
  path: string | null
  line: number | null
  kind: 'syntax' | 'schema' | 'validation' | 'coverage'
}

export interface WorkflowSummary {
  node_count: number
  tool_count: number
  source_count: number
}

/** Response from POST /api/v1/agent-designer/validate. */
export interface ValidateResponse {
  valid: boolean
  errors: ValidationErrorItem[]
  workflow_summary: WorkflowSummary | null
}
