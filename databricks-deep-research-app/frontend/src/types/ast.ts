/**
 * TypeScript types for the Agent Designer workflow AST.
 *
 * These types mirror the canonical Pydantic shapes defined in:
 *   - databricks-deep-research/src/databricks_deep_research/workflow/definition.py
 *   - databricks-deep-research-app/src/deep_research/agent_designer/registry.py
 *   - databricks-deep-research-app/src/deep_research/api/v1/agent_designer.py
 */

// ---------------------------------------------------------------------------
// Node types
// ---------------------------------------------------------------------------

/**
 * The 8 supported workflow node kinds, matching NodeType enum in definition.py.
 * Leaf types: agent, tool, subworkflow (no children).
 * Composite types: sequence, parallel, loop, conditional, plan_and_execute (carry children).
 */
export type NodeType =
  | 'sequence'
  | 'parallel'
  | 'loop'
  | 'conditional'
  | 'agent'
  | 'tool'
  | 'subworkflow'
  | 'plan_and_execute';

// ---------------------------------------------------------------------------
// Path type
// ---------------------------------------------------------------------------

/**
 * Dot-separated string path into an AST.
 * Examples: 'root', 'root.children.0', 'root.children.0.children.1'
 * For plan_and_execute body: 'root.children.0.config.body' or
 * 'root.children.0.config.body.children.0' (per mutations.py semantics).
 */
export type BlockPath = string;

// ---------------------------------------------------------------------------
// Block (WorkflowNode)
// ---------------------------------------------------------------------------

/**
 * A single node in the workflow tree.
 *
 * Structural rules (matching WorkflowNode in definition.py):
 * - Composite types (sequence, parallel, loop, conditional) use `children` for
 *   their body nodes.
 * - `plan_and_execute` stores its body in `config.body` (a nested Block or
 *   sequence wrapper), per the mutations.py add_block semantics. Children of
 *   the body live at config.body.children when body is a sequence.
 * - Leaf types (agent, tool, subworkflow) have no children.
 * - For agent blocks, `config.tools` lists declared tool names bound to that agent.
 */
export interface Block {
  id: string;
  type: NodeType;
  label: string;
  config: Record<string, unknown>;
  children?: Block[];
}

// ---------------------------------------------------------------------------
// Tool declaration
// ---------------------------------------------------------------------------

/**
 * A tool declared in the workflow's top-level tools section.
 * Matches ToolDeclaration in definition.py and declare_tool() in mutations.py.
 */
export interface ToolDecl {
  name: string;
  kind: string;
  config: Record<string, unknown>;
  description?: string;
}

export interface SourceDecl {
  name: string;
  kind: string;
  endpoint?: string;
  description?: string;
  query_strategy?: Record<string, unknown>;
  metadata?: Record<string, unknown>;
}

export interface ServicePrincipalRunAs {
  service_principal_id: string;
}

// ---------------------------------------------------------------------------
// AST (WorkflowDefinition subset)
// ---------------------------------------------------------------------------

/**
 * The top-level workflow AST as exchanged between the Designer UI and the API.
 * Matches the canonical WorkflowDefinition wire shape from the framework.
 */
export interface AST {
  id: string;
  name: string;
  description?: string;
  schema_version?: number;
  version: number;
  root: Block;
  tools: ToolDecl[];
  pools?: Array<Record<string, unknown>>;
  sources?: SourceDecl[];
  models?: Record<string, unknown>;
  required_inputs?: string[];
  output_keys?: string[];
  token_budget?: number;
  timeout_seconds?: number;
  run_as?: 'caller' | ServicePrincipalRunAs;
  /**
   * Saved research depth/effort default for this agent. Scales researcher tool
   * budgets + loop iterations at runtime (proportional). Absent / 'standard' =
   * no change. A per-turn chat selection overrides this.
   */
  research_effort?: 'light' | 'standard' | 'deep';
  /** Forward-compatible storage for future WorkflowDefinition fields. */
  [key: string]: unknown;
}

// ---------------------------------------------------------------------------
// Validation error
// ---------------------------------------------------------------------------

/**
 * A single validation error returned by POST /api/v1/agent-designer/validate.
 * Matches ValidationErrorItem in agent_designer.py.
 */
export interface ValidationError {
  message: string;
  path: string | null;
  line: number | null;
  kind: 'syntax' | 'schema' | 'validation' | 'coverage';
}
