/**
 * astFieldDiff — AST-aware semantic diff for the Designer chat mutation preview.
 *
 * The old PendingMutationCard diff only walked the node `type:label` tree, so
 * config/prompt/tool edits (the most common chat edits, and the only ones the
 * active deterministic-blueprint backend supports) rendered as "0 edits".
 *
 * This computes a SEMANTIC diff by matching blocks by `id` and tools by `name`
 * (NOT a raw recursive key diff — that mis-renders array reorders/insertions as
 * giant changes). It surfaces:
 *   - per-node config field changes (system_prompt, user_prompt_template,
 *     model_tier, max_tool_calls, bound tools, …)
 *   - added / removed nodes (structural)
 *   - top-level tool declaration add / remove / config changes
 *   - workflow-level name / description changes
 *
 * It is intentionally separate from `useAstMerge` (which stays the conflict-merge
 * engine with its own raw-leaf semantics — see Codex review correction #4).
 */

import type { AST, Block } from '@/types/ast';

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

export type ChangeKind = 'added' | 'removed' | 'modified';
export type ScopeKind = 'node' | 'tool' | 'workflow';

/** A single human-meaningful field-level change. */
export interface AstFieldChange {
  /** Display label of the owning entity: node label, `Tool: <name>`, or `Workflow`. */
  scope: string;
  scopeKind: ScopeKind;
  /** Stable key for the owning entity (block id / tool name / "workflow"). */
  scopeKey: string;
  /** Humanized field name, e.g. "System prompt", "Model tier", "Bound tools". */
  field: string;
  /** Raw dot-path, for React keys + noise detection. */
  rawPath: string;
  kind: ChangeKind;
  oldValue: unknown;
  newValue: unknown;
}

/** A node added to or removed from the tree (matched by block id). */
export interface AstStructuralChange {
  kind: 'node_added' | 'node_removed';
  nodeId: string;
  label: string;
  nodeType: string;
}

export interface AstDiffResult {
  fieldChanges: AstFieldChange[];
  structural: AstStructuralChange[];
  addedNodeCount: number;
  removedNodeCount: number;
  /** Count of field changes that are NOT pure identity/noise churn. */
  meaningfulCount: number;
}

// ---------------------------------------------------------------------------
// Equality + value helpers
// ---------------------------------------------------------------------------

/** Order-insensitive (for object keys) structural deep-equal. */
export function deepEqual(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (a === null || b === null || a === undefined || b === undefined) return false;
  const aArr = Array.isArray(a);
  const bArr = Array.isArray(b);
  if (aArr !== bArr) return false;
  if (aArr && bArr) {
    if (a.length !== b.length) return false;
    return a.every((v, i) => deepEqual(v, b[i]));
  }
  if (typeof a === 'object' && typeof b === 'object') {
    const aRec = a as Record<string, unknown>;
    const bRec = b as Record<string, unknown>;
    const keys = new Set([...Object.keys(aRec), ...Object.keys(bRec)]);
    for (const k of keys) {
      if (!deepEqual(aRec[k], bRec[k])) return false;
    }
    return true;
  }
  return false;
}

/** Short, human-readable rendering of a config value for a diff row. */
export function formatDiffValue(value: unknown, max = 160): string {
  if (value === undefined) return '(none)';
  if (value === null) return '(null)';
  if (typeof value === 'string') {
    return value.length > max ? value.slice(0, max) + '…' : value;
  }
  if (Array.isArray(value)) {
    // Arrays of primitives (e.g. bound tool names) read best comma-joined.
    if (value.every((v) => typeof v === 'string' || typeof v === 'number')) {
      const joined = value.join(', ');
      return joined.length > max ? joined.slice(0, max) + '…' : joined || '(empty)';
    }
  }
  const json = JSON.stringify(value);
  return json.length > max ? json.slice(0, max) + '…' : json;
}

// ---------------------------------------------------------------------------
// Field humanization + noise detection
// ---------------------------------------------------------------------------

const FIELD_LABELS: Record<string, string> = {
  system_prompt: 'System prompt',
  user_prompt_template: 'User prompt template',
  model_tier: 'Model tier',
  max_tool_calls: 'Max tool calls',
  output_format: 'Output format',
  provider: 'Search provider',
  model: 'Search model',
  model_family: 'Search model family',
  tools: 'Bound tools',
  subtype: 'Subtype',
  error_handling: 'Error handling',
  budget_seconds: 'Budget (seconds)',
  description: 'Description',
  name: 'Name',
  label: 'Label',
  run_as: 'Run as',
  required_inputs: 'Required inputs',
  output_keys: 'Output keys',
};

function titleCase(token: string): string {
  return token
    .replace(/[_-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

/** Humanize a config sub-path (e.g. `config.system_prompt` → "System prompt"). */
export function humanizeField(key: string): string {
  // Strip a leading `config.` so node config paths read cleanly.
  const trimmed = key.replace(/^config\./, '');
  const segs = trimmed.split('.');
  return segs.map((s) => FIELD_LABELS[s] ?? titleCase(s)).join(' › ');
}

/**
 * True for identity / ordering churn that should not count as a user-meaningful
 * edit (block ids, node_id echoes). Matching by id already avoids most of this;
 * this is a backstop for the canonical full-AST card.
 */
export function isNoiseField(rawPath: string): boolean {
  const leaf = rawPath.split('.').pop() ?? '';
  return leaf === 'id' || leaf === 'node_id';
}

// ---------------------------------------------------------------------------
// Block enumeration (matches blockPath.resolveBlock traversal semantics)
// ---------------------------------------------------------------------------

interface IndexedBlock {
  block: Block;
  path: string;
}

function indexBlocks(root: Block | undefined): Map<string, IndexedBlock> {
  const out = new Map<string, IndexedBlock>();
  if (!root) return out;
  const visit = (block: Block, path: string): void => {
    if (block && typeof block.id === 'string') {
      out.set(block.id, { block, path });
    }
    const children = block.children ?? [];
    children.forEach((child, i) => visit(child, `${path}.children.${i}`));
    // plan_and_execute body lives in config.body as a nested Block.
    const body = (block.config as Record<string, unknown> | undefined)?.['body'];
    if (isBlockLike(body)) {
      visit(body as Block, `${path}.config.body`);
    }
  };
  visit(root, 'root');
  return out;
}

function isBlockLike(value: unknown): value is Block {
  return (
    value !== null &&
    typeof value === 'object' &&
    !Array.isArray(value) &&
    typeof (value as Record<string, unknown>).id === 'string' &&
    typeof (value as Record<string, unknown>).type === 'string'
  );
}

// ---------------------------------------------------------------------------
// Config diff (excludes structural keys handled by traversal)
// ---------------------------------------------------------------------------

const STRUCTURAL_CONFIG_KEYS = new Set(['body', 'children', 'evaluator']);

function diffNodeConfig(
  oldBlock: Block,
  newBlock: Block,
  scope: string,
  scopeKey: string,
  nodePath: string,
  out: AstFieldChange[],
): void {
  // Label is a top-level Block field, not config.
  if (!deepEqual(oldBlock.label, newBlock.label)) {
    out.push({
      scope,
      scopeKind: 'node',
      scopeKey,
      field: 'Label',
      rawPath: `${nodePath}.label`,
      kind: 'modified',
      oldValue: oldBlock.label,
      newValue: newBlock.label,
    });
  }

  const oldCfg = (oldBlock.config ?? {}) as Record<string, unknown>;
  const newCfg = (newBlock.config ?? {}) as Record<string, unknown>;
  const keys = new Set([...Object.keys(oldCfg), ...Object.keys(newCfg)]);
  for (const key of keys) {
    if (STRUCTURAL_CONFIG_KEYS.has(key)) continue;
    const oldVal = oldCfg[key];
    const newVal = newCfg[key];
    if (deepEqual(oldVal, newVal)) continue;
    out.push({
      scope,
      scopeKind: 'node',
      scopeKey,
      field: humanizeField(key),
      rawPath: `${nodePath}.config.${key}`,
      kind: oldVal === undefined ? 'added' : newVal === undefined ? 'removed' : 'modified',
      oldValue: oldVal,
      newValue: newVal,
    });
  }
}

// ---------------------------------------------------------------------------
// Tool diff (top-level AST.tools, matched by name)
// ---------------------------------------------------------------------------

function diffTools(oldAst: AST, newAst: AST, out: AstFieldChange[]): void {
  const oldTools = new Map((oldAst.tools ?? []).map((t) => [t.name, t]));
  const newTools = new Map((newAst.tools ?? []).map((t) => [t.name, t]));
  const names = new Set([...oldTools.keys(), ...newTools.keys()]);
  for (const name of names) {
    const oldTool = oldTools.get(name);
    const newTool = newTools.get(name);
    const scope = `Tool: ${name}`;
    const scopeKey = `tool:${name}`;
    if (!oldTool && newTool) {
      out.push({
        scope,
        scopeKind: 'tool',
        scopeKey,
        field: 'Tool declaration',
        rawPath: `tools.${name}`,
        kind: 'added',
        oldValue: undefined,
        newValue: newTool.kind,
      });
      continue;
    }
    if (oldTool && !newTool) {
      out.push({
        scope,
        scopeKind: 'tool',
        scopeKey,
        field: 'Tool declaration',
        rawPath: `tools.${name}`,
        kind: 'removed',
        oldValue: oldTool.kind,
        newValue: undefined,
      });
      continue;
    }
    if (oldTool && newTool) {
      if (!deepEqual(oldTool.kind, newTool.kind)) {
        out.push({
          scope, scopeKind: 'tool', scopeKey, field: 'Kind',
          rawPath: `tools.${name}.kind`, kind: 'modified',
          oldValue: oldTool.kind, newValue: newTool.kind,
        });
      }
      if (!deepEqual(oldTool.config, newTool.config)) {
        out.push({
          scope, scopeKind: 'tool', scopeKey, field: 'Config',
          rawPath: `tools.${name}.config`, kind: 'modified',
          oldValue: oldTool.config, newValue: newTool.config,
        });
      }
      if (!deepEqual(oldTool.description, newTool.description)) {
        out.push({
          scope, scopeKind: 'tool', scopeKey, field: 'Description',
          rawPath: `tools.${name}.description`, kind: 'modified',
          oldValue: oldTool.description, newValue: newTool.description,
        });
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Workflow-level diff (name / description only — everything else is noise)
// ---------------------------------------------------------------------------

function diffWorkflowMeta(oldAst: AST, newAst: AST, out: AstFieldChange[]): void {
  const oldRec = oldAst as unknown as Record<string, unknown>;
  const newRec = newAst as unknown as Record<string, unknown>;
  // name/description PLUS the top-level fields editable via update_workflow_meta
  // (run_as / required_inputs / output_keys) so an edit to them is visible.
  for (const key of [
    'name',
    'description',
    'run_as',
    'required_inputs',
    'output_keys',
  ] as const) {
    const oldVal = oldRec[key];
    const newVal = newRec[key];
    if (!deepEqual(oldVal, newVal)) {
      out.push({
        scope: 'Workflow',
        scopeKind: 'workflow',
        scopeKey: 'workflow',
        field: humanizeField(key),
        rawPath: key,
        kind: oldVal === undefined ? 'added' : newVal === undefined ? 'removed' : 'modified',
        oldValue: oldVal,
        newValue: newVal,
      });
    }
  }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/**
 * Compute a semantic, AST-aware diff between two workflow ASTs.
 * Blocks are matched by `id`, tools by `name`.
 */
export function computeAstFieldDiff(
  oldAst: AST | null | undefined,
  newAst: AST | null | undefined,
): AstDiffResult {
  const fieldChanges: AstFieldChange[] = [];
  const structural: AstStructuralChange[] = [];

  if (!newAst) {
    return { fieldChanges, structural, addedNodeCount: 0, removedNodeCount: 0, meaningfulCount: 0 };
  }
  // No prior AST (initial propose) — treat every node as added, no field rows.
  const oldIndex = oldAst ? indexBlocks(oldAst.root) : new Map<string, IndexedBlock>();
  const newIndex = indexBlocks(newAst.root);

  for (const [id, { block }] of newIndex) {
    if (!oldIndex.has(id)) {
      structural.push({
        kind: 'node_added',
        nodeId: id,
        label: block.label || block.type,
        nodeType: block.type,
      });
    }
  }
  for (const [id, { block }] of oldIndex) {
    if (!newIndex.has(id)) {
      structural.push({
        kind: 'node_removed',
        nodeId: id,
        label: block.label || block.type,
        nodeType: block.type,
      });
    }
  }

  // Field changes only for nodes present in BOTH (added/removed are structural).
  for (const [id, { block: newBlock, path }] of newIndex) {
    const old = oldIndex.get(id);
    if (!old) continue;
    diffNodeConfig(old.block, newBlock, newBlock.label || newBlock.type, id, path, fieldChanges);
  }

  if (oldAst) {
    diffTools(oldAst, newAst, fieldChanges);
    diffWorkflowMeta(oldAst, newAst, fieldChanges);
  }

  const addedNodeCount = structural.filter((s) => s.kind === 'node_added').length;
  const removedNodeCount = structural.filter((s) => s.kind === 'node_removed').length;
  const meaningfulCount = fieldChanges.filter((c) => !isNoiseField(c.rawPath)).length;

  return { fieldChanges, structural, addedNodeCount, removedNodeCount, meaningfulCount };
}
