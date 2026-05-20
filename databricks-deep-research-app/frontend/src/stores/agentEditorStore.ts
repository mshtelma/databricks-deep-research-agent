/**
 * Zustand store for the Agent Designer editor.
 *
 * Mutation semantics mirror backend mutations.py (pure transforms: input AST
 * is never mutated; a new AST is produced and replaces state).
 *
 * Deep clone strategy: structuredClone() (available in jsdom + all modern
 * browsers + Node 17+). Chosen over Immer to keep the bundle smaller — the
 * agent designer chunk is already isolated in vite.config.ts.
 *
 * Path format: dot-separated string matching mutations.py _split_path.
 *   'root'                              — root Block
 *   'root.children.0'                  — first child of root
 *   'root.children.0.config.body'      — plan_and_execute body
 */

import { create } from 'zustand';
import type { AST, Block, BlockPath, NodeType, ToolDecl, ValidationError } from '@/types/ast';
import type { AgentV2Response, ModelTier } from '@/types/agentDesigner';
import { resolveBlock, isDescendant } from '@/lib/blockPath';
import { normalizeWorkflowAst } from '@/lib/workflowAst';

// ---------------------------------------------------------------------------
// State shape
// ---------------------------------------------------------------------------

export interface AgentEditorState {
  ast: AST | null;
  agentId: string | null;
  etag: string | null;
  selectedPath: BlockPath | null;
  isDirty: boolean;
  validationErrors: ValidationError[];
}

// ---------------------------------------------------------------------------
// Action signatures
// ---------------------------------------------------------------------------

export interface AgentEditorActions {
  /** Load an agent into the editor; clears dirty state. */
  load(payload: { agent: AgentV2Response; etag: string | null }): void;

  /** Replace AST wholesale (used by chat-applied mutations). Marks dirty. */
  setAst(ast: AST): void;

  /** Set the currently selected block path. */
  select(path: BlockPath | null): void;

  /**
   * Append a new Block to parentPath's children.
   * Returns the new block's path, or null if parentPath is invalid.
   * Marks isDirty on success.
   */
  addBlock(
    parentPath: BlockPath,
    nodeType: NodeType,
    label: string,
    config?: Record<string, unknown>,
  ): BlockPath | null;

  /**
   * Shallow-merge patches into the block at path.
   * Returns false if path is missing.
   * Marks isDirty on success.
   */
  updateBlock(path: BlockPath, patches: Partial<Block>): boolean;

  /**
   * Remove the block at path.
   * Returns false if path === 'root' or path is missing.
   * Marks isDirty on success.
   */
  deleteBlock(path: BlockPath): boolean;

  /**
   * Move block at `from` to `to`.
   * position defaults to 'inside' for composite targets, 'after' otherwise.
   * Returns false if to is a descendant of from (cycle) or from === to.
   * Marks isDirty on success.
   */
  moveBlock(
    from: BlockPath,
    to: BlockPath,
    position?: 'before' | 'after' | 'inside',
  ): boolean;

  /**
   * Append a new ToolDecl to ast.tools.
   * Returns false on duplicate name.
   * Marks isDirty on success.
   */
  declareTool(kind: string, name: string, config?: Record<string, unknown>): boolean;

  /**
   * Update a declared tool. If the name changes, bound agent references are
   * renamed with it.
   */
  updateTool(
    name: string,
    patch: Partial<Pick<ToolDecl, 'name' | 'kind' | 'config' | 'description'>>,
  ): boolean;

  /**
   * Remove a ToolDecl by name and scrub from all agent config.tools arrays.
   * Returns false if no such tool exists.
   * Marks isDirty on success.
   */
  removeTool(name: string): boolean;

  /**
   * Add toolName to block.config.tools at blockPath.
   * Returns false if block is not an agent or toolName is not declared.
   * Marks isDirty on success.
   */
  bindToolToBlock(blockPath: BlockPath, toolName: string): boolean;

  /**
   * Set block.config.model_tier on an agent block.
   * Returns false if block is not an agent.
   * Marks isDirty on success.
   */
  setModelTier(blockPath: BlockPath, tier: ModelTier): boolean;

  /** Replace validation errors (does not affect isDirty). */
  markValidationErrors(errors: ValidationError[]): void;

  /** Clear isDirty and update etag (called after successful save). */
  markClean(newEtag: string | null): void;
}

// ---------------------------------------------------------------------------
// Composite node types (matching mutations.py _COMPOSITE_TYPES)
// ---------------------------------------------------------------------------

const COMPOSITE_TYPES: ReadonlySet<NodeType> = new Set<NodeType>([
  'sequence',
  'parallel',
  'loop',
  'conditional',
  'plan_and_execute',
]);

// ---------------------------------------------------------------------------
// Internal path helpers (matching mutations.py _split_path semantics)
// ---------------------------------------------------------------------------

function splitPath(path: BlockPath): Array<string | number> {
  return path.split('.').map((seg) => (/^\d+$/.test(seg) ? parseInt(seg, 10) : seg));
}

/**
 * Navigate into a raw object (AST as unknown) using a dot-path and return
 * the value, or null if any segment is missing.
 */
function getAtRaw(obj: unknown, path: string): unknown {
  const segs = splitPath(path);
  let cur: unknown = obj;
  for (const seg of segs) {
    if (cur === null || cur === undefined) return null;
    if (typeof seg === 'number') {
      if (!Array.isArray(cur)) return null;
      cur = cur[seg];
    } else {
      if (typeof cur !== 'object' || Array.isArray(cur)) return null;
      cur = (cur as Record<string, unknown>)[seg];
    }
  }
  return cur ?? null;
}

/**
 * Set a value at path in a cloned object, matching mutations.py _set_at.
 * The top-level object must be an object (AST dict).
 */
function setAtRaw(
  root: Record<string, unknown>,
  path: string,
  value: unknown,
): Record<string, unknown> {
  const segs = splitPath(path);
  const cloned = structuredClone(root);
  let cur: unknown = cloned;
  for (let i = 0; i < segs.length - 1; i++) {
    const seg = segs[i]!;
    if (typeof seg === 'number') {
      cur = (cur as unknown[])[seg];
    } else {
      cur = (cur as Record<string, unknown>)[seg];
    }
    if (cur === null || cur === undefined) return cloned; // path missing
  }
  const last = segs[segs.length - 1]!;
  if (typeof last === 'number') {
    (cur as unknown[])[last] = value;
  } else {
    (cur as Record<string, unknown>)[last] = value;
  }
  return cloned;
}

// ---------------------------------------------------------------------------
// Internal tree-walk helper for tool reference cleanup
// ---------------------------------------------------------------------------

function updateToolRefsInBlock(
  block: Block,
  transform: (toolName: string) => string | null,
): Block {
  const result: Block = { ...block };
  if (Array.isArray(result.children)) {
    result.children = result.children.map((child) => updateToolRefsInBlock(child, transform));
  }
  if (
    result.config &&
    Array.isArray((result.config as Record<string, unknown>)['tools'])
  ) {
    const configTools = (result.config as Record<string, unknown>)['tools'] as string[];
    const filtered = configTools
      .map(transform)
      .filter((item): item is string => typeof item === 'string' && item.length > 0);
    result.config = {
      ...(result.config as Record<string, unknown>),
      tools: filtered.length > 0 ? filtered : undefined,
    };
  }
  return result;
}

function defaultCondition(): Record<string, unknown> {
  return { kind: 'key_equals', state_key: 'intent', value: 'yes' };
}

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

export const initialState: AgentEditorState = {
  ast: null,
  agentId: null,
  etag: null,
  selectedPath: null,
  isDirty: false,
  validationErrors: [],
};

export const useAgentEditorStore = create<AgentEditorState & AgentEditorActions>()(
  (set, get) => ({
    ...initialState,

    // -----------------------------------------------------------------------
    // load
    // -----------------------------------------------------------------------
    load({ agent, etag }) {
      set({
        ast: normalizeWorkflowAst(agent.definition, agent.name),
        agentId: agent.id,
        etag,
        selectedPath: null,
        isDirty: false,
        validationErrors: [],
      });
    },

    // -----------------------------------------------------------------------
    // setAst
    // -----------------------------------------------------------------------
    setAst(ast) {
      set({ ast: normalizeWorkflowAst(ast), isDirty: true, validationErrors: [] });
    },

    // -----------------------------------------------------------------------
    // select
    // -----------------------------------------------------------------------
    select(path) {
      set({ selectedPath: path });
    },

    // -----------------------------------------------------------------------
    // addBlock
    // -----------------------------------------------------------------------
    addBlock(parentPath, nodeType, label, config = {}) {
      const { ast } = get();
      if (!ast) return null;

      const newAst = structuredClone(ast);

      // Handle plan_and_execute body paths
      if (parentPath.endsWith('config.body')) {
        return _addToPlanBody(set, newAst, parentPath, nodeType, label, config);
      }

      // Normal composite: find parent and append to children
      const parentBlock = resolveBlock(newAst, parentPath);
      if (!parentBlock) return null;

      // Only composite types can receive children
      if (!COMPOSITE_TYPES.has(parentBlock.type)) return null;

      const newBlock: Block = {
        id: crypto.randomUUID(),
        type: nodeType,
        label,
        config,
        children: [],
      };

      const children = Array.isArray(parentBlock.children) ? parentBlock.children : [];
      const newIndex = children.length;

      if (parentBlock.type === 'conditional') {
        const existingConditions = Array.isArray(parentBlock.config.conditions)
          ? parentBlock.config.conditions as Array<Record<string, unknown>>
          : [];
        parentBlock.config = {
          ...parentBlock.config,
          conditions: [...existingConditions, defaultCondition()],
          default_branch: newIndex,
        };
      }

      // Navigate to parent and push child
      const childrenPath = parentPath + '.children';
      const updated = setAtRaw(
        newAst as unknown as Record<string, unknown>,
        childrenPath,
        [...children, newBlock],
      ) as unknown as AST;

      const newPath: BlockPath = `${childrenPath}.${newIndex}`;
      set({ ast: updated, isDirty: true });
      return newPath;
    },

    // -----------------------------------------------------------------------
    // updateBlock
    // -----------------------------------------------------------------------
    updateBlock(path, patches) {
      const { ast } = get();
      if (!ast) return false;

      const block = resolveBlock(ast, path);
      if (!block) return false;

      const newAst = structuredClone(ast);
      // resolveBlock returns a live reference into the cloned tree, so
      // Object.assign(target, patches) mutates the clone in-place — safe because
      // newAst is brand-new and not yet shared with any React consumer.
      const target = resolveBlock(newAst, path);
      if (!target) return false;

      Object.assign(target, patches);
      set({ ast: newAst, isDirty: true });
      return true;
    },

    // -----------------------------------------------------------------------
    // updateTool
    // -----------------------------------------------------------------------
    updateTool(name, patch) {
      const { ast } = get();
      if (!ast) return false;

      const existing = ast.tools ?? [];
      const current = existing.find((t) => t.name === name);
      if (!current) return false;

      const nextName = patch.name?.trim() || name;
      if (nextName !== name && existing.some((t) => t.name === nextName)) return false;

      const tools = existing.map((tool) => {
        if (tool.name !== name) return tool;
        return {
          ...tool,
          ...patch,
          name: nextName,
          config: patch.config ?? tool.config,
        };
      });

      const root = nextName === name
        ? ast.root
        : updateToolRefsInBlock(structuredClone(ast.root), (toolName) => (
            toolName === name ? nextName : toolName
          ));

      set({ ast: { ...ast, tools, root }, isDirty: true });
      return true;
    },

    // -----------------------------------------------------------------------
    // deleteBlock
    // -----------------------------------------------------------------------
    deleteBlock(path) {
      if (path === 'root') return false;

      const { ast } = get();
      if (!ast) return false;

      const segs = splitPath(path);
      const last = segs[segs.length - 1];

      // Last segment must be a numeric index (node lives in a children list)
      if (typeof last !== 'number') return false;

      const block = resolveBlock(ast, path);
      if (!block) return false;

      const parentPath = segs.slice(0, segs.length - 1).map(String).join('.');
      const parentList = getAtRaw(ast as unknown, parentPath);
      if (!Array.isArray(parentList)) return false;

      const parentBlockPath = segs.slice(0, segs.length - 2).map(String).join('.');
      const parentBlock = resolveBlock(ast, parentBlockPath);
      const newList = [...parentList];
      newList.splice(last, 1);

      let updated = setAtRaw(
        ast as unknown as Record<string, unknown>,
        parentPath,
        newList,
      ) as unknown as AST;

      if (parentBlock?.type === 'conditional') {
        const conditions = Array.isArray(parentBlock.config.conditions)
          ? [...parentBlock.config.conditions as Array<Record<string, unknown>>]
          : [];
        if (last < conditions.length) {
          conditions.splice(last, 1);
        }
        const maxConditions = Math.max(0, newList.length - 1);
        const trimmedConditions = conditions.slice(0, maxConditions);
        const previousDefault = typeof parentBlock.config.default_branch === 'number'
          ? parentBlock.config.default_branch
          : parentList.length - 1;
        const defaultBranch = Math.max(
          0,
          Math.min(newList.length - 1, previousDefault > last ? previousDefault - 1 : previousDefault),
        );
        updated = setAtRaw(
          updated as unknown as Record<string, unknown>,
          `${parentBlockPath}.config`,
          {
            ...parentBlock.config,
            conditions: trimmedConditions,
            default_branch: defaultBranch,
          },
        ) as unknown as AST;
      }

      set({ ast: updated, isDirty: true });
      return true;
    },

    // -----------------------------------------------------------------------
    // moveBlock
    // -----------------------------------------------------------------------
    moveBlock(from, to, position) {
      if (from === to) return false;

      const { ast } = get();
      if (!ast) return false;

      // Cycle check: to must not be equal to or a descendant of from
      if (to === from || isDescendant(from, to)) return false;

      const nodeToMove = resolveBlock(ast, from);
      if (!nodeToMove) return false;

      const destBlock = resolveBlock(ast, to);
      if (!destBlock) return false;

      // Resolve effective position
      const isComposite = COMPOSITE_TYPES.has(destBlock.type);
      const effectivePosition = position ?? (isComposite ? 'inside' : 'after');

      // Step 1: delete from source
      const fromSegs = splitPath(from);
      const fromLast = fromSegs[fromSegs.length - 1];
      if (typeof fromLast !== 'number') return false;

      const fromParentPath = fromSegs.slice(0, fromSegs.length - 1).map(String).join('.');
      const fromParentList = getAtRaw(ast as unknown, fromParentPath);
      if (!Array.isArray(fromParentList)) return false;

      const clonedNode = structuredClone(nodeToMove);
      let afterDelete = setAtRaw(
        ast as unknown as Record<string, unknown>,
        fromParentPath,
        fromParentList.filter((_, i) => i !== fromLast),
      ) as unknown as AST;

      // Step 2: adjust to path after deletion (mirrors mutations.py _adjust_path_after_deletion)
      const adjustedTo = _adjustPathAfterDeletion(to, from);

      // Step 3: insert at destination
      if (effectivePosition === 'inside') {
        // Append as last child of target
        const destBlockAfter = resolveBlock(afterDelete, adjustedTo);
        if (!destBlockAfter) return false;
        const destChildren = Array.isArray(destBlockAfter.children) ? destBlockAfter.children : [];
        const childrenPath = adjustedTo + '.children';
        afterDelete = setAtRaw(
          afterDelete as unknown as Record<string, unknown>,
          childrenPath,
          [...destChildren, clonedNode],
        ) as unknown as AST;
      } else {
        // 'before' or 'after' — insert as sibling of target
        const toSegs = splitPath(adjustedTo);
        const toLast = toSegs[toSegs.length - 1];
        if (typeof toLast !== 'number') return false;
        const toParentPath = toSegs.slice(0, toSegs.length - 1).map(String).join('.');
        const toParentList = getAtRaw(afterDelete as unknown, toParentPath);
        if (!Array.isArray(toParentList)) return false;
        const insertIndex = effectivePosition === 'before' ? toLast : toLast + 1;
        const newList = [...toParentList];
        newList.splice(insertIndex, 0, clonedNode);
        afterDelete = setAtRaw(
          afterDelete as unknown as Record<string, unknown>,
          toParentPath,
          newList,
        ) as unknown as AST;
      }

      set({ ast: afterDelete, isDirty: true });
      return true;
    },

    // -----------------------------------------------------------------------
    // declareTool
    // -----------------------------------------------------------------------
    declareTool(kind, name, config = {}) {
      const { ast } = get();
      if (!ast) return false;

      const existing = ast.tools ?? [];
      if (existing.some((t) => t.name === name)) return false;

      const newAst: AST = {
        ...ast,
        tools: [...existing, { kind, name, config }],
      };
      set({ ast: newAst, isDirty: true });
      return true;
    },

    // -----------------------------------------------------------------------
    // removeTool
    // -----------------------------------------------------------------------
    removeTool(name) {
      const { ast } = get();
      if (!ast) return false;

      const existing = ast.tools ?? [];
      if (!existing.some((t) => t.name === name)) return false;

      const newAst: AST = {
        ...ast,
        tools: existing.filter((t) => t.name !== name),
        root: updateToolRefsInBlock(
          structuredClone(ast.root),
          (toolName) => (toolName === name ? null : toolName),
        ),
      };
      set({ ast: newAst, isDirty: true });
      return true;
    },

    // -----------------------------------------------------------------------
    // bindToolToBlock
    // -----------------------------------------------------------------------
    bindToolToBlock(blockPath, toolName) {
      const { ast } = get();
      if (!ast) return false;

      const block = resolveBlock(ast, blockPath);
      if (!block) return false;

      if (block.type !== 'agent') return false;

      const declared = (ast.tools ?? []).map((t) => t.name);
      if (!declared.includes(toolName)) return false;

      const existing = Array.isArray(block.config.tools) ? block.config.tools as string[] : [];
      if (existing.includes(toolName)) return true; // already bound, no-op

      const newAst = structuredClone(ast);
      const target = resolveBlock(newAst, blockPath);
      if (!target) return false;

      target.config = {
        ...target.config,
        tools: [...existing, toolName],
      };
      set({ ast: newAst, isDirty: true });
      return true;
    },

    // -----------------------------------------------------------------------
    // setModelTier
    // -----------------------------------------------------------------------
    setModelTier(blockPath, tier) {
      const { ast } = get();
      if (!ast) return false;

      const block = resolveBlock(ast, blockPath);
      if (!block) return false;

      if (block.type !== 'agent') return false;

      const newAst = structuredClone(ast);
      const target = resolveBlock(newAst, blockPath);
      if (!target) return false;

      target.config = { ...(target.config as Record<string, unknown>), model_tier: tier };
      set({ ast: newAst, isDirty: true });
      return true;
    },

    // -----------------------------------------------------------------------
    // markValidationErrors
    // -----------------------------------------------------------------------
    markValidationErrors(errors) {
      set({ validationErrors: errors });
    },

    // -----------------------------------------------------------------------
    // markClean
    // -----------------------------------------------------------------------
    markClean(newEtag) {
      set({ isDirty: false, etag: newEtag });
    },
  }),
);

// ---------------------------------------------------------------------------
// Module-level helpers (not part of Zustand store)
// ---------------------------------------------------------------------------

/**
 * Handle addBlock for plan_and_execute body paths.
 * Mirrors mutations.py::_add_to_plan_body.
 */
function _addToPlanBody(
  set: (partial: Partial<AgentEditorState>) => void,
  ast: AST,
  bodyPath: BlockPath,
  nodeType: NodeType,
  label: string,
  config: Record<string, unknown>,
): BlockPath | null {
  const newBlock: Block = {
    id: crypto.randomUUID(),
    type: nodeType,
    label,
    config,
    children: [],
  };

  const currentBody = getAtRaw(ast as unknown, bodyPath);

  let updated: AST;
  let newPath: BlockPath;

  if (currentBody === null || currentBody === undefined) {
    // Empty body — set directly
    updated = setAtRaw(
      ast as unknown as Record<string, unknown>,
      bodyPath,
      newBlock,
    ) as unknown as AST;
    newPath = bodyPath;
  } else if (typeof currentBody === 'object' && !Array.isArray(currentBody)) {
    const bodyBlock = currentBody as Block;
    if (bodyBlock.type === 'sequence') {
      // Already a sequence — append to its children
      const existing = Array.isArray(bodyBlock.children) ? bodyBlock.children : [];
      const idx = existing.length;
      updated = setAtRaw(
        ast as unknown as Record<string, unknown>,
        bodyPath + '.children',
        [...existing, newBlock],
      ) as unknown as AST;
      newPath = `${bodyPath}.children.${idx}`;
    } else {
      // Single non-sequence — wrap both in a new sequence
      const wrapper: Block = {
        id: crypto.randomUUID(),
        type: 'sequence',
        label: 'body',
        config: {},
        children: [bodyBlock, newBlock],
      };
      updated = setAtRaw(
        ast as unknown as Record<string, unknown>,
        bodyPath,
        wrapper,
      ) as unknown as AST;
      newPath = `${bodyPath}.children.1`;
    }
  } else {
    return null;
  }

  set({ ast: updated, isDirty: true });
  return newPath;
}

/**
 * Adjust a path after a sibling deletion.
 * Mirrors mutations.py::_adjust_path_after_deletion.
 */
function _adjustPathAfterDeletion(pathToAdjust: string, deletedPath: string): string {
  const deletedSegs = splitPath(deletedPath);
  const adjustSegs = splitPath(pathToAdjust);

  const deletedLast = deletedSegs[deletedSegs.length - 1];
  if (typeof deletedLast !== 'number') return pathToAdjust;

  const deletedParent = deletedSegs.slice(0, deletedSegs.length - 1);
  const depth = deletedParent.length;

  if (adjustSegs.length > depth) {
    const prefix = adjustSegs.slice(0, depth);
    const match = prefix.every((s, i) => String(s) === String(deletedParent[i]));
    if (match) {
      const segAtDepth = adjustSegs[depth];
      if (typeof segAtDepth === 'number' && segAtDepth > deletedLast) {
        const newSegs = [...adjustSegs];
        newSegs[depth] = segAtDepth - 1;
        return newSegs.map(String).join('.');
      }
    }
  }

  return pathToAdjust;
}
