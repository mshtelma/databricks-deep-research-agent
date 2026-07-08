/**
 * Tests for agentEditorStore.
 *
 * Zustand stores work outside React — actions are called directly via
 * useAgentEditorStore.getState(). No act() wrapper needed.
 *
 * The store is reset in beforeEach via setState(initialState) so each test
 * starts from a clean slate.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { useAgentEditorStore, initialState } from '../agentEditorStore';
import { createDraftWorkflow } from '@/lib/workflowAst';
import type { AST, Block } from '@/types/ast';
import type { AgentV2Response } from '@/types/agentDesigner';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeAst(): AST {
  return {
    ...createDraftWorkflow('Test Workflow'),
    id: 'wf',
    root: { id: 'r', type: 'sequence', label: 'root', config: {}, children: [] },
  };
}

function makeAgentBlock(id = 'a1'): Block {
  return {
    id,
    type: 'agent',
    label: 'My Agent',
    config: {},
    children: [],
  };
}

function makeAgentResponse(ast: AST): AgentV2Response {
  return {
    id: 'agent-123',
    owner_id: 'user-1',
    name: 'Test Agent',
    description: null,
    avatar_url: null,
    visibility: 'private',
    definition: ast as unknown as Record<string, unknown>,
    schema_version: 1,
    etag: 'etag-v1',
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-01-01T00:00:00Z',
  };
}

function store() {
  return useAgentEditorStore.getState();
}

// ---------------------------------------------------------------------------
// Reset store before each test
// ---------------------------------------------------------------------------

beforeEach(() => {
  useAgentEditorStore.setState(initialState);
});

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('agentEditorStore — initial state', () => {
  it('has null ast, null agentId, isDirty=false', () => {
    const s = store();
    expect(s.ast).toBeNull();
    expect(s.agentId).toBeNull();
    expect(s.isDirty).toBe(false);
    expect(s.etag).toBeNull();
    expect(s.selectedPath).toBeNull();
    expect(s.validationErrors).toEqual([]);
  });
});

describe('load', () => {
  it('sets ast and etag, clears dirty and validationErrors', () => {
    const ast = makeAst();
    const agent = makeAgentResponse(ast);

    // Pre-dirty the store
    useAgentEditorStore.setState({ isDirty: true, validationErrors: [{ message: 'err', path: null, line: null, kind: 'syntax' }] });

    store().load({ agent, etag: 'etag-v2' });

    const s = store();
    expect(s.ast).not.toBeNull();
    expect(s.agentId).toBe('agent-123');
    expect(s.etag).toBe('etag-v2');
    expect(s.isDirty).toBe(false);
    expect(s.validationErrors).toEqual([]);
    expect(s.selectedPath).toBeNull();
  });
});

describe('addBlock', () => {
  it('returns new path, marks dirty, ast contains new block at returned path', () => {
    const ast = makeAst();
    useAgentEditorStore.setState({ ast, isDirty: false });

    const newPath = store().addBlock('root', 'agent', 'My Agent');

    expect(newPath).toBe('root.children.0');
    const s = store();
    expect(s.isDirty).toBe(true);
    expect(s.ast?.root.children).toHaveLength(1);
    expect(s.ast?.root.children?.[0]?.type).toBe('agent');
    expect(s.ast?.root.children?.[0]?.label).toBe('My Agent');
    expect(typeof s.ast?.root.children?.[0]?.id).toBe('string');
  });

  it('replaces generic researcher labels on inserted blocks', () => {
    const ast = makeAst();
    useAgentEditorStore.setState({ ast, isDirty: false });

    store().addBlock('root', 'agent', 'Researcher 1', { subtype: 'researcher' });

    expect(store().ast?.root.children?.[0]?.label).toBe('Evidence Researcher');
  });

  it('addBlock to invalid parent returns null and does not mutate', () => {
    const ast = makeAst();
    useAgentEditorStore.setState({ ast, isDirty: false });

    // 'root.children.99' does not exist
    const result = store().addBlock('root.children.99', 'agent', 'Unreachable');

    expect(result).toBeNull();
    expect(store().isDirty).toBe(false);
  });

  it('addBlock to a leaf (agent) node returns null — leaves have no children', () => {
    const ast: AST = {
      ...makeAst(),
      root: {
        ...makeAst().root,
        children: [makeAgentBlock()],
      },
    };
    useAgentEditorStore.setState({ ast, isDirty: false });

    // agent is a leaf — cannot add children to it
    const result = store().addBlock('root.children.0', 'agent', 'Nested');
    expect(result).toBeNull();
    expect(store().isDirty).toBe(false);
  });

  it('addBlock appends sequentially — second block gets index 1', () => {
    const ast = makeAst();
    useAgentEditorStore.setState({ ast });
    store().addBlock('root', 'agent', 'First');
    const path2 = store().addBlock('root', 'agent', 'Second');
    expect(path2).toBe('root.children.1');
    expect(store().ast?.root.children).toHaveLength(2);
  });
});

describe('updateBlock', () => {
  it('applies patches to resolved block and marks dirty', () => {
    const ast: AST = {
      ...makeAst(),
      root: {
        ...makeAst().root,
        children: [makeAgentBlock()],
      },
    };
    useAgentEditorStore.setState({ ast, isDirty: false });

    const ok = store().updateBlock('root.children.0', { label: 'Updated Label' });

    expect(ok).toBe(true);
    expect(store().isDirty).toBe(true);
    expect(store().ast?.root.children?.[0]?.label).toBe('Updated Label');
  });

  it('returns false and does not mutate when path is missing', () => {
    const ast = makeAst();
    useAgentEditorStore.setState({ ast, isDirty: false });

    const ok = store().updateBlock('root.children.99', { label: 'Ghost' });

    expect(ok).toBe(false);
    expect(store().isDirty).toBe(false);
  });
});

describe('deleteBlock', () => {
  it('removes a block and marks dirty', () => {
    const ast: AST = {
      ...makeAst(),
      root: {
        ...makeAst().root,
        children: [makeAgentBlock('a1'), makeAgentBlock('a2')],
      },
    };
    useAgentEditorStore.setState({ ast, isDirty: false });

    const ok = store().deleteBlock('root.children.0');

    expect(ok).toBe(true);
    expect(store().isDirty).toBe(true);
    expect(store().ast?.root.children).toHaveLength(1);
    expect(store().ast?.root.children?.[0]?.id).toBe('a2');
  });

  it('deleteBlock("root") returns false — cannot delete root', () => {
    const ast = makeAst();
    useAgentEditorStore.setState({ ast, isDirty: false });

    const ok = store().deleteBlock('root');

    expect(ok).toBe(false);
    expect(store().isDirty).toBe(false);
    expect(store().ast?.root).toBeDefined();
  });
});

describe('moveBlock', () => {
  it('reorders within siblings (move first to after second)', () => {
    const childA: Block = { id: 'a', type: 'agent', label: 'A', config: {}, children: [] };
    const childB: Block = { id: 'b', type: 'agent', label: 'B', config: {}, children: [] };
    const ast: AST = {
      ...makeAst(),
      root: { ...makeAst().root, children: [childA, childB] },
    };
    useAgentEditorStore.setState({ ast, isDirty: false });

    // Move child A (index 0) to after child B (index 1) → A goes to position 1
    const ok = store().moveBlock('root.children.0', 'root.children.1', 'after');

    expect(ok).toBe(true);
    expect(store().isDirty).toBe(true);
    const children = store().ast?.root.children ?? [];
    // After move: [B, A]
    expect(children[0]?.id).toBe('b');
    expect(children[1]?.id).toBe('a');
  });

  it('returns false when to is a descendant of from (cycle prevention)', () => {
    const grandchild: Block = { id: 'gc', type: 'agent', label: 'GC', config: {}, children: [] };
    const child: Block = { id: 'c', type: 'sequence', label: 'Child', config: {}, children: [grandchild] };
    const ast: AST = {
      ...makeAst(),
      root: { ...makeAst().root, children: [child] },
    };
    useAgentEditorStore.setState({ ast, isDirty: false });

    const ok = store().moveBlock('root.children.0', 'root.children.0.children.0');

    expect(ok).toBe(false);
    expect(store().isDirty).toBe(false);
  });
});

describe('tool declaration lifecycle references', () => {
  it('renames local tool step refs with the declaration', () => {
    const ast: AST = {
      ...makeAst(),
      tools: [{ kind: 'decorated', name: 'old_tool', config: { import: 'pkg.mod:old_tool' } }],
      root: {
        ...makeAst().root,
        children: [
          {
            id: 'tool-1',
            type: 'tool',
            label: 'Call tool',
            config: { ref: { type: 'builtin', name: 'old_tool' } },
            children: [],
          },
        ],
      },
    };
    useAgentEditorStore.setState({ ast, isDirty: false });

    expect(store().updateTool('old_tool', { name: 'new_tool' })).toBe(true);

    const toolBlock = store().ast?.root.children?.[0];
    expect(toolBlock?.config.ref).toEqual({ type: 'builtin', name: 'new_tool' });
  });

  it('does not rewrite direct external tool step refs', () => {
    const ast: AST = {
      ...makeAst(),
      tools: [{ kind: 'uc_function', name: 'pct_change', config: { function_name: 'main.metrics.pct_change' } }],
      root: {
        ...makeAst().root,
        children: [
          {
            id: 'tool-1',
            type: 'tool',
            label: 'Direct UC call',
            config: { ref: { type: 'uc_function', name: 'main.metrics.pct_change' } },
            children: [],
          },
        ],
      },
    };
    useAgentEditorStore.setState({ ast, isDirty: false });

    expect(store().updateTool('pct_change', { name: 'pct_change_v2' })).toBe(true);

    const toolBlock = store().ast?.root.children?.[0];
    expect(toolBlock?.config.ref).toEqual({
      type: 'uc_function',
      name: 'main.metrics.pct_change',
    });
  });

  it('preserves direct external refs in agent tool bindings', () => {
    const directRef = { type: 'uc_function', name: 'main.metrics.pct_change' };
    const ast: AST = {
      ...makeAst(),
      tools: [{ kind: 'uc_function', name: 'pct_change', config: { function_name: 'main.metrics.pct_change' } }],
      root: {
        ...makeAst().root,
        children: [
          {
            id: 'agent-1',
            type: 'agent',
            label: 'Agent',
            config: { tools: ['pct_change', directRef] },
            children: [],
          },
        ],
      },
    };
    useAgentEditorStore.setState({ ast, isDirty: false });

    expect(store().updateTool('pct_change', { name: 'pct_change_v2' })).toBe(true);

    const agentBlock = store().ast?.root.children?.[0];
    expect(agentBlock?.config.tools).toEqual(['pct_change_v2', directRef]);
  });

  it('clears local tool step refs when a declaration is removed', () => {
    const ast: AST = {
      ...makeAst(),
      tools: [{ kind: 'decorated', name: 'old_tool', config: { import: 'pkg.mod:old_tool' } }],
      root: {
        ...makeAst().root,
        children: [
          {
            id: 'tool-1',
            type: 'tool',
            label: 'Call tool',
            config: { ref: { type: 'builtin', name: 'old_tool' } },
            children: [],
          },
        ],
      },
    };
    useAgentEditorStore.setState({ ast, isDirty: false });

    expect(store().removeTool('old_tool')).toBe(true);

    const toolBlock = store().ast?.root.children?.[0];
    expect(toolBlock?.config.ref).toEqual({ type: 'builtin', name: '' });
  });

  it('updates local refs inside plan_and_execute config.body', () => {
    const ast: AST = {
      ...makeAst(),
      tools: [{ kind: 'decorated', name: 'old_tool', config: { import: 'pkg.mod:old_tool' } }],
      root: {
        ...makeAst().root,
        children: [
          {
            id: 'plan-1',
            type: 'plan_and_execute',
            label: 'Plan',
            config: {
              body: {
                id: 'tool-1',
                type: 'tool',
                label: 'Call tool',
                config: { ref: { type: 'builtin', name: 'old_tool' } },
                children: [],
              },
            },
            children: [],
          },
        ],
      },
    };
    useAgentEditorStore.setState({ ast, isDirty: false });

    expect(store().updateTool('old_tool', { name: 'new_tool' })).toBe(true);

    const planBlock = store().ast?.root.children?.[0];
    const body = planBlock?.config.body as Block;
    expect(body.config.ref).toEqual({ type: 'builtin', name: 'new_tool' });
  });

  it('updates local tool bindings inside plan_and_execute planner and evaluator configs', () => {
    const directRef = { type: 'uc_function', name: 'main.metrics.pct_change' };
    const ast: AST = {
      ...makeAst(),
      tools: [{ kind: 'decorated', name: 'old_tool', config: { import: 'pkg.mod:old_tool' } }],
      root: {
        ...makeAst().root,
        children: [
          {
            id: 'plan-1',
            type: 'plan_and_execute',
            label: 'Plan',
            config: {
              planner: { tools: ['old_tool', directRef] },
              evaluator: { tools: ['old_tool'] },
              body: null,
            },
            children: [],
          },
        ],
      },
    };
    useAgentEditorStore.setState({ ast, isDirty: false });

    expect(store().updateTool('old_tool', { name: 'new_tool' })).toBe(true);

    const planBlock = store().ast?.root.children?.[0];
    expect((planBlock?.config.planner as Record<string, unknown>).tools).toEqual([
      'new_tool',
      directRef,
    ]);
    expect((planBlock?.config.evaluator as Record<string, unknown>).tools).toEqual(['new_tool']);

    expect(store().removeTool('new_tool')).toBe(true);

    const updatedPlanBlock = store().ast?.root.children?.[0];
    expect((updatedPlanBlock?.config.planner as Record<string, unknown>).tools).toEqual([
      directRef,
    ]);
    expect((updatedPlanBlock?.config.evaluator as Record<string, unknown>).tools).toBeUndefined();
  });
});

describe('declareTool', () => {
  it('appends a tool and marks dirty', () => {
    const ast = makeAst();
    useAgentEditorStore.setState({ ast, isDirty: false });

    const ok = store().declareTool('web_search', 'my_search', { depth: 3 });

    expect(ok).toBe(true);
    expect(store().isDirty).toBe(true);
    const tools = store().ast?.tools ?? [];
    expect(tools).toHaveLength(1);
    expect(tools[0]?.name).toBe('my_search');
    expect(tools[0]?.kind).toBe('web_search');
    expect(tools[0]?.config).toEqual({ depth: 3 });
  });

  it('returns false on duplicate name', () => {
    const ast: AST = { ...makeAst(), tools: [{ kind: 'web_search', name: 'my_search', config: {} }] };
    useAgentEditorStore.setState({ ast, isDirty: false });

    const ok = store().declareTool('web_search', 'my_search');

    expect(ok).toBe(false);
    expect(store().isDirty).toBe(false);
    expect(store().ast?.tools).toHaveLength(1);
  });
});

describe('updateTool', () => {
  it('updates config and marks dirty', () => {
    const ast: AST = {
      ...makeAst(),
      tools: [{ kind: 'vector_search', name: 'search', config: { index_name: 'old' } }],
    };
    useAgentEditorStore.setState({ ast, isDirty: false });

    const ok = store().updateTool('search', { config: { index_name: 'new' } });

    expect(ok).toBe(true);
    expect(store().isDirty).toBe(true);
    expect(store().ast?.tools[0]?.config).toEqual({ index_name: 'new' });
  });

  it('renames bound references when tool name changes', () => {
    const agentBlock: Block = {
      id: 'ag1',
      type: 'agent',
      label: 'Agent',
      config: { tools: ['old_search'] },
      children: [],
    };
    const ast: AST = {
      ...makeAst(),
      tools: [{ kind: 'web_search', name: 'old_search', config: {} }],
      root: { ...makeAst().root, children: [agentBlock] },
    };
    useAgentEditorStore.setState({ ast, isDirty: false });

    const ok = store().updateTool('old_search', { name: 'new_search' });

    expect(ok).toBe(true);
    expect(store().ast?.tools[0]?.name).toBe('new_search');
    expect(store().ast?.root.children?.[0]?.config.tools).toEqual(['new_search']);
  });
});

describe('bindToolToBlock', () => {
  it('returns false when block is not an agent', () => {
    const ast: AST = {
      ...makeAst(),
      tools: [{ kind: 'web_search', name: 'search', config: {} }],
      root: { id: 'r', type: 'sequence', label: 'root', config: {}, children: [] },
    };
    useAgentEditorStore.setState({ ast, isDirty: false });

    const ok = store().bindToolToBlock('root', 'search');

    expect(ok).toBe(false);
    expect(store().isDirty).toBe(false);
  });

  it('adds tool name to block.config.tools on an agent block', () => {
    const agentBlock = makeAgentBlock('ag1');
    const ast: AST = {
      ...makeAst(),
      tools: [{ kind: 'web_search', name: 'search', config: {} }],
      root: { ...makeAst().root, children: [agentBlock] },
    };
    useAgentEditorStore.setState({ ast, isDirty: false });

    const ok = store().bindToolToBlock('root.children.0', 'search');

    expect(ok).toBe(true);
    expect(store().isDirty).toBe(true);
    expect(store().ast?.root.children?.[0]?.config.tools).toContain('search');
  });
});

describe('removeTool', () => {
  it('removes from ast.tools AND from referencing block.config.tools, marks dirty', () => {
    const agentBlock: Block = {
      id: 'ag1',
      type: 'agent',
      label: 'Agent',
      config: { tools: ['search', 'other'] },
      children: [],
    };
    const ast: AST = {
      ...makeAst(),
      tools: [
        { kind: 'web_search', name: 'search', config: {} },
        { kind: 'web_search', name: 'other', config: {} },
      ],
      root: { ...makeAst().root, children: [agentBlock] },
    };
    useAgentEditorStore.setState({ ast, isDirty: false });

    const ok = store().removeTool('search');

    expect(ok).toBe(true);
    expect(store().isDirty).toBe(true);
    // Removed from ast.tools
    expect(store().ast?.tools.map((t) => t.name)).not.toContain('search');
    expect(store().ast?.tools).toHaveLength(1);
    // Removed from block.config.tools
    const blockTools = (store().ast?.root.children?.[0]?.config.tools as string[] | undefined) ?? [];
    expect(blockTools).not.toContain('search');
    expect(blockTools).toContain('other');
  });
});

describe('setModelTier', () => {
  it('returns false on non-agent block', () => {
    const ast = makeAst(); // root is a sequence
    useAgentEditorStore.setState({ ast, isDirty: false });

    const ok = store().setModelTier('root', 'analytical');

    expect(ok).toBe(false);
    expect(store().isDirty).toBe(false);
  });

  it('updates config.model_tier on an agent block and marks dirty', () => {
    const agentBlock = makeAgentBlock('ag1');
    const ast: AST = {
      ...makeAst(),
      root: { ...makeAst().root, children: [agentBlock] },
    };
    useAgentEditorStore.setState({ ast, isDirty: false });

    const ok = store().setModelTier('root.children.0', 'complex');

    expect(ok).toBe(true);
    expect(store().isDirty).toBe(true);
    const config = store().ast?.root.children?.[0]?.config as Record<string, unknown>;
    expect(config?.['model_tier']).toBe('complex');
  });
});

describe('markClean', () => {
  it('clears isDirty and sets new etag', () => {
    useAgentEditorStore.setState({ isDirty: true, etag: 'old-etag' });

    store().markClean('new-etag');

    expect(store().isDirty).toBe(false);
    expect(store().etag).toBe('new-etag');
  });

  it('accepts null etag', () => {
    useAgentEditorStore.setState({ isDirty: true, etag: 'old-etag' });
    store().markClean(null);
    expect(store().isDirty).toBe(false);
    expect(store().etag).toBeNull();
  });
});

describe('setAst', () => {
  it('replaces ast wholesale, marks dirty, clears validationErrors', () => {
    const ast1 = makeAst();
    const ast2: AST = { ...makeAst(), version: 2 };
    useAgentEditorStore.setState({
      ast: ast1,
      isDirty: false,
      validationErrors: [{ message: 'old error', path: null, line: null, kind: 'schema' }],
    });

    store().setAst(ast2);

    expect(store().ast?.version).toBe(2);
    expect(store().isDirty).toBe(true);
    expect(store().validationErrors).toEqual([]);
  });
});

describe('markValidationErrors', () => {
  it('sets validation errors without affecting isDirty', () => {
    useAgentEditorStore.setState({ isDirty: false });
    const errors = [{ message: 'bad', path: 'root', line: 1, kind: 'validation' as const }];
    store().markValidationErrors(errors);
    expect(store().validationErrors).toEqual(errors);
    expect(store().isDirty).toBe(false);
  });
});

describe('select', () => {
  it('sets selectedPath', () => {
    store().select('root.children.0');
    expect(store().selectedPath).toBe('root.children.0');
  });

  it('clears selectedPath when passed null', () => {
    useAgentEditorStore.setState({ selectedPath: 'root.children.0' });
    store().select(null);
    expect(store().selectedPath).toBeNull();
  });
});

describe('setPendingChatSeed', () => {
  it('sets pendingChatSeed to the provided text', () => {
    store().setPendingChatSeed('Please fix these validation issues');
    expect(store().pendingChatSeed).toBe('Please fix these validation issues');
  });

  it('clears pendingChatSeed when called with null', () => {
    useAgentEditorStore.setState({ pendingChatSeed: 'some seed' });
    store().setPendingChatSeed(null);
    expect(store().pendingChatSeed).toBeNull();
  });

  it('initialState has pendingChatSeed as null', () => {
    useAgentEditorStore.setState(initialState);
    expect(store().pendingChatSeed).toBeNull();
  });
});
