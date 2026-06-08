import { describe, it, expect } from 'vitest';
import {
  pathSegments,
  depth,
  parentPath,
  childIndex,
  appendChildPath,
  isDescendant,
  resolveBlock,
} from '@/lib/blockPath';
import { createDraftWorkflow } from '@/lib/workflowAst';
import type { AST, Block } from '@/types/ast';

// ---------------------------------------------------------------------------
// Fixture helpers
// ---------------------------------------------------------------------------

function makeBlock(id: string, overrides: Partial<Block> = {}): Block {
  return {
    id,
    type: 'agent',
    label: id,
    config: {},
    ...overrides,
  };
}

/**
 * Minimal AST used across most tests:
 *
 *   root (sequence)
 *     children[0]: blockA (agent)
 *       children[0]: blockB (agent)
 *       children[1]: blockC (agent)
 *     children[1]: blockD (agent)
 */
function makeSimpleAST(): AST {
  const blockB = makeBlock('blockB');
  const blockC = makeBlock('blockC');
  const blockA = makeBlock('blockA', {
    type: 'sequence',
    children: [blockB, blockC],
  });
  const blockD = makeBlock('blockD');
  const root = makeBlock('root', {
    type: 'sequence',
    children: [blockA, blockD],
  });
  return { ...createDraftWorkflow('Test Workflow'), root };
}

/**
 * AST with a plan_and_execute node whose body is a sequence.
 *
 * Path layout (matching mutations.py body semantics):
 *   root.children.0                     — plan_and_execute node
 *   root.children.0.config.body         — sequence wrapper
 *   root.children.0.config.body.children.0 — bodyChild0
 *   root.children.0.config.body.children.1 — bodyChild1
 */
function makePlanExecuteAST(): AST {
  const bodyChild0 = makeBlock('bodyChild0');
  const bodyChild1 = makeBlock('bodyChild1');
  const bodySequence: Block = {
    id: 'bodySeq',
    type: 'sequence',
    label: 'body',
    config: {},
    children: [bodyChild0, bodyChild1],
  };
  const planNode: Block = {
    id: 'plan1',
    type: 'plan_and_execute',
    label: 'Plan & Execute',
    config: { body: bodySequence },
  };
  const root = makeBlock('root', {
    type: 'sequence',
    children: [planNode],
  });
  return { ...createDraftWorkflow('Test Workflow'), root };
}

// ---------------------------------------------------------------------------
// resolveBlock
// ---------------------------------------------------------------------------

describe('resolveBlock', () => {
  it('resolveBlock for root returns root block', () => {
    const ast = makeSimpleAST();
    const result = resolveBlock(ast, 'root');
    expect(result).toBe(ast.root);
  });

  it('resolveBlock for root.children.0 returns first child of root', () => {
    const ast = makeSimpleAST();
    const result = resolveBlock(ast, 'root.children.0');
    expect(result).toBe(ast.root.children![0]);
    expect(result?.id).toBe('blockA');
  });

  it('resolveBlock for root.children.0.children.1 returns nested child', () => {
    const ast = makeSimpleAST();
    const result = resolveBlock(ast, 'root.children.0.children.1');
    const firstChild = ast.root.children![0]!;
    expect(result).toBe(firstChild.children![1]);
    expect(result?.id).toBe('blockC');
  });

  it('resolveBlock for missing path returns null', () => {
    const ast = makeSimpleAST();
    expect(resolveBlock(ast, 'root.children.99')).toBeNull();
    expect(resolveBlock(ast, 'root.children.0.children.99')).toBeNull();
  });

  it('resolveBlock through plan_and_execute body works — path root.children.0.config.body.children.0 (mutations.py config.body semantics)', () => {
    const ast = makePlanExecuteAST();
    // plan_and_execute body lives in config.body per mutations.py line 187:
    //   if parent_path.endswith("config.body"): ...
    // Children of the body sequence are at config.body.children.N
    const result = resolveBlock(ast, 'root.children.0.config.body.children.0');
    expect(result).not.toBeNull();
    expect(result?.id).toBe('bodyChild0');
  });

  it('resolveBlock through plan_and_execute body — second child at root.children.0.config.body.children.1', () => {
    const ast = makePlanExecuteAST();
    const result = resolveBlock(ast, 'root.children.0.config.body.children.1');
    expect(result).not.toBeNull();
    expect(result?.id).toBe('bodyChild1');
  });

  it('resolveBlock returns actual reference, not a clone', () => {
    const ast = makeSimpleAST();
    const result = resolveBlock(ast, 'root.children.1');
    expect(result).toBe(ast.root.children![1]);
  });
});

// ---------------------------------------------------------------------------
// parentPath
// ---------------------------------------------------------------------------

describe('parentPath', () => {
  it("parentPath of 'root' returns null", () => {
    expect(parentPath('root')).toBeNull();
  });

  it("parentPath of 'root.children.0' returns 'root'", () => {
    expect(parentPath('root.children.0')).toBe('root');
  });

  it("parentPath of 'root.children.0.children.1' returns 'root.children.0'", () => {
    expect(parentPath('root.children.0.children.1')).toBe('root.children.0');
  });
});

// ---------------------------------------------------------------------------
// isDescendant
// ---------------------------------------------------------------------------

describe('isDescendant', () => {
  it('self is NOT a descendant of itself', () => {
    expect(isDescendant('root', 'root')).toBe(false);
    expect(isDescendant('root.children.0', 'root.children.0')).toBe(false);
  });

  it("'root' is ancestor of 'root.children.0'", () => {
    expect(isDescendant('root', 'root.children.0')).toBe(true);
  });

  it("'root.children.0' and 'root.children.1' — neither is descendant of the other", () => {
    expect(isDescendant('root.children.0', 'root.children.1')).toBe(false);
    expect(isDescendant('root.children.1', 'root.children.0')).toBe(false);
  });

  it("deep ancestor is ancestor of deep descendant", () => {
    expect(isDescendant('root', 'root.children.0.children.1')).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// depth
// ---------------------------------------------------------------------------

describe('depth', () => {
  it("depth of 'root' is 0", () => {
    expect(depth('root')).toBe(0);
  });

  it("depth of 'root.children.0' is 1", () => {
    expect(depth('root.children.0')).toBe(1);
  });

  it("depth of 'root.children.0.children.1' is 2", () => {
    expect(depth('root.children.0.children.1')).toBe(2);
  });
});

// ---------------------------------------------------------------------------
// childIndex
// ---------------------------------------------------------------------------

describe('childIndex', () => {
  it("childIndex of 'root.children.5' is 5", () => {
    expect(childIndex('root.children.5')).toBe(5);
  });

  it("childIndex of 'root' is null", () => {
    expect(childIndex('root')).toBeNull();
  });

  it("childIndex of 'root.children.0.children.3' is 3", () => {
    expect(childIndex('root.children.0.children.3')).toBe(3);
  });
});

// ---------------------------------------------------------------------------
// appendChildPath
// ---------------------------------------------------------------------------

describe('appendChildPath', () => {
  it('appendChildPath produces correct dot-form', () => {
    expect(appendChildPath('root', 0)).toBe('root.children.0');
    expect(appendChildPath('root.children.0', 2)).toBe('root.children.0.children.2');
  });
});

// ---------------------------------------------------------------------------
// pathSegments
// ---------------------------------------------------------------------------

describe('pathSegments', () => {
  it("pathSegments of 'root.children.0' returns ['root', 'children', '0']", () => {
    expect(pathSegments('root.children.0')).toEqual(['root', 'children', '0']);
  });

  it("pathSegments of 'root' returns ['root']", () => {
    expect(pathSegments('root')).toEqual(['root']);
  });
});
