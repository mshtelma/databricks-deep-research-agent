import { describe, expect, it } from 'vitest';

import {
  computeAstFieldDiff,
  deepEqual,
  humanizeField,
  isNoiseField,
} from '../astFieldDiff';
import { createDraftWorkflow } from '../workflowAst';
import type { AST, Block } from '@/types/ast';

function agentBlock(id: string, label: string, config: Record<string, unknown>): Block {
  return { id, type: 'agent', label, config };
}

function workflowWith(children: Block[], tools: AST['tools'] = []): AST {
  const base = createDraftWorkflow('Test');
  return {
    ...base,
    id: 'wf-stable',
    root: { id: 'root-stable', type: 'sequence', label: 'Workflow', config: {}, children },
    tools,
  };
}

describe('computeAstFieldDiff', () => {
  it('surfaces update_workflow_meta field changes (run_as / output_keys)', () => {
    const oldAst = {
      ...workflowWith([agentBlock('a1', 'R', {})]),
      run_as: 'caller',
      output_keys: ['report'],
    } as unknown as AST;
    const newAst = {
      ...workflowWith([agentBlock('a1', 'R', {})]),
      run_as: 'sp-123',
      output_keys: ['report', 'summary'],
    } as unknown as AST;

    const diff = computeAstFieldDiff(oldAst, newAst);
    const fields = diff.fieldChanges
      .filter((c) => c.scopeKind === 'workflow')
      .map((c) => c.field);
    expect(fields).toContain('Run as');
    expect(fields).toContain('Output keys');
  });

  it('surfaces a system_prompt rewrite as one node field change', () => {
    const oldAst = workflowWith([agentBlock('a1', 'Researcher', { system_prompt: 'old' })]);
    const newAst = workflowWith([agentBlock('a1', 'Researcher', { system_prompt: 'new and longer' })]);

    const diff = computeAstFieldDiff(oldAst, newAst);

    expect(diff.fieldChanges).toHaveLength(1);
    expect(diff.fieldChanges[0]).toMatchObject({
      scope: 'Researcher',
      scopeKind: 'node',
      field: 'System prompt',
      kind: 'modified',
      oldValue: 'old',
      newValue: 'new and longer',
    });
    expect(diff.meaningfulCount).toBe(1);
  });

  it('surfaces a model_tier change', () => {
    const oldAst = workflowWith([agentBlock('a1', 'Synth', { model_tier: 'analytical' })]);
    const newAst = workflowWith([agentBlock('a1', 'Synth', { model_tier: 'complex' })]);

    const diff = computeAstFieldDiff(oldAst, newAst);
    expect(diff.fieldChanges).toEqual([
      expect.objectContaining({ field: 'Model tier', oldValue: 'analytical', newValue: 'complex' }),
    ]);
  });

  it('surfaces a tool binding change (config.tools) as "Bound tools"', () => {
    const oldAst = workflowWith([agentBlock('a1', 'Researcher', { tools: [] })]);
    const newAst = workflowWith([agentBlock('a1', 'Researcher', { tools: ['vector_search'] })]);

    const diff = computeAstFieldDiff(oldAst, newAst);
    expect(diff.fieldChanges).toEqual([
      expect.objectContaining({ field: 'Bound tools', kind: 'modified' }),
    ]);
  });

  it('surfaces a declared tool as an added tool change', () => {
    const oldAst = workflowWith([agentBlock('a1', 'R', {})], []);
    const newAst = workflowWith(
      [agentBlock('a1', 'R', {})],
      [{ name: 'vs', kind: 'vector_search', config: {} }],
    );

    const diff = computeAstFieldDiff(oldAst, newAst);
    expect(diff.fieldChanges).toEqual([
      expect.objectContaining({ scopeKind: 'tool', field: 'Tool declaration', kind: 'added' }),
    ]);
  });

  it('detects removed nodes for the Fix #5 banner', () => {
    const oldAst = workflowWith([
      agentBlock('a1', 'Keep', {}),
      agentBlock('a2', 'Drop', {}),
    ]);
    const newAst = workflowWith([agentBlock('a1', 'Keep', {})]);

    const diff = computeAstFieldDiff(oldAst, newAst);
    expect(diff.removedNodeCount).toBe(1);
    expect(diff.structural).toEqual([
      expect.objectContaining({ kind: 'node_removed', label: 'Drop' }),
    ]);
  });

  it('detects added nodes structurally without per-field rows', () => {
    const oldAst = workflowWith([agentBlock('a1', 'Keep', {})]);
    const newAst = workflowWith([
      agentBlock('a1', 'Keep', {}),
      agentBlock('a2', 'New', { system_prompt: 'x' }),
    ]);

    const diff = computeAstFieldDiff(oldAst, newAst);
    expect(diff.addedNodeCount).toBe(1);
    // The added node is reported structurally, not as N config field rows.
    expect(diff.fieldChanges).toHaveLength(0);
  });

  it('reports no changes when ASTs are semantically equal (key order differs)', () => {
    const oldAst = workflowWith([agentBlock('a1', 'R', { system_prompt: 'p', model_tier: 't' })]);
    const newAst = workflowWith([agentBlock('a1', 'R', { model_tier: 't', system_prompt: 'p' })]);

    const diff = computeAstFieldDiff(oldAst, newAst);
    expect(diff.fieldChanges).toHaveLength(0);
    expect(diff.meaningfulCount).toBe(0);
  });

  it('diffs plan_and_execute body nodes (config.body)', () => {
    const oldRoot: Block = {
      id: 'root-stable', type: 'sequence', label: 'W', config: {},
      children: [{
        id: 'pae', type: 'plan_and_execute', label: 'PAE',
        config: { body: agentBlock('inner', 'Inner', { system_prompt: 'old' }) },
      }],
    };
    const newRoot: Block = JSON.parse(JSON.stringify(oldRoot));
    const innerBody = newRoot.children![0]!.config.body as Block;
    innerBody.config.system_prompt = 'new';

    const oldAst: AST = { ...createDraftWorkflow('T'), id: 'wf', root: oldRoot, tools: [] };
    const newAst: AST = { ...createDraftWorkflow('T'), id: 'wf', root: newRoot, tools: [] };

    const diff = computeAstFieldDiff(oldAst, newAst);
    expect(diff.fieldChanges).toEqual([
      expect.objectContaining({ scope: 'Inner', field: 'System prompt' }),
    ]);
  });

  it('treats a null old AST (initial propose) as all-added, no field rows', () => {
    const newAst = workflowWith([agentBlock('a1', 'R', { system_prompt: 'p' })]);
    const diff = computeAstFieldDiff(null, newAst);
    expect(diff.fieldChanges).toHaveLength(0);
    expect(diff.addedNodeCount).toBeGreaterThanOrEqual(1);
  });
});

describe('helpers', () => {
  it('humanizeField maps known and unknown keys', () => {
    expect(humanizeField('system_prompt')).toBe('System prompt');
    expect(humanizeField('config.model_tier')).toBe('Model tier');
    expect(humanizeField('some_custom_key')).toBe('Some Custom Key');
  });

  it('isNoiseField flags id/node_id leaves only', () => {
    expect(isNoiseField('root.children.0.id')).toBe(true);
    expect(isNoiseField('root.children.0.config.node_id')).toBe(true);
    expect(isNoiseField('root.children.0.config.system_prompt')).toBe(false);
  });

  it('deepEqual is order-insensitive for object keys', () => {
    expect(deepEqual({ a: 1, b: 2 }, { b: 2, a: 1 })).toBe(true);
    expect(deepEqual([1, 2], [2, 1])).toBe(false);
    expect(deepEqual({ a: 1 }, { a: 2 })).toBe(false);
  });
});
