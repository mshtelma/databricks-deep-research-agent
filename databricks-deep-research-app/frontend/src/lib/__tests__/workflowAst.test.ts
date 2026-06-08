import { describe, expect, it } from 'vitest';

import { normalizeWorkflowAst } from '../workflowAst';

describe('normalizeWorkflowAst', () => {
  it('preserves plan-and-execute evaluator as agent config', () => {
    const ast = normalizeWorkflowAst({
      id: 'wf',
      name: 'Workflow',
      version: 1,
      required_inputs: ['query'],
      output_keys: ['report'],
      root: {
        id: 'root',
        type: 'sequence',
        label: 'Root',
        config: {},
        children: [
          {
            id: 'plan',
            type: 'plan_and_execute',
            label: 'Plan',
            config: {
              planner: {
                subtype: 'planner',
                output_key: 'research_plan',
                system_prompt: 'Plan.',
                user_prompt_template: '{query}',
              },
              evaluator: {
                subtype: 'reflector',
                output_key: 'evaluation',
                system_prompt: 'Evaluate.',
                user_prompt_template: '{query}',
              },
              body: {
                id: 'body',
                type: 'sequence',
                label: 'Body',
                config: {},
                children: [],
              },
            },
            children: [],
          },
        ],
      },
    });

    const planNode = ast.root.children?.[0];
    expect(planNode).toBeDefined();
    if (!planNode) throw new Error('Expected plan_and_execute node');
    const evaluator = planNode.config['evaluator'] as Record<string, unknown>;

    expect(evaluator['subtype']).toBe('reflector');
    expect(evaluator['output_key']).toBe('evaluation');
    expect(evaluator).not.toHaveProperty('type');
    expect(evaluator).not.toHaveProperty('children');
  });

  it('unwraps legacy evaluator nodes back to agent config', () => {
    const ast = normalizeWorkflowAst({
      id: 'wf',
      name: 'Workflow',
      version: 1,
      root: {
        id: 'root',
        type: 'sequence',
        label: 'Root',
        config: {},
        children: [
          {
            id: 'plan',
            type: 'plan_and_execute',
            label: 'Plan',
            config: {
              planner: { subtype: 'planner' },
              evaluator: {
                id: 'evaluator',
                type: 'agent',
                label: 'Evaluator',
                config: {
                  subtype: 'reflector',
                  output_key: 'evaluation',
                },
                children: [],
              },
              body: {
                id: 'body',
                type: 'sequence',
                label: 'Body',
                config: {},
                children: [],
              },
            },
            children: [],
          },
        ],
      },
    });

    const evaluator = ast.root.children?.[0]!.config['evaluator'] as Record<string, unknown>;

    expect(evaluator).toEqual({
      subtype: 'reflector',
      output_key: 'evaluation',
    });
  });
});
