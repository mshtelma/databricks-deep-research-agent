import { describe, expect, it } from 'vitest';

import { buildDesignerSavePayload } from '../agentDesignerSave';
import { createDraftWorkflow } from '../workflowAst';

describe('buildDesignerSavePayload', () => {
  it('preserves designer-generated workflow name and description for new agents', () => {
    const ast = {
      ...createDraftWorkflow(),
      name: 'Launch readiness workflow',
      description: 'Assess user-provided launch ideas and produce a readiness brief',
    };

    const payload = buildDesignerSavePayload(ast, {
      isNew: true,
      localName: '',
      localDescription: '',
    });

    expect(payload.definition.name).toBe('Launch readiness workflow');
    expect(payload.definition.description).toBe(
      'Assess user-provided launch ideas and produce a readiness brief',
    );
    expect(payload.name).toBe('Launch readiness workflow');
    expect(payload.description).toBe(
      'Assess user-provided launch ideas and produce a readiness brief',
    );
  });

  it('does not strip applied designer intent from existing Untitled Agent definitions', () => {
    const ast = {
      ...createDraftWorkflow(),
      name: 'Workflow review brief',
      description: 'Review user-provided workflow ideas and produce an implementation brief',
    };

    const payload = buildDesignerSavePayload(ast, {
      isNew: false,
      agentName: 'Untitled Agent',
      agentDescription: null,
    });

    expect(payload.definition.name).toBe('Workflow review brief');
    expect(payload.definition.description).toBe(
      'Review user-provided workflow ideas and produce an implementation brief',
    );
    expect(payload.name).toBe('Workflow review brief');
    expect(payload.description).toBe(
      'Review user-provided workflow ideas and produce an implementation brief',
    );
  });

  it('keeps explicit existing agent metadata while preserving workflow description', () => {
    const ast = {
      ...createDraftWorkflow(),
      name: 'Designer workflow name',
      description: 'Designer workflow instructions',
    };

    const payload = buildDesignerSavePayload(ast, {
      isNew: false,
      agentName: 'Workflow Analyst',
      agentDescription: 'Visible list description',
    });

    expect(payload.definition.name).toBe('Designer workflow name');
    expect(payload.definition.description).toBe('Designer workflow instructions');
    expect(payload.name).toBe('Workflow Analyst');
    expect(payload.description).toBe('Visible list description');
  });
});
