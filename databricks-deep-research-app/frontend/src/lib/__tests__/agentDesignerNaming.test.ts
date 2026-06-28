import { describe, expect, it } from 'vitest';
import { semanticNodeLabel } from '../agentDesignerNaming';

describe('semanticNodeLabel', () => {
  it('replaces generic researcher ordinals with a role-specific fallback', () => {
    expect(
      semanticNodeLabel('agent', 'Researcher 2', { subtype: 'researcher' }),
    ).toBe('Evidence Researcher');
  });

  it('keeps meaningful labels unchanged', () => {
    expect(
      semanticNodeLabel('agent', 'Treasury Calendar Evidence', { subtype: 'researcher' }),
    ).toBe('Treasury Calendar Evidence');
  });

  it('uses tool refs for generic tool labels', () => {
    expect(semanticNodeLabel('tool', 'Tool 1', { ref: 'web_research' })).toBe(
      'Web Research Tool',
    );
  });
});
