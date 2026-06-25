import { describe, expect, it } from 'vitest';
import type { AvailableSource } from '@/types/dataSources';
import {
  deriveEnabledSourceIdsForSubmit,
  deriveQueryModeFromComposerState,
  deriveSourceScopeFromComposerSources,
} from '../sourceRouting';

const sources: AvailableSource[] = [
  {
    id: 'web_search',
    name: 'Web Search',
    type: 'web_search',
    description: null,
    isEnabled: true,
  },
  {
    id: 'vs:catalog.schema.index',
    name: 'Finance Docs',
    type: 'vector_search',
    description: null,
    isEnabled: true,
  },
];

describe('MessageInput source routing', () => {
  it('routes Answer plus MCP-only to lightweight search, not model-only simple', () => {
    expect(
      deriveQueryModeFromComposerState('answer', { web: false, ent: false, mcp: true }),
    ).toBe('web_search');
  });

  it('routes Answer with no sources to model-only simple', () => {
    expect(
      deriveQueryModeFromComposerState('answer', { web: false, ent: false, mcp: false }),
    ).toBe('simple');
  });

  it('routes Deep mode to deep research regardless of selected sources', () => {
    expect(
      deriveQueryModeFromComposerState('deep', { web: false, ent: false, mcp: true }),
    ).toBe('deep_research');
  });

  it('maps MCP-only selection to no-web scope and no enabled built-in sources', () => {
    const composerSources = { web: false, ent: false, mcp: true };

    expect(deriveSourceScopeFromComposerSources(composerSources)).toBe('enterprise_only');
    expect(deriveEnabledSourceIdsForSubmit(sources, composerSources)).toEqual([]);
  });

  it('keeps selected enterprise sources only when the enterprise channel is enabled', () => {
    expect(
      deriveEnabledSourceIdsForSubmit(sources, { web: false, ent: true, mcp: true }),
    ).toEqual(['vs:catalog.schema.index']);
  });

  it('keeps web search only when the web channel is enabled', () => {
    expect(
      deriveEnabledSourceIdsForSubmit(sources, { web: true, ent: false, mcp: true }),
    ).toEqual(['web_search']);
  });
});
