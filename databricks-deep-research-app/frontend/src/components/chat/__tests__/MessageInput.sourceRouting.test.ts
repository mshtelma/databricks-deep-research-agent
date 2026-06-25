import { describe, expect, it } from 'vitest';
import type { AvailableSource } from '@/types/dataSources';
import {
  deriveEnabledMcpServerNamesForSubmit,
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
  {
    id: 'mcp:tavily_mcp',
    name: 'tavily_mcp',
    type: 'mcp_server',
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

  it('maps MCP-only selection to no-web scope and selected MCP source IDs', () => {
    const composerSources = { web: false, ent: false, mcp: true };

    expect(deriveSourceScopeFromComposerSources(composerSources)).toBe('enterprise_only');
    expect(deriveEnabledSourceIdsForSubmit(sources, composerSources)).toEqual([
      'mcp:tavily_mcp',
    ]);
    expect(deriveEnabledMcpServerNamesForSubmit(sources, composerSources)).toEqual([
      'tavily_mcp',
    ]);
  });

  it('maps Web plus MCP to all scope so MCP is not disabled as web-only', () => {
    expect(
      deriveSourceScopeFromComposerSources({ web: true, ent: false, mcp: true }),
    ).toBe('all');
  });

  it('keeps selected enterprise sources only when the enterprise channel is enabled', () => {
    expect(
      deriveEnabledSourceIdsForSubmit(sources, { web: false, ent: true, mcp: false }),
    ).toEqual(['vs:catalog.schema.index']);
  });

  it('keeps web search only when the web channel is enabled', () => {
    expect(
      deriveEnabledSourceIdsForSubmit(sources, { web: true, ent: false, mcp: false }),
    ).toEqual(['web_search']);
  });

  it('keeps MCP sources only when the MCP channel is enabled', () => {
    expect(
      deriveEnabledSourceIdsForSubmit(sources, { web: false, ent: true, mcp: true }),
    ).toEqual(['vs:catalog.schema.index', 'mcp:tavily_mcp']);

    expect(
      deriveEnabledMcpServerNamesForSubmit(sources, {
        web: false,
        ent: true,
        mcp: false,
      }),
    ).toEqual([]);
  });
});
