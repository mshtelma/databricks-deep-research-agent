import type { AvailableSource, SourceScope } from '@/types/dataSources';
import type { QueryMode } from '@/types';

/** VariantA retrieval channels: Web, Enterprise, and MCP. */
export interface ComposerSources {
  web: boolean;
  ent: boolean;
  mcp: boolean;
}

/** VariantA primary mode: Answer (quick) vs Deep Research. */
export type ComposerMode = 'answer' | 'deep';

export function deriveQueryModeFromComposerState(
  mode: ComposerMode,
  sources: ComposerSources,
): QueryMode {
  if (mode === 'deep') return 'deep_research';
  return sources.web || sources.ent || sources.mcp ? 'web_search' : 'simple';
}

function isEnterpriseAvailableSource(source: AvailableSource): boolean {
  return source.type !== 'web_search' && source.type !== 'uploaded_file';
}

export function deriveSourceScopeFromComposerSources(sources: ComposerSources): SourceScope {
  if (sources.web && sources.ent) return 'all';
  if (sources.web) return 'web_only';
  // MCP-only still needs the legacy no-web scope. Fine-grained enabledSources
  // then decides whether any non-MCP enterprise tools are attached.
  return 'enterprise_only';
}

export function deriveEnabledSourceIdsForSubmit(
  availableSources: AvailableSource[],
  sources: ComposerSources,
): string[] {
  return availableSources
    .filter((source) => source.isEnabled)
    .filter((source) => {
      if (source.type === 'web_search') return sources.web;
      if (isEnterpriseAvailableSource(source)) return sources.ent;
      return false;
    })
    .map((source) => source.id);
}
