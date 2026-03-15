/**
 * SourcesSection - Browse-style source configuration for custom agents.
 *
 * Features:
 * - Search by name/description
 * - Filter by source type
 * - Grouped by source type with collapsible headers
 * - Tri-state per source: neutral / enabled (always include) / disabled (always exclude)
 * - Capability badges and descriptions
 * - Stale source ID detection with clear button
 * - Refresh discovery button
 */

import * as React from 'react';
import { cn } from '@/lib/utils';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import {
  SearchIcon,
  RefreshIcon,
  ChevronIcon,
  SourceDatabaseIcon,
  SourceChartIcon,
  SourceBotIcon,
  CheckIcon,
  XIcon,
} from './sourceIcons';
import type { DiscoveredSource, DataSourceType } from '@/types/discovery';
import type { AgentSourceConfig } from '@/types/customAgents';
import type { SourceScope } from '@/types/dataSources';

// =============================================================================
// Props
// =============================================================================

interface SourcesSectionProps {
  config: AgentSourceConfig;
  sources: DiscoveredSource[];
  onChange: (updates: Partial<AgentSourceConfig>) => void;
  disabled: boolean;
  isLoadingSources?: boolean;
  sourcesError?: Error | null;
  onRefresh: () => void;
  isRefreshing: boolean;
}

// =============================================================================
// Constants
// =============================================================================

type SourceState = 'enabled' | 'disabled' | 'neutral';

interface SourceCategory {
  type: DataSourceType;
  label: string;
  icon: React.ReactNode;
  description: string;
}

const SOURCE_CATEGORIES: SourceCategory[] = [
  {
    type: 'vector_search',
    label: 'Vector Search',
    icon: <SourceDatabaseIcon className="h-4 w-4" />,
    description: 'Semantic search over indexed documents',
  },
  {
    type: 'genie',
    label: 'Genie Spaces',
    icon: <SourceChartIcon className="h-4 w-4" />,
    description: 'Natural language queries on structured data',
  },
  {
    type: 'knowledge_assistant',
    label: 'Serving Endpoints',
    icon: <SourceBotIcon className="h-4 w-4" />,
    description: 'Chat-compatible model endpoints',
  },
];

const SCOPE_OPTIONS: { value: SourceScope; label: string; description: string }[] = [
  { value: 'all', label: 'All Sources', description: 'Search both enterprise and web sources' },
  { value: 'enterprise_only', label: 'Enterprise Only', description: 'Search only internal enterprise data sources' },
  { value: 'web_only', label: 'Web Only', description: 'Search only public web sources' },
];

// =============================================================================
// Component
// =============================================================================

export function SourcesSection({
  config,
  sources,
  onChange,
  disabled,
  isLoadingSources,
  sourcesError,
  onRefresh,
  isRefreshing,
}: SourcesSectionProps) {
  const [searchQuery, setSearchQuery] = React.useState('');
  const [typeFilter, setTypeFilter] = React.useState<DataSourceType | 'all'>('all');
  const [expandedTypes, setExpandedTypes] = React.useState<Set<string>>(
    new Set(SOURCE_CATEGORIES.map((c) => c.type))
  );

  // Ready sources only
  const readySources = React.useMemo(
    () => sources.filter((s) => s.status === 'ready'),
    [sources]
  );

  // Filtered sources
  const filteredSources = React.useMemo(() => {
    let result = readySources;
    if (typeFilter !== 'all') {
      result = result.filter((s) => s.source_type === typeFilter);
    }
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      result = result.filter(
        (s) =>
          s.name.toLowerCase().includes(q) ||
          (s.description && s.description.toLowerCase().includes(q))
      );
    }
    return result;
  }, [readySources, typeFilter, searchQuery]);

  // Grouped by type
  const groupedSources = React.useMemo(() => {
    const groups: Partial<Record<DataSourceType, DiscoveredSource[]>> = {};
    filteredSources.forEach((s) => {
      (groups[s.source_type] ??= []).push(s);
    });
    return groups;
  }, [filteredSources]);

  // Stale IDs — configured but no longer in discovery
  const readyIds = React.useMemo(
    () => new Set(readySources.map((s) => s.source_id)),
    [readySources]
  );
  const staleIds = React.useMemo(() => {
    if (readySources.length === 0) return []; // Don't flag stale if discovery hasn't loaded
    return [
      ...config.enabledSources.filter((id) => !readyIds.has(id)),
      ...config.disabledSources.filter((id) => !readyIds.has(id)),
    ];
  }, [config.enabledSources, config.disabledSources, readyIds, readySources.length]);

  // Tri-state helpers
  const getSourceState = (sourceId: string): SourceState => {
    if (config.enabledSources.includes(sourceId)) return 'enabled';
    if (config.disabledSources.includes(sourceId)) return 'disabled';
    return 'neutral';
  };

  const cycleSourceState = (sourceId: string, targetState: SourceState) => {
    const currentState = getSourceState(sourceId);
    const enabledSet = new Set(config.enabledSources);
    const disabledSet = new Set(config.disabledSources);
    // Clear from both
    enabledSet.delete(sourceId);
    disabledSet.delete(sourceId);
    // If clicking already-active state, go neutral; otherwise apply target
    if (currentState !== targetState) {
      if (targetState === 'enabled') enabledSet.add(sourceId);
      if (targetState === 'disabled') disabledSet.add(sourceId);
    }
    onChange({
      enabledSources: Array.from(enabledSet),
      disabledSources: Array.from(disabledSet),
    });
  };

  const toggleType = (type: string) => {
    setExpandedTypes((prev) => {
      const next = new Set(prev);
      if (next.has(type)) next.delete(type);
      else next.add(type);
      return next;
    });
  };

  const clearStaleIds = () => {
    onChange({
      enabledSources: config.enabledSources.filter((id) => readyIds.has(id)),
      disabledSources: config.disabledSources.filter((id) => readyIds.has(id)),
    });
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h3 className="text-lg font-medium">Sources</h3>
        <p className="text-sm text-muted-foreground mt-1">
          Configure which data sources this agent can access.
        </p>
      </div>

      {/* Source Scope */}
      <div className="space-y-3">
        <label className="text-sm font-medium">Source Scope</label>
        <div className="grid grid-cols-3 gap-2">
          {SCOPE_OPTIONS.map((option) => (
            <button
              key={option.value}
              type="button"
              onClick={() => onChange({ scope: option.value })}
              disabled={disabled}
              className={cn(
                'p-3 rounded-lg border text-left transition-colors',
                config.scope === option.value
                  ? 'border-primary bg-primary/10'
                  : 'border-input hover:border-primary/50'
              )}
            >
              <p className="font-medium text-sm">{option.label}</p>
              <p className="text-xs text-muted-foreground mt-0.5">{option.description}</p>
            </button>
          ))}
        </div>
      </div>

      {/* Search / Filter / Refresh toolbar */}
      <div className="flex gap-2">
        <div className="relative flex-1">
          <SearchIcon className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search sources..."
            className="pl-9"
            disabled={disabled}
          />
        </div>
        <select
          value={typeFilter}
          onChange={(e) => setTypeFilter(e.target.value as DataSourceType | 'all')}
          disabled={disabled}
          className="rounded-md border border-input bg-background px-3 py-2 text-sm"
        >
          <option value="all">All Types</option>
          {SOURCE_CATEGORIES.map((c) => (
            <option key={c.type} value={c.type}>
              {c.label}
            </option>
          ))}
        </select>
        <Button
          variant="outline"
          size="icon"
          onClick={onRefresh}
          disabled={isRefreshing || disabled}
          title="Refresh sources"
        >
          <RefreshIcon className={cn('h-4 w-4', isRefreshing && 'animate-spin')} />
        </Button>
      </div>

      {/* Loading / Error / Empty states */}
      {isLoadingSources && (
        <p className="text-sm text-muted-foreground animate-pulse">
          Discovering available sources...
        </p>
      )}
      {sourcesError && !isLoadingSources && (
        <p className="text-sm text-destructive">
          Failed to discover data sources.
        </p>
      )}
      {!isLoadingSources && !sourcesError && readySources.length === 0 && (
        <p className="text-sm text-muted-foreground">
          No data sources discovered in this workspace.
        </p>
      )}

      {/* No search results */}
      {!isLoadingSources && readySources.length > 0 && filteredSources.length === 0 && (
        <p className="text-sm text-muted-foreground text-center py-4">
          No sources match your search
        </p>
      )}

      {/* Grouped source list */}
      {filteredSources.length > 0 && (
        <div className="space-y-3 max-h-[400px] overflow-y-auto">
          {SOURCE_CATEGORIES.map((category) => {
            const categorySources = groupedSources[category.type];
            if (!categorySources?.length) return null;
            const isExpanded = expandedTypes.has(category.type);
            const enabledCount = categorySources.filter(
              (s) => getSourceState(s.source_id) === 'enabled'
            ).length;
            const disabledCount = categorySources.filter(
              (s) => getSourceState(s.source_id) === 'disabled'
            ).length;

            return (
              <div key={category.type} className="border rounded-lg">
                {/* Category header */}
                <button
                  type="button"
                  onClick={() => toggleType(category.type)}
                  className={cn(
                    'w-full flex items-center justify-between p-3 text-left',
                    'hover:bg-muted/50 transition-colors rounded-t-lg',
                    !isExpanded && 'rounded-b-lg'
                  )}
                >
                  <div className="flex items-center gap-3">
                    <span className="text-muted-foreground">{category.icon}</span>
                    <div>
                      <h4 className="font-medium text-sm">{category.label}</h4>
                      <p className="text-xs text-muted-foreground">{category.description}</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    {enabledCount > 0 && (
                      <span className="text-xs px-1.5 py-0.5 rounded-full bg-green-100 text-green-700 dark:bg-green-900/40 dark:text-green-300">
                        +{enabledCount}
                      </span>
                    )}
                    {disabledCount > 0 && (
                      <span className="text-xs px-1.5 py-0.5 rounded-full bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-300">
                        -{disabledCount}
                      </span>
                    )}
                    <span className="text-xs bg-muted px-2 py-0.5 rounded-full">
                      {categorySources.length}
                    </span>
                    <ChevronIcon
                      className={cn('h-4 w-4 transition-transform', isExpanded && 'rotate-180')}
                    />
                  </div>
                </button>

                {/* Source items */}
                {isExpanded && (
                  <div className="p-3 pt-0 border-t space-y-1">
                    {categorySources.map((source) => (
                      <SourceRow
                        key={source.source_id}
                        source={source}
                        state={getSourceState(source.source_id)}
                        onEnable={() => cycleSourceState(source.source_id, 'enabled')}
                        onDisable={() => cycleSourceState(source.source_id, 'disabled')}
                        disabled={disabled}
                      />
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* Stale IDs warning */}
      {staleIds.length > 0 && (
        <div className="rounded-md border border-yellow-300 bg-yellow-50 dark:border-yellow-700 dark:bg-yellow-950/30 p-3">
          <div className="flex items-center justify-between">
            <p className="text-sm text-yellow-800 dark:text-yellow-200">
              {staleIds.length} configured source{staleIds.length > 1 ? 's are' : ' is'} no longer
              available in your workspace.
            </p>
            <Button
              variant="outline"
              size="sm"
              onClick={clearStaleIds}
              disabled={disabled}
              className="shrink-0 ml-3"
            >
              Clear stale
            </Button>
          </div>
        </div>
      )}

      {/* Footer */}
      <div className="text-xs text-muted-foreground">
        {config.enabledSources.length} enabled &middot; {config.disabledSources.length} disabled &middot;{' '}
        {readySources.length} total available
      </div>
    </div>
  );
}

// =============================================================================
// SourceRow
// =============================================================================

interface SourceRowProps {
  source: DiscoveredSource;
  state: SourceState;
  onEnable: () => void;
  onDisable: () => void;
  disabled: boolean;
}

function SourceRow({ source, state, onEnable, onDisable, disabled }: SourceRowProps) {
  const metadata = source.metadata as Record<string, unknown>;
  const isKA = metadata?.is_knowledge_assistant === true;

  return (
    <div
      className={cn(
        'flex items-start gap-3 p-2 rounded-md transition-colors',
        state === 'enabled' && 'bg-green-50 dark:bg-green-950/30',
        state === 'disabled' && 'bg-red-50 dark:bg-red-950/30'
      )}
    >
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium truncate">{source.name}</span>
          {source.source_type === 'knowledge_assistant' && (
            <span className="text-xs px-1.5 py-0.5 rounded bg-muted">
              {isKA ? 'KA' : 'Endpoint'}
            </span>
          )}
        </div>
        {source.description && (
          <p className="text-xs text-muted-foreground truncate mt-0.5">{source.description}</p>
        )}
        {source.capabilities && source.capabilities.length > 0 && (
          <div className="flex gap-1 mt-1 flex-wrap">
            {source.capabilities.slice(0, 3).map((cap) => (
              <span key={cap} className="text-xs px-1 py-0.5 rounded bg-muted/50">
                {cap}
              </span>
            ))}
          </div>
        )}
      </div>
      <div className="flex items-center gap-1 shrink-0 mt-0.5">
        <button
          type="button"
          onClick={onEnable}
          disabled={disabled}
          title={state === 'enabled' ? 'Remove from enabled' : 'Always include this source'}
          aria-label={state === 'enabled' ? 'Remove from enabled' : 'Enable source'}
          className={cn(
            'p-1 rounded transition-colors',
            state === 'enabled'
              ? 'bg-green-200 text-green-800 dark:bg-green-800 dark:text-green-200'
              : 'text-muted-foreground/40 hover:text-green-600 hover:bg-green-50 dark:hover:bg-green-950/50',
            disabled && 'opacity-50 cursor-not-allowed'
          )}
        >
          <CheckIcon className="h-3.5 w-3.5" />
        </button>
        <button
          type="button"
          onClick={onDisable}
          disabled={disabled}
          title={state === 'disabled' ? 'Remove from disabled' : 'Always exclude this source'}
          aria-label={state === 'disabled' ? 'Remove from disabled' : 'Disable source'}
          className={cn(
            'p-1 rounded transition-colors',
            state === 'disabled'
              ? 'bg-red-200 text-red-800 dark:bg-red-800 dark:text-red-200'
              : 'text-muted-foreground/40 hover:text-red-600 hover:bg-red-50 dark:hover:bg-red-950/50',
            disabled && 'opacity-50 cursor-not-allowed'
          )}
        >
          <XIcon className="h-3.5 w-3.5" />
        </button>
      </div>
    </div>
  );
}

export default SourcesSection;
