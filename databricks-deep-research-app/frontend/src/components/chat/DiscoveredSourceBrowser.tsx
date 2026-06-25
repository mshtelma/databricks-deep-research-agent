/**
 * DiscoveredSourceBrowser - Browse and select from discovered data sources.
 *
 * Features:
 * - Multi-select with checkboxes
 * - Search/filter by name
 * - Type filter dropdown
 * - "Show all serving endpoints" toggle
 * - Refresh button
 * - Grouped by source type
 */

import * as React from 'react';
import { Boxes } from 'lucide-react';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import type { DiscoveredSource, DataSourceType } from '@/types/discovery';

interface DiscoveredSourceBrowserProps {
  /** Currently selected source IDs */
  selectedIds: string[];
  /** Callback when selection changes */
  onSelectionChange: (ids: string[]) => void;
  /** Pre-fetched sources from parent — no internal fetch */
  sources: DiscoveredSource[];
  /** Whether discovery is loading */
  isLoading: boolean;
  /** Discovery error, if any */
  error: Error | null;
  /** Re-run the discovery query (retry on error) */
  onRefetch: () => void;
  /** Force backend cache invalidation + refetch */
  onRefresh: () => void;
  /** Whether refresh mutation is in flight */
  isRefreshing: boolean;
  /** Filter to specific source types */
  allowedTypes?: DataSourceType[];
  /** Maximum height */
  maxHeight?: string;
  /** Additional CSS classes */
  className?: string;
}

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
    icon: <DatabaseIcon className="h-4 w-4" />,
    description: 'Semantic search over indexed documents',
  },
  {
    type: 'delta_table',
    label: 'Delta Tables',
    icon: <DatabaseIcon className="h-4 w-4" />,
    description: 'Structured tables queried through SQL warehouses',
  },
  {
    type: 'genie',
    label: 'Genie Spaces',
    icon: <ChartIcon className="h-4 w-4" />,
    description: 'Natural language queries on structured data',
  },
  {
    type: 'knowledge_assistant',
    label: 'Serving Endpoints',
    icon: <BotIcon className="h-4 w-4" />,
    description: 'Chat-compatible model endpoints',
  },
  {
    type: 'web_search',
    label: 'Web Search',
    icon: <GlobeIcon className="h-4 w-4" />,
    description: 'Public web search',
  },
  {
    type: 'mcp_server',
    label: 'MCP Servers',
    icon: <Boxes className="h-4 w-4" />,
    description: 'Model Context Protocol tool servers',
  },
];

export function DiscoveredSourceBrowser({
  selectedIds,
  onSelectionChange,
  sources,
  isLoading,
  error,
  onRefetch,
  onRefresh,
  isRefreshing,
  allowedTypes,
  maxHeight = '400px',
  className,
}: DiscoveredSourceBrowserProps) {
  const [searchQuery, setSearchQuery] = React.useState('');
  const [typeFilter, setTypeFilter] = React.useState<DataSourceType | 'all'>('all');
  const [expandedTypes, setExpandedTypes] = React.useState<Set<string>>(
    new Set(SOURCE_CATEGORIES.map((c) => c.type))
  );
  const [showAllEndpoints, setShowAllEndpoints] = React.useState(false);

  // Filter sources
  const filteredSources = React.useMemo(() => {
    let filtered = sources.filter((s) => s.status === 'ready');

    // Filter non-KA endpoints unless showAllEndpoints is toggled
    if (!showAllEndpoints) {
      filtered = filtered.filter((s) => {
        if (s.source_type !== 'knowledge_assistant') return true;
        const metadata = s.metadata as Record<string, unknown>;
        return metadata?.is_knowledge_assistant === true;
      });
    }

    // Type filter
    if (typeFilter !== 'all') {
      filtered = filtered.filter((s) => s.source_type === typeFilter);
    }

    // Allowed types filter
    if (allowedTypes && allowedTypes.length > 0) {
      filtered = filtered.filter((s) => allowedTypes.includes(s.source_type));
    }

    // Search filter
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(
        (s) =>
          s.name.toLowerCase().includes(query) ||
          s.source_id.toLowerCase().includes(query) ||
          (s.description && s.description.toLowerCase().includes(query))
      );
    }

    return filtered;
  }, [sources, showAllEndpoints, typeFilter, allowedTypes, searchQuery]);

  // Group by type
  const groupedSources = React.useMemo(() => {
    const groups: Record<DataSourceType, DiscoveredSource[]> = {
      vector_search: [],
      delta_table: [],
      genie: [],
      knowledge_assistant: [],
      web_search: [],
      uploaded_file: [],
      mcp_server: [],
      custom: [],
    };
    filteredSources.forEach((s) => {
      groups[s.source_type]?.push(s);
    });
    return groups;
  }, [filteredSources]);

  const toggleSource = (sourceId: string) => {
    if (selectedIds.includes(sourceId)) {
      onSelectionChange(selectedIds.filter((id) => id !== sourceId));
    } else {
      onSelectionChange([...selectedIds, sourceId]);
    }
  };

  const toggleType = (type: string) => {
    setExpandedTypes((prev) => {
      const next = new Set(prev);
      if (next.has(type)) {
        next.delete(type);
      } else {
        next.add(type);
      }
      return next;
    });
  };

  const handleRefresh = () => {
    onRefresh();
  };

  const filteredCategories = SOURCE_CATEGORIES.filter(
    (cat) => !allowedTypes || allowedTypes.length === 0 || allowedTypes.includes(cat.type)
  );

  if (error) {
    return (
      <div className={cn('p-8 text-center', className)}>
        <p className="text-sm text-destructive mb-2">Failed to load sources</p>
        <Button variant="outline" size="sm" onClick={() => onRefetch()}>
          Retry
        </Button>
      </div>
    );
  }

  return (
    <div className={cn('flex flex-col', className)}>
      {/* Header: Search + Filter + Refresh */}
      <div className="p-3 border-b space-y-3">
        <div className="flex gap-2">
          <div className="relative flex-1">
            <SearchIcon className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search sources..."
              className="pl-9"
            />
          </div>
          <select
            value={typeFilter}
            onChange={(e) => setTypeFilter(e.target.value as DataSourceType | 'all')}
            className="rounded-md border border-input bg-background px-3 py-2 text-sm"
          >
            <option value="all">All Types</option>
            {filteredCategories.map((cat) => (
              <option key={cat.type} value={cat.type}>
                {cat.label}
              </option>
            ))}
          </select>
          <Button
            variant="outline"
            size="icon"
            onClick={handleRefresh}
            disabled={isRefreshing}
            title="Refresh sources"
          >
            <RefreshIcon
              className={cn(
                'h-4 w-4',
                isRefreshing && 'animate-spin'
              )}
            />
          </Button>
        </div>

        {/* Show all endpoints toggle (client-side filter) */}
        <label className="flex items-center gap-2 text-sm cursor-pointer">
          <input
            type="checkbox"
            checked={showAllEndpoints}
            onChange={(e) => setShowAllEndpoints(e.target.checked)}
            className="rounded border-input"
          />
          <span className="text-muted-foreground">
            Show all serving endpoints (not just Knowledge Assistants)
          </span>
        </label>
      </div>

      {/* Loading state */}
      {isLoading && (
        <div className="p-8 text-center text-sm text-muted-foreground">
          Discovering sources...
        </div>
      )}

      {/* Sources list */}
      {!isLoading && (
        <div className="flex-1 overflow-y-auto" style={{ maxHeight }}>
          <div className="p-3 space-y-3">
            {filteredCategories.map((category) => {
              const categorySources = groupedSources[category.type];
              if (categorySources.length === 0) return null;

              const isExpanded = expandedTypes.has(category.type);

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
                        <p className="text-xs text-muted-foreground">
                          {category.description}
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-xs bg-muted px-2 py-0.5 rounded-full">
                        {categorySources.filter((s) => selectedIds.includes(s.source_id)).length}/{categorySources.length}
                      </span>
                      <ChevronIcon
                        className={cn(
                          'h-4 w-4 transition-transform',
                          isExpanded && 'rotate-180'
                        )}
                      />
                    </div>
                  </button>

                  {/* Source items */}
                  {isExpanded && (
                    <div className="p-3 pt-0 border-t space-y-1">
                      {categorySources.map((source) => (
                        <SourceItem
                          key={source.source_id}
                          source={source}
                          isSelected={selectedIds.includes(source.source_id)}
                          onToggle={() => toggleSource(source.source_id)}
                        />
                      ))}
                    </div>
                  )}
                </div>
              );
            })}

            {filteredSources.length === 0 && (
              <div className="p-8 text-center text-sm text-muted-foreground">
                No sources match your filters
              </div>
            )}
          </div>
        </div>
      )}

      {/* Footer: Selection count */}
      <div className="p-3 border-t text-xs text-muted-foreground">
        {selectedIds.length} of {sources.filter((s) => s.status === 'ready').length} sources selected
      </div>
    </div>
  );
}

interface SourceItemProps {
  source: DiscoveredSource;
  isSelected: boolean;
  onToggle: () => void;
}

function SourceItem({ source, isSelected, onToggle }: SourceItemProps) {
  const metadata = source.metadata as Record<string, unknown>;
  const isKA = metadata?.is_knowledge_assistant === true;

  return (
    <label
      className={cn(
        'flex items-start gap-3 p-2 rounded-md cursor-pointer',
        'hover:bg-muted/50 transition-colors',
        isSelected && 'bg-primary/5'
      )}
    >
      <input
        type="checkbox"
        checked={isSelected}
        onChange={onToggle}
        className="mt-0.5 rounded border-input"
      />
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className={cn('text-sm font-medium truncate', !isSelected && 'text-muted-foreground')}>
            {source.name}
          </span>
          {source.source_type === 'knowledge_assistant' && (
            <span className="text-xs px-1.5 py-0.5 rounded bg-muted">
              {isKA ? 'KA' : 'Endpoint'}
            </span>
          )}
        </div>
        {source.description && (
          <p className="text-xs text-muted-foreground truncate mt-0.5">
            {source.description}
          </p>
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
    </label>
  );
}

// Icons (inline SVGs for self-containment)
function SearchIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="11" cy="11" r="8" />
      <path d="m21 21-4.3-4.3" />
    </svg>
  );
}

function RefreshIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
      <path d="M3 3v5h5" />
      <path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16" />
      <path d="M16 21h5v-5" />
    </svg>
  );
}

function ChevronIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="m6 9 6 6 6-6" />
    </svg>
  );
}

function DatabaseIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <ellipse cx="12" cy="5" rx="9" ry="3" />
      <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" />
      <path d="M3 12c0 1.66 4 3 9 3s9-1.34 9-3" />
    </svg>
  );
}

function ChartIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M3 3v18h18" />
      <path d="m19 9-5 5-4-4-3 3" />
    </svg>
  );
}

function BotIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M12 8V4H8" />
      <rect width="16" height="12" x="4" y="8" rx="2" />
      <path d="M2 14h2M20 14h2M15 13v2M9 13v2" />
    </svg>
  );
}

function GlobeIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <circle cx="12" cy="12" r="10" />
      <path d="M2 12h20M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
    </svg>
  );
}

export default DiscoveredSourceBrowser;
