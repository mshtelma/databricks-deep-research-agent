/**
 * SourceList - Display discovered data sources grouped by type with collapsible sections.
 *
 * Features:
 * - Type grouping with collapsible sections
 * - Source count per type
 * - Status indicators (ready/syncing/unavailable/error)
 * - Refresh button for cache invalidation
 * - Loading and error states
 *
 * Task: T010m [US9a]
 */

import React, { useState, useMemo, useCallback } from 'react';
import { useDiscoveredSources, useRefreshDiscovery } from '@/hooks/useDiscoveredSources';
import { DiscoveredSourceCard } from './DiscoveredSourceCard';
import type { DiscoveredSource, DataSourceType, DiscoveryStatus } from '@/types/discovery';
import { getSourceTypeLabel } from '@/types/discovery';

interface SourceListProps {
  /** Callback when a source is selected */
  onSourceSelect?: (source: DiscoveredSource) => void;
  /** Currently selected source IDs */
  selectedSourceIds?: string[];
  /** Filter by specific source types */
  allowedTypes?: DataSourceType[];
  /** Show detailed metadata in cards */
  showMetadata?: boolean;
  /** Enable expandable metadata panel */
  expandable?: boolean;
  /** Compact mode with smaller cards */
  compact?: boolean;
  /** Additional CSS class */
  className?: string;
}

/** Type group configuration */
interface TypeGroup {
  type: DataSourceType;
  label: string;
  icon: React.ReactNode;
  description: string;
}

const TYPE_GROUPS: TypeGroup[] = [
  {
    type: 'vector_search',
    label: 'Vector Search',
    icon: <DatabaseIcon className="h-5 w-5" />,
    description: 'Semantic search over indexed documents',
  },
  {
    type: 'delta_table',
    label: 'Delta Tables',
    icon: <DatabaseIcon className="h-5 w-5" />,
    description: 'Structured tables queried through SQL warehouses',
  },
  {
    type: 'genie',
    label: 'Genie Spaces',
    icon: <SparklesIcon className="h-5 w-5" />,
    description: 'Natural language queries on structured data',
  },
  {
    type: 'knowledge_assistant',
    label: 'Knowledge Assistants',
    icon: <UserCircleIcon className="h-5 w-5" />,
    description: 'Domain expert AI assistants',
  },
  {
    type: 'web_search',
    label: 'Web Search',
    icon: <GlobeIcon className="h-5 w-5" />,
    description: 'Search the public web',
  },
  {
    type: 'uploaded_file',
    label: 'Uploaded Files',
    icon: <DocumentIcon className="h-5 w-5" />,
    description: 'User-uploaded documents',
  },
];

export const SourceList: React.FC<SourceListProps> = ({
  onSourceSelect,
  selectedSourceIds = [],
  allowedTypes,
  showMetadata = true,
  expandable = true,
  compact = false,
  className = '',
}) => {
  const [expandedTypes, setExpandedTypes] = useState<Set<DataSourceType>>(
    new Set(['vector_search', 'delta_table', 'genie', 'knowledge_assistant'])
  );
  const [searchQuery, setSearchQuery] = useState('');

  const { data, isLoading, error } = useDiscoveredSources();
  const { mutate: refresh, isPending: isRefreshing } = useRefreshDiscovery();

  // Filter type groups
  const filteredGroups = useMemo(() => {
    if (!allowedTypes || allowedTypes.length === 0) return TYPE_GROUPS;
    return TYPE_GROUPS.filter((g) => allowedTypes.includes(g.type));
  }, [allowedTypes]);

  // Group sources by type
  const sourcesByType = useMemo(() => {
    const grouped: Record<DataSourceType, DiscoveredSource[]> = {
      vector_search: [],
      delta_table: [],
      genie: [],
      knowledge_assistant: [],
      web_search: [],
      uploaded_file: [],
      custom: [],
    };

    if (!data?.sources) return grouped;

    for (const source of data.sources) {
      const type = source.source_type as DataSourceType;
      if (grouped[type]) {
        grouped[type].push(source);
      }
    }

    // Filter by search query if present
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      for (const type of Object.keys(grouped) as DataSourceType[]) {
        grouped[type] = grouped[type].filter(
          (s) =>
            s.name.toLowerCase().includes(query) ||
            s.description?.toLowerCase().includes(query) ||
            s.source_id.toLowerCase().includes(query)
        );
      }
    }

    return grouped;
  }, [data?.sources, searchQuery]);

  // Calculate status counts
  const statusCounts = useMemo(() => {
    const counts: Record<DiscoveryStatus, number> = {
      ready: 0,
      syncing: 0,
      unavailable: 0,
      error: 0,
    };

    if (!data?.sources) return counts;

    for (const source of data.sources) {
      const status = source.status as DiscoveryStatus;
      if (counts[status] !== undefined) {
        counts[status]++;
      }
    }

    return counts;
  }, [data?.sources]);

  const toggleType = useCallback((type: DataSourceType) => {
    setExpandedTypes((prev) => {
      const next = new Set(prev);
      if (next.has(type)) {
        next.delete(type);
      } else {
        next.add(type);
      }
      return next;
    });
  }, []);

  const handleRefresh = useCallback(() => {
    refresh(undefined);
  }, [refresh]);

  const isSelected = (sourceId: string) => selectedSourceIds.includes(sourceId);

  // Loading state
  if (isLoading) {
    return (
      <div className={`flex flex-col items-center justify-center py-12 ${className}`}>
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mb-4"></div>
        <p className="text-sm text-gray-500">Discovering available data sources...</p>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className={`bg-red-50 border border-red-200 rounded-lg p-4 ${className}`}>
        <div className="flex items-start gap-3">
          <ExclamationCircleIcon className="h-5 w-5 text-red-500 flex-shrink-0 mt-0.5" />
          <div>
            <h3 className="text-sm font-medium text-red-800">Failed to discover sources</h3>
            <p className="text-sm text-red-700 mt-1">{error.message}</p>
            <button
              onClick={handleRefresh}
              className="mt-2 text-sm text-red-600 hover:text-red-800 underline"
            >
              Try again
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={`flex flex-col gap-4 ${className}`}>
      {/* Header with search and refresh */}
      <div className="flex items-center justify-between gap-4">
        {/* Search input */}
        <div className="flex-1 relative">
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search sources..."
            className="w-full pl-10 pr-4 py-2 text-sm border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <SearchIcon className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400" />
        </div>

        {/* Status summary */}
        <div className="flex items-center gap-2">
          {statusCounts.ready > 0 && (
            <span className="flex items-center gap-1 text-xs text-green-600">
              <span className="h-2 w-2 rounded-full bg-green-500"></span>
              {statusCounts.ready} ready
            </span>
          )}
          {statusCounts.syncing > 0 && (
            <span className="flex items-center gap-1 text-xs text-yellow-600">
              <span className="h-2 w-2 rounded-full bg-yellow-500"></span>
              {statusCounts.syncing} syncing
            </span>
          )}
          {statusCounts.error > 0 && (
            <span className="flex items-center gap-1 text-xs text-red-600">
              <span className="h-2 w-2 rounded-full bg-red-500"></span>
              {statusCounts.error} errors
            </span>
          )}
        </div>

        {/* Refresh button */}
        <button
          onClick={handleRefresh}
          disabled={isRefreshing}
          className="flex items-center gap-2 px-3 py-2 text-sm text-gray-600 hover:text-gray-900 border border-gray-300 rounded-lg hover:bg-gray-50 disabled:opacity-50"
          title="Refresh sources"
        >
          <RefreshIcon className={`h-4 w-4 ${isRefreshing ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {/* Cache info */}
      {data?.cached && (
        <div className="text-xs text-gray-500">
          Showing cached results. Last discovered:{' '}
          {new Date(data.discovered_at).toLocaleTimeString()}
          {data.cache_expires_at && (
            <span> · Expires: {new Date(data.cache_expires_at).toLocaleTimeString()}</span>
          )}
        </div>
      )}

      {/* Errors from partial failures */}
      {data?.errors && data.errors.length > 0 && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3">
          <div className="flex items-start gap-2">
            <ExclamationTriangleIcon className="h-4 w-4 text-yellow-500 flex-shrink-0 mt-0.5" />
            <div className="text-xs text-yellow-700">
              <span className="font-medium">Some sources could not be discovered:</span>
              <ul className="mt-1 list-disc list-inside">
                {data.errors.map((err, idx) => (
                  <li key={idx}>
                    {getSourceTypeLabel(err.source_type)}: {err.error_message}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}

      {/* Type groups */}
      <div className="space-y-3">
        {filteredGroups.map((group) => {
          const sources = sourcesByType[group.type] || [];
          const isExpanded = expandedTypes.has(group.type);
          const hasReadySources = sources.some((s) => s.status === 'ready');

          return (
            <div key={group.type} className="border border-gray-200 rounded-lg overflow-hidden">
              {/* Group header */}
              <button
                type="button"
                onClick={() => toggleType(group.type)}
                className={`
                  w-full flex items-center justify-between p-4 text-left
                  hover:bg-gray-50 transition-colors
                  ${sources.length === 0 ? 'opacity-50' : ''}
                `}
              >
                <div className="flex items-center gap-3">
                  <span className={`${hasReadySources ? 'text-blue-600' : 'text-gray-400'}`}>
                    {group.icon}
                  </span>
                  <div>
                    <h3 className="font-medium text-sm text-gray-900">{group.label}</h3>
                    <p className="text-xs text-gray-500">{group.description}</p>
                  </div>
                </div>

                <div className="flex items-center gap-3">
                  {/* Source count badge */}
                  <span
                    className={`
                    px-2 py-1 text-xs font-medium rounded-full
                    ${sources.length > 0 ? 'bg-blue-100 text-blue-700' : 'bg-gray-100 text-gray-500'}
                  `}
                  >
                    {sources.length}
                  </span>

                  {/* Expand/collapse chevron */}
                  <ChevronIcon
                    className={`h-5 w-5 text-gray-400 transition-transform ${
                      isExpanded ? 'rotate-180' : ''
                    }`}
                  />
                </div>
              </button>

              {/* Sources list */}
              {isExpanded && sources.length > 0 && (
                <div className="border-t border-gray-200 p-3 space-y-2 bg-gray-50">
                  {sources.map((source) => (
                    <DiscoveredSourceCard
                      key={source.source_id}
                      source={source}
                      isSelected={isSelected(source.source_id)}
                      onClick={() => onSourceSelect?.(source)}
                      showMetadata={showMetadata}
                      expandable={expandable}
                      compact={compact}
                    />
                  ))}
                </div>
              )}

              {/* Empty state for expanded groups */}
              {isExpanded && sources.length === 0 && (
                <div className="border-t border-gray-200 p-6 text-center bg-gray-50">
                  <p className="text-sm text-gray-500">No {group.label.toLowerCase()} available</p>
                  <p className="text-xs text-gray-400 mt-1">
                    {group.type === 'vector_search' &&
                      "You don't have access to any Vector Search indexes"}
                    {group.type === 'genie' && "You don't have access to any Genie spaces"}
                    {group.type === 'knowledge_assistant' &&
                      'No Knowledge Assistant endpoints found'}
                  </p>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Total count */}
      <div className="text-xs text-gray-500 text-center">
        {data?.total_count || 0} total sources discovered
      </div>
    </div>
  );
};

// Icons
function SearchIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      fill="none"
      stroke="currentColor"
      viewBox="0 0 24 24"
      strokeWidth={2}
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 5.196a7.5 7.5 0 0010.607 10.607z"
      />
    </svg>
  );
}

function RefreshIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      fill="none"
      stroke="currentColor"
      viewBox="0 0 24 24"
      strokeWidth={2}
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
      />
    </svg>
  );
}

function ChevronIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      fill="none"
      stroke="currentColor"
      viewBox="0 0 24 24"
      strokeWidth={2}
    >
      <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
    </svg>
  );
}

function DatabaseIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      fill="none"
      stroke="currentColor"
      viewBox="0 0 24 24"
      strokeWidth={2}
    >
      <ellipse cx="12" cy="5" rx="9" ry="3" />
      <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" />
      <path d="M3 12c0 1.66 4 3 9 3s9-1.34 9-3" />
    </svg>
  );
}

function SparklesIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      fill="none"
      stroke="currentColor"
      viewBox="0 0 24 24"
      strokeWidth={2}
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 00-2.456 2.456zM16.894 20.567L16.5 21.75l-.394-1.183a2.25 2.25 0 00-1.423-1.423L13.5 18.75l1.183-.394a2.25 2.25 0 001.423-1.423l.394-1.183.394 1.183a2.25 2.25 0 001.423 1.423l1.183.394-1.183.394a2.25 2.25 0 00-1.423 1.423z"
      />
    </svg>
  );
}

function UserCircleIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      fill="none"
      stroke="currentColor"
      viewBox="0 0 24 24"
      strokeWidth={2}
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M17.982 18.725A7.488 7.488 0 0012 15.75a7.488 7.488 0 00-5.982 2.975m11.963 0a9 9 0 10-11.963 0m11.963 0A8.966 8.966 0 0112 21a8.966 8.966 0 01-5.982-2.275M15 9.75a3 3 0 11-6 0 3 3 0 016 0z"
      />
    </svg>
  );
}

function GlobeIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      fill="none"
      stroke="currentColor"
      viewBox="0 0 24 24"
      strokeWidth={2}
    >
      <circle cx="12" cy="12" r="10" />
      <path d="M2 12h20" />
      <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
    </svg>
  );
}

function DocumentIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      fill="none"
      stroke="currentColor"
      viewBox="0 0 24 24"
      strokeWidth={2}
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m2.25 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z"
      />
    </svg>
  );
}

function ExclamationCircleIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      fill="none"
      stroke="currentColor"
      viewBox="0 0 24 24"
      strokeWidth={2}
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z"
      />
    </svg>
  );
}

function ExclamationTriangleIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      fill="none"
      stroke="currentColor"
      viewBox="0 0 24 24"
      strokeWidth={2}
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z"
      />
    </svg>
  );
}

export default SourceList;
