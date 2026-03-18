/**
 * DataSourceSelector - Main dropdown component for selecting discovered data sources.
 *
 * Features:
 * - Grouped by source type (Vector Search, Genie, Assistants)
 * - Search/filter by source name
 * - Loading state during initial discovery
 * - Refresh button for manual cache invalidation
 * - Multi-select support
 *
 * Task: T010l [US9a]
 */

import React, { useState, useMemo, useCallback } from 'react';
import { useDiscoveredSources, useRefreshDiscovery } from '@/hooks/useDiscoveredSources';
import type { DiscoveredSource, DataSourceType } from '@/types/discovery';
import { getSourceTypeLabel, getStatusLabel, getStatusColor } from '@/types/discovery';

interface DataSourceSelectorProps {
  /** Currently selected source IDs */
  selectedSourceIds?: string[];
  /** Callback when selection changes */
  onSelectionChange?: (sourceIds: string[]) => void;
  /** Enable multi-select (default: true) */
  multiSelect?: boolean;
  /** Filter by specific source types */
  allowedTypes?: DataSourceType[];
  /** Placeholder text */
  placeholder?: string;
  /** Disabled state */
  disabled?: boolean;
  /** Show status indicators */
  showStatus?: boolean;
  /** Compact mode (smaller padding) */
  compact?: boolean;
}

export const DataSourceSelector: React.FC<DataSourceSelectorProps> = ({
  selectedSourceIds = [],
  onSelectionChange,
  multiSelect = true,
  allowedTypes,
  placeholder = 'Select data sources...',
  disabled = false,
  showStatus = true,
  compact = false,
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  const { data, isLoading, error } = useDiscoveredSources();
  const { mutate: refresh, isPending: isRefreshing } = useRefreshDiscovery();

  // Filter sources by allowed types and search query
  const filteredSources = useMemo(() => {
    let sources = data?.sources || [];

    // Filter by allowed types
    if (allowedTypes && allowedTypes.length > 0) {
      sources = sources.filter((s) => allowedTypes.includes(s.source_type as DataSourceType));
    }

    // Filter by search query
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      sources = sources.filter(
        (s) =>
          s.name.toLowerCase().includes(query) ||
          s.description?.toLowerCase().includes(query) ||
          s.source_id.toLowerCase().includes(query)
      );
    }

    return sources;
  }, [data?.sources, allowedTypes, searchQuery]);

  // Group filtered sources by type
  const groupedSources = useMemo(() => {
    const groups: Record<string, DiscoveredSource[]> = {};

    for (const source of filteredSources) {
      const type = source.source_type;
      if (!groups[type]) {
        groups[type] = [];
      }
      groups[type].push(source);
    }

    return groups;
  }, [filteredSources]);

  const handleToggleSource = useCallback(
    (sourceId: string) => {
      if (!onSelectionChange) return;

      if (multiSelect) {
        const newSelection = selectedSourceIds.includes(sourceId)
          ? selectedSourceIds.filter((id) => id !== sourceId)
          : [...selectedSourceIds, sourceId];
        onSelectionChange(newSelection);
      } else {
        onSelectionChange(selectedSourceIds.includes(sourceId) ? [] : [sourceId]);
        setIsOpen(false);
      }
    },
    [selectedSourceIds, onSelectionChange, multiSelect]
  );

  const handleRefresh = useCallback(() => {
    refresh(undefined);
  }, [refresh]);

  const selectedCount = selectedSourceIds.length;
  const totalCount = data?.total_count || 0;

  // Render selected sources summary
  const renderSelectedSummary = () => {
    if (selectedCount === 0) {
      return <span className="text-gray-500">{placeholder}</span>;
    }

    if (selectedCount === 1) {
      const source = data?.sources.find((s) => s.source_id === selectedSourceIds[0]);
      return <span className="truncate">{source?.name || selectedSourceIds[0]}</span>;
    }

    return <span>{selectedCount} sources selected</span>;
  };

  return (
    <div className="relative">
      {/* Trigger button */}
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        disabled={disabled || isLoading}
        className={`
          w-full flex items-center justify-between
          border border-gray-300 rounded-md shadow-sm
          bg-white text-left
          hover:border-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500
          disabled:opacity-50 disabled:cursor-not-allowed
          ${compact ? 'px-3 py-1.5 text-sm' : 'px-4 py-2'}
        `}
      >
        <div className="flex items-center gap-2 min-w-0">
          {isLoading ? (
            <span className="text-gray-500">Discovering sources...</span>
          ) : (
            renderSelectedSummary()
          )}
        </div>

        <div className="flex items-center gap-2 ml-2 flex-shrink-0">
          {/* Refresh button */}
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              handleRefresh();
            }}
            disabled={isRefreshing}
            className="p-1 text-gray-400 hover:text-gray-600 rounded"
            title="Refresh sources"
          >
            <svg
              className={`w-4 h-4 ${isRefreshing ? 'animate-spin' : ''}`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
              />
            </svg>
          </button>

          {/* Dropdown arrow */}
          <svg
            className={`w-5 h-5 text-gray-400 transition-transform ${isOpen ? 'rotate-180' : ''}`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </div>
      </button>

      {/* Dropdown panel */}
      {isOpen && (
        <div className="absolute z-50 mt-1 w-full bg-white border border-gray-200 rounded-md shadow-lg max-h-80 overflow-hidden">
          {/* Search input */}
          <div className="p-2 border-b border-gray-200">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search sources..."
              className="w-full px-3 py-1.5 text-sm border border-gray-300 rounded focus:outline-none focus:ring-1 focus:ring-blue-500"
              autoFocus
            />
          </div>

          {/* Error state */}
          {error && (
            <div className="p-4 text-center text-red-600 text-sm">
              Failed to load sources: {error.message}
            </div>
          )}

          {/* Empty state */}
          {!isLoading && !error && filteredSources.length === 0 && (
            <div className="p-4 text-center text-gray-500 text-sm">
              {searchQuery ? 'No sources match your search' : 'No sources available'}
            </div>
          )}

          {/* Source list */}
          <div className="overflow-y-auto max-h-60">
            {Object.entries(groupedSources).map(([type, sources]) => (
              <div key={type}>
                {/* Type header */}
                <div className="sticky top-0 px-3 py-1.5 bg-gray-50 text-xs font-semibold text-gray-600 border-b border-gray-100">
                  {getSourceTypeLabel(type as DataSourceType)} ({sources.length})
                </div>

                {/* Sources in this type */}
                {sources.map((source) => (
                  <button
                    key={source.source_id}
                    type="button"
                    onClick={() => handleToggleSource(source.source_id)}
                    className={`
                      w-full flex items-center gap-3 px-3 py-2 text-left
                      hover:bg-gray-50 transition-colors
                      ${selectedSourceIds.includes(source.source_id) ? 'bg-blue-50' : ''}
                    `}
                  >
                    {/* Checkbox for multi-select */}
                    {multiSelect && (
                      <input
                        type="checkbox"
                        checked={selectedSourceIds.includes(source.source_id)}
                        readOnly
                        className="h-4 w-4 text-blue-600 rounded border-gray-300"
                      />
                    )}

                    {/* Source info */}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="font-medium text-sm truncate">{source.name}</span>
                        {showStatus && (
                          <span
                            className={`text-xs ${getStatusColor(source.status)}`}
                            title={getStatusLabel(source.status)}
                          >
                            {source.status === 'ready' ? '●' : '○'}
                          </span>
                        )}
                      </div>
                      {source.description && (
                        <p className="text-xs text-gray-500 truncate">{source.description}</p>
                      )}
                    </div>

                    {/* Capabilities badges */}
                    <div className="flex gap-1 flex-shrink-0">
                      {source.capabilities.slice(0, 2).map((cap) => (
                        <span
                          key={cap}
                          className="px-1.5 py-0.5 text-xs bg-gray-100 text-gray-600 rounded"
                        >
                          {cap}
                        </span>
                      ))}
                    </div>
                  </button>
                ))}
              </div>
            ))}
          </div>

          {/* Footer with stats */}
          {data && (
            <div className="px-3 py-2 bg-gray-50 border-t border-gray-200 text-xs text-gray-500 flex justify-between">
              <span>
                {totalCount} sources available
                {data.cached && ' (cached)'}
              </span>
              {data.errors && data.errors.length > 0 && (
                <span className="text-yellow-600">{data.errors.length} type(s) had errors</span>
              )}
            </div>
          )}
        </div>
      )}

      {/* Click outside to close */}
      {isOpen && (
        <div className="fixed inset-0 z-40" onClick={() => setIsOpen(false)} aria-hidden="true" />
      )}
    </div>
  );
};

export default DataSourceSelector;
