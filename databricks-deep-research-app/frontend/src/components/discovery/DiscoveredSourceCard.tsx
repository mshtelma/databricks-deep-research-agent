/**
 * DiscoveredSourceCard - Card component for displaying a discovered data source.
 *
 * Features:
 * - Display name, type icon, and description (truncated)
 * - Status indicator badge (ready/syncing/unavailable/error)
 * - Expandable metadata panel showing:
 *   - Query types supported (ANN, HYBRID, FULL_TEXT)
 *   - Filter columns available
 *   - Endpoint details
 *   - Index information for Vector Search
 *   - Space details for Genie
 * - Click to select
 * - Capabilities badges
 *
 * Task: T010n [US9a]
 */

import React, { useState, useCallback } from 'react';
import type {
  DiscoveredSource,
  DataSourceType,
  DiscoveryStatus,
  VectorSearchMetadata,
  GenieSpaceMetadata,
  ServingEndpointMetadata,
} from '@/types/discovery';
import { getStatusLabel, getStatusColor, getQueryTypeLabel, parseSourceMetadata } from '@/types/discovery';

interface DiscoveredSourceCardProps {
  /** The discovered source to display */
  source: DiscoveredSource;
  /** Whether the source is currently selected */
  isSelected?: boolean;
  /** Click handler */
  onClick?: () => void;
  /** Show detailed metadata */
  showMetadata?: boolean;
  /** Allow expanding metadata panel */
  expandable?: boolean;
  /** Compact mode with less padding */
  compact?: boolean;
  /** Additional CSS class */
  className?: string;
}

export const DiscoveredSourceCard: React.FC<DiscoveredSourceCardProps> = ({
  source,
  isSelected = false,
  onClick,
  showMetadata = true,
  expandable = true,
  compact = false,
  className = '',
}) => {
  const [isExpanded, setIsExpanded] = useState(false);

  const handleClick = useCallback(() => {
    onClick?.();
  }, [onClick]);

  const handleToggleExpand = useCallback(
    (e: React.MouseEvent) => {
      if (expandable) {
        e.stopPropagation();
        setIsExpanded((prev) => !prev);
      }
    },
    [expandable]
  );

  const parsedMetadata = parseSourceMetadata(source);

  return (
    <div
      className={`
        bg-white border rounded-lg transition-all
        ${isSelected ? 'border-blue-500 ring-2 ring-blue-200' : 'border-gray-200 hover:border-gray-300'}
        ${onClick ? 'cursor-pointer' : ''}
        ${compact ? 'p-2' : 'p-3'}
        ${className}
      `}
      onClick={handleClick}
    >
      {/* Main content row */}
      <div className="flex items-start gap-3">
        {/* Type icon */}
        <div
          className={`
          flex-shrink-0 p-2 rounded-lg
          ${isSelected ? 'bg-blue-100 text-blue-600' : 'bg-gray-100 text-gray-500'}
        `}
        >
          <SourceTypeIcon type={source.source_type as DataSourceType} className="h-5 w-5" />
        </div>

        {/* Info */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between gap-2">
            <h4 className="font-medium text-sm text-gray-900 truncate">{source.name}</h4>
            <StatusBadge status={source.status} />
          </div>

          {source.description && (
            <p
              className={`text-xs text-gray-500 mt-0.5 ${compact ? 'truncate' : 'line-clamp-2'}`}
              title={source.description}
            >
              {source.description}
            </p>
          )}

          {/* Capabilities badges */}
          {source.capabilities.length > 0 && showMetadata && (
            <div className="flex flex-wrap gap-1 mt-2">
              {source.capabilities.slice(0, compact ? 2 : 4).map((cap) => (
                <span
                  key={cap}
                  className="px-1.5 py-0.5 text-xs bg-gray-100 text-gray-600 rounded"
                >
                  {cap}
                </span>
              ))}
              {source.capabilities.length > (compact ? 2 : 4) && (
                <span className="px-1.5 py-0.5 text-xs bg-gray-100 text-gray-500 rounded">
                  +{source.capabilities.length - (compact ? 2 : 4)}
                </span>
              )}
            </div>
          )}
        </div>

        {/* Expand button */}
        {expandable && showMetadata && (
          <button
            type="button"
            onClick={handleToggleExpand}
            className="flex-shrink-0 p-1 text-gray-400 hover:text-gray-600 rounded"
            title={isExpanded ? 'Collapse details' : 'Expand details'}
          >
            <ChevronIcon
              className={`h-4 w-4 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
            />
          </button>
        )}
      </div>

      {/* Expanded metadata panel */}
      {isExpanded && showMetadata && (
        <div className="mt-3 pt-3 border-t border-gray-100 space-y-3">
          {/* Endpoint info */}
          <div className="flex items-center gap-2 text-xs">
            <span className="text-gray-500">Endpoint:</span>
            <code className="px-1.5 py-0.5 bg-gray-100 rounded text-gray-700 font-mono">
              {source.endpoint_name}
            </code>
          </div>

          {/* Vector Search specific metadata */}
          {parsedMetadata.vectorSearch && (
            <VectorSearchDetails metadata={parsedMetadata.vectorSearch} />
          )}

          {/* Genie specific metadata */}
          {parsedMetadata.genie && <GenieDetails metadata={parsedMetadata.genie} />}

          {/* Serving endpoint specific metadata */}
          {parsedMetadata.servingEndpoint && (
            <ServingEndpointDetails metadata={parsedMetadata.servingEndpoint} />
          )}

          {/* Discovery timestamp */}
          <div className="text-xs text-gray-400">
            Discovered: {new Date(source.discovered_at).toLocaleString()}
          </div>
        </div>
      )}
    </div>
  );
};

/** Status badge component */
function StatusBadge({ status }: { status: DiscoveryStatus }) {
  const colorClass = getStatusColor(status);
  const label = getStatusLabel(status);

  const bgColors: Record<string, string> = {
    'text-green-600': 'bg-green-100',
    'text-yellow-600': 'bg-yellow-100',
    'text-gray-500': 'bg-gray-100',
    'text-red-600': 'bg-red-100',
  };

  return (
    <span
      className={`
        inline-flex items-center gap-1 px-2 py-0.5 text-xs font-medium rounded-full
        ${bgColors[colorClass] || 'bg-gray-100'} ${colorClass}
      `}
    >
      <StatusDot status={status} />
      {label}
    </span>
  );
}

/** Status dot indicator */
function StatusDot({ status }: { status: DiscoveryStatus }) {
  const dotColors: Record<DiscoveryStatus, string> = {
    ready: 'bg-green-500',
    syncing: 'bg-yellow-500 animate-pulse',
    unavailable: 'bg-gray-400',
    error: 'bg-red-500',
  };

  return <span className={`h-1.5 w-1.5 rounded-full ${dotColors[status]}`} />;
}

/** Vector Search metadata details */
function VectorSearchDetails({ metadata }: { metadata: VectorSearchMetadata }) {
  return (
    <div className="space-y-2">
      {/* Index info */}
      <div className="text-xs">
        <span className="text-gray-500">Index:</span>
        <code className="ml-2 px-1.5 py-0.5 bg-gray-100 rounded text-gray-700 font-mono">
          {metadata.index_name}
        </code>
      </div>

      {/* Query types */}
      <div className="text-xs">
        <span className="text-gray-500">Query Types:</span>
        <div className="flex flex-wrap gap-1 mt-1">
          {metadata.supported_query_types.map((qt) => (
            <span key={qt} className="px-1.5 py-0.5 bg-blue-50 text-blue-700 rounded text-xs">
              {getQueryTypeLabel(qt)}
            </span>
          ))}
        </div>
      </div>

      {/* Filter columns */}
      {metadata.filter_columns.length > 0 && (
        <div className="text-xs">
          <span className="text-gray-500">Filter Columns:</span>
          <div className="flex flex-wrap gap-1 mt-1">
            {metadata.filter_columns.slice(0, 5).map((col) => (
              <span
                key={col.name}
                className="px-1.5 py-0.5 bg-gray-100 text-gray-600 rounded text-xs"
                title={`Type: ${col.data_type}`}
              >
                {col.name}
              </span>
            ))}
            {metadata.filter_columns.length > 5 && (
              <span className="px-1.5 py-0.5 bg-gray-100 text-gray-500 rounded text-xs">
                +{metadata.filter_columns.length - 5} more
              </span>
            )}
          </div>
        </div>
      )}

      {/* Embedding info */}
      {metadata.embedding_model && (
        <div className="text-xs">
          <span className="text-gray-500">Embedding Model:</span>
          <span className="ml-2 text-gray-700">{metadata.embedding_model}</span>
          {metadata.embedding_dimension && (
            <span className="text-gray-500 ml-1">({metadata.embedding_dimension}d)</span>
          )}
        </div>
      )}

      {/* Row count */}
      {metadata.row_count !== undefined && (
        <div className="text-xs">
          <span className="text-gray-500">Rows:</span>
          <span className="ml-2 text-gray-700">{metadata.row_count.toLocaleString()}</span>
        </div>
      )}

      {/* Reranking support */}
      {metadata.supports_reranking && (
        <div className="text-xs flex items-center gap-1">
          <CheckIcon className="h-3 w-3 text-green-500" />
          <span className="text-gray-600">Reranking supported</span>
        </div>
      )}
    </div>
  );
}

/** Genie space metadata details */
function GenieDetails({ metadata }: { metadata: GenieSpaceMetadata }) {
  return (
    <div className="space-y-2">
      {/* Space ID */}
      <div className="text-xs">
        <span className="text-gray-500">Space ID:</span>
        <code className="ml-2 px-1.5 py-0.5 bg-gray-100 rounded text-gray-700 font-mono">
          {metadata.space_id}
        </code>
      </div>

      {/* Title */}
      {metadata.title && (
        <div className="text-xs">
          <span className="text-gray-500">Title:</span>
          <span className="ml-2 text-gray-700">{metadata.title}</span>
        </div>
      )}

      {/* Warehouse */}
      {metadata.warehouse_id && (
        <div className="text-xs">
          <span className="text-gray-500">Warehouse:</span>
          <code className="ml-2 px-1.5 py-0.5 bg-gray-100 rounded text-gray-700 font-mono text-xs">
            {metadata.warehouse_id}
          </code>
        </div>
      )}

      {/* Owner */}
      {metadata.owner && (
        <div className="text-xs">
          <span className="text-gray-500">Owner:</span>
          <span className="ml-2 text-gray-700">{metadata.owner}</span>
        </div>
      )}

      {/* Capabilities */}
      {metadata.capabilities.length > 0 && (
        <div className="text-xs">
          <span className="text-gray-500">Capabilities:</span>
          <div className="flex flex-wrap gap-1 mt-1">
            {metadata.capabilities.map((cap) => (
              <span key={cap} className="px-1.5 py-0.5 bg-purple-50 text-purple-700 rounded text-xs">
                {cap}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

/** Serving endpoint metadata details */
function ServingEndpointDetails({ metadata }: { metadata: ServingEndpointMetadata }) {
  return (
    <div className="space-y-2">
      {/* Endpoint name */}
      <div className="text-xs">
        <span className="text-gray-500">Endpoint:</span>
        <code className="ml-2 px-1.5 py-0.5 bg-gray-100 rounded text-gray-700 font-mono">
          {metadata.endpoint_name}
        </code>
      </div>

      {/* Endpoint type */}
      <div className="text-xs">
        <span className="text-gray-500">Type:</span>
        <span className="ml-2 text-gray-700">{metadata.endpoint_type}</span>
      </div>

      {/* State */}
      <div className="text-xs flex items-center gap-2">
        <span className="text-gray-500">State:</span>
        <span
          className={`
          px-1.5 py-0.5 rounded text-xs font-medium
          ${metadata.state === 'READY' ? 'bg-green-100 text-green-700' : ''}
          ${metadata.state === 'PENDING' ? 'bg-yellow-100 text-yellow-700' : ''}
          ${metadata.state === 'NOT_READY' ? 'bg-red-100 text-red-700' : ''}
        `}
        >
          {metadata.state}
        </span>
      </div>

      {/* Assistant type */}
      {metadata.assistant_type && (
        <div className="text-xs">
          <span className="text-gray-500">Assistant Type:</span>
          <span className="ml-2 text-gray-700">{metadata.assistant_type}</span>
        </div>
      )}

      {/* Creator */}
      {metadata.creator && (
        <div className="text-xs">
          <span className="text-gray-500">Creator:</span>
          <span className="ml-2 text-gray-700">{metadata.creator}</span>
        </div>
      )}

      {/* Tags */}
      {Object.keys(metadata.tags).length > 0 && (
        <div className="text-xs">
          <span className="text-gray-500">Tags:</span>
          <div className="flex flex-wrap gap-1 mt-1">
            {Object.entries(metadata.tags)
              .slice(0, 4)
              .map(([key, value]) => (
                <span
                  key={key}
                  className="px-1.5 py-0.5 bg-gray-100 text-gray-600 rounded text-xs"
                  title={`${key}: ${value}`}
                >
                  {key}={value}
                </span>
              ))}
          </div>
        </div>
      )}
    </div>
  );
}

// Icons
function SourceTypeIcon({ type, className }: { type: DataSourceType; className?: string }) {
  switch (type) {
    case 'vector_search':
      return (
        <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <ellipse cx="12" cy="5" rx="9" ry="3" strokeWidth={2} />
          <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" strokeWidth={2} />
          <path d="M3 12c0 1.66 4 3 9 3s9-1.34 9-3" strokeWidth={2} />
        </svg>
      );
    case 'genie':
      return (
        <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09z"
          />
        </svg>
      );
    case 'knowledge_assistant':
      return (
        <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M17.982 18.725A7.488 7.488 0 0012 15.75a7.488 7.488 0 00-5.982 2.975m11.963 0a9 9 0 10-11.963 0m11.963 0A8.966 8.966 0 0112 21a8.966 8.966 0 01-5.982-2.275M15 9.75a3 3 0 11-6 0 3 3 0 016 0z"
          />
        </svg>
      );
    case 'web_search':
      return (
        <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <circle cx="12" cy="12" r="10" strokeWidth={2} />
          <path d="M2 12h20" strokeWidth={2} />
          <path
            d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"
            strokeWidth={2}
          />
        </svg>
      );
    case 'uploaded_file':
      return (
        <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m2.25 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z"
          />
        </svg>
      );
    default:
      return (
        <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M9.879 7.519c1.171-1.025 3.071-1.025 4.242 0 1.172 1.025 1.172 2.687 0 3.712-.203.179-.43.326-.67.442-.745.361-1.45.999-1.45 1.827v.75M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-9 5.25h.008v.008H12v-.008z"
          />
        </svg>
      );
  }
}

function ChevronIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
    </svg>
  );
}

function CheckIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 12.75l6 6 9-13.5" />
    </svg>
  );
}

export default DiscoveredSourceCard;
