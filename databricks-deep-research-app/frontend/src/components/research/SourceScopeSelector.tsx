/**
 * SourceScopeSelector - Toggle group for selecting source scope.
 *
 * Features (T042):
 * - Toggle group with options: Enterprise Only | Web Only | All
 * - Expandable section showing individual source toggles
 * - Source descriptions and relevance hints when available
 */

import * as React from 'react';
import { cn } from '@/lib/utils';
import type {
  SourceScope,
  AvailableSource,
  DataSourceType,
} from '@/types/dataSources';

interface SourceScopeSelectorProps {
  selectedScope: SourceScope;
  onScopeChange: (scope: SourceScope) => void;
  availableSources?: AvailableSource[];
  onSourceToggle?: (sourceId: string, enabled: boolean) => void;
  disabled?: boolean;
  className?: string;
  compact?: boolean;
}

const SCOPE_OPTIONS: { value: SourceScope; label: string; description: string }[] = [
  {
    value: 'enterprise_only',
    label: 'Enterprise Only',
    description: 'Search only internal enterprise data sources',
  },
  {
    value: 'web_only',
    label: 'Web Only',
    description: 'Search only public web sources',
  },
  {
    value: 'all',
    label: 'All Sources',
    description: 'Search both enterprise and web sources',
  },
];

export function SourceScopeSelector({
  selectedScope,
  onScopeChange,
  availableSources = [],
  onSourceToggle,
  disabled = false,
  className,
  compact = false,
}: SourceScopeSelectorProps) {
  const [isExpanded, setIsExpanded] = React.useState(false);

  // Filter sources based on scope
  const filteredSources = React.useMemo(() => {
    if (selectedScope === 'all') return availableSources;
    if (selectedScope === 'enterprise_only') {
      return availableSources.filter(
        (s) => s.type !== 'web_search' && s.type !== 'uploaded_file'
      );
    }
    if (selectedScope === 'web_only') {
      return availableSources.filter((s) => s.type === 'web_search');
    }
    return availableSources;
  }, [availableSources, selectedScope]);

  // Group sources by type
  const groupedSources = React.useMemo(() => {
    const groups: Record<string, AvailableSource[]> = {};
    filteredSources.forEach((source) => {
      const key = source.type;
      if (!groups[key]) groups[key] = [];
      groups[key].push(source);
    });
    return groups;
  }, [filteredSources]);

  const hasSourcesForExpansion =
    availableSources.length > 0 && onSourceToggle !== undefined;

  return (
    <div className={cn('space-y-2', className)}>
      {/* Main toggle group */}
      <div className="flex items-center gap-1">
        {!compact && (
          <span className="text-xs text-muted-foreground mr-1">Sources:</span>
        )}
        <div className="flex gap-1 rounded-md border border-input p-0.5 bg-muted/50">
          {SCOPE_OPTIONS.map((option) => (
            <button
              key={option.value}
              type="button"
              onClick={() => onScopeChange(option.value)}
              disabled={disabled}
              title={option.description}
              className={cn(
                'px-2 py-1 text-xs rounded transition-colors',
                'focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring',
                selectedScope === option.value
                  ? 'bg-background text-foreground shadow-sm'
                  : 'text-muted-foreground hover:text-foreground hover:bg-background/50',
                disabled && 'cursor-not-allowed opacity-50'
              )}
            >
              {option.label}
            </button>
          ))}
        </div>

        {/* Expand button for source toggles */}
        {hasSourcesForExpansion && (
          <button
            type="button"
            onClick={() => setIsExpanded(!isExpanded)}
            disabled={disabled}
            className={cn(
              'ml-1 p-1 rounded text-muted-foreground hover:text-foreground hover:bg-muted/50',
              'focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring',
              disabled && 'cursor-not-allowed opacity-50'
            )}
            title={isExpanded ? 'Hide source options' : 'Show source options'}
          >
            <ChevronIcon
              className={cn(
                'h-4 w-4 transition-transform',
                isExpanded && 'rotate-180'
              )}
            />
          </button>
        )}
      </div>

      {/* Expanded source toggles */}
      {isExpanded && hasSourcesForExpansion && (
        <div className="rounded-md border border-input bg-muted/30 p-3 space-y-3">
          {Object.entries(groupedSources).map(([type, sources]) => (
            <div key={type}>
              <div className="flex items-center gap-2 mb-2">
                <SourceTypeIcon type={type as DataSourceType} className="h-4 w-4" />
                <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
                  {getSourceTypeLabel(type as DataSourceType)}
                </span>
              </div>
              <div className="space-y-2 ml-6">
                {sources.map((source) => (
                  <SourceToggleItem
                    key={source.id}
                    source={source}
                    onToggle={(enabled) => onSourceToggle!(source.id, enabled)}
                    disabled={disabled}
                  />
                ))}
              </div>
            </div>
          ))}

          {filteredSources.length === 0 && (
            <p className="text-sm text-muted-foreground text-center py-2">
              No sources available for this scope
            </p>
          )}
        </div>
      )}
    </div>
  );
}

interface SourceToggleItemProps {
  source: AvailableSource;
  onToggle: (enabled: boolean) => void;
  disabled?: boolean;
}

function SourceToggleItem({ source, onToggle, disabled }: SourceToggleItemProps) {
  return (
    <label
      className={cn(
        'flex items-start gap-2 cursor-pointer group',
        disabled && 'cursor-not-allowed opacity-50'
      )}
    >
      <input
        type="checkbox"
        checked={source.isEnabled}
        onChange={(e) => onToggle(e.target.checked)}
        disabled={disabled}
        className="mt-0.5 rounded border-input"
      />
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span
            className={cn(
              'text-sm font-medium',
              source.isEnabled ? 'text-foreground' : 'text-muted-foreground',
              'group-hover:text-foreground'
            )}
          >
            {source.name}
          </span>
          {source.relevanceHint && (
            <span className="px-1.5 py-0.5 rounded text-xs bg-primary/10 text-primary">
              {source.relevanceHint}
            </span>
          )}
        </div>
        {source.description && (
          <p className="text-xs text-muted-foreground mt-0.5 line-clamp-1">
            {source.description}
          </p>
        )}
      </div>
    </label>
  );
}

function getSourceTypeLabel(type: DataSourceType): string {
  const labels: Record<DataSourceType, string> = {
    vector_search: 'Vector Search',
    delta_table: 'Delta Table',
    genie: 'Genie',
    knowledge_assistant: 'Knowledge Assistant',
    web_search: 'Web Search',
    uploaded_file: 'Uploaded Files',
    custom: 'Custom',
  };
  return labels[type] || type;
}

function SourceTypeIcon({
  type,
  className,
}: {
  type: DataSourceType;
  className?: string;
}) {
  switch (type) {
    case 'vector_search':
      return <SearchIcon className={cn('text-blue-600', className)} />;
    case 'delta_table':
      return <DatabaseIcon className={cn('text-cyan-600', className)} />;
    case 'genie':
      return <DatabaseIcon className={cn('text-purple-600', className)} />;
    case 'knowledge_assistant':
      return <BrainIcon className={cn('text-emerald-600', className)} />;
    case 'web_search':
      return <GlobeIcon className={cn('text-orange-600', className)} />;
    case 'uploaded_file':
      return <FileIcon className={cn('text-gray-600', className)} />;
    default:
      return <CubeIcon className={cn('text-slate-600', className)} />;
  }
}

// Icons
function ChevronIcon({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <path d="m6 9 6 6 6-6" />
    </svg>
  );
}

function SearchIcon({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <circle cx="11" cy="11" r="8" />
      <path d="m21 21-4.3-4.3" />
    </svg>
  );
}

function DatabaseIcon({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <ellipse cx="12" cy="5" rx="9" ry="3" />
      <path d="M3 5v14a9 3 0 0 0 18 0V5" />
      <path d="M3 12a9 3 0 0 0 18 0" />
    </svg>
  );
}

function BrainIcon({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 1.98-3A2.5 2.5 0 0 1 9.5 2Z" />
      <path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-1.98-3A2.5 2.5 0 0 0 14.5 2Z" />
    </svg>
  );
}

function GlobeIcon({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <circle cx="12" cy="12" r="10" />
      <path d="M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20" />
      <path d="M2 12h20" />
    </svg>
  );
}

function FileIcon({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z" />
      <polyline points="14 2 14 8 20 8" />
    </svg>
  );
}

function CubeIcon({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <path d="m21 16-9 5-9-5V8l9-5 9 5v8z" />
      <path d="m3.27 6.96 8.73 4.84 8.73-4.84" />
      <line x1="12" x2="12" y1="22" y2="11.8" />
    </svg>
  );
}

export default SourceScopeSelector;
