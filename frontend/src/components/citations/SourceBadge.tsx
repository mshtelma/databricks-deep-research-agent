/**
 * SourceBadge component - Visual indicator for source types
 *
 * Displays a badge indicating the type of source (web, vector search,
 * knowledge assistant, or custom). Used in citation cards and source lists.
 */

import React from 'react';
import type { SourceType } from '@/types/index';

interface SourceBadgeProps {
  /** Type of source */
  sourceType: SourceType;
  /** Additional CSS classes */
  className?: string;
  /** Show icon only (no text) */
  iconOnly?: boolean;
}

/**
 * Get display config for a source type
 */
function getSourceTypeConfig(sourceType: SourceType): {
  label: string;
  icon: string;
  colorClass: string;
} {
  switch (sourceType) {
    case 'web':
      return {
        label: 'Web',
        icon: '🌐',
        colorClass: 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200',
      };
    case 'vector_search':
      return {
        label: 'Vector Search',
        icon: '🔍',
        colorClass: 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200',
      };
    case 'knowledge_assistant':
      return {
        label: 'Knowledge Assistant',
        icon: '🤖',
        colorClass: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200',
      };
    case 'custom':
      return {
        label: 'Custom',
        icon: '⚙️',
        colorClass: 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200',
      };
    default:
      return {
        label: 'Unknown',
        icon: '❓',
        colorClass: 'bg-gray-100 text-gray-600 dark:bg-gray-700 dark:text-gray-400',
      };
  }
}

export const SourceBadge: React.FC<SourceBadgeProps> = ({
  sourceType,
  className = '',
  iconOnly = false,
}) => {
  const config = getSourceTypeConfig(sourceType);

  return (
    <span
      data-testid="source-badge"
      data-source-type={sourceType}
      className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium ${config.colorClass} ${className}`}
      title={config.label}
    >
      <span role="img" aria-label={config.label}>{config.icon}</span>
      {!iconOnly && <span>{config.label}</span>}
    </span>
  );
};

export default SourceBadge;
