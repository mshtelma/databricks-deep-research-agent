/**
 * CitationMarker component - Superscript citation indicator
 *
 * Displays an inline superscript citation marker that can be clicked
 * or hovered to show the evidence card. Color-coded by verification status.
 *
 * Supports human-readable citation keys like [Arxiv], [Zhipu], [Github-2]
 * as well as legacy numeric indices for backwards compatibility.
 */

import React from 'react';
import { VerificationVerdict } from '@/types/citation';

interface CitationMarkerProps {
  /** Human-readable citation key (e.g., "Arxiv", "Zhipu", "Github-2") */
  citationKey: string;
  /** Legacy numeric index for backwards compatibility */
  index?: number;
  /** Source title (for link-based citations) */
  title?: string;
  /** Source URL (for link-based citations) */
  url?: string;
  /** Verification verdict for color coding */
  verdict?: VerificationVerdict | null;
  /** Handler for click to show evidence card or open link */
  onClick?: () => void;
  /** Mouse enter handler - passes event for element reference */
  onMouseEnter?: (e: React.MouseEvent<HTMLElement>) => void;
  /** Mouse leave handler - passes event for element reference */
  onMouseLeave?: (e: React.MouseEvent<HTMLElement>) => void;
  /** Whether the evidence card is currently visible */
  isActive?: boolean;
  /** Whether citation data is not yet loaded (shows dimmed state) */
  isUnresolved?: boolean;
}

/**
 * Get Tailwind color class for verdict
 */
function getVerdictColorClass(verdict?: VerificationVerdict | null): string {
  if (!verdict) return 'text-blue-600 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300';

  const colorMap: Record<VerificationVerdict, string> = {
    supported: 'text-green-600 hover:text-green-700 dark:text-green-400 dark:hover:text-green-300',
    partial: 'text-amber-600 hover:text-amber-700 dark:text-amber-400 dark:hover:text-amber-300',
    unsupported: 'text-red-600 hover:text-red-700 dark:text-red-400 dark:hover:text-red-300',
    contradicted: 'text-purple-600 hover:text-purple-700 dark:text-purple-400 dark:hover:text-purple-300',
  };

  return colorMap[verdict] || 'text-blue-600 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300';
}

/**
 * Get background color class for active state
 */
function getActiveBackgroundClass(verdict?: VerificationVerdict | null): string {
  if (!verdict) return 'bg-blue-100 dark:bg-blue-900/30';

  const bgMap: Record<VerificationVerdict, string> = {
    supported: 'bg-green-100 dark:bg-green-900/30',
    partial: 'bg-amber-100 dark:bg-amber-900/30',
    unsupported: 'bg-red-100 dark:bg-red-900/30',
    contradicted: 'bg-purple-100 dark:bg-purple-900/30',
  };

  return bgMap[verdict] || 'bg-blue-100 dark:bg-blue-900/30';
}

/**
 * Truncate URL for display
 */
function truncateUrl(url: string, maxLength = 50): string {
  if (url.length <= maxLength) return url;
  return url.slice(0, maxLength - 3) + '...';
}

/**
 * Build tooltip text for citation
 */
function buildTooltip(title?: string, url?: string, verdict?: VerificationVerdict | null): string {
  const parts: string[] = [];

  if (title) {
    parts.push(title);
  }

  if (url) {
    parts.push(truncateUrl(url));
  }

  if (verdict) {
    parts.push(`Status: ${verdict}`);
  }

  if (parts.length === 0) {
    return 'Click to view source';
  }

  return parts.join('\n');
}

export const CitationMarker: React.FC<CitationMarkerProps> = ({
  citationKey,
  index,
  title,
  url,
  verdict,
  onClick,
  onMouseEnter,
  onMouseLeave,
  isActive = false,
  isUnresolved = false,
}) => {
  // Use dimmed styling when citation data hasn't loaded yet
  const colorClass = isUnresolved
    ? 'text-gray-400 dark:text-gray-500'
    : getVerdictColorClass(verdict);
  const activeClass = isActive && !isUnresolved ? getActiveBackgroundClass(verdict) : '';
  const tooltipText = isUnresolved
    ? 'Loading citation data...'
    : buildTooltip(title, url, verdict);

  // Display key for UI - prefer human-readable key, fallback to index
  const displayKey = citationKey || (index !== undefined ? String(index) : '?');

  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation();

    // Always open URL in new tab if available (primary action)
    if (url) {
      window.open(url, '_blank', 'noopener,noreferrer');
    }

    // Also call onClick handler for any additional handling
    onClick?.();
  };

  return (
    <sup
      data-testid={`citation-marker-${citationKey}`}
      className={`
        inline-flex items-center justify-center
        cursor-pointer select-none
        text-xs font-medium
        ml-0.5 mr-1 px-1.5 py-0.5 rounded
        transition-colors duration-150
        ${colorClass}
        ${activeClass}
        ${isActive && !isUnresolved ? 'ring-1 ring-current' : ''}
        ${isUnresolved ? 'opacity-60 bg-gray-100 dark:bg-gray-800' : 'hover:underline'}
      `}
      title={tooltipText}
      onClick={handleClick}
      onMouseEnter={(e) => onMouseEnter?.(e)}
      onMouseLeave={(e) => onMouseLeave?.(e)}
      role="button"
      tabIndex={0}
      aria-label={`Citation ${displayKey}${title ? `: ${title}` : ''}${verdict ? `, ${verdict}` : ''}`}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          handleClick(e as unknown as React.MouseEvent);
        }
      }}
    >
      [{displayKey}]
    </sup>
  );
};

export default CitationMarker;
