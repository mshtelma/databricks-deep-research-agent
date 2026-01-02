/**
 * SourceMetadata component - Source information display
 *
 * Displays metadata about the source document including title,
 * URL, author, date, and content type.
 */

import React from 'react';
import { SourceMetadata as SourceMetadataType } from '@/types/citation';

interface SourceMetadataProps {
  /** Source metadata object */
  source: SourceMetadataType;
  /** Whether to show full URL or truncated */
  showFullUrl?: boolean;
  /** Whether to show content type badge */
  showContentType?: boolean;
}

/**
 * Truncate URL for display
 */
function truncateUrl(url: string, maxLength: number = 50): string {
  if (url.length <= maxLength) return url;

  try {
    const parsed = new URL(url);
    const domain = parsed.hostname;
    const path = parsed.pathname;

    if (domain.length + path.length <= maxLength) {
      return `${domain}${path}`;
    }

    // Truncate path
    const truncatedPath = path.slice(0, maxLength - domain.length - 3) + '...';
    return `${domain}${truncatedPath}`;
  } catch {
    return url.slice(0, maxLength - 3) + '...';
  }
}

/**
 * Get icon for content type
 */
function getContentTypeIcon(contentType: string | null): string {
  if (!contentType) return '📄';

  const typeMap: Record<string, string> = {
    pdf: '📕',
    html: '🌐',
    text: '📝',
    json: '📊',
    xml: '📋',
  };

  return typeMap[contentType.toLowerCase()] || '📄';
}

export const SourceMetadata: React.FC<SourceMetadataProps> = ({
  source,
  showFullUrl = false,
  showContentType = true,
}) => {
  const displayUrl = source.url
    ? showFullUrl
      ? source.url
      : truncateUrl(source.url)
    : null;

  return (
    <div data-testid="source-metadata" className="space-y-1">
      {/* Title */}
      {source.title && (
        <h4 className="font-medium text-gray-900 dark:text-gray-100 line-clamp-2">
          {source.title}
        </h4>
      )}

      {/* URL */}
      {displayUrl && (
        <a
          data-testid="source-metadata-url"
          href={source.url || '#'}
          target="_blank"
          rel="noopener noreferrer"
          className="text-sm text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300 hover:underline truncate block"
          title={source.url || undefined}
        >
          {displayUrl}
        </a>
      )}

      {/* Author and Date row */}
      <div className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
        {source.author && (
          <span className="flex items-center gap-1">
            <span className="text-gray-400">👤</span>
            {source.author}
          </span>
        )}
        {source.author && source.publishedDate && (
          <span className="text-gray-300">•</span>
        )}
        {source.publishedDate && (
          <span className="flex items-center gap-1">
            <span className="text-gray-400">📅</span>
            {source.publishedDate}
          </span>
        )}
      </div>

      {/* Content type and pages */}
      {showContentType && (source.contentType || source.totalPages) && (
        <div className="flex items-center gap-2 text-xs text-gray-500 dark:text-gray-500">
          {source.contentType && (
            <span className="flex items-center gap-1 bg-gray-100 dark:bg-gray-700 px-1.5 py-0.5 rounded">
              {getContentTypeIcon(source.contentType)}
              {source.contentType.toUpperCase()}
            </span>
          )}
          {source.totalPages && (
            <span className="flex items-center gap-1">
              <span>📑</span>
              {source.totalPages} pages
            </span>
          )}
        </div>
      )}
    </div>
  );
};

export default SourceMetadata;
