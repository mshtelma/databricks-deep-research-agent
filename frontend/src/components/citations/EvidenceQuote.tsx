/**
 * EvidenceQuote component - Quote display with highlighting
 *
 * Displays the supporting quote from a source with optional
 * keyword highlighting and truncation with "Show more".
 */

import React, { useState, useMemo } from 'react';

interface EvidenceQuoteProps {
  /** The quote text to display */
  quoteText: string;
  /** Optional keywords to highlight in the quote */
  highlightKeywords?: string[];
  /** Maximum characters before truncation */
  maxLength?: number;
  /** Section heading for context */
  sectionHeading?: string | null;
  /** Start offset for location info */
  startOffset?: number | null;
  /** End offset for location info */
  endOffset?: number | null;
}

/**
 * Highlight keywords in text
 */
function highlightText(
  text: string,
  keywords: string[]
): React.ReactNode[] {
  if (!keywords || keywords.length === 0) {
    return [text];
  }

  // Create regex pattern for all keywords
  const escapedKeywords = keywords.map((kw) =>
    kw.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  );
  const pattern = new RegExp(`(${escapedKeywords.join('|')})`, 'gi');

  const parts = text.split(pattern);

  return parts.map((part, index) => {
    const isKeyword = keywords.some(
      (kw) => kw.toLowerCase() === part.toLowerCase()
    );

    if (isKeyword) {
      return (
        <mark
          key={`kw-${index}-${part.slice(0, 10)}`}
          className="bg-yellow-200 dark:bg-yellow-800 px-0.5 rounded"
        >
          {part}
        </mark>
      );
    }

    return part;
  });
}

export const EvidenceQuote: React.FC<EvidenceQuoteProps> = ({
  quoteText,
  highlightKeywords = [],
  maxLength = 300,
  sectionHeading,
  startOffset,
  endOffset,
}) => {
  const [isExpanded, setIsExpanded] = useState(false);

  const shouldTruncate = quoteText.length > maxLength;
  const displayText = useMemo(() => {
    if (!shouldTruncate || isExpanded) {
      return quoteText;
    }
    return quoteText.slice(0, maxLength).trimEnd() + '...';
  }, [quoteText, maxLength, shouldTruncate, isExpanded]);

  const highlightedContent = useMemo(
    () => highlightText(displayText, highlightKeywords),
    [displayText, highlightKeywords]
  );

  return (
    <div className="space-y-2">
      {/* Section heading if available */}
      {sectionHeading && (
        <div className="text-xs text-gray-500 dark:text-gray-400 flex items-center gap-1">
          <span>§</span>
          <span>{sectionHeading}</span>
        </div>
      )}

      {/* Quote with styling */}
      <blockquote data-testid="evidence-quote" className="relative pl-4 py-2 border-l-4 border-gray-300 dark:border-gray-600 bg-gray-50 dark:bg-gray-800 rounded-r">
        {/* Opening quote mark */}
        <span className="absolute top-0 left-1 text-2xl text-gray-300 dark:text-gray-600 font-serif">
          "
        </span>

        <p className="text-sm text-gray-700 dark:text-gray-300 pl-3 pr-2 leading-relaxed">
          {highlightedContent}
        </p>

        {/* Closing quote mark */}
        <span className="absolute bottom-0 right-2 text-2xl text-gray-300 dark:text-gray-600 font-serif">
          "
        </span>
      </blockquote>

      {/* Show more/less button */}
      {shouldTruncate && (
        <button
          data-testid="evidence-quote-expand"
          onClick={() => setIsExpanded(!isExpanded)}
          className="text-xs text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300 font-medium"
          aria-expanded={isExpanded}
          aria-label={isExpanded ? 'Show less of the quote' : 'Show more of the quote'}
        >
          {isExpanded ? 'Show less' : 'Show more'}
        </button>
      )}

      {/* Location info */}
      {(startOffset !== null || endOffset !== null) && (
        <div className="text-xs text-gray-400 dark:text-gray-500">
          Location: chars {startOffset ?? '?'}-{endOffset ?? '?'}
        </div>
      )}
    </div>
  );
};

export default EvidenceQuote;
