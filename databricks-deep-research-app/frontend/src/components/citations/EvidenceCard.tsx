/**
 * EvidenceCard component - Full evidence card popover
 *
 * Displays comprehensive evidence information including:
 * - Source metadata (title, URL, author, date)
 * - Supporting quote with highlighting
 * - Verification verdict badge
 * - Location context (section, page)
 */

import React from 'react';
import { Citation, VerificationVerdict } from '@/types/citation';
import { SourceMetadata } from './SourceMetadata';
import { EvidenceQuote } from './EvidenceQuote';
import { VerificationBadge } from './VerificationBadge';

interface EvidenceCardProps {
  /** Citation data with evidence span and source (optional for claims without citations) */
  citation?: Citation | null;
  /** Claim text for context */
  claimText?: string;
  /** Verification verdict for the claim */
  verdict?: VerificationVerdict | null;
  /** Keywords to highlight in the quote */
  highlightKeywords?: string[];
  /** Whether the card is in a popover (affects styling) */
  isPopover?: boolean;
  /** Close handler for popover mode */
  onClose?: () => void;
}

export const EvidenceCard: React.FC<EvidenceCardProps> = React.memo(({
  citation,
  claimText,
  verdict,
  highlightKeywords = [],
  isPopover = false,
  onClose,
}) => {
  // Handle optional citation - extract values with defaults
  const evidenceSpan = citation?.evidenceSpan;
  const source = evidenceSpan?.source;
  const confidenceScore = citation?.confidenceScore ?? null;
  const isPrimary = citation?.isPrimary ?? false;

  return (
    <div
      data-testid="evidence-card"
      className={`
        bg-white dark:bg-gray-900
        border border-gray-200 dark:border-gray-700
        rounded-lg shadow-lg
        ${isPopover ? 'max-w-xl w-[32rem]' : 'w-full'}
      `}
    >
      {/* Header with verdict and close button */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-100 dark:border-gray-800">
        <div className="flex items-center gap-2">
          {verdict && <VerificationBadge verdict={verdict} size="sm" />}
          {isPrimary && (
            <span className="text-xs text-gray-500 dark:text-gray-400 bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">
              Primary
            </span>
          )}
          {confidenceScore !== null && (
            <span className="text-xs text-gray-400 dark:text-gray-500">
              {Math.round(confidenceScore * 100)}% confidence
            </span>
          )}
        </div>
        {isPopover && onClose && (
          <button
            data-testid="evidence-card-close"
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 p-1"
            aria-label="Close"
          >
            <svg
              className="w-4 h-4"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        )}
      </div>

      {/* Source metadata */}
      {source && (
        <div className="px-4 py-3 border-b border-gray-100 dark:border-gray-800">
          <SourceMetadata source={source} />
        </div>
      )}

      {/* Evidence quote - only show if we have evidence */}
      {evidenceSpan ? (
        <div className="px-4 py-3">
          <div className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-2 uppercase tracking-wide">
            Supporting Evidence
          </div>
          <EvidenceQuote
            quoteText={evidenceSpan.quoteText}
            highlightKeywords={highlightKeywords}
            sectionHeading={evidenceSpan.sectionHeading}
            startOffset={evidenceSpan.startOffset}
            endOffset={evidenceSpan.endOffset}
            maxLength={isPopover ? 400 : 600}
          />
        </div>
      ) : (
        <div className="px-4 py-3">
          <div className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-2 uppercase tracking-wide">
            Evidence
          </div>
          <p className="text-sm text-gray-500 dark:text-gray-400 italic">
            No detailed evidence available for this claim.
          </p>
        </div>
      )}

      {/* Claim preview (if provided and in popover) */}
      {isPopover && claimText && (
        <div className="px-4 py-3 bg-gray-50 dark:bg-gray-800 rounded-b-lg">
          <div className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-1 uppercase tracking-wide">
            Claim
          </div>
          <p className="text-sm text-gray-700 dark:text-gray-300 line-clamp-4">
            {claimText}
          </p>
        </div>
      )}

      {/* Footer with additional info */}
      {evidenceSpan?.hasNumericContent && (
        <div className="px-4 py-2 bg-blue-50 dark:bg-blue-900/20 rounded-b-lg border-t border-blue-100 dark:border-blue-800">
          <span className="text-xs text-blue-600 dark:text-blue-400 flex items-center gap-1">
            <span aria-label="Contains numeric data">📊</span>
            Contains numeric data
          </span>
        </div>
      )}
    </div>
  );
});

EvidenceCard.displayName = 'EvidenceCard';

export default EvidenceCard;
