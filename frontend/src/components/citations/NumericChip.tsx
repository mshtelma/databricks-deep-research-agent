/**
 * NumericChip component - Inline numeric value indicator
 *
 * Displays numeric claims as inline "chips" that can be expanded
 * to show the full numeric detail with normalization and derivation.
 */

import React, { useState } from 'react';
import type { NumericClaimDetail, DerivationType } from '@/types/citation';

interface NumericChipProps {
  /** The raw value displayed inline */
  rawValue: string;
  /** Full numeric claim details */
  detail: NumericClaimDetail | null;
  /** Whether QA verification matched */
  qaVerified?: boolean;
  /** Click handler to expand details */
  onClick?: () => void;
}

/**
 * Format a normalized value with appropriate units
 * Exported for use in NumericDetails and other components
 */
export function formatNormalizedValue(value: number | null, unit: string | null): string {
  if (value === null) return 'N/A';

  // Format large numbers with suffixes
  if (Math.abs(value) >= 1_000_000_000_000) {
    return `${(value / 1_000_000_000_000).toFixed(2)}T ${unit || ''}`.trim();
  }
  if (Math.abs(value) >= 1_000_000_000) {
    return `${(value / 1_000_000_000).toFixed(2)}B ${unit || ''}`.trim();
  }
  if (Math.abs(value) >= 1_000_000) {
    return `${(value / 1_000_000).toFixed(2)}M ${unit || ''}`.trim();
  }
  if (Math.abs(value) >= 1_000) {
    return `${(value / 1_000).toFixed(2)}K ${unit || ''}`.trim();
  }

  return `${value.toLocaleString()} ${unit || ''}`.trim();
}

/**
 * Get color classes based on derivation type and QA status
 */
function getChipColorClasses(
  derivationType: DerivationType,
  qaVerified?: boolean
): string {
  if (derivationType === 'computed') {
    return 'bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-300 border-amber-300';
  }

  if (qaVerified === false) {
    return 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300 border-red-300';
  }

  if (qaVerified === true) {
    return 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300 border-green-300';
  }

  return 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300 border-blue-300';
}

export const NumericChip: React.FC<NumericChipProps> = ({
  rawValue,
  detail,
  qaVerified,
  onClick,
}) => {
  const [isHovered, setIsHovered] = useState(false);

  const derivationType = detail?.derivationType || 'direct';
  const colorClasses = getChipColorClasses(derivationType, qaVerified);

  // Icon based on derivation type
  const icon = derivationType === 'computed' ? (
    <span title="Derived value" className="mr-1">
      <svg
        className="w-3 h-3 inline-block"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M9 7h6m0 10v-3m-3 3h.01M9 17h.01M9 14h.01M12 14h.01M15 11h.01M12 11h.01M9 11h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z"
        />
      </svg>
    </span>
  ) : null;

  return (
    <span
      data-testid="numeric-chip"
      className={`
        inline-flex items-center
        px-1.5 py-0.5
        text-xs font-medium
        border rounded-md
        cursor-pointer
        transition-all duration-150
        ${colorClasses}
        ${isHovered ? 'ring-1 ring-current shadow-sm' : ''}
      `}
      onClick={onClick}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          onClick?.();
        }
      }}
      aria-label={`Numeric value: ${rawValue}${derivationType === 'computed' ? ' (derived)' : ''}`}
    >
      {icon}
      {rawValue}
      {qaVerified !== undefined && (
        <span className="ml-1">
          {qaVerified ? (
            <svg className="w-3 h-3 text-green-600" fill="currentColor" viewBox="0 0 20 20">
              <path
                fillRule="evenodd"
                d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                clipRule="evenodd"
              />
            </svg>
          ) : (
            <svg className="w-3 h-3 text-red-600" fill="currentColor" viewBox="0 0 20 20">
              <path
                fillRule="evenodd"
                d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z"
                clipRule="evenodd"
              />
            </svg>
          )}
        </span>
      )}
    </span>
  );
};

export default NumericChip;
