/**
 * VerificationBadge component - Verdict indicator badge
 *
 * Displays a color-coded badge showing the verification verdict
 * for a claim (Supported, Partially Supported, Unsupported, Contradicted).
 */

import React from 'react';
import {
  VerificationVerdict,
  VERDICT_LABELS,
} from '@/types/citation';

interface VerificationBadgeProps {
  /** Verification verdict */
  verdict: VerificationVerdict;
  /** Optional size variant */
  size?: 'sm' | 'md' | 'lg';
  /** Whether to show the full label or just an icon */
  showLabel?: boolean;
}

/**
 * Get Tailwind classes for badge styling based on verdict
 */
function getVerdictStyles(verdict: VerificationVerdict): {
  bg: string;
  text: string;
  border: string;
  icon: string;
} {
  const styles: Record<VerificationVerdict, { bg: string; text: string; border: string; icon: string }> = {
    supported: {
      bg: 'bg-green-50',
      text: 'text-green-700',
      border: 'border-green-200',
      icon: '✓',
    },
    partial: {
      bg: 'bg-amber-50',
      text: 'text-amber-700',
      border: 'border-amber-200',
      icon: '~',
    },
    unsupported: {
      bg: 'bg-red-50',
      text: 'text-red-700',
      border: 'border-red-200',
      icon: '?',
    },
    contradicted: {
      bg: 'bg-purple-50',
      text: 'text-purple-700',
      border: 'border-purple-200',
      icon: '✗',
    },
  };

  return styles[verdict];
}

/**
 * Get size classes
 */
function getSizeClasses(size: 'sm' | 'md' | 'lg'): string {
  const sizeMap = {
    sm: 'text-xs px-1.5 py-0.5',
    md: 'text-sm px-2 py-1',
    lg: 'text-base px-3 py-1.5',
  };

  return sizeMap[size];
}

export const VerificationBadge: React.FC<VerificationBadgeProps> = ({
  verdict,
  size = 'sm',
  showLabel = true,
}) => {
  const styles = getVerdictStyles(verdict);
  const sizeClasses = getSizeClasses(size);
  const label = VERDICT_LABELS[verdict];

  return (
    <span
      data-testid={`verification-badge-${verdict}`}
      className={`
        inline-flex items-center gap-1
        font-medium rounded-full border
        ${styles.bg} ${styles.text} ${styles.border}
        ${sizeClasses}
      `}
      title={label}
    >
      <span className="font-bold">{styles.icon}</span>
      {showLabel && <span>{label}</span>}
    </span>
  );
};

export default VerificationBadge;
