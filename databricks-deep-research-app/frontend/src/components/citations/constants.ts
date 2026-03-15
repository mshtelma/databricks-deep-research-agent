/**
 * Constants for citation components
 */

export const CITATION_CONSTANTS = {
  /** Maximum length for quote truncation */
  QUOTE_TRUNCATE_LENGTH: 300,

  /** Maximum length for URL display */
  URL_TRUNCATE_LENGTH: 50,

  /** Maximum claims to show in verification section */
  MAX_CLAIMS_TO_SHOW: 10,

  /** Maximum width for popover */
  POPOVER_MAX_WIDTH: 384,

  /** Quote truncation for popover mode */
  POPOVER_QUOTE_LENGTH: 250,

  /** Quote truncation for inline mode */
  INLINE_QUOTE_LENGTH: 400,
} as const;

/** Verdict color mappings for Tailwind classes */
export const VERDICT_COLORS = {
  supported: 'text-green-600 dark:text-green-400',
  partial: 'text-amber-600 dark:text-amber-400',
  unsupported: 'text-red-600 dark:text-red-400',
  contradicted: 'text-purple-600 dark:text-purple-400',
} as const;

/** Verdict background color mappings */
export const VERDICT_BG_COLORS = {
  supported: 'bg-green-50 dark:bg-green-900/20',
  partial: 'bg-amber-50 dark:bg-amber-900/20',
  unsupported: 'bg-red-50 dark:bg-red-900/20',
  contradicted: 'bg-purple-50 dark:bg-purple-900/20',
} as const;

/** Human-readable verdict labels */
export const VERDICT_LABELS = {
  supported: 'Supported',
  partial: 'Partially Supported',
  unsupported: 'Unsupported',
  contradicted: 'Contradicted',
} as const;

/** Confidence level colors */
export const CONFIDENCE_COLORS = {
  high: 'text-green-600 dark:text-green-400',
  medium: 'text-amber-600 dark:text-amber-400',
  low: 'text-red-600 dark:text-red-400',
} as const;

export type VerificationVerdict = keyof typeof VERDICT_COLORS;
export type ConfidenceLevel = keyof typeof CONFIDENCE_COLORS;
