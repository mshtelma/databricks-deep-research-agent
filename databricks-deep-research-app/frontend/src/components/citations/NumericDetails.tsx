/**
 * NumericDetails component - Expanded view of numeric claim details
 *
 * Shows when a NumericChip is clicked:
 * - Exact quote containing the number
 * - Normalization (e.g., "$3.2B" -> 3,200,000,000 USD)
 * - Assumptions (currency year, exchange rate, rounding)
 * - Derivation information for computed values
 */

import React from 'react';
import type { NumericClaimDetail, ComputationStep } from '@/types/citation';

interface NumericDetailsProps {
  /** The numeric claim detail to display */
  detail: NumericClaimDetail;
  /** The claim text for context */
  claimText?: string;
  /** Handler to close the details panel */
  onClose?: () => void;
}

/**
 * Format a number with locale-specific separators
 */
function formatNumber(value: number | null): string {
  if (value === null) return 'N/A';
  return value.toLocaleString('en-US', {
    maximumFractionDigits: 2,
  });
}

/**
 * Format a computation step for display
 */
function formatComputationStep(step: ComputationStep, index: number): React.ReactNode {
  return (
    <div
      key={index}
      className="flex items-start gap-2 text-sm bg-gray-50 dark:bg-gray-800 p-2 rounded"
    >
      <span className="flex-shrink-0 w-6 h-6 flex items-center justify-center bg-amber-200 dark:bg-amber-800 text-amber-800 dark:text-amber-200 rounded-full text-xs font-medium">
        {index + 1}
      </span>
      <div className="flex-1">
        <div className="font-medium text-gray-700 dark:text-gray-300">
          {step.operation}
        </div>
        <div className="text-gray-500 dark:text-gray-400 text-xs mt-1">
          Inputs: {step.inputs.map((input, i) => (
            <span key={i} className="mx-1">
              {formatNumber(input.value)}
            </span>
          ))}
        </div>
        <div className="text-gray-600 dark:text-gray-300 text-xs mt-1">
          Result: <strong>{formatNumber(step.result)}</strong>
        </div>
      </div>
    </div>
  );
}

export const NumericDetails: React.FC<NumericDetailsProps> = ({
  detail,
  claimText,
  onClose,
}) => {
  const {
    rawValue,
    normalizedValue,
    unit,
    entityReference,
    derivationType,
    computationDetails,
    assumptions,
    qaVerification,
  } = detail;

  return (
    <div data-testid="numeric-details" className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg max-w-md w-full">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-100 dark:border-gray-800">
        <div className="flex items-center gap-2">
          <span className={`
            px-2 py-0.5 text-xs font-medium rounded-full
            ${derivationType === 'computed'
              ? 'bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-300'
              : 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300'}
          `}>
            {derivationType === 'computed' ? 'Derived' : 'Direct Quote'}
          </span>
        </div>
        {onClose && (
          <button
            data-testid="numeric-details-close"
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 p-1"
            aria-label="Close"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        )}
      </div>

      {/* Main content */}
      <div className="px-4 py-3 space-y-4">
        {/* Raw value */}
        <div>
          <div className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide mb-1">
            Value as Stated
          </div>
          <div className="text-lg font-semibold text-gray-900 dark:text-white">
            {rawValue}
          </div>
        </div>

        {/* Normalized value */}
        {normalizedValue !== null && (
          <div>
            <div className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide mb-1">
              Normalized Value
            </div>
            <div className="text-sm text-gray-700 dark:text-gray-300">
              {formatNumber(normalizedValue)} {unit || ''}
            </div>
          </div>
        )}

        {/* Entity reference */}
        {entityReference && (
          <div>
            <div className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide mb-1">
              Entity
            </div>
            <div className="text-sm text-gray-700 dark:text-gray-300">
              {entityReference}
            </div>
          </div>
        )}

        {/* Computation details for derived values */}
        {derivationType === 'computed' && computationDetails && (
          <div>
            <div className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide mb-2">
              Calculation Steps
            </div>
            <div className="space-y-2">
              {computationDetails.steps.map((step, index) =>
                formatComputationStep(step, index)
              )}
            </div>
            {computationDetails.formula && (
              <div className="mt-2 text-xs text-gray-500 dark:text-gray-400 font-mono bg-gray-50 dark:bg-gray-800 p-2 rounded">
                {computationDetails.formula}
              </div>
            )}
          </div>
        )}

        {/* Assumptions */}
        {assumptions && Object.keys(assumptions).some((k) => assumptions[k] !== null) && (
          <div>
            <div className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide mb-2">
              Assumptions
            </div>
            <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
              {assumptions.currencyYear && (
                <li className="flex items-center gap-2">
                  <span className="w-2 h-2 bg-gray-400 rounded-full" />
                  Currency year: {assumptions.currencyYear}
                </li>
              )}
              {assumptions.exchangeRate && (
                <li className="flex items-center gap-2">
                  <span className="w-2 h-2 bg-gray-400 rounded-full" />
                  Exchange rate: {assumptions.exchangeRate}
                </li>
              )}
              {assumptions.roundingMethod && (
                <li className="flex items-center gap-2">
                  <span className="w-2 h-2 bg-gray-400 rounded-full" />
                  Rounding: {assumptions.roundingMethod}
                </li>
              )}
            </ul>
          </div>
        )}

        {/* QA Verification results */}
        {qaVerification && qaVerification.length > 0 && (
          <div>
            <div className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide mb-2">
              QA Verification
            </div>
            <div className="space-y-2">
              {qaVerification.map((qa, index) => (
                <div
                  key={index}
                  className={`
                    text-sm p-2 rounded border-l-2
                    ${qa.match
                      ? 'border-green-500 bg-green-50 dark:bg-green-900/20'
                      : 'border-red-500 bg-red-50 dark:bg-red-900/20'}
                  `}
                >
                  <div className="font-medium text-gray-700 dark:text-gray-300">
                    Q: {qa.question}
                  </div>
                  <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    Claim: {qa.claimAnswer}
                  </div>
                  <div className="text-xs text-gray-500 dark:text-gray-400">
                    Evidence: {qa.evidenceAnswer}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Claim context footer */}
      {claimText && (
        <div className="px-4 py-3 bg-gray-50 dark:bg-gray-800 rounded-b-lg border-t border-gray-100 dark:border-gray-700">
          <div className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
            Full Claim
          </div>
          <p className="text-sm text-gray-700 dark:text-gray-300 line-clamp-3">
            {claimText}
          </p>
        </div>
      )}
    </div>
  );
};

export default NumericDetails;
