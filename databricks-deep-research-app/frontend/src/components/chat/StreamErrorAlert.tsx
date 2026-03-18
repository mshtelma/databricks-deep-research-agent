import { useState } from 'react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';

interface StreamErrorAlertProps {
  error: Error;
  errorCode?: string;
  stackTrace?: string;
  errorType?: string;
  recoverable?: boolean;
  onRetry?: () => void;
  onDismiss?: () => void;
}

/**
 * Displays research errors inline with an expandable stack trace.
 *
 * Since Databricks Apps logs get purged, this component shows the full
 * traceback in a collapsible panel so users can debug issues.
 */
export function StreamErrorAlert({
  error,
  errorCode,
  stackTrace,
  errorType,
  recoverable,
  onRetry,
  onDismiss,
}: StreamErrorAlertProps) {
  const [showDetails, setShowDetails] = useState(false);
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    if (!stackTrace) return;

    const textToCopy = [
      `Error: ${error.message}`,
      errorType && `Type: ${errorType}`,
      errorCode && `Code: ${errorCode}`,
      '',
      'Stack Trace:',
      stackTrace,
    ].filter(Boolean).join('\n');

    try {
      await navigator.clipboard.writeText(textToCopy);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (e) {
      console.error('Failed to copy to clipboard:', e);
    }
  };

  return (
    <div className="bg-destructive/10 border border-destructive/30 rounded-lg p-4 my-4">
      {/* Header */}
      <div className="flex items-start gap-3">
        {/* Error Icon */}
        <svg
          className="w-5 h-5 text-destructive shrink-0 mt-0.5"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          strokeWidth={2}
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
          />
        </svg>

        <div className="flex-1 min-w-0">
          <h3 className="font-medium text-destructive">
            Research Failed
          </h3>
          <p className="text-sm text-destructive/80 mt-1 break-words">
            {error.message}
          </p>
        </div>

        {onDismiss && (
          <button
            onClick={onDismiss}
            className="shrink-0 text-destructive/60 hover:text-destructive transition-colors"
            aria-label="Dismiss error"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        )}
      </div>

      {/* Actions */}
      <div className="flex flex-wrap gap-2 mt-3">
        {recoverable && onRetry && (
          <Button variant="destructive" size="sm" onClick={onRetry}>
            <svg className="w-4 h-4 mr-1.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
            Retry
          </Button>
        )}
        {stackTrace && (
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setShowDetails(!showDetails)}
            className="text-destructive/80 hover:text-destructive hover:bg-destructive/10"
          >
            {showDetails ? 'Hide' : 'Show'} Technical Details
            <svg
              className={cn("ml-1.5 w-4 h-4 transition-transform", showDetails && "rotate-180")}
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </Button>
        )}
      </div>

      {/* Collapsible Stack Trace */}
      {showDetails && stackTrace && (
        <div className="mt-3">
          <div className="flex justify-between items-center mb-2">
            <span className="text-xs text-muted-foreground">
              {errorType && <span className="font-medium">{errorType}</span>}
              {errorType && errorCode && ' • '}
              {errorCode && <span className="font-mono">{errorCode}</span>}
            </span>
            <Button
              variant="ghost"
              size="sm"
              onClick={handleCopy}
              className="h-7 px-2 text-xs"
            >
              {copied ? (
                <>
                  <svg className="w-3 h-3 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                  Copied!
                </>
              ) : (
                <>
                  <svg className="w-3 h-3 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                  </svg>
                  Copy
                </>
              )}
            </Button>
          </div>
          <pre className="bg-zinc-900 text-red-300 p-3 rounded text-xs overflow-x-auto max-h-64 overflow-y-auto font-mono whitespace-pre-wrap break-words">
            {stackTrace}
          </pre>
        </div>
      )}
    </div>
  );
}
