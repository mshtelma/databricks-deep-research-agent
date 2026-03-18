import * as React from 'react';
import { cn } from '@/lib/utils';

interface FeedbackButtonsProps {
  currentRating?: 'positive' | 'negative' | null;
  onFeedback: (rating: 'positive' | 'negative') => void;
  onErrorReport?: () => void;
  disabled?: boolean;
  className?: string;
}

export function FeedbackButtons({
  currentRating,
  onFeedback,
  onErrorReport,
  disabled = false,
  className,
}: FeedbackButtonsProps) {
  const [submitting, setSubmitting] = React.useState(false);

  const handleFeedback = async (rating: 'positive' | 'negative') => {
    if (disabled || submitting) return;

    setSubmitting(true);
    try {
      onFeedback(rating);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className={cn('flex items-center gap-1', className)}>
      {/* Thumbs up */}
      <button
        type="button"
        onClick={() => handleFeedback('positive')}
        disabled={disabled || submitting}
        aria-label="Good response"
        aria-pressed={currentRating === 'positive'}
        className={cn(
          'p-1.5 rounded transition-colors',
          'hover:bg-green-100 dark:hover:bg-green-900/30',
          'focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring',
          'disabled:cursor-not-allowed disabled:opacity-50',
          currentRating === 'positive' && 'bg-green-100 dark:bg-green-900/30 text-green-600'
        )}
      >
        <ThumbsUpIcon
          className={cn(
            'w-4 h-4',
            currentRating === 'positive' ? 'text-green-600' : 'text-muted-foreground'
          )}
          filled={currentRating === 'positive'}
        />
      </button>

      {/* Thumbs down */}
      <button
        type="button"
        onClick={() => handleFeedback('negative')}
        disabled={disabled || submitting}
        aria-label="Bad response"
        aria-pressed={currentRating === 'negative'}
        className={cn(
          'p-1.5 rounded transition-colors',
          'hover:bg-red-100 dark:hover:bg-red-900/30',
          'focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring',
          'disabled:cursor-not-allowed disabled:opacity-50',
          currentRating === 'negative' && 'bg-red-100 dark:bg-red-900/30 text-red-600'
        )}
      >
        <ThumbsDownIcon
          className={cn(
            'w-4 h-4',
            currentRating === 'negative' ? 'text-red-600' : 'text-muted-foreground'
          )}
          filled={currentRating === 'negative'}
        />
      </button>

      {/* Error report button (shown after negative feedback) */}
      {currentRating === 'negative' && onErrorReport && (
        <button
          type="button"
          onClick={onErrorReport}
          disabled={disabled}
          className={cn(
            'ml-1 px-2 py-1 text-xs rounded transition-colors',
            'text-muted-foreground hover:text-foreground hover:bg-muted',
            'focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring'
          )}
        >
          Report issue
        </button>
      )}
    </div>
  );
}

interface IconProps {
  className?: string;
  filled?: boolean;
}

function ThumbsUpIcon({ className, filled = false }: IconProps) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill={filled ? 'currentColor' : 'none'}
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <path d="M7 10v12" />
      <path d="M15 5.88 14 10h5.83a2 2 0 0 1 1.92 2.56l-2.33 8A2 2 0 0 1 17.5 22H4a2 2 0 0 1-2-2v-8a2 2 0 0 1 2-2h2.76a2 2 0 0 0 1.79-1.11L12 2a3.13 3.13 0 0 1 3 3.88Z" />
    </svg>
  );
}

function ThumbsDownIcon({ className, filled = false }: IconProps) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill={filled ? 'currentColor' : 'none'}
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <path d="M17 14V2" />
      <path d="M9 18.12 10 14H4.17a2 2 0 0 1-1.92-2.56l2.33-8A2 2 0 0 1 6.5 2H20a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2h-2.76a2 2 0 0 0-1.79 1.11L12 22a3.13 3.13 0 0 1-3-3.88Z" />
    </svg>
  );
}
