import * as React from 'react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

interface ClarificationQuestion {
  question: string;
  options?: string[];
}

interface ClarificationDialogProps {
  question: ClarificationQuestion;
  onSubmit: (answer: string) => void;
  onSkip?: () => void;
  className?: string;
}

export function ClarificationDialog({
  question,
  onSubmit,
  onSkip,
  className,
}: ClarificationDialogProps) {
  const [answer, setAnswer] = React.useState('');
  const [selectedOption, setSelectedOption] = React.useState<string | null>(null);

  const handleSubmit = () => {
    const response = selectedOption || answer.trim();
    if (response) {
      onSubmit(response);
      setAnswer('');
      setSelectedOption(null);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div
      className={cn(
        'rounded-lg border border-amber-200 bg-amber-50 dark:border-amber-800 dark:bg-amber-950/50 p-4',
        className
      )}
    >
      <div className="flex items-start gap-3">
        <div className="flex-shrink-0">
          <span className="inline-flex h-8 w-8 items-center justify-center rounded-full bg-amber-100 dark:bg-amber-900">
            <svg
              className="h-4 w-4 text-amber-600 dark:text-amber-400"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
          </span>
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-amber-800 dark:text-amber-200 mb-2">
            Clarification needed
          </p>
          <p className="text-sm text-amber-700 dark:text-amber-300 mb-3">
            {question.question}
          </p>

          {question.options && question.options.length > 0 ? (
            <div className="space-y-2 mb-3">
              {question.options.map((option, index) => (
                <button
                  key={index}
                  type="button"
                  onClick={() => setSelectedOption(option)}
                  className={cn(
                    'w-full text-left px-3 py-2 rounded-md text-sm transition-colors',
                    selectedOption === option
                      ? 'bg-amber-200 dark:bg-amber-800 text-amber-900 dark:text-amber-100'
                      : 'bg-white dark:bg-amber-900/50 text-amber-700 dark:text-amber-300 hover:bg-amber-100 dark:hover:bg-amber-800/50'
                  )}
                >
                  {option}
                </button>
              ))}
              <button
                type="button"
                onClick={() => setSelectedOption(null)}
                className={cn(
                  'w-full text-left px-3 py-2 rounded-md text-sm transition-colors',
                  selectedOption === null && answer.trim()
                    ? 'bg-amber-200 dark:bg-amber-800 text-amber-900 dark:text-amber-100'
                    : 'bg-white dark:bg-amber-900/50 text-amber-700 dark:text-amber-300 hover:bg-amber-100 dark:hover:bg-amber-800/50'
                )}
              >
                Other (type your answer)
              </button>
            </div>
          ) : null}

          {(!question.options || selectedOption === null) && (
            <textarea
              value={answer}
              onChange={(e) => setAnswer(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Type your answer..."
              rows={2}
              className={cn(
                'w-full resize-none rounded-md border border-amber-300 dark:border-amber-700 bg-white dark:bg-amber-900/50',
                'px-3 py-2 text-sm text-amber-900 dark:text-amber-100',
                'placeholder:text-amber-400 dark:placeholder:text-amber-600',
                'focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-amber-500',
                'mb-3'
              )}
            />
          )}

          <div className="flex gap-2">
            <Button
              size="sm"
              onClick={handleSubmit}
              disabled={!selectedOption && !answer.trim()}
              className="bg-amber-600 hover:bg-amber-700 text-white"
            >
              Answer
            </Button>
            {onSkip && (
              <Button
                size="sm"
                variant="ghost"
                onClick={onSkip}
                className="text-amber-600 hover:text-amber-700 hover:bg-amber-100 dark:hover:bg-amber-900"
              >
                Skip
              </Button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
