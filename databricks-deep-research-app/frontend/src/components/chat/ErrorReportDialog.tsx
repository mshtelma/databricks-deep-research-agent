import * as React from 'react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

export type ErrorCategory =
  | 'incorrect_info'
  | 'missing_info'
  | 'confusing'
  | 'off_topic'
  | 'other';

const ERROR_CATEGORIES: { value: ErrorCategory; label: string }[] = [
  { value: 'incorrect_info', label: 'Incorrect information' },
  { value: 'missing_info', label: 'Missing important information' },
  { value: 'confusing', label: 'Confusing or unclear' },
  { value: 'off_topic', label: 'Off-topic response' },
  { value: 'other', label: 'Other issue' },
];

interface ErrorReportDialogProps {
  isOpen: boolean;
  onClose: () => void;
  onSubmit: (report: { category: ErrorCategory; details: string }) => void;
  isSubmitting?: boolean;
}

export function ErrorReportDialog({
  isOpen,
  onClose,
  onSubmit,
  isSubmitting = false,
}: ErrorReportDialogProps) {
  const [category, setCategory] = React.useState<ErrorCategory>('incorrect_info');
  const [details, setDetails] = React.useState('');
  const dialogRef = React.useRef<HTMLDivElement>(null);
  const textareaRef = React.useRef<HTMLTextAreaElement>(null);

  // Reset form when dialog opens
  React.useEffect(() => {
    if (isOpen) {
      setCategory('incorrect_info');
      setDetails('');
      setTimeout(() => textareaRef.current?.focus(), 100);
    }
  }, [isOpen]);

  // Close on escape key
  React.useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  // Close on click outside
  React.useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (dialogRef.current && !dialogRef.current.contains(e.target as Node) && isOpen) {
        onClose();
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [isOpen, onClose]);

  const handleSubmit = () => {
    onSubmit({ category, details: details.trim() });
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div className="fixed inset-0 bg-black/50" aria-hidden="true" />

      {/* Dialog */}
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="error-report-title"
        className={cn(
          'relative z-50 w-full max-w-md rounded-lg bg-background p-6 shadow-lg',
          'animate-in fade-in-0 zoom-in-95'
        )}
      >
        <h3 id="error-report-title" className="text-lg font-semibold mb-2">
          Report an Issue
        </h3>

        <p className="text-sm text-muted-foreground mb-4">
          Help us improve by describing what went wrong with this response.
        </p>

        {/* Category selection */}
        <div className="space-y-2 mb-4">
          <label className="text-sm font-medium">What type of issue?</label>
          <div className="space-y-2">
            {ERROR_CATEGORIES.map((cat) => (
              <label
                key={cat.value}
                className={cn(
                  'flex items-center gap-3 p-2 rounded-md border cursor-pointer transition-colors',
                  category === cat.value
                    ? 'border-primary bg-primary/5'
                    : 'border-border hover:border-primary/50'
                )}
              >
                <input
                  type="radio"
                  name="error-category"
                  value={cat.value}
                  checked={category === cat.value}
                  onChange={() => setCategory(cat.value)}
                  disabled={isSubmitting}
                  className="accent-primary"
                />
                <span className="text-sm">{cat.label}</span>
              </label>
            ))}
          </div>
        </div>

        {/* Details textarea */}
        <div className="mb-4">
          <label htmlFor="error-details" className="text-sm font-medium block mb-2">
            Additional details (optional)
          </label>
          <textarea
            ref={textareaRef}
            id="error-details"
            value={details}
            onChange={(e) => setDetails(e.target.value)}
            disabled={isSubmitting}
            placeholder="Describe the issue in more detail..."
            rows={4}
            className={cn(
              'w-full resize-none rounded-md border border-input bg-transparent px-3 py-2 text-sm',
              'placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring',
              'disabled:cursor-not-allowed disabled:opacity-50'
            )}
          />
        </div>

        <div className="flex justify-end gap-3">
          <Button variant="outline" onClick={onClose} disabled={isSubmitting}>
            Cancel
          </Button>
          <Button onClick={handleSubmit} disabled={isSubmitting}>
            {isSubmitting ? 'Submitting...' : 'Submit Report'}
          </Button>
        </div>
      </div>
    </div>
  );
}
