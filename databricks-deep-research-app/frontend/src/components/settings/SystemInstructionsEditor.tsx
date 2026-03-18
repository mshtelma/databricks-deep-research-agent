import * as React from 'react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

interface SystemInstructionsEditorProps {
  value: string;
  onChange: (value: string) => void;
  onSave: () => void;
  isSaving?: boolean;
  hasChanges?: boolean;
  className?: string;
}

const PLACEHOLDER = `Enter custom instructions that will be applied to all research responses.

Examples:
- "Always cite sources with numbered references"
- "Use formal academic language"
- "Focus on recent developments (last 2 years)"
- "Include pros and cons for comparisons"`;

const CHARACTER_LIMIT = 2000;

export function SystemInstructionsEditor({
  value,
  onChange,
  onSave,
  isSaving = false,
  hasChanges = false,
  className,
}: SystemInstructionsEditorProps) {
  const textareaRef = React.useRef<HTMLTextAreaElement>(null);
  const characterCount = value.length;
  const isOverLimit = characterCount > CHARACTER_LIMIT;

  // Auto-resize textarea
  React.useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = `${Math.min(textarea.scrollHeight, 300)}px`;
    }
  }, [value]);

  return (
    <div className={cn('space-y-4', className)}>
      <div>
        <label
          htmlFor="system-instructions"
          className="text-sm font-medium block mb-2"
        >
          System Instructions
        </label>
        <p className="text-xs text-muted-foreground mb-3">
          These instructions will be applied to all research responses. They help
          customize the agent's behavior, tone, and output format.
        </p>
        <textarea
          ref={textareaRef}
          id="system-instructions"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          disabled={isSaving}
          placeholder={PLACEHOLDER}
          className={cn(
            'w-full min-h-[150px] resize-none rounded-md border bg-transparent px-3 py-2 text-sm',
            'placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1',
            'disabled:cursor-not-allowed disabled:opacity-50',
            isOverLimit
              ? 'border-red-500 focus-visible:ring-red-500'
              : 'border-input focus-visible:ring-ring'
          )}
        />
        <div className="flex justify-between items-center mt-2">
          <span
            className={cn(
              'text-xs',
              isOverLimit ? 'text-red-500' : 'text-muted-foreground'
            )}
          >
            {characterCount} / {CHARACTER_LIMIT} characters
          </span>
          {isOverLimit && (
            <span className="text-xs text-red-500">
              Please reduce your instructions
            </span>
          )}
        </div>
      </div>

      {/* Suggestions */}
      <div className="space-y-2">
        <p className="text-xs font-medium text-muted-foreground">
          Quick suggestions:
        </p>
        <div className="flex flex-wrap gap-2">
          {[
            'Be concise',
            'Use bullet points',
            'Cite all sources',
            'Academic tone',
            'Simple language',
          ].map((suggestion) => (
            <button
              key={suggestion}
              type="button"
              onClick={() => {
                const newValue = value ? `${value}\n${suggestion}` : suggestion;
                onChange(newValue);
              }}
              disabled={isSaving}
              className={cn(
                'px-2 py-1 text-xs rounded-full border transition-colors',
                'hover:bg-muted focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring',
                'disabled:cursor-not-allowed disabled:opacity-50'
              )}
            >
              + {suggestion}
            </button>
          ))}
        </div>
      </div>

      {/* Save button */}
      <div className="flex justify-end">
        <Button
          onClick={onSave}
          disabled={isSaving || !hasChanges || isOverLimit}
        >
          {isSaving ? (
            <>
              <SpinnerIcon className="w-4 h-4 mr-2 animate-spin" />
              Saving...
            </>
          ) : (
            'Save Changes'
          )}
        </Button>
      </div>
    </div>
  );
}

function SpinnerIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none">
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="4"
      />
      <path
        className="opacity-75"
        fill="currentColor"
        d="m4 12a8 8 0 0 1 8-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 0 1 4 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
      />
    </svg>
  );
}
