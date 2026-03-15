import { cn } from '@/lib/utils';
import type { QueryMode } from '@/types';

interface QueryModeSelectorProps {
  value: QueryMode;
  onChange: (mode: QueryMode) => void;
  disabled?: boolean;
  className?: string;
}

const MODE_OPTIONS: { value: QueryMode; label: string; icon: string; description: string }[] = [
  {
    value: 'simple',
    label: 'Simple',
    icon: '\u26A1', // Lightning bolt
    description: 'Quick LLM response without web search (~3s)',
  },
  {
    value: 'web_search',
    label: 'Web Search',
    icon: '\uD83D\uDD0D', // Magnifying glass
    description: 'Fast answer with 2-5 web sources (~15s)',
  },
  {
    value: 'deep_research',
    label: 'Deep Research',
    icon: '\uD83D\uDD2C', // Microscope
    description: 'Comprehensive multi-step research (~2min)',
  },
];

export function QueryModeSelector({
  value,
  onChange,
  disabled = false,
  className,
}: QueryModeSelectorProps) {
  return (
    <div className={cn('flex items-center gap-1', className)}>
      <span className="text-xs text-muted-foreground mr-1">Mode:</span>
      <div className="flex gap-1 rounded-md border border-input p-0.5 bg-muted/50">
        {MODE_OPTIONS.map((option) => (
          <button
            key={option.value}
            type="button"
            onClick={() => onChange(option.value)}
            disabled={disabled}
            title={option.description}
            data-testid={`mode-${option.value}`}
            className={cn(
              'px-2 py-1 text-xs rounded transition-colors',
              'focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring',
              value === option.value
                ? 'bg-background text-foreground shadow-sm'
                : 'text-muted-foreground hover:text-foreground hover:bg-background/50',
              disabled && 'cursor-not-allowed opacity-50'
            )}
          >
            <span className="mr-1" aria-hidden="true">{option.icon}</span>
            {option.label}
          </button>
        ))}
      </div>
    </div>
  );
}

export function QueryModeBadge({ mode }: { mode: QueryMode }) {
  const option = MODE_OPTIONS.find((o) => o.value === mode);
  if (!option) return null;

  return (
    <span
      className={cn(
        'inline-flex items-center px-2 py-0.5 rounded text-xs font-medium',
        mode === 'simple' && 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200',
        mode === 'web_search' && 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200',
        mode === 'deep_research' && 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200'
      )}
    >
      {option.label}
    </span>
  );
}
