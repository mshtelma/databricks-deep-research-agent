import { cn } from '@/lib/utils';

export type ResearchDepth = 'auto' | 'light' | 'medium' | 'extended';

interface ResearchDepthSelectorProps {
  value: ResearchDepth;
  onChange: (depth: ResearchDepth) => void;
  disabled?: boolean;
  className?: string;
}

const DEPTH_OPTIONS: { value: ResearchDepth; label: string; description: string }[] = [
  { value: 'auto', label: 'Auto', description: 'System chooses based on query complexity' },
  { value: 'light', label: 'Light', description: 'Quick search (1-3 steps)' },
  { value: 'medium', label: 'Medium', description: 'Balanced research (3-6 steps)' },
  { value: 'extended', label: 'Extended', description: 'Deep research (5-10 steps)' },
];

export function ResearchDepthSelector({
  value,
  onChange,
  disabled = false,
  className,
}: ResearchDepthSelectorProps) {
  return (
    <div className={cn('flex items-center gap-1', className)}>
      <span className="text-xs text-muted-foreground mr-1">Depth:</span>
      <div className="flex gap-1 rounded-md border border-input p-0.5 bg-muted/50">
        {DEPTH_OPTIONS.map((option) => (
          <button
            key={option.value}
            type="button"
            onClick={() => onChange(option.value)}
            disabled={disabled}
            title={option.description}
            className={cn(
              'px-2 py-1 text-xs rounded transition-colors',
              'focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring',
              value === option.value
                ? 'bg-background text-foreground shadow-sm'
                : 'text-muted-foreground hover:text-foreground hover:bg-background/50',
              disabled && 'cursor-not-allowed opacity-50'
            )}
          >
            {option.label}
          </button>
        ))}
      </div>
    </div>
  );
}

export function ResearchDepthBadge({ depth }: { depth: ResearchDepth }) {
  const option = DEPTH_OPTIONS.find((o) => o.value === depth);
  if (!option || depth === 'auto') return null;

  return (
    <span
      className={cn(
        'inline-flex items-center px-2 py-0.5 rounded text-xs font-medium',
        depth === 'light' && 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200',
        depth === 'medium' && 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200',
        depth === 'extended' && 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200'
      )}
    >
      {option.label}
    </span>
  );
}
