import { cn } from '@/lib/utils';

export type StepType = 'search' | 'fetch' | 'analyze' | 'reflect';

export interface ReasoningStepData {
  id: string;
  type: StepType;
  title: string;
  description?: string;
  status: 'pending' | 'in_progress' | 'completed' | 'failed';
  timestamp?: string;
  details?: {
    searchQueries?: string[];
    urls?: string[];
    observation?: string;
    sourcesFound?: number;
  };
}

interface ReasoningStepProps {
  step: ReasoningStepData;
  isExpanded?: boolean;
  onToggle?: () => void;
  className?: string;
}

export function ReasoningStep({
  step,
  isExpanded = false,
  onToggle,
  className,
}: ReasoningStepProps) {
  const hasDetails =
    step.details?.searchQueries?.length ||
    step.details?.urls?.length ||
    step.details?.observation;

  return (
    <div
      className={cn(
        'rounded-md border transition-colors',
        step.status === 'in_progress' && 'border-primary/50 bg-primary/5',
        step.status === 'completed' && 'border-green-500/30 bg-green-50/50 dark:bg-green-950/20',
        step.status === 'failed' && 'border-red-500/30 bg-red-50/50 dark:bg-red-950/20',
        step.status === 'pending' && 'border-muted',
        className
      )}
    >
      {/* Header */}
      <button
        type="button"
        onClick={onToggle}
        disabled={!hasDetails}
        className={cn(
          'w-full flex items-center gap-3 p-3 text-left',
          hasDetails && 'cursor-pointer hover:bg-muted/50'
        )}
      >
        <StepIcon type={step.type} status={step.status} />
        <div className="flex-1 min-w-0">
          <p className={cn(
            'text-sm font-medium',
            step.status === 'completed' && 'text-green-700 dark:text-green-400',
            step.status === 'failed' && 'text-red-700 dark:text-red-400'
          )}>
            {step.title}
          </p>
          {step.description && (
            <p className="text-xs text-muted-foreground mt-0.5 truncate">
              {step.description}
            </p>
          )}
        </div>
        {step.status === 'completed' && step.details?.sourcesFound !== undefined && (
          <span className="text-xs text-muted-foreground">
            {step.details.sourcesFound} source{step.details.sourcesFound !== 1 ? 's' : ''}
          </span>
        )}
        {hasDetails && (
          <ChevronIcon className={cn(
            'w-4 h-4 text-muted-foreground transition-transform',
            isExpanded && 'rotate-180'
          )} />
        )}
      </button>

      {/* Expanded Details */}
      {isExpanded && hasDetails && (
        <div className="px-3 pb-3 pt-0 border-t border-muted">
          {step.details?.searchQueries && step.details.searchQueries.length > 0 && (
            <div className="mt-3">
              <p className="text-xs font-medium text-muted-foreground mb-1.5 flex items-center gap-1">
                <SearchIcon className="w-3 h-3" />
                Search Queries
              </p>
              <ul className="space-y-1">
                {step.details.searchQueries.map((query, i) => (
                  <li
                    key={i}
                    className="text-xs bg-muted/50 px-2 py-1 rounded font-mono"
                  >
                    "{query}"
                  </li>
                ))}
              </ul>
            </div>
          )}

          {step.details?.urls && step.details.urls.length > 0 && (
            <div className="mt-3">
              <p className="text-xs font-medium text-muted-foreground mb-1.5 flex items-center gap-1">
                <LinkIcon className="w-3 h-3" />
                Fetched URLs
              </p>
              <ul className="space-y-1">
                {step.details.urls.map((url, i) => (
                  <li key={i}>
                    <a
                      href={url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-xs text-primary hover:underline truncate block"
                    >
                      {url}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {step.details?.observation && (
            <div className="mt-3">
              <p className="text-xs font-medium text-muted-foreground mb-1.5 flex items-center gap-1">
                <NotesIcon className="w-3 h-3" />
                Observation
              </p>
              <p className="text-xs text-muted-foreground bg-muted/50 p-2 rounded">
                {step.details.observation}
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

interface StepIconProps {
  type: StepType;
  status: ReasoningStepData['status'];
}

function StepIcon({ type, status }: StepIconProps) {
  const iconClass = cn(
    'w-5 h-5',
    status === 'in_progress' && 'text-primary animate-pulse',
    status === 'completed' && 'text-green-500',
    status === 'failed' && 'text-red-500',
    status === 'pending' && 'text-muted-foreground'
  );

  if (status === 'in_progress') {
    return <SpinnerIcon className={iconClass} />;
  }

  switch (type) {
    case 'search':
      return <SearchIcon className={iconClass} />;
    case 'fetch':
      return <LinkIcon className={iconClass} />;
    case 'analyze':
      return <AnalyzeIcon className={iconClass} />;
    case 'reflect':
      return <ReflectIcon className={iconClass} />;
    default:
      return <CircleIcon className={iconClass} />;
  }
}

// Icons

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

function SearchIcon({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <circle cx="11" cy="11" r="8" />
      <path d="m21 21-4.3-4.3" />
    </svg>
  );
}

function LinkIcon({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71" />
      <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71" />
    </svg>
  );
}

function AnalyzeIcon({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
      <polyline points="3.27 6.96 12 12.01 20.73 6.96" />
      <line x1="12" x2="12" y1="22.08" y2="12" />
    </svg>
  );
}

function ReflectIcon({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <circle cx="12" cy="12" r="10" />
      <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" />
      <path d="M12 17h.01" />
    </svg>
  );
}

function NotesIcon({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <path d="M16 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V8Z" />
      <path d="M15 3v4a2 2 0 0 0 2 2h4" />
      <path d="M10 12h4" />
      <path d="M10 16h4" />
    </svg>
  );
}

function CircleIcon({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <circle cx="12" cy="12" r="10" />
    </svg>
  );
}

function ChevronIcon({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <path d="m6 9 6 6 6-6" />
    </svg>
  );
}
