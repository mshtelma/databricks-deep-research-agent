import { cn } from '@/lib/utils';

type AgentStatus =
  | 'idle'
  | 'classifying'
  | 'planning'
  | 'researching'
  | 'reflecting'
  | 'synthesizing'
  | 'complete'
  | 'error';

interface AgentStatusIndicatorProps {
  status: AgentStatus;
  currentAgent?: string;
  className?: string;
}

export function AgentStatusIndicator({
  status,
  currentAgent,
  className,
}: AgentStatusIndicatorProps) {
  const statusInfo = getStatusInfo(status, currentAgent);

  if (status === 'idle') {
    return null;
  }

  return (
    <div
      data-testid="research-status"
      className={cn(
        'flex items-center gap-2 px-3 py-1.5 rounded-full text-sm',
        statusInfo.bgClass,
        className
      )}
    >
      {statusInfo.icon}
      <span className={statusInfo.textClass}>{statusInfo.label}</span>
    </div>
  );
}

function getStatusInfo(status: AgentStatus, currentAgent?: string) {
  switch (status) {
    case 'classifying':
      return {
        label: 'Analyzing query...',
        icon: <SpinnerIcon className="w-4 h-4 animate-spin text-blue-500" />,
        bgClass: 'bg-blue-50 dark:bg-blue-950',
        textClass: 'text-blue-600 dark:text-blue-400',
      };
    case 'planning':
      return {
        label: 'Creating research plan...',
        icon: <SpinnerIcon className="w-4 h-4 animate-spin text-purple-500" />,
        bgClass: 'bg-purple-50 dark:bg-purple-950',
        textClass: 'text-purple-600 dark:text-purple-400',
      };
    case 'researching':
      return {
        label: currentAgent || 'Researching...',
        icon: <SearchIcon className="w-4 h-4 text-amber-500 animate-pulse" />,
        bgClass: 'bg-amber-50 dark:bg-amber-950',
        textClass: 'text-amber-600 dark:text-amber-400',
      };
    case 'reflecting':
      return {
        label: 'Evaluating progress...',
        icon: <SpinnerIcon className="w-4 h-4 animate-spin text-cyan-500" />,
        bgClass: 'bg-cyan-50 dark:bg-cyan-950',
        textClass: 'text-cyan-600 dark:text-cyan-400',
      };
    case 'synthesizing':
      return {
        label: 'Writing report...',
        icon: <PenIcon className="w-4 h-4 text-green-500 animate-pulse" />,
        bgClass: 'bg-green-50 dark:bg-green-950',
        textClass: 'text-green-600 dark:text-green-400',
      };
    case 'complete':
      return {
        label: 'Complete',
        icon: <CheckIcon className="w-4 h-4 text-green-500" />,
        bgClass: 'bg-green-50 dark:bg-green-950',
        textClass: 'text-green-600 dark:text-green-400',
      };
    case 'error':
      return {
        label: 'Error occurred',
        icon: <ErrorIcon className="w-4 h-4 text-red-500" />,
        bgClass: 'bg-red-50 dark:bg-red-950',
        textClass: 'text-red-600 dark:text-red-400',
      };
    default:
      return {
        label: 'Processing...',
        icon: <SpinnerIcon className="w-4 h-4 animate-spin text-gray-500" />,
        bgClass: 'bg-gray-50 dark:bg-gray-900',
        textClass: 'text-gray-600 dark:text-gray-400',
      };
  }
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

function PenIcon({ className }: { className?: string }) {
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
      <path d="M12 20h9" />
      <path d="M16.5 3.5a2.121 2.121 0 1 1 3 3L7 19l-4 1 1-4Z" />
    </svg>
  );
}

function CheckIcon({ className }: { className?: string }) {
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
      <path d="M20 6 9 17l-5-5" />
    </svg>
  );
}

function ErrorIcon({ className }: { className?: string }) {
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
      <path d="m15 9-6 6" />
      <path d="m9 9 6 6" />
    </svg>
  );
}
