import * as React from 'react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

interface MessageActionsProps {
  onEdit?: () => void;
  onCopy?: () => void;
  onRegenerate?: () => void;
  onDelete?: () => void;
  showEdit?: boolean;
  showRegenerate?: boolean;
  showDelete?: boolean;
  className?: string;
}

export function MessageActions({
  onEdit,
  onCopy,
  onRegenerate,
  onDelete,
  showEdit = true,
  showRegenerate = false,
  showDelete = false,
  className,
}: MessageActionsProps) {
  const [isOpen, setIsOpen] = React.useState(false);
  const [copied, setCopied] = React.useState(false);
  const menuRef = React.useRef<HTMLDivElement>(null);

  // Close menu on click outside
  React.useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setIsOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleCopy = () => {
    onCopy?.();
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
    setIsOpen(false);
  };

  const handleAction = (action: () => void) => {
    action();
    setIsOpen(false);
  };

  return (
    <div ref={menuRef} className={cn('relative inline-flex', className)}>
      <Button
        variant="ghost"
        size="sm"
        onClick={() => setIsOpen(!isOpen)}
        className="h-8 w-8 p-0 hover:bg-muted"
        aria-label="Message actions"
      >
        <MoreIcon className="h-4 w-4" />
      </Button>

      {isOpen && (
        <div
          className={cn(
            'absolute right-0 top-full mt-1 z-50',
            'min-w-[160px] rounded-md border bg-popover p-1 shadow-md',
            'animate-in fade-in-0 zoom-in-95'
          )}
        >
          {onCopy && (
            <button
              onClick={handleCopy}
              className={cn(
                'w-full flex items-center gap-2 px-2 py-1.5 text-sm rounded-sm',
                'hover:bg-accent hover:text-accent-foreground',
                'focus:bg-accent focus:text-accent-foreground focus:outline-none'
              )}
            >
              {copied ? (
                <>
                  <CheckIcon className="h-4 w-4 text-green-500" />
                  Copied!
                </>
              ) : (
                <>
                  <CopyIcon className="h-4 w-4" />
                  Copy
                </>
              )}
            </button>
          )}

          {showEdit && onEdit && (
            <button
              onClick={() => handleAction(onEdit)}
              className={cn(
                'w-full flex items-center gap-2 px-2 py-1.5 text-sm rounded-sm',
                'hover:bg-accent hover:text-accent-foreground',
                'focus:bg-accent focus:text-accent-foreground focus:outline-none'
              )}
            >
              <EditIcon className="h-4 w-4" />
              Edit
            </button>
          )}

          {showRegenerate && onRegenerate && (
            <button
              data-testid="regenerate-response"
              onClick={() => handleAction(onRegenerate)}
              className={cn(
                'w-full flex items-center gap-2 px-2 py-1.5 text-sm rounded-sm',
                'hover:bg-accent hover:text-accent-foreground',
                'focus:bg-accent focus:text-accent-foreground focus:outline-none'
              )}
            >
              <RefreshIcon className="h-4 w-4" />
              Regenerate
            </button>
          )}

          {showDelete && onDelete && (
            <>
              <div className="my-1 h-px bg-border" />
              <button
                onClick={() => handleAction(onDelete)}
                className={cn(
                  'w-full flex items-center gap-2 px-2 py-1.5 text-sm rounded-sm',
                  'text-destructive hover:bg-destructive/10',
                  'focus:bg-destructive/10 focus:outline-none'
                )}
              >
                <TrashIcon className="h-4 w-4" />
                Delete
              </button>
            </>
          )}
        </div>
      )}
    </div>
  );
}

// Icons
function MoreIcon({ className }: { className?: string }) {
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
      <circle cx="12" cy="12" r="1" />
      <circle cx="19" cy="12" r="1" />
      <circle cx="5" cy="12" r="1" />
    </svg>
  );
}

function CopyIcon({ className }: { className?: string }) {
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
      <rect width="14" height="14" x="8" y="8" rx="2" ry="2" />
      <path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2" />
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
      <polyline points="20 6 9 17 4 12" />
    </svg>
  );
}

function EditIcon({ className }: { className?: string }) {
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
      <path d="M17 3a2.85 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z" />
      <path d="m15 5 4 4" />
    </svg>
  );
}

function RefreshIcon({ className }: { className?: string }) {
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
      <path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
      <path d="M3 3v5h5" />
      <path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16" />
      <path d="M16 16h5v5" />
    </svg>
  );
}

function TrashIcon({ className }: { className?: string }) {
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
      <path d="M3 6h18" />
      <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6" />
      <path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2" />
    </svg>
  );
}
