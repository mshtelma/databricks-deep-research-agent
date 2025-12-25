import * as React from 'react';
import { cn } from '@/lib/utils';

interface CopyButtonProps {
  text: string;
  onCopy?: () => void;
  className?: string;
  variant?: 'icon' | 'text';
  label?: string;
}

export function CopyButton({
  text,
  onCopy,
  className,
  variant = 'icon',
  label = 'Copy',
}: CopyButtonProps) {
  const [copied, setCopied] = React.useState(false);
  const timeoutRef = React.useRef<NodeJS.Timeout | null>(null);

  // Clear timeout on unmount
  React.useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      onCopy?.();

      // Reset copied state after 2 seconds
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
      timeoutRef.current = setTimeout(() => {
        setCopied(false);
      }, 2000);
    } catch (err) {
      console.error('Failed to copy to clipboard:', err);
    }
  };

  if (variant === 'text') {
    return (
      <button
        type="button"
        onClick={handleCopy}
        className={cn(
          'inline-flex items-center gap-1 px-2 py-1 text-xs rounded transition-colors',
          'text-muted-foreground hover:text-foreground hover:bg-muted',
          'focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring',
          className
        )}
        aria-label={copied ? 'Copied!' : label}
      >
        {copied ? (
          <>
            <CheckIcon className="w-3.5 h-3.5 text-green-500" />
            <span className="text-green-600">Copied!</span>
          </>
        ) : (
          <>
            <CopyIcon className="w-3.5 h-3.5" />
            <span>{label}</span>
          </>
        )}
      </button>
    );
  }

  // Icon variant (default)
  return (
    <button
      type="button"
      onClick={handleCopy}
      className={cn(
        'p-1.5 rounded transition-colors',
        'text-muted-foreground hover:text-foreground hover:bg-muted',
        'focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring',
        copied && 'text-green-500',
        className
      )}
      aria-label={copied ? 'Copied!' : label}
    >
      {copied ? (
        <CheckIcon className="w-4 h-4" />
      ) : (
        <CopyIcon className="w-4 h-4" />
      )}
    </button>
  );
}

// Icons

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
