/**
 * MessageExportMenu - Dropdown menu for exporting agent messages
 *
 * Provides options to:
 * - Export Report: Download synthesis as markdown
 * - Verification Report: Download claims + verdicts as markdown (if claims exist)
 * - Copy to Clipboard: Copy report markdown to clipboard
 */

import * as React from 'react';
import { messagesApi } from '@/api/client';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

interface MessageExportMenuProps {
  messageId: string;
  hasClaims: boolean;
  className?: string;
}

type LoadingState = 'report' | 'verification' | 'copy' | null;

/**
 * Download a string as a file
 */
function downloadMarkdown(content: string, filename: string): void {
  const blob = new Blob([content], { type: 'text/markdown' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/**
 * Show a toast notification (simple implementation)
 */
function showToast(message: string, type: 'success' | 'error' = 'success'): void {
  // Create toast element
  const toast = document.createElement('div');
  toast.className = cn(
    'fixed bottom-4 right-4 px-4 py-2 rounded-lg shadow-lg z-50 text-sm font-medium',
    'animate-in slide-in-from-bottom-2 fade-in duration-200',
    type === 'success'
      ? 'bg-green-600 text-white'
      : 'bg-red-600 text-white'
  );
  toast.textContent = message;
  document.body.appendChild(toast);

  // Remove after 3 seconds
  setTimeout(() => {
    toast.classList.add('animate-out', 'fade-out', 'slide-out-to-bottom-2');
    setTimeout(() => {
      document.body.removeChild(toast);
    }, 200);
  }, 3000);
}

export function MessageExportMenu({
  messageId,
  hasClaims,
  className,
}: MessageExportMenuProps) {
  const [isOpen, setIsOpen] = React.useState(false);
  const [isLoading, setIsLoading] = React.useState<LoadingState>(null);
  const menuRef = React.useRef<HTMLDivElement>(null);

  // Close menu when clicking outside
  React.useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setIsOpen(false);
      }
    };

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
      return () => document.removeEventListener('mousedown', handleClickOutside);
    }
  }, [isOpen]);

  // Close menu on escape key
  React.useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        setIsOpen(false);
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [isOpen]);

  const handleExportReport = async () => {
    setIsLoading('report');
    try {
      const { content, filename } = await messagesApi.exportReport(messageId);
      downloadMarkdown(content, filename);
      showToast('Report downloaded');
      setIsOpen(false);
    } catch (error) {
      console.error('Export report failed:', error);
      showToast('Failed to export report', 'error');
    } finally {
      setIsLoading(null);
    }
  };

  const handleExportVerification = async () => {
    setIsLoading('verification');
    try {
      const { content, filename } = await messagesApi.exportProvenance(messageId);
      downloadMarkdown(content, filename);
      showToast('Verification report downloaded');
      setIsOpen(false);
    } catch (error) {
      console.error('Export verification failed:', error);
      showToast('Failed to export verification report', 'error');
    } finally {
      setIsLoading(null);
    }
  };

  const handleCopyToClipboard = async () => {
    setIsLoading('copy');
    try {
      const { content } = await messagesApi.exportReport(messageId);
      await navigator.clipboard.writeText(content);
      showToast('Copied to clipboard');
      setIsOpen(false);
    } catch (error) {
      console.error('Copy to clipboard failed:', error);
      showToast('Failed to copy to clipboard', 'error');
    } finally {
      setIsLoading(null);
    }
  };

  return (
    <div ref={menuRef} className={cn('relative', className)}>
      <Button
        variant="ghost"
        size="icon"
        className="h-8 w-8 text-muted-foreground hover:text-foreground"
        onClick={() => setIsOpen(!isOpen)}
        aria-haspopup="true"
        aria-expanded={isOpen}
        aria-label="Export options"
        data-testid="message-export-menu-trigger"
      >
        <MoreVerticalIcon className="h-4 w-4" />
      </Button>

      {isOpen && (
        <div
          className={cn(
            'absolute right-0 top-full mt-1 z-50',
            'w-48 rounded-md bg-popover border shadow-lg',
            'animate-in fade-in-0 zoom-in-95 duration-100'
          )}
          role="menu"
          aria-orientation="vertical"
        >
          {/* Export Report */}
          <button
            className={cn(
              'w-full flex items-center gap-2 px-3 py-2 text-sm text-left',
              'hover:bg-accent hover:text-accent-foreground',
              'focus:bg-accent focus:text-accent-foreground focus:outline-none',
              'disabled:opacity-50 disabled:cursor-not-allowed',
              'first:rounded-t-md'
            )}
            onClick={handleExportReport}
            disabled={isLoading !== null}
            role="menuitem"
            data-testid="export-report-button"
          >
            {isLoading === 'report' ? (
              <SpinnerIcon className="h-4 w-4 animate-spin" />
            ) : (
              <FileTextIcon className="h-4 w-4" />
            )}
            <span>Export Report</span>
          </button>

          {/* Verification Report - only if claims exist */}
          {hasClaims && (
            <button
              className={cn(
                'w-full flex items-center gap-2 px-3 py-2 text-sm text-left',
                'hover:bg-accent hover:text-accent-foreground',
                'focus:bg-accent focus:text-accent-foreground focus:outline-none',
                'disabled:opacity-50 disabled:cursor-not-allowed'
              )}
              onClick={handleExportVerification}
              disabled={isLoading !== null}
              role="menuitem"
              data-testid="export-verification-button"
            >
              {isLoading === 'verification' ? (
                <SpinnerIcon className="h-4 w-4 animate-spin" />
              ) : (
                <CheckCircleIcon className="h-4 w-4" />
              )}
              <span>Verification Report</span>
            </button>
          )}

          {/* Separator */}
          <div className="h-px bg-border my-1" role="separator" />

          {/* Copy to Clipboard */}
          <button
            className={cn(
              'w-full flex items-center gap-2 px-3 py-2 text-sm text-left',
              'hover:bg-accent hover:text-accent-foreground',
              'focus:bg-accent focus:text-accent-foreground focus:outline-none',
              'disabled:opacity-50 disabled:cursor-not-allowed',
              'last:rounded-b-md'
            )}
            onClick={handleCopyToClipboard}
            disabled={isLoading !== null}
            role="menuitem"
            data-testid="copy-to-clipboard-button"
          >
            {isLoading === 'copy' ? (
              <SpinnerIcon className="h-4 w-4 animate-spin" />
            ) : (
              <ClipboardIcon className="h-4 w-4" />
            )}
            <span>Copy to Clipboard</span>
          </button>
        </div>
      )}
    </div>
  );
}

// Icon components
function MoreVerticalIcon({ className }: { className?: string }) {
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
      <circle cx="12" cy="5" r="1" />
      <circle cx="12" cy="19" r="1" />
    </svg>
  );
}

function FileTextIcon({ className }: { className?: string }) {
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
      <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z" />
      <polyline points="14 2 14 8 20 8" />
      <line x1="16" x2="8" y1="13" y2="13" />
      <line x1="16" x2="8" y1="17" y2="17" />
      <line x1="10" x2="8" y1="9" y2="9" />
    </svg>
  );
}

function CheckCircleIcon({ className }: { className?: string }) {
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
      <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
      <polyline points="22 4 12 14.01 9 11.01" />
    </svg>
  );
}

function ClipboardIcon({ className }: { className?: string }) {
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
      <rect width="8" height="4" x="8" y="2" rx="1" ry="1" />
      <path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2" />
    </svg>
  );
}

function SpinnerIcon({ className }: { className?: string }) {
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
      <path d="M21 12a9 9 0 1 1-6.219-8.56" />
    </svg>
  );
}

export default MessageExportMenu;
