import * as React from 'react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

export type ExportFormat = 'markdown' | 'json';

interface ExportChatDialogProps {
  isOpen: boolean;
  chatTitle: string | null;
  onClose: () => void;
  onExport: (format: ExportFormat, includeMetadata: boolean) => void;
  isExporting?: boolean;
}

export function ExportChatDialog({
  isOpen,
  chatTitle,
  onClose,
  onExport,
  isExporting = false,
}: ExportChatDialogProps) {
  const [format, setFormat] = React.useState<ExportFormat>('markdown');
  const [includeMetadata, setIncludeMetadata] = React.useState(true);
  const dialogRef = React.useRef<HTMLDivElement>(null);

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

  const handleExport = () => {
    onExport(format, includeMetadata);
  };

  if (!isOpen) return null;

  const formatOptions: { value: ExportFormat; label: string; description: string }[] = [
    {
      value: 'markdown',
      label: 'Markdown',
      description: 'Best for reading and sharing. Includes formatting.',
    },
    {
      value: 'json',
      label: 'JSON',
      description: 'Complete data export. Best for programmatic use.',
    },
  ];

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div className="fixed inset-0 bg-black/50" aria-hidden="true" />

      {/* Dialog */}
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="export-dialog-title"
        className={cn(
          'relative z-50 w-full max-w-md rounded-lg bg-background p-6 shadow-lg',
          'animate-in fade-in-0 zoom-in-95'
        )}
      >
        <h3 id="export-dialog-title" className="text-lg font-semibold mb-2">
          Export Chat
        </h3>

        <p className="text-sm text-muted-foreground mb-4">
          Export "{chatTitle || 'Untitled Chat'}" to a file.
        </p>

        {/* Format selection */}
        <div className="space-y-3 mb-4">
          <label className="text-sm font-medium">Format</label>
          <div className="space-y-2">
            {formatOptions.map((option) => (
              <label
                key={option.value}
                className={cn(
                  'flex items-start gap-3 p-3 rounded-lg border cursor-pointer transition-colors',
                  format === option.value
                    ? 'border-primary bg-primary/5'
                    : 'border-border hover:border-primary/50'
                )}
              >
                <input
                  type="radio"
                  name="format"
                  value={option.value}
                  checked={format === option.value}
                  onChange={() => setFormat(option.value)}
                  className="mt-0.5"
                  disabled={isExporting}
                />
                <div>
                  <span className="font-medium text-sm">{option.label}</span>
                  <p className="text-xs text-muted-foreground mt-0.5">
                    {option.description}
                  </p>
                </div>
              </label>
            ))}
          </div>
        </div>

        {/* Include metadata checkbox */}
        <label className="flex items-center gap-2 mb-6 cursor-pointer">
          <input
            type="checkbox"
            checked={includeMetadata}
            onChange={(e) => setIncludeMetadata(e.target.checked)}
            disabled={isExporting}
            className="rounded border-input"
          />
          <span className="text-sm">Include metadata (timestamps, sources)</span>
        </label>

        <div className="flex justify-end gap-3">
          <Button variant="outline" onClick={onClose} disabled={isExporting}>
            Cancel
          </Button>
          <Button onClick={handleExport} disabled={isExporting}>
            {isExporting ? (
              <>
                <SpinnerIcon className="w-4 h-4 mr-2 animate-spin" />
                Exporting...
              </>
            ) : (
              <>
                <DownloadIcon className="w-4 h-4 mr-2" />
                Export
              </>
            )}
          </Button>
        </div>
      </div>
    </div>
  );
}

function DownloadIcon({ className }: { className?: string }) {
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
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="7 10 12 15 17 10" />
      <line x1="12" x2="12" y1="15" y2="3" />
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
