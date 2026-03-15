import { useState, useCallback } from 'react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import type { UploadedFile, FileType, ProcessingStatus } from '@/types/files';
import {
  formatFileSize,
  FILE_TYPE_LABELS,
  PROCESSING_STATUS_LABELS,
  getProcessingStatusColor,
} from '@/types/files';

interface UploadedFileListProps {
  /** List of uploaded files */
  files: UploadedFile[];
  /** Callback when preview is requested */
  onPreview?: (file: UploadedFile) => void;
  /** Callback when delete is requested */
  onDelete?: (file: UploadedFile) => void;
  /** Whether the list is loading */
  isLoading?: boolean;
  /** Show compact chip/pill layout instead of full cards */
  compact?: boolean;
  /** Custom className */
  className?: string;
}

/**
 * List of uploaded files with status indicators and actions.
 *
 * Features:
 * - File cards with icon, name, size, status
 * - Processing status indicator
 * - Preview and delete buttons
 * - Delete confirmation
 * - Loading state
 */
export function UploadedFileList({
  files,
  onPreview,
  onDelete,
  isLoading = false,
  compact = false,
  className,
}: UploadedFileListProps) {
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);

  const handleDeleteClick = useCallback((file: UploadedFile) => {
    setDeleteConfirmId(file.id);
  }, []);

  const handleDeleteConfirm = useCallback(
    (file: UploadedFile) => {
      onDelete?.(file);
      setDeleteConfirmId(null);
    },
    [onDelete]
  );

  const handleDeleteCancel = useCallback(() => {
    setDeleteConfirmId(null);
  }, []);

  if (isLoading) {
    return (
      <div className={cn('space-y-3', className)}>
        {[1, 2, 3].map((i) => (
          <Card key={i} className="animate-pulse">
            <CardContent className="p-4">
              <div className="flex items-center gap-3">
                <div className="h-10 w-10 rounded-lg bg-muted" />
                <div className="flex-1 space-y-2">
                  <div className="h-4 w-1/3 rounded bg-muted" />
                  <div className="h-3 w-1/4 rounded bg-muted" />
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    );
  }

  if (files.length === 0) {
    if (compact) return null;
    return (
      <div
        className={cn(
          'flex flex-col items-center justify-center rounded-lg border border-dashed p-8 text-center',
          className
        )}
      >
        <EmptyFilesIcon className="mb-3 h-10 w-10 text-muted-foreground/50" />
        <p className="text-sm text-muted-foreground">No files uploaded yet</p>
        <p className="mt-1 text-xs text-muted-foreground/70">
          Upload files above to use them in your research
        </p>
      </div>
    );
  }

  if (compact) {
    return (
      <div className={cn('flex flex-wrap gap-1.5', className)}>
        {files.map((file) => (
          <CompactFileChip
            key={file.id}
            file={file}
            onDelete={onDelete ? () => onDelete(file) : undefined}
          />
        ))}
      </div>
    );
  }

  return (
    <div className={cn('space-y-2', className)}>
      <div className="flex items-center justify-between">
        <p className="text-xs font-medium text-muted-foreground">
          {files.length} file{files.length !== 1 ? 's' : ''} uploaded
        </p>
      </div>

      <div className="space-y-2">
        {files.map((file) => (
          <FileCard
            key={file.id}
            file={file}
            onPreview={onPreview}
            onDelete={onDelete ? handleDeleteClick : undefined}
            isDeleteConfirming={deleteConfirmId === file.id}
            onDeleteConfirm={handleDeleteConfirm}
            onDeleteCancel={handleDeleteCancel}
          />
        ))}
      </div>
    </div>
  );
}

interface FileCardProps {
  file: UploadedFile;
  onPreview?: (file: UploadedFile) => void;
  onDelete?: (file: UploadedFile) => void;
  isDeleteConfirming?: boolean;
  onDeleteConfirm?: (file: UploadedFile) => void;
  onDeleteCancel?: () => void;
}

/**
 * Individual file card component.
 */
function FileCard({
  file,
  onPreview,
  onDelete,
  isDeleteConfirming,
  onDeleteConfirm,
  onDeleteCancel,
}: FileCardProps) {
  const isReady = file.processingStatus === 'ready';
  const isFailed = file.processingStatus === 'failed';
  const isProcessing =
    file.processingStatus === 'pending' || file.processingStatus === 'processing';

  return (
    <Card
      className={cn(
        'transition-shadow hover:shadow-sm',
        isFailed && 'border-destructive/50 bg-destructive/5'
      )}
    >
      <CardContent className="p-3">
        <div className="flex items-start gap-3">
          {/* File Type Icon */}
          <div
            className={cn(
              'flex h-10 w-10 shrink-0 items-center justify-center rounded-lg',
              getFileTypeBackground(file.fileType)
            )}
          >
            <FileTypeIcon type={file.fileType} className="h-5 w-5" />
          </div>

          {/* File Info */}
          <div className="min-w-0 flex-1">
            <div className="flex items-start justify-between gap-2">
              <div className="min-w-0 flex-1">
                <p className="truncate text-sm font-medium">{file.filename}</p>
                <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                  <span>{FILE_TYPE_LABELS[file.fileType]}</span>
                  <span>-</span>
                  <span>{formatFileSize(file.fileSize)}</span>
                  {file.chunkCount > 0 && isReady && (
                    <>
                      <span>-</span>
                      <span>
                        {file.chunkCount} chunk{file.chunkCount !== 1 ? 's' : ''}
                      </span>
                    </>
                  )}
                </div>
              </div>

              {/* Status Badge */}
              <ProcessingStatusBadge status={file.processingStatus} />
            </div>

            {/* Error Message */}
            {isFailed && !!file.metadata?.error && (
              <p className="mt-2 text-xs text-destructive">{String(file.metadata.error)}</p>
            )}

            {/* Actions */}
            {!isDeleteConfirming && (
              <div className="mt-2 flex items-center gap-2">
                {onPreview && isReady && (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => onPreview(file)}
                    className="h-7 px-2 text-xs"
                  >
                    <PreviewIcon className="mr-1 h-3.5 w-3.5" />
                    Preview
                  </Button>
                )}
                {onDelete && (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => onDelete(file)}
                    className="h-7 px-2 text-xs text-destructive hover:text-destructive"
                    disabled={isProcessing}
                  >
                    <TrashIcon className="mr-1 h-3.5 w-3.5" />
                    Remove
                  </Button>
                )}
              </div>
            )}

            {/* Delete Confirmation */}
            {isDeleteConfirming && (
              <div className="mt-2 flex items-center gap-2 rounded-md bg-destructive/10 p-2">
                <p className="flex-1 text-xs text-destructive">Delete this file?</p>
                <Button
                  variant="destructive"
                  size="sm"
                  onClick={() => onDeleteConfirm?.(file)}
                  className="h-6 px-2 text-xs"
                >
                  Delete
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={onDeleteCancel}
                  className="h-6 px-2 text-xs"
                >
                  Cancel
                </Button>
              </div>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

/**
 * Compact file chip for inline display.
 */
function CompactFileChip({
  file,
  onDelete,
}: {
  file: UploadedFile;
  onDelete?: () => void;
}) {
  const isReady = file.processingStatus === 'ready';
  const isFailed = file.processingStatus === 'failed';
  const isProcessing =
    file.processingStatus === 'pending' || file.processingStatus === 'processing';

  const borderColor = isFailed
    ? 'border-red-400 dark:border-red-600'
    : isProcessing
      ? 'border-blue-400 dark:border-blue-600'
      : isReady
        ? 'border-green-400 dark:border-green-600'
        : 'border-border';

  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs',
        borderColor
      )}
    >
      {isProcessing && <LoadingSpinner className="h-3 w-3" />}
      {isReady && <MiniCheckIcon className="h-3 w-3 text-green-600 dark:text-green-400" />}
      {isFailed && <MiniErrorIcon className="h-3 w-3 text-red-600 dark:text-red-400" />}
      <span className="max-w-[120px] truncate">{file.filename}</span>
      <span className="text-muted-foreground">{formatFileSize(file.fileSize)}</span>
      {onDelete && (
        <button
          type="button"
          onClick={onDelete}
          className="ml-0.5 rounded-full p-0.5 text-muted-foreground hover:text-foreground hover:bg-accent transition-colors"
          aria-label={`Remove ${file.filename}`}
        >
          <MiniXIcon className="h-3 w-3" />
        </button>
      )}
    </span>
  );
}

/**
 * Processing status badge component.
 */
function ProcessingStatusBadge({ status }: { status: ProcessingStatus }) {
  const color = getProcessingStatusColor(status);
  const label = PROCESSING_STATUS_LABELS[status];

  const colorClasses: Record<string, string> = {
    green: 'bg-green-100 text-green-800 dark:bg-green-900/50 dark:text-green-400',
    yellow: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/50 dark:text-yellow-400',
    blue: 'bg-blue-100 text-blue-800 dark:bg-blue-900/50 dark:text-blue-400',
    red: 'bg-red-100 text-red-800 dark:bg-red-900/50 dark:text-red-400',
  };

  return (
    <Badge variant="secondary" className={cn('text-xs font-normal', colorClasses[color])}>
      {(status === 'pending' || status === 'processing') && (
        <LoadingSpinner className="mr-1 h-3 w-3" />
      )}
      {status === 'ready' && <CheckIcon className="mr-1 h-3 w-3" />}
      {status === 'failed' && <ErrorIcon className="mr-1 h-3 w-3" />}
      {label}
    </Badge>
  );
}

/**
 * Get background color class for file type.
 */
function getFileTypeBackground(type: FileType): string {
  const backgrounds: Record<FileType, string> = {
    pdf: 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400',
    txt: 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-400',
    md: 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400',
    docx: 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400',
  };
  return backgrounds[type] || 'bg-muted text-muted-foreground';
}

/**
 * File type icon component.
 */
function FileTypeIcon({ type, className }: { type: FileType; className?: string }) {
  switch (type) {
    case 'pdf':
      return (
        <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z" />
          <polyline points="14 2 14 8 20 8" />
          <path d="M10 12h4" />
          <path d="M10 16h4" />
        </svg>
      );
    case 'txt':
      return (
        <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z" />
          <polyline points="14 2 14 8 20 8" />
          <line x1="8" x2="16" y1="13" y2="13" />
          <line x1="8" x2="14" y1="17" y2="17" />
        </svg>
      );
    case 'md':
      return (
        <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z" />
          <polyline points="14 2 14 8 20 8" />
          <text x="8" y="16" fontSize="6" fontWeight="bold" fill="currentColor" stroke="none">
            M
          </text>
        </svg>
      );
    case 'docx':
      return (
        <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z" />
          <polyline points="14 2 14 8 20 8" />
          <text x="7" y="16" fontSize="5" fontWeight="bold" fill="currentColor" stroke="none">
            W
          </text>
        </svg>
      );
    default:
      return (
        <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z" />
          <polyline points="14 2 14 8 20 8" />
        </svg>
      );
  }
}

/**
 * Empty files icon component.
 */
function EmptyFilesIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z" />
      <polyline points="14 2 14 8 20 8" />
      <path d="M12 18v-6" />
      <path d="M9 15l3-3 3 3" />
    </svg>
  );
}

/**
 * Preview icon component.
 */
function PreviewIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
      <circle cx="12" cy="12" r="3" />
    </svg>
  );
}

/**
 * Trash icon component.
 */
function TrashIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M3 6h18" />
      <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6" />
      <path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2" />
    </svg>
  );
}

/**
 * Loading spinner component.
 */
function LoadingSpinner({ className }: { className?: string }) {
  return (
    <svg
      className={cn('animate-spin', className)}
      xmlns="http://www.w3.org/2000/svg"
      fill="none"
      viewBox="0 0 24 24"
    >
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path
        className="opacity-75"
        fill="currentColor"
        d="m4 12a8 8 0 0 1 8-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 0 1 4 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
      />
    </svg>
  );
}

/**
 * Check icon component.
 */
function CheckIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <polyline points="20 6 9 17 4 12" />
    </svg>
  );
}

/**
 * Error icon component.
 */
function ErrorIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <circle cx="12" cy="12" r="10" />
      <line x1="15" x2="9" y1="9" y2="15" />
      <line x1="9" x2="15" y1="9" y2="15" />
    </svg>
  );
}

/**
 * Mini check icon for compact chips.
 */
function MiniCheckIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.5"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <polyline points="20 6 9 17 4 12" />
    </svg>
  );
}

/**
 * Mini error icon for compact chips.
 */
function MiniErrorIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.5"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <circle cx="12" cy="12" r="10" />
      <line x1="12" x2="12" y1="8" y2="12" />
      <line x1="12" x2="12.01" y1="16" y2="16" />
    </svg>
  );
}

/**
 * Mini X icon for compact chip remove button.
 */
function MiniXIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.5"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <line x1="18" x2="6" y1="6" y2="18" />
      <line x1="6" x2="18" y1="6" y2="18" />
    </svg>
  );
}
