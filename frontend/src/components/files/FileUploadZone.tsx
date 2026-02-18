import * as React from 'react';
import { useCallback, useState, useRef } from 'react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import type { FileType, UploadProgress } from '@/types/files';
import {
  ACCEPTED_FILE_EXTENSIONS,
  DEFAULT_MAX_FILE_SIZE,
  DEFAULT_MAX_FILES,
  validateFile,
  formatFileSize,
  FILE_TYPE_LABELS,
} from '@/types/files';

interface FileUploadZoneProps {
  /** Callback when files are selected */
  onFilesSelected: (files: File[]) => void;
  /** Maximum number of files allowed */
  maxFiles?: number;
  /** Maximum file size in bytes */
  maxSizeBytes?: number;
  /** Accepted file types */
  acceptedTypes?: FileType[];
  /** Whether upload is in progress */
  isUploading?: boolean;
  /** Current upload progress for files */
  uploadProgress?: UploadProgress[];
  /** Whether the zone is disabled */
  disabled?: boolean;
  /** Callback to close the upload zone */
  onClose?: () => void;
  /** Custom className */
  className?: string;
}

interface FileError {
  filename: string;
  error: string;
}

/**
 * Drag-and-drop file upload zone with visual feedback.
 *
 * Supports:
 * - Drag and drop files
 * - Click to browse
 * - File type validation
 * - File size validation
 * - Upload progress indicators
 * - Error states
 */
export function FileUploadZone({
  onFilesSelected,
  maxFiles = DEFAULT_MAX_FILES,
  maxSizeBytes = DEFAULT_MAX_FILE_SIZE,
  acceptedTypes,
  isUploading = false,
  uploadProgress = [],
  disabled = false,
  onClose,
  className,
}: FileUploadZoneProps) {
  const [isDragOver, setIsDragOver] = useState(false);
  const [fileErrors, setFileErrors] = useState<FileError[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const acceptedExtensions = acceptedTypes
    ? acceptedTypes.map((t) => `.${t}`).join(',')
    : ACCEPTED_FILE_EXTENSIONS;

  const handleDragEnter = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      if (!disabled && !isUploading) {
        setIsDragOver(true);
      }
    },
    [disabled, isUploading]
  );

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const validateAndFilterFiles = useCallback(
    (files: FileList | File[]): { valid: File[]; errors: FileError[] } => {
      const fileArray = Array.from(files);
      const valid: File[] = [];
      const errors: FileError[] = [];

      // Check max files limit
      if (fileArray.length > maxFiles) {
        errors.push({
          filename: 'Multiple files',
          error: `Maximum ${maxFiles} files allowed`,
        });
        return { valid: [], errors };
      }

      for (const file of fileArray) {
        const validation = validateFile(file, maxSizeBytes);
        if (validation.valid) {
          valid.push(file);
        } else {
          errors.push({
            filename: file.name,
            error: validation.error || 'Invalid file',
          });
        }
      }

      return { valid, errors };
    },
    [maxFiles, maxSizeBytes]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragOver(false);

      if (disabled || isUploading) return;

      const { files } = e.dataTransfer;
      if (files.length === 0) return;

      const { valid, errors } = validateAndFilterFiles(files);
      setFileErrors(errors);

      if (valid.length > 0) {
        onFilesSelected(valid);
      }
    },
    [disabled, isUploading, validateAndFilterFiles, onFilesSelected]
  );

  const handleFileInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const { files } = e.target;
      if (!files || files.length === 0) return;

      const { valid, errors } = validateAndFilterFiles(files);
      setFileErrors(errors);

      if (valid.length > 0) {
        onFilesSelected(valid);
      }

      // Reset input so same file can be selected again
      e.target.value = '';
    },
    [validateAndFilterFiles, onFilesSelected]
  );

  const handleClick = useCallback(() => {
    if (!disabled && !isUploading) {
      fileInputRef.current?.click();
    }
  }, [disabled, isUploading]);

  const clearErrors = useCallback(() => {
    setFileErrors([]);
  }, []);

  const supportedTypesText = acceptedTypes
    ? acceptedTypes.map((t) => FILE_TYPE_LABELS[t]).join(', ')
    : 'PDF, TXT, Markdown, Word documents';

  return (
    <div className={cn('space-y-3', className)}>
      {/* Drop Zone */}
      <div
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        onClick={handleClick}
        role="button"
        tabIndex={disabled || isUploading ? -1 : 0}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            handleClick();
          }
        }}
        className={cn(
          'relative flex flex-col items-center justify-center rounded-lg border-2 border-dashed p-4 transition-colors',
          'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2',
          isDragOver
            ? 'border-primary bg-primary/5'
            : 'border-muted-foreground/25 hover:border-primary/50',
          (disabled || isUploading) && 'cursor-not-allowed opacity-50',
          !disabled && !isUploading && 'cursor-pointer'
        )}
      >
        {/* Close button */}
        {onClose && (
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              onClose();
            }}
            className="absolute top-2 right-2 rounded-sm p-0.5 text-muted-foreground hover:text-foreground hover:bg-accent transition-colors"
            aria-label="Close upload zone"
          >
            <XCloseIcon className="h-4 w-4" />
          </button>
        )}

        {/* Upload Icon */}
        <UploadIcon className={cn('mb-2 h-8 w-8 text-muted-foreground', isDragOver && 'text-primary')} />

        {/* Main Text */}
        <div className="text-center">
          <p className="text-sm font-medium text-foreground">
            {isDragOver ? 'Drop files here' : 'Drag and drop files here'}
          </p>
          <p className="mt-1 text-xs text-muted-foreground">
            or <span className="text-primary underline">click to browse</span>
          </p>
        </div>

        {/* File Type & Size Info */}
        <div className="mt-2 flex flex-wrap items-center justify-center gap-x-4 gap-y-2 text-xs text-muted-foreground">
          <span className="flex items-center gap-1">
            <FileTypeIcon className="h-3.5 w-3.5" />
            {supportedTypesText}
          </span>
          <span className="flex items-center gap-1">
            <SizeIcon className="h-3.5 w-3.5" />
            Max {formatFileSize(maxSizeBytes)} per file
          </span>
          <span className="flex items-center gap-1">
            <CountIcon className="h-3.5 w-3.5" />
            Up to {maxFiles} files
          </span>
        </div>

        {/* Hidden File Input */}
        <input
          ref={fileInputRef}
          type="file"
          accept={acceptedExtensions}
          multiple={maxFiles > 1}
          onChange={handleFileInputChange}
          className="hidden"
          disabled={disabled || isUploading}
        />
      </div>

      {/* Upload Progress */}
      {uploadProgress.length > 0 && (
        <div className="space-y-2">
          <p className="text-xs font-medium text-muted-foreground">Uploading...</p>
          {uploadProgress.map((progress) => (
            <UploadProgressItem key={progress.fileId} progress={progress} />
          ))}
        </div>
      )}

      {/* Error Messages */}
      {fileErrors.length > 0 && (
        <div className="rounded-md bg-destructive/10 p-3">
          <div className="flex items-start justify-between">
            <div className="flex items-start gap-2">
              <ErrorIcon className="mt-0.5 h-4 w-4 text-destructive" />
              <div className="text-sm text-destructive">
                <p className="font-medium">Upload errors</p>
                <ul className="mt-1 list-disc pl-4 text-xs">
                  {fileErrors.map((error, index) => (
                    <li key={index}>
                      <span className="font-medium">{error.filename}:</span> {error.error}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={(e) => {
                e.stopPropagation();
                clearErrors();
              }}
              className="h-6 px-2 text-xs"
            >
              Dismiss
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * Single upload progress item.
 */
function UploadProgressItem({ progress }: { progress: UploadProgress }) {
  const statusColors = {
    uploading: 'bg-primary',
    processing: 'bg-amber-500',
    completed: 'bg-green-500',
    failed: 'bg-red-500',
  };

  return (
    <div className="rounded-md border bg-background p-3">
      <div className="flex items-center justify-between text-sm">
        <span className="truncate font-medium">{progress.filename}</span>
        <span className="ml-2 text-xs text-muted-foreground">
          {progress.status === 'uploading' && `${progress.progress}%`}
          {progress.status === 'processing' && 'Processing...'}
          {progress.status === 'completed' && 'Done'}
          {progress.status === 'failed' && 'Failed'}
        </span>
      </div>
      <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-muted">
        <div
          className={cn('h-full transition-all duration-300', statusColors[progress.status])}
          style={{ width: `${progress.progress}%` }}
        />
      </div>
      {progress.error && <p className="mt-1 text-xs text-destructive">{progress.error}</p>}
    </div>
  );
}

/**
 * Upload icon component.
 */
function UploadIcon({ className }: { className?: string }) {
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
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="17 8 12 3 7 8" />
      <line x1="12" x2="12" y1="3" y2="15" />
    </svg>
  );
}

/**
 * File type icon component.
 */
function FileTypeIcon({ className }: { className?: string }) {
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
      <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z" />
      <polyline points="14 2 14 8 20 8" />
    </svg>
  );
}

/**
 * Size icon component.
 */
function SizeIcon({ className }: { className?: string }) {
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
      <path d="M8 12h8" />
      <path d="M12 8v8" />
    </svg>
  );
}

/**
 * Count icon component.
 */
function CountIcon({ className }: { className?: string }) {
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
      <rect width="7" height="7" x="3" y="3" rx="1" />
      <rect width="7" height="7" x="14" y="3" rx="1" />
      <rect width="7" height="7" x="14" y="14" rx="1" />
      <rect width="7" height="7" x="3" y="14" rx="1" />
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
      <line x1="12" x2="12" y1="8" y2="12" />
      <line x1="12" x2="12.01" y1="16" y2="16" />
    </svg>
  );
}

/**
 * X close icon component.
 */
function XCloseIcon({ className }: { className?: string }) {
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
      <line x1="18" x2="6" y1="6" y2="18" />
      <line x1="6" x2="18" y1="6" y2="18" />
    </svg>
  );
}
