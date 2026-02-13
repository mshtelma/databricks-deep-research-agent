/**
 * File upload types for enterprise data source management.
 *
 * This module defines types for:
 * - File type enumerations
 * - Processing status tracking
 * - Uploaded file metadata
 * - File preview and chunks
 */

/** Supported file types for upload */
export type FileType = 'pdf' | 'txt' | 'md' | 'docx';

/** File processing status */
export type ProcessingStatus = 'pending' | 'processing' | 'ready' | 'failed';

/** Uploaded file metadata (matches backend UploadedFileResponse camelCase) */
export interface UploadedFile {
  id: string;
  ownerId: string;
  sessionId: string | null;
  filename: string;
  fileType: FileType;
  fileSize: number;
  storagePath: string;
  processingStatus: ProcessingStatus;
  chunkCount: number;
  expiresAt: string | null;
  metadata: Record<string, unknown>;
  createdAt: string;
  updatedAt: string;
}

/** File preview response */
export interface FilePreview {
  fileId: string;
  filename: string;
  fileType: FileType;
  previewText: string;
  totalChunks: number;
  sampleChunks: FileChunk[];
}

/** Individual file chunk */
export interface FileChunk {
  chunkIndex: number;
  content: string;
  tokenCount: number | null;
  metadata: Record<string, unknown> | null;
}

/** Upload progress tracking */
export interface UploadProgress {
  fileId: string;
  filename: string;
  progress: number; // 0-100
  status: 'uploading' | 'processing' | 'completed' | 'failed';
  error?: string;
}

/** API response for file list (matches backend UploadedFileListResponse) */
export interface FileListResponse {
  items: UploadedFile[];
  total: number;
  limit: number;
  offset: number;
}

// =============================================================================
// Display Utilities
// =============================================================================

/** Human-readable labels for file types */
export const FILE_TYPE_LABELS: Record<FileType, string> = {
  pdf: 'PDF Document',
  txt: 'Text File',
  md: 'Markdown',
  docx: 'Word Document',
};

/** MIME types for supported file formats */
export const FILE_TYPE_MIME_TYPES: Record<FileType, string[]> = {
  pdf: ['application/pdf'],
  txt: ['text/plain'],
  md: ['text/markdown', 'text/x-markdown'],
  docx: ['application/vnd.openxmlformats-officedocument.wordprocessingml.document'],
};

/** File extensions for supported formats */
export const FILE_TYPE_EXTENSIONS: Record<FileType, string> = {
  pdf: '.pdf',
  txt: '.txt',
  md: '.md',
  docx: '.docx',
};

/** All accepted file extensions for file input */
export const ACCEPTED_FILE_EXTENSIONS = '.pdf,.txt,.md,.docx';

/** All accepted MIME types for file input */
export const ACCEPTED_MIME_TYPES = [
  'application/pdf',
  'text/plain',
  'text/markdown',
  'text/x-markdown',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
].join(',');

/** Default max file size in bytes (10MB) */
export const DEFAULT_MAX_FILE_SIZE = 10 * 1024 * 1024;

/** Default max number of files */
export const DEFAULT_MAX_FILES = 10;

/** Processing status colors */
export const PROCESSING_STATUS_COLORS: Record<ProcessingStatus, string> = {
  pending: 'yellow',
  processing: 'blue',
  ready: 'green',
  failed: 'red',
};

/** Processing status labels */
export const PROCESSING_STATUS_LABELS: Record<ProcessingStatus, string> = {
  pending: 'Pending',
  processing: 'Processing',
  ready: 'Ready',
  failed: 'Failed',
};

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * Get file type from filename extension.
 */
export function getFileTypeFromFilename(filename: string): FileType | null {
  const ext = filename.split('.').pop()?.toLowerCase();
  switch (ext) {
    case 'pdf':
      return 'pdf';
    case 'txt':
      return 'txt';
    case 'md':
      return 'md';
    case 'docx':
      return 'docx';
    default:
      return null;
  }
}

/**
 * Get file type from MIME type.
 */
export function getFileTypeFromMimeType(mimeType: string): FileType | null {
  for (const [type, mimeTypes] of Object.entries(FILE_TYPE_MIME_TYPES)) {
    if (mimeTypes.includes(mimeType)) {
      return type as FileType;
    }
  }
  return null;
}

/**
 * Format file size for display.
 */
export function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
}

/**
 * Check if a file is valid for upload.
 */
export function validateFile(
  file: File,
  maxSizeBytes: number = DEFAULT_MAX_FILE_SIZE
): { valid: boolean; error?: string } {
  // Check file size
  if (file.size > maxSizeBytes) {
    return {
      valid: false,
      error: `File size exceeds ${formatFileSize(maxSizeBytes)} limit`,
    };
  }

  // Check file type
  const fileType = getFileTypeFromFilename(file.name) || getFileTypeFromMimeType(file.type);
  if (!fileType) {
    return {
      valid: false,
      error: 'Unsupported file type. Please upload PDF, TXT, MD, or DOCX files.',
    };
  }

  return { valid: true };
}

/**
 * Get processing status color class.
 */
export function getProcessingStatusColor(
  status: ProcessingStatus
): 'green' | 'yellow' | 'blue' | 'red' {
  const colors: Record<ProcessingStatus, 'green' | 'yellow' | 'blue' | 'red'> = {
    pending: 'yellow',
    processing: 'blue',
    ready: 'green',
    failed: 'red',
  };
  return colors[status];
}
