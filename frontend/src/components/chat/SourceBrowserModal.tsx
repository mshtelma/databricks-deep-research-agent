/**
 * SourceBrowserModal - Modal wrapper for DiscoveredSourceBrowser.
 *
 * Features:
 * - Full-screen modal with proper focus management
 * - Apply/Cancel buttons
 * - Local selection state (doesn't commit until Apply)
 * - Escape key to close
 */

import * as React from 'react';
import { Button } from '@/components/ui/button';
import { DiscoveredSourceBrowser } from './DiscoveredSourceBrowser';
import { cn } from '@/lib/utils';
import type { DiscoveredSource } from '@/types/discovery';

interface SourceBrowserModalProps {
  isOpen: boolean;
  onClose: () => void;
  /** Currently selected source IDs (passed from parent state) */
  initialSelectedIds: string[];
  /** Callback when user clicks Apply */
  onApply: (selectedIds: string[]) => void;
  /** Discovery data from parent */
  sources: DiscoveredSource[];
  isDiscoveryLoading: boolean;
  discoveryError: Error | null;
  onRefetch: () => void;
  onRefresh: () => void;
  isRefreshing: boolean;
}

export function SourceBrowserModal({
  isOpen,
  onClose,
  initialSelectedIds,
  onApply,
  sources,
  isDiscoveryLoading,
  discoveryError,
  onRefetch,
  onRefresh,
  isRefreshing,
}: SourceBrowserModalProps) {
  // Local state for selection (doesn't affect parent until Apply)
  const [localSelectedIds, setLocalSelectedIds] = React.useState<string[]>(initialSelectedIds);
  const dialogRef = React.useRef<HTMLDivElement>(null);

  // Reset local state when modal opens
  React.useEffect(() => {
    if (isOpen) {
      setLocalSelectedIds(initialSelectedIds);
    }
  }, [isOpen, initialSelectedIds]);

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

  const handleApply = () => {
    onApply(localSelectedIds);
    onClose();
  };

  const handleSelectNone = () => {
    setLocalSelectedIds([]);
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div className="fixed inset-0 bg-black/50" aria-hidden="true" />

      {/* Dialog */}
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="source-browser-title"
        className={cn(
          'relative z-50 w-full max-w-2xl rounded-lg bg-background shadow-lg',
          'max-h-[85vh] flex flex-col',
          'animate-in fade-in-0 zoom-in-95'
        )}
      >
        {/* Header */}
        <div className="p-4 pb-0 flex items-center justify-between">
          <h3 id="source-browser-title" className="text-lg font-semibold">
            Select Data Sources
          </h3>
          <button
            type="button"
            onClick={onClose}
            className="p-1 rounded-md hover:bg-muted transition-colors"
            aria-label="Close"
          >
            <XIcon className="h-5 w-5" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-hidden">
          <DiscoveredSourceBrowser
            selectedIds={localSelectedIds}
            onSelectionChange={setLocalSelectedIds}
            sources={sources}
            isLoading={isDiscoveryLoading}
            error={discoveryError}
            onRefetch={onRefetch}
            onRefresh={onRefresh}
            isRefreshing={isRefreshing}
            maxHeight="calc(85vh - 180px)"
          />
        </div>

        {/* Footer */}
        <div className="p-4 pt-2 border-t flex justify-between">
          <Button variant="ghost" size="sm" onClick={handleSelectNone}>
            Clear Selection
          </Button>
          <div className="flex gap-2">
            <Button variant="outline" onClick={onClose}>
              Cancel
            </Button>
            <Button onClick={handleApply}>
              Apply ({localSelectedIds.length} selected)
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}

// Icon
function XIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
    >
      <path d="M18 6 6 18M6 6l12 12" />
    </svg>
  );
}

export default SourceBrowserModal;
