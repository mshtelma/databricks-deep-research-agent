import * as React from 'react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

interface EditMessageModalProps {
  isOpen: boolean;
  originalContent: string;
  onClose: () => void;
  onSave: (newContent: string) => void;
  isSaving?: boolean;
}

export function EditMessageModal({
  isOpen,
  originalContent,
  onClose,
  onSave,
  isSaving = false,
}: EditMessageModalProps) {
  const [content, setContent] = React.useState(originalContent);
  const textareaRef = React.useRef<HTMLTextAreaElement>(null);
  const dialogRef = React.useRef<HTMLDivElement>(null);

  // Reset content when modal opens
  React.useEffect(() => {
    if (isOpen) {
      setContent(originalContent);
      // Focus textarea after a short delay to ensure it's rendered
      setTimeout(() => textareaRef.current?.focus(), 100);
    }
  }, [isOpen, originalContent]);

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

  const handleSave = () => {
    if (content.trim() && content !== originalContent) {
      onSave(content.trim());
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      handleSave();
    }
  };

  if (!isOpen) return null;

  const hasChanges = content.trim() !== originalContent;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div className="fixed inset-0 bg-black/50" aria-hidden="true" />

      {/* Dialog */}
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="edit-message-title"
        className={cn(
          'relative z-50 w-full max-w-lg rounded-lg bg-background p-6 shadow-lg',
          'animate-in fade-in-0 zoom-in-95'
        )}
      >
        <h3 id="edit-message-title" className="text-lg font-semibold mb-4">
          Edit Message
        </h3>

        <p className="text-sm text-muted-foreground mb-3">
          Editing this message will invalidate all subsequent messages and trigger a new research.
        </p>

        <textarea
          ref={textareaRef}
          value={content}
          onChange={(e) => setContent(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={isSaving}
          rows={5}
          className={cn(
            'w-full resize-none rounded-md border border-input bg-transparent px-3 py-2 text-sm',
            'placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring',
            'disabled:cursor-not-allowed disabled:opacity-50',
            'min-h-[120px]'
          )}
        />

        <p className="text-xs text-muted-foreground mt-2 mb-4">
          Press {navigator.platform.includes('Mac') ? 'Cmd' : 'Ctrl'}+Enter to save
        </p>

        <div className="flex justify-end gap-3">
          <Button variant="outline" onClick={onClose} disabled={isSaving}>
            Cancel
          </Button>
          <Button
            onClick={handleSave}
            disabled={!hasChanges || !content.trim() || isSaving}
          >
            {isSaving ? 'Saving...' : 'Save & Regenerate'}
          </Button>
        </div>
      </div>
    </div>
  );
}
