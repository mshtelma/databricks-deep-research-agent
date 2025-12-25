import * as React from 'react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

interface RenameChatDialogProps {
  isOpen: boolean;
  currentTitle: string;
  onClose: () => void;
  onSave: (newTitle: string) => void;
  isSaving?: boolean;
}

export function RenameChatDialog({
  isOpen,
  currentTitle,
  onClose,
  onSave,
  isSaving = false,
}: RenameChatDialogProps) {
  const [title, setTitle] = React.useState(currentTitle);
  const inputRef = React.useRef<HTMLInputElement>(null);
  const dialogRef = React.useRef<HTMLDivElement>(null);

  // Reset title when dialog opens
  React.useEffect(() => {
    if (isOpen) {
      setTitle(currentTitle);
      setTimeout(() => {
        inputRef.current?.focus();
        inputRef.current?.select();
      }, 100);
    }
  }, [isOpen, currentTitle]);

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
    if (title.trim()) {
      onSave(title.trim());
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleSave();
    }
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
        aria-labelledby="rename-dialog-title"
        className={cn(
          'relative z-50 w-full max-w-md rounded-lg bg-background p-6 shadow-lg',
          'animate-in fade-in-0 zoom-in-95'
        )}
      >
        <h3 id="rename-dialog-title" className="text-lg font-semibold mb-4">
          Rename Chat
        </h3>

        <input
          ref={inputRef}
          type="text"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={isSaving}
          placeholder="Enter chat title..."
          className={cn(
            'w-full rounded-md border border-input bg-transparent px-3 py-2 text-sm',
            'placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring',
            'disabled:cursor-not-allowed disabled:opacity-50',
            'mb-4'
          )}
        />

        <div className="flex justify-end gap-3">
          <Button variant="outline" onClick={onClose} disabled={isSaving}>
            Cancel
          </Button>
          <Button
            onClick={handleSave}
            disabled={!title.trim() || isSaving}
          >
            {isSaving ? 'Saving...' : 'Save'}
          </Button>
        </div>
      </div>
    </div>
  );
}
