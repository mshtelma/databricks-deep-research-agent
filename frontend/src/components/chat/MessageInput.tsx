import * as React from 'react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

interface MessageInputProps {
  onSubmit: (message: string) => void;
  onStop?: () => void;
  isLoading?: boolean;
  placeholder?: string;
  disabled?: boolean;
}

export function MessageInput({
  onSubmit,
  onStop,
  isLoading = false,
  placeholder = 'Ask a research question...',
  disabled = false,
}: MessageInputProps) {
  const [message, setMessage] = React.useState('');
  const textareaRef = React.useRef<HTMLTextAreaElement>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (message.trim() && !isLoading && !disabled) {
      onSubmit(message.trim());
      setMessage('');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  // Auto-resize textarea
  React.useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`;
    }
  }, [message]);

  return (
    <form onSubmit={handleSubmit} className="flex gap-2 p-4 border-t bg-background">
      <textarea
        data-testid="message-input"
        ref={textareaRef}
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={placeholder}
        disabled={disabled || isLoading}
        rows={1}
        className={cn(
          'flex-1 resize-none rounded-md border border-input bg-transparent px-3 py-2 text-sm shadow-sm transition-colors',
          'placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring',
          'disabled:cursor-not-allowed disabled:opacity-50',
          'min-h-[40px] max-h-[200px]'
        )}
      />
      {isLoading && onStop ? (
        <Button
          data-testid="stop-button"
          type="button"
          variant="outline"
          onClick={onStop}
          className="self-end"
        >
          Stop
        </Button>
      ) : (
        <Button
          data-testid="send-button"
          type="submit"
          disabled={!message.trim() || isLoading || disabled}
          className="self-end"
        >
          Send
        </Button>
      )}
    </form>
  );
}
