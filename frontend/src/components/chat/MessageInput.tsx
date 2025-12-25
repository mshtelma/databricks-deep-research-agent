import * as React from 'react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { ResearchDepthSelector, type ResearchDepth } from './ResearchDepthSelector';

interface MessageInputProps {
  onSubmit: (message: string, researchDepth?: ResearchDepth) => void;
  onStop?: () => void;
  isLoading?: boolean;
  placeholder?: string;
  disabled?: boolean;
  showDepthSelector?: boolean;
}

export function MessageInput({
  onSubmit,
  onStop,
  isLoading = false,
  placeholder = 'Ask a research question...',
  disabled = false,
  showDepthSelector = true,
}: MessageInputProps) {
  const [message, setMessage] = React.useState('');
  const [researchDepth, setResearchDepth] = React.useState<ResearchDepth>('auto');
  const textareaRef = React.useRef<HTMLTextAreaElement>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (message.trim() && !isLoading && !disabled) {
      onSubmit(message.trim(), researchDepth);
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
    <form onSubmit={handleSubmit} className="border-t bg-background">
      {showDepthSelector && (
        <div className="px-4 pt-2">
          <ResearchDepthSelector
            value={researchDepth}
            onChange={setResearchDepth}
            disabled={disabled || isLoading}
          />
        </div>
      )}
      <div className="flex gap-2 p-4 pt-2">
        <textarea
          data-testid="message-input"
          ref={textareaRef}
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={disabled || isLoading}
          rows={1}
          aria-label="Message input"
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
      </div>
    </form>
  );
}
