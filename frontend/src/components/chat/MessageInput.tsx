import * as React from 'react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { ResearchDepthSelector, type ResearchDepth } from './ResearchDepthSelector';
import { QueryModeSelector } from './QueryModeSelector';
import type { QueryMode } from '@/types';

interface MessageInputProps {
  onSubmit: (message: string, queryMode?: QueryMode, researchDepth?: ResearchDepth, verifySources?: boolean) => void;
  onStop?: () => void;
  isLoading?: boolean;
  placeholder?: string;
  disabled?: boolean;
  showModeSelector?: boolean;
  showDepthSelector?: boolean;
}

export function MessageInput({
  onSubmit,
  onStop,
  isLoading = false,
  placeholder = 'Ask a research question...',
  disabled = false,
  showModeSelector = true,
  showDepthSelector = true,
}: MessageInputProps) {
  const [message, setMessage] = React.useState('');
  const [queryMode, setQueryMode] = React.useState<QueryMode>('simple');
  const [researchDepth, setResearchDepth] = React.useState<ResearchDepth>('auto');
  // Default: true for deep_research (thorough), false for web_search (speed)
  const [verifySources, setVerifySources] = React.useState<boolean>(false);
  const textareaRef = React.useRef<HTMLTextAreaElement>(null);

  // Reset verifySources when query mode changes
  React.useEffect(() => {
    setVerifySources(queryMode === 'deep_research');
  }, [queryMode]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (message.trim() && !isLoading && !disabled) {
      onSubmit(message.trim(), queryMode, researchDepth, verifySources);
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

  // Show depth selector only when Deep Research mode is selected
  const shouldShowDepthSelector = showDepthSelector && queryMode === 'deep_research';
  // Show verify sources checkbox when web_search or deep_research is selected
  const shouldShowVerifyCheckbox = queryMode === 'web_search' || queryMode === 'deep_research';

  return (
    <form onSubmit={handleSubmit} className="border-t bg-background">
      <div className="px-4 pt-2 flex flex-wrap gap-4 items-center">
        {showModeSelector && (
          <QueryModeSelector
            value={queryMode}
            onChange={setQueryMode}
            disabled={disabled || isLoading}
          />
        )}
        {shouldShowDepthSelector && (
          <ResearchDepthSelector
            value={researchDepth}
            onChange={setResearchDepth}
            disabled={disabled || isLoading}
          />
        )}
        {shouldShowVerifyCheckbox && (
          <label className="flex items-center gap-1.5 text-xs text-muted-foreground cursor-pointer select-none">
            <input
              type="checkbox"
              checked={verifySources}
              onChange={(e) => setVerifySources(e.target.checked)}
              disabled={disabled || isLoading}
              className="h-3.5 w-3.5 rounded border-input cursor-pointer accent-primary"
            />
            <span>Verify sources</span>
          </label>
        )}
      </div>
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
