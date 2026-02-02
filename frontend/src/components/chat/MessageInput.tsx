import * as React from 'react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { ResearchDepthSelector, type ResearchDepth } from './ResearchDepthSelector';
import { QueryModeSelector } from './QueryModeSelector';
import { useQueryMode } from '@/hooks';
import type { QueryMode } from '@/types';
import type { InputConfig } from '@/core/plugins/types';

interface MessageInputProps {
  onSubmit: (message: string, queryMode?: QueryMode, researchDepth?: ResearchDepth, verifySources?: boolean, outputType?: string) => void;
  onStop?: () => void;
  isLoading?: boolean;
  placeholder?: string;
  disabled?: boolean;
  showModeSelector?: boolean;
  showDepthSelector?: boolean;
  /** Plugin-provided input configuration (overrides individual props) */
  inputConfig?: InputConfig;
}

export function MessageInput({
  onSubmit,
  onStop,
  isLoading = false,
  placeholder = 'Ask a research question...',
  disabled = false,
  showModeSelector = true,
  showDepthSelector = true,
  inputConfig,
}: MessageInputProps) {
  const [message, setMessage] = React.useState('');

  // Resolve effective configuration (inputConfig overrides props)
  const effectiveShowModeSelector = inputConfig?.showModeSelector ?? showModeSelector ?? true;
  const effectiveShowDepthSelector = inputConfig?.showDepthSelector ?? showDepthSelector ?? true;
  const effectiveShowVerifySources = inputConfig?.showVerifySources ?? true;
  const effectivePlaceholder = inputConfig?.placeholder ?? placeholder ?? 'Ask a research question...';

  // Use hook for persistence (localStorage + optional API sync)
  // Only sync with preferences when mode selector is visible
  const { mode: storedMode, setMode: setStoredMode } = useQueryMode({
    initialMode: 'simple',
    syncWithPreferences: effectiveShowModeSelector, // Only sync when visible
  });

  // Effective query mode: plugin default when selector hidden, else user's choice
  const queryMode = effectiveShowModeSelector
    ? storedMode
    : (inputConfig?.defaultQueryMode ?? 'deep_research');

  // Only allow mode changes when selector is visible
  const setQueryMode = effectiveShowModeSelector ? setStoredMode : () => {};

  // Use plugin default for research depth when selector is hidden
  const [researchDepth, setResearchDepth] = React.useState<ResearchDepth>(
    inputConfig?.defaultResearchDepth ?? 'auto'
  );

  // Default: use plugin config if selector hidden, else true for deep_research
  const [verifySources, setVerifySources] = React.useState<boolean>(
    !effectiveShowVerifySources
      ? (inputConfig?.defaultVerifySources ?? true)
      : false
  );
  const textareaRef = React.useRef<HTMLTextAreaElement>(null);

  // Reset verifySources when query mode changes (only when selector is visible)
  // Default to OFF - user must explicitly enable source verification
  React.useEffect(() => {
    if (effectiveShowVerifySources) {
      setVerifySources(false);
    }
  }, [queryMode, effectiveShowVerifySources]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (message.trim() && !isLoading && !disabled) {
      onSubmit(message.trim(), queryMode, researchDepth, verifySources, inputConfig?.defaultOutputType);
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

  // Show depth selector only when Deep Research mode is selected AND selector is enabled
  const shouldShowDepthSelector = effectiveShowDepthSelector && queryMode === 'deep_research';
  // Show verify sources checkbox when web_search or deep_research is selected AND checkbox is enabled
  const shouldShowVerifyCheckbox = effectiveShowVerifySources && (queryMode === 'web_search' || queryMode === 'deep_research');

  return (
    <form onSubmit={handleSubmit} className="border-t bg-background">
      <div className="px-4 pt-2 flex flex-wrap gap-4 items-center">
        {effectiveShowModeSelector && (
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
          placeholder={effectivePlaceholder}
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
