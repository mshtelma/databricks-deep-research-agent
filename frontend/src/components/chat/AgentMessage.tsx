import * as React from 'react';
import { Message, Source } from '@/types';
import { cn } from '@/lib/utils';
import { Card, CardContent } from '@/components/ui/card';
import { MarkdownRenderer } from '@/components/common';

interface AgentMessageProps {
  message: Message;
  sources?: Source[];
  isStreaming?: boolean;
  onRegenerate?: () => void;
  className?: string;
}

export function AgentMessage({
  message,
  sources = [],
  isStreaming = false,
  onRegenerate,
  className,
}: AgentMessageProps) {
  const [showSources, setShowSources] = React.useState(false);

  return (
    <div data-testid="agent-response" className={cn('flex justify-start', className)}>
      <Card className="max-w-[90%] bg-muted">
        <CardContent className="p-4">
          {/* Message content with markdown rendering */}
          <div className="relative">
            <MarkdownRenderer content={message.content} />
            {isStreaming && (
              <span className="inline-block w-2 h-4 bg-primary animate-pulse ml-1 align-text-bottom" />
            )}
          </div>

          {/* Sources section */}
          {sources.length > 0 && (
            <div className="mt-4 pt-4 border-t">
              <button
                onClick={() => setShowSources(!showSources)}
                className="text-sm text-muted-foreground hover:text-foreground flex items-center gap-1"
              >
                <SourcesIcon className="w-4 h-4" />
                {sources.length} source{sources.length !== 1 ? 's' : ''}
                <ChevronIcon
                  className={cn(
                    'w-4 h-4 transition-transform',
                    showSources && 'rotate-180'
                  )}
                />
              </button>

              {showSources && (
                <div className="mt-2 space-y-2">
                  {sources.map((source, index) => (
                    <SourceCard key={source.url || index} source={source} index={index + 1} />
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Timestamp and regenerate */}
          {!isStreaming && (
            <div className="flex items-center justify-between mt-2">
              <span className="text-xs text-muted-foreground">
                {new Date(message.created_at).toLocaleTimeString()}
              </span>
              {onRegenerate && (
                <button
                  data-testid="regenerate-response"
                  onClick={onRegenerate}
                  className="text-xs text-muted-foreground hover:text-foreground transition-colors"
                >
                  Regenerate
                </button>
              )}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function SourceCard({ source, index }: { source: Source; index: number }) {
  return (
    <a
      data-testid="citation"
      href={source.url}
      target="_blank"
      rel="noopener noreferrer"
      className="block p-2 rounded border hover:bg-background transition-colors"
    >
      <div className="flex items-start gap-2">
        <span className="text-xs bg-primary/10 text-primary px-1.5 py-0.5 rounded">
          [{index}]
        </span>
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium truncate">{source.title || source.url}</p>
          {source.snippet && (
            <p className="text-xs text-muted-foreground line-clamp-2 mt-1">
              {source.snippet}
            </p>
          )}
        </div>
      </div>
    </a>
  );
}

function SourcesIcon({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20" />
    </svg>
  );
}

function ChevronIcon({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <path d="m6 9 6 6 6-6" />
    </svg>
  );
}
