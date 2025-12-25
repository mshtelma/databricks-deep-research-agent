import * as React from 'react';
import { Message } from '@/types';
import { UserMessage } from './UserMessage';
import { AgentMessage } from './AgentMessage';
import { cn } from '@/lib/utils';

interface MessageListProps {
  messages: Message[];
  streamingContent?: string;
  isStreaming?: boolean;
  isLoading?: boolean;
  className?: string;
}

export function MessageList({
  messages,
  streamingContent,
  isStreaming = false,
  isLoading = false,
  className,
}: MessageListProps) {
  const messagesEndRef = React.useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom on new messages
  React.useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamingContent]);

  // Show welcome screen only if no messages, not streaming, and no completed streaming content
  if (messages.length === 0 && !isStreaming && !streamingContent) {
    return (
      <div className={cn('flex-1 flex items-center justify-center', className)}>
        <div className="text-center text-muted-foreground">
          <h2 className="text-2xl font-semibold mb-2">Deep Research Agent</h2>
          <p className="max-w-md">
            Ask any research question and I'll search the web, analyze sources,
            and synthesize a comprehensive answer with citations.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div data-testid="message-list" className={cn('flex-1 overflow-y-auto p-4 space-y-4', className)}>
      {messages.map((message) =>
        message.role === 'user' ? (
          <UserMessage key={message.id} message={message} />
        ) : (
          <AgentMessage key={message.id} message={message} />
        )
      )}

      {/* Loading indicator (before streaming starts) */}
      {isLoading && !streamingContent && (
        <div data-testid="loading-indicator" className="flex justify-start">
          <div className="bg-muted rounded-lg px-4 py-2 text-sm text-muted-foreground animate-pulse">
            Thinking...
          </div>
        </div>
      )}

      {/* Streaming message (shown while streaming) */}
      {isStreaming && streamingContent && (
        <div data-testid="streaming-indicator">
          <AgentMessage
            message={{
              id: 'streaming',
              chat_id: '',
              role: 'agent',
              content: streamingContent,
              created_at: new Date().toISOString(),
              is_edited: false,
            }}
            isStreaming={true}
          />
        </div>
      )}

      {/* Completed streamed message (shown after streaming completes, before messages are refreshed) */}
      {!isStreaming && streamingContent && messages.filter(m => m.role === 'agent').length === 0 && (
        <AgentMessage
          message={{
            id: 'completed-stream',
            chat_id: '',
            role: 'agent',
            content: streamingContent,
            created_at: new Date().toISOString(),
            is_edited: false,
          }}
          isStreaming={false}
        />
      )}

      <div ref={messagesEndRef} />
    </div>
  );
}
