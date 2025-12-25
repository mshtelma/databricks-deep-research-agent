import { Message } from '@/types';
import { cn } from '@/lib/utils';

interface UserMessageProps {
  message: Message;
  className?: string;
}

export function UserMessage({ message, className }: UserMessageProps) {
  return (
    <div data-testid="user-message" className={cn('flex justify-end', className)}>
      <div className="max-w-[80%] rounded-lg bg-primary text-primary-foreground px-4 py-2">
        <p className="whitespace-pre-wrap">{message.content}</p>
        <span className="text-xs opacity-70 mt-1 block">
          {new Date(message.created_at).toLocaleTimeString()}
        </span>
      </div>
    </div>
  );
}
