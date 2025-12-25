import { Chat } from '@/types';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

interface ChatSidebarProps {
  chats: Chat[];
  currentChatId?: string;
  onSelectChat: (chatId: string) => void;
  onNewChat: () => void;
  isLoading?: boolean;
  className?: string;
}

export function ChatSidebar({
  chats,
  currentChatId,
  onSelectChat,
  onNewChat,
  isLoading = false,
  className,
}: ChatSidebarProps) {
  return (
    <aside className={cn('w-64 border-r bg-muted/40 flex flex-col', className)}>
      {/* Header */}
      <div className="p-4 border-b">
        <Button data-testid="new-chat-button" onClick={onNewChat} className="w-full" variant="outline">
          <PlusIcon className="w-4 h-4 mr-2" />
          New Chat
        </Button>
      </div>

      {/* Chat list */}
      <div data-testid="chat-list" className="flex-1 overflow-y-auto p-2 space-y-1">
        {isLoading ? (
          <div className="p-4 text-center text-muted-foreground">
            Loading chats...
          </div>
        ) : chats.length === 0 ? (
          <div className="p-4 text-center text-muted-foreground">
            No chats yet
          </div>
        ) : (
          chats.map((chat) => (
            <ChatListItem
              key={chat.id}
              chat={chat}
              isSelected={chat.id === currentChatId}
              onClick={() => onSelectChat(chat.id)}
            />
          ))
        )}
      </div>
    </aside>
  );
}

interface ChatListItemProps {
  chat: Chat;
  isSelected: boolean;
  onClick: () => void;
}

function ChatListItem({ chat, isSelected, onClick }: ChatListItemProps) {
  return (
    <button
      data-testid={`chat-item-${chat.id}`}
      onClick={onClick}
      className={cn(
        'w-full text-left p-3 rounded-lg transition-colors',
        'hover:bg-accent hover:text-accent-foreground',
        isSelected && 'bg-accent text-accent-foreground'
      )}
    >
      <p className={cn(
        "font-medium truncate text-sm",
        !chat.title && "italic text-muted-foreground"
      )}>
        {chat.title || 'New chat...'}
      </p>
      <p className="text-xs text-muted-foreground mt-1">
        {new Date(chat.updated_at).toLocaleDateString()}
      </p>
    </button>
  );
}

function PlusIcon({ className }: { className?: string }) {
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
      <path d="M5 12h14" />
      <path d="M12 5v14" />
    </svg>
  );
}
