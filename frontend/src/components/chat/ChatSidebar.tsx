import * as React from 'react';
import { Chat } from '@/types';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { ChatSearchInput } from './ChatSearchInput';
import { ActiveJobsIndicator } from '@/components/jobs/ActiveJobsIndicator';

type StatusFilter = 'active' | 'archived' | 'all';
type ChatListEntry = Chat & { isDraft?: boolean };

interface ChatSidebarProps {
  chats: ChatListEntry[];
  currentChatId?: string;
  onSelectChat: (chatId: string) => void;
  onNewChat: () => void;
  onRenameChat?: (chatId: string, newTitle: string) => void;
  onArchiveChat?: (chatId: string) => void;
  onRestoreChat?: (chatId: string) => void;
  onDeleteChat?: (chatId: string) => void;
  onExportChat?: (chatId: string) => void;
  searchQuery: string;
  onSearchQueryChange: (query: string) => void;
  statusFilter: StatusFilter;
  onStatusFilterChange: (status: StatusFilter) => void;
  isLoading?: boolean;
  className?: string;
}

export function ChatSidebar({
  chats,
  currentChatId,
  onSelectChat,
  onNewChat,
  onRenameChat,
  onArchiveChat,
  onRestoreChat,
  onDeleteChat,
  onExportChat,
  searchQuery,
  onSearchQueryChange,
  statusFilter,
  onStatusFilterChange,
  isLoading = false,
  className,
}: ChatSidebarProps) {
  // Filter chats by status and search query
  const filteredChats = React.useMemo(() => {
    return chats.filter((chat) => {
      // Status filter
      if (statusFilter !== 'all' && chat.status !== statusFilter) {
        return false;
      }

      // Search filter
      if (searchQuery) {
        const query = searchQuery.toLowerCase();
        const title = (chat.title || '').toLowerCase();
        return title.includes(query);
      }

      return true;
    });
  }, [chats, statusFilter, searchQuery]);

  return (
    <aside className={cn('w-64 border-r bg-muted/40 flex flex-col', className)}>
      {/* Header */}
      <div className="p-4 border-b space-y-3">
        <Button data-testid="new-chat-button" onClick={onNewChat} className="w-full" variant="outline">
          <PlusIcon className="w-4 h-4 mr-2" />
          New Chat
        </Button>

        <ChatSearchInput
          value={searchQuery}
          onChange={onSearchQueryChange}
          placeholder="Search chats..."
        />

        {/* Status filter tabs */}
        <div className="flex gap-1 p-1 bg-muted rounded-md">
          <StatusFilterTab
            label="Active"
            isActive={statusFilter === 'active'}
            onClick={() => onStatusFilterChange('active')}
          />
          <StatusFilterTab
            label="Archived"
            isActive={statusFilter === 'archived'}
            onClick={() => onStatusFilterChange('archived')}
          />
          <StatusFilterTab
            label="All"
            isActive={statusFilter === 'all'}
            onClick={() => onStatusFilterChange('all')}
          />
        </div>
      </div>

      {/* Chat list */}
      <div data-testid="chat-list" className="flex-1 overflow-y-auto p-2 space-y-1">
        {isLoading ? (
          <div data-testid="chat-list-loading" className="p-4 text-center text-muted-foreground">
            Loading chats...
          </div>
        ) : filteredChats.length === 0 ? (
          <div data-testid="chat-list-empty" className="p-4 text-center text-muted-foreground">
            {searchQuery ? 'No matching chats' : statusFilter === 'archived' ? 'No archived chats' : 'No chats yet'}
          </div>
        ) : (
          filteredChats.map((chat) => (
            <ChatListItem
              key={chat.id}
              chat={chat}
              isSelected={chat.id === currentChatId}
              onClick={() => onSelectChat(chat.id)}
              onRename={
                !chat.isDraft && onRenameChat ? (title) => onRenameChat(chat.id, title) : undefined
              }
              onArchive={
                !chat.isDraft && onArchiveChat ? () => onArchiveChat(chat.id) : undefined
              }
              onRestore={
                !chat.isDraft && onRestoreChat ? () => onRestoreChat(chat.id) : undefined
              }
              onDelete={
                !chat.isDraft && onDeleteChat ? () => onDeleteChat(chat.id) : undefined
              }
              onExport={
                !chat.isDraft && onExportChat ? () => onExportChat(chat.id) : undefined
              }
            />
          ))
        )}
      </div>

      {/* Active Jobs Indicator - shown at bottom of sidebar */}
      <div className="p-2 border-t">
        <ActiveJobsIndicator onNavigateToChat={onSelectChat} />
      </div>
    </aside>
  );
}

interface StatusFilterTabProps {
  label: string;
  isActive: boolean;
  onClick: () => void;
}

function StatusFilterTab({ label, isActive, onClick }: StatusFilterTabProps) {
  return (
    <button
      type="button"
      data-testid={`status-filter-${label.toLowerCase()}`}
      onClick={onClick}
      className={cn(
        'flex-1 px-2 py-1 text-xs font-medium rounded transition-colors',
        isActive
          ? 'bg-background text-foreground shadow-sm'
          : 'text-muted-foreground hover:text-foreground'
      )}
    >
      {label}
    </button>
  );
}

interface ChatListItemProps {
  chat: ChatListEntry;
  isSelected: boolean;
  onClick: () => void;
  onRename?: (newTitle: string) => void;
  onArchive?: () => void;
  onRestore?: () => void;
  onDelete?: () => void;
  onExport?: () => void;
}

function ChatListItem({
  chat,
  isSelected,
  onClick,
  onRename,
  onArchive,
  onRestore,
  onDelete,
  onExport,
}: ChatListItemProps) {
  const [showMenu, setShowMenu] = React.useState(false);
  const menuRef = React.useRef<HTMLDivElement>(null);
  const buttonRef = React.useRef<HTMLButtonElement>(null);

  // Close menu on click outside
  React.useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (
        menuRef.current &&
        !menuRef.current.contains(e.target as Node) &&
        buttonRef.current &&
        !buttonRef.current.contains(e.target as Node)
      ) {
        setShowMenu(false);
      }
    };

    if (showMenu) {
      document.addEventListener('mousedown', handleClickOutside);
      return () => document.removeEventListener('mousedown', handleClickOutside);
    }
  }, [showMenu]);

  // Close menu on escape
  React.useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && showMenu) {
        setShowMenu(false);
      }
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [showMenu]);

  const handleMenuAction = (action: () => void) => {
    setShowMenu(false);
    action();
  };

  const isArchived = chat.status === 'archived';
  const isDraft = !!chat.isDraft;

  return (
    <div className="relative group">
      <button
        data-testid={`chat-item-${chat.id}`}
        onClick={onClick}
        className={cn(
          'w-full text-left p-3 rounded-lg transition-colors pr-10',
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
        <div className="flex items-center gap-2 mt-1">
          <span className="text-xs text-muted-foreground">
            {new Date(chat.updated_at).toLocaleDateString()}
          </span>
          {isDraft && (
            <span className="text-xs px-1.5 py-0.5 rounded bg-muted text-muted-foreground">
              Draft
            </span>
          )}
          {isArchived && (
            <span className="text-xs px-1.5 py-0.5 rounded bg-muted text-muted-foreground">
              Archived
            </span>
          )}
        </div>
      </button>

      {/* Context menu trigger */}
      {!isDraft && (
        <button
          ref={buttonRef}
          type="button"
          data-testid={`chat-menu-trigger-${chat.id}`}
          onClick={(e) => {
            e.stopPropagation();
            setShowMenu(!showMenu);
          }}
          className={cn(
            'absolute right-2 top-1/2 -translate-y-1/2 p-1 rounded',
            'opacity-0 group-hover:opacity-100 focus:opacity-100',
            'hover:bg-accent text-muted-foreground hover:text-foreground',
            'transition-opacity'
          )}
          aria-label="Chat options"
          aria-haspopup="menu"
          aria-expanded={showMenu}
          aria-controls={`chat-menu-${chat.id}`}
        >
          <MoreIcon className="w-4 h-4" />
        </button>
      )}

      {/* Context menu */}
      {showMenu && (
        <div
          ref={menuRef}
          id={`chat-menu-${chat.id}`}
          role="menu"
          aria-label="Chat actions"
          className={cn(
            'absolute right-0 top-full z-50 mt-1 w-48',
            'rounded-md border bg-popover p-1 shadow-md',
            'animate-in fade-in-0 zoom-in-95'
          )}
        >
          {onRename && (
            <ContextMenuItem
              data-testid="chat-action-rename"
              icon={<EditIcon className="w-4 h-4" />}
              label="Rename"
              onClick={() => {
                const newTitle = prompt('Enter new title:', chat.title || '');
                if (newTitle !== null && newTitle.trim()) {
                  handleMenuAction(() => onRename(newTitle.trim()));
                } else {
                  setShowMenu(false);
                }
              }}
            />
          )}
          {onExport && (
            <ContextMenuItem
              data-testid="chat-action-export"
              icon={<ExportIcon className="w-4 h-4" />}
              label="Export"
              onClick={() => handleMenuAction(onExport)}
            />
          )}
          {isArchived ? (
            onRestore && (
              <ContextMenuItem
                data-testid="chat-action-restore"
                icon={<RestoreIcon className="w-4 h-4" />}
                label="Unarchive"
                onClick={() => handleMenuAction(onRestore)}
              />
            )
          ) : (
            onArchive && (
              <ContextMenuItem
                data-testid="chat-action-archive"
                icon={<ArchiveIcon className="w-4 h-4" />}
                label="Archive"
                onClick={() => handleMenuAction(onArchive)}
              />
            )
          )}
          {onDelete && (
            <>
              <div className="my-1 h-px bg-border" />
              <ContextMenuItem
                data-testid="chat-action-delete"
                icon={<TrashIcon className="w-4 h-4" />}
                label="Delete"
                onClick={() => handleMenuAction(onDelete)}
                variant="destructive"
              />
            </>
          )}
        </div>
      )}
    </div>
  );
}

interface ContextMenuItemProps {
  icon: React.ReactNode;
  label: string;
  onClick: () => void;
  variant?: 'default' | 'destructive';
  'data-testid'?: string;
}

function ContextMenuItem({ icon, label, onClick, variant = 'default', 'data-testid': dataTestId }: ContextMenuItemProps) {
  return (
    <button
      type="button"
      data-testid={dataTestId}
      onClick={onClick}
      className={cn(
        'flex w-full items-center gap-2 rounded-sm px-2 py-1.5 text-sm',
        'transition-colors cursor-pointer',
        variant === 'destructive'
          ? 'text-destructive hover:bg-destructive/10'
          : 'hover:bg-accent hover:text-accent-foreground'
      )}
    >
      {icon}
      {label}
    </button>
  );
}

// Icons

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

function MoreIcon({ className }: { className?: string }) {
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
      <circle cx="12" cy="12" r="1" />
      <circle cx="12" cy="5" r="1" />
      <circle cx="12" cy="19" r="1" />
    </svg>
  );
}

function EditIcon({ className }: { className?: string }) {
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
      <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
      <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
    </svg>
  );
}

function ArchiveIcon({ className }: { className?: string }) {
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
      <rect width="20" height="5" x="2" y="3" rx="1" />
      <path d="M4 8v11a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8" />
      <path d="M10 12h4" />
    </svg>
  );
}

function RestoreIcon({ className }: { className?: string }) {
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
      <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
      <path d="M3 3v5h5" />
    </svg>
  );
}

function ExportIcon({ className }: { className?: string }) {
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
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="7 10 12 15 17 10" />
      <line x1="12" x2="12" y1="15" y2="3" />
    </svg>
  );
}

function TrashIcon({ className }: { className?: string }) {
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
      <path d="M3 6h18" />
      <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6" />
      <path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2" />
      <line x1="10" x2="10" y1="11" y2="17" />
      <line x1="14" x2="14" y1="11" y2="17" />
    </svg>
  );
}
