import * as React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { Chat } from '@/types';
import { cn } from '@/lib/utils';
import { ChatSearchInput } from './ChatSearchInput';
import { ActiveJobsIndicator } from '@/components/jobs/ActiveJobsIndicator';
import { UserProfile } from '@/components/user';
import { IncognitoSection } from '@/components/incognito';
import { useIncognitoSessionStatus, useCreateIncognitoChat } from '@/hooks';
import { ComponentRegistry } from '@/core/plugins';

type StatusFilter = 'active' | 'archived' | 'all';
type ChatListEntry = Chat & { isDraft?: boolean };

interface ChatSidebarProps {
  chats: ChatListEntry[];
  currentChatId?: string;
  onSelectChat: (chatId: string) => void;
  onNewChat: () => void;
  onNewIncognitoChat?: () => void;
  onRenameChat?: (chatId: string, newTitle: string) => void;
  onArchiveChat?: (chatId: string) => void;
  onRestoreChat?: (chatId: string) => void;
  onDeleteChat?: (chatId: string) => void;
  onExportChat?: (chatId: string) => void;
  /** Called when user hovers over a chat (for prefetching messages) */
  onHoverChat?: (chatId: string) => void;
  searchQuery: string;
  onSearchQueryChange: (query: string) => void;
  statusFilter: StatusFilter;
  onStatusFilterChange: (status: StatusFilter) => void;
  isLoading?: boolean;
  className?: string;
  /** Collapsed state — when true, render a 44px icon rail */
  collapsed?: boolean;
  /** Toggle the collapsed state */
  onToggleCollapsed?: () => void;
}

export function ChatSidebar({
  chats,
  currentChatId,
  onSelectChat,
  onNewChat,
  onNewIncognitoChat,
  onRenameChat,
  onArchiveChat,
  onRestoreChat,
  onDeleteChat,
  onExportChat,
  onHoverChat,
  searchQuery,
  onSearchQueryChange,
  statusFilter,
  onStatusFilterChange,
  isLoading = false,
  className,
  collapsed = false,
  onToggleCollapsed,
}: ChatSidebarProps) {
  const navigate = useNavigate();
  const location = useLocation();
  const [showNewChatMenu, setShowNewChatMenu] = React.useState(false);
  const [enableAuxQueries, setEnableAuxQueries] = React.useState(false);
  const newChatButtonRef = React.useRef<HTMLDivElement>(null);
  const activateAuxQueries = React.useCallback(() => {
    setEnableAuxQueries(true);
  }, []);
  const { data: sessionStatus } = useIncognitoSessionStatus({
    enabled: enableAuxQueries || showNewChatMenu,
  });
  const createIncognito = useCreateIncognitoChat();

  // Defer non-essential sidebar queries so main chat UI is interactive first.
  React.useEffect(() => {
    const timer = setTimeout(() => {
      setEnableAuxQueries(true);
    }, 1200);
    return () => clearTimeout(timer);
  }, []);

  // Close menu on click outside
  React.useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (
        newChatButtonRef.current &&
        !newChatButtonRef.current.contains(e.target as Node)
      ) {
        setShowNewChatMenu(false);
      }
    };

    if (showNewChatMenu) {
      document.addEventListener('mousedown', handleClickOutside);
      return () => document.removeEventListener('mousedown', handleClickOutside);
    }
  }, [showNewChatMenu]);

  const canCreateIncognito = !sessionStatus || sessionStatus.chatCount < sessionStatus.maxChats;

  const handleNewIncognitoChat = async () => {
    activateAuxQueries();
    setShowNewChatMenu(false);
    if (onNewIncognitoChat) {
      onNewIncognitoChat();
    } else {
      try {
        const chat = await createIncognito.mutateAsync({});
        onSelectChat(chat.id);
      } catch (error) {
        console.error('Failed to create incognito chat:', error);
      }
    }
  };
  // Filter chats by status and search query
  const filteredChats = React.useMemo(() => {
    return chats.filter((chat) => {
      if (statusFilter !== 'all' && chat.status !== statusFilter) {
        return false;
      }
      if (searchQuery) {
        const query = searchQuery.toLowerCase();
        const title = (chat.title || '').toLowerCase();
        return title.includes(query);
      }
      return true;
    });
  }, [chats, statusFilter, searchQuery]);

  const showChatListLoading = isLoading && filteredChats.length === 0;

  const sidebarConfig = ComponentRegistry.getSidebarConfig();
  const showAgentsLink = sidebarConfig?.showAgentsLink !== false;
  const showTemplatesLink = sidebarConfig?.showTemplatesLink !== false;
  const onAgentsRoute = location.pathname.startsWith('/agents') || location.pathname.startsWith('/designer');
  const onTemplatesRoute = location.pathname.startsWith('/templates');

  // ---------------------------------------------------------------------
  // Collapsed (44px) rail
  // ---------------------------------------------------------------------
  if (collapsed) {
    return (
      <aside
        className={cn(
          'db-root flex w-11 shrink-0 flex-col items-center gap-2 border-r border-db-gray-lines bg-white pt-2.5 font-db-sans',
          className,
        )}
        onMouseEnter={activateAuxQueries}
      >
        <button
          type="button"
          onClick={onToggleCollapsed}
          aria-label="Expand sidebar"
          title="Expand sidebar"
          className="rounded p-1.5 text-db-gray-text hover:bg-db-oat-medium hover:text-db-navy-800"
        >
          <ChevRightSmall className="h-3.5 w-3.5" />
        </button>
        <button
          type="button"
          onClick={onNewChat}
          aria-label="New chat"
          title="New chat"
          className="rounded p-1.5 text-db-navy-800 hover:bg-db-oat-medium"
        >
          <PlusIcon className="h-3.5 w-3.5" />
        </button>
        <div className="my-1 h-px w-7 bg-db-gray-lines" />
        {showAgentsLink && (
          <button
            type="button"
            onClick={() => navigate('/agents')}
            aria-label="Agents"
            title="Agents"
            className={cn(
              'rounded p-1.5 hover:bg-db-oat-medium',
              onAgentsRoute ? 'bg-db-oat-medium text-db-navy-800' : 'text-db-gray-text hover:text-db-navy-800',
            )}
          >
            <AgentIcon className="h-4 w-4" />
          </button>
        )}
        {showTemplatesLink && (
          <button
            type="button"
            onClick={() => navigate('/templates')}
            aria-label="Templates"
            title="Templates"
            className={cn(
              'rounded p-1.5 hover:bg-db-oat-medium',
              onTemplatesRoute ? 'bg-db-oat-medium text-db-navy-800' : 'text-db-gray-text hover:text-db-navy-800',
            )}
          >
            <TemplateIcon className="h-4 w-4" />
          </button>
        )}
        <div className="flex-1" />
        <CollapsedAvatar />
      </aside>
    );
  }

  // ---------------------------------------------------------------------
  // Expanded (240px) sidebar
  // ---------------------------------------------------------------------
  return (
    <aside
      className={cn(
        'db-root flex w-60 shrink-0 flex-col border-r border-db-gray-lines bg-white font-db-sans text-db-navy-800',
        className,
      )}
      onMouseEnter={activateAuxQueries}
      onFocusCapture={activateAuxQueries}
    >
      {/* Header: New Chat + dropdown + collapse */}
      <div className="flex items-center gap-1.5 px-2.5 pb-2 pt-2.5">
        <div ref={newChatButtonRef} className="relative flex flex-1 items-center gap-1">
          <button
            type="button"
            data-testid="new-chat-button"
            onClick={onNewChat}
            className="flex flex-1 items-center justify-center gap-1.5 rounded-db-md border border-db-gray-lines bg-white px-2.5 py-1.5 text-[13px] font-medium text-db-navy-800 transition-colors hover:border-db-navy-300 hover:bg-db-oat-medium"
          >
            <PlusIcon className="h-3 w-3" />
            New chat
          </button>
          <button
            type="button"
            data-testid="new-chat-menu-trigger"
            onClick={() => {
              const next = !showNewChatMenu;
              setShowNewChatMenu(next);
              if (next) activateAuxQueries();
            }}
            aria-label="More options"
            aria-expanded={showNewChatMenu}
            className="rounded-db-md border border-db-gray-lines bg-white px-2 py-1.5 text-db-gray-text transition-colors hover:bg-db-oat-medium hover:text-db-navy-800"
          >
            <ChevronDownIcon className="h-3 w-3" />
          </button>

          {showNewChatMenu && (
            <div className="absolute left-0 right-0 top-full z-50 mt-1 rounded-db-md border border-db-gray-lines bg-white p-1 shadow-db-md animate-in fade-in-0 zoom-in-95">
              <button
                type="button"
                data-testid="new-incognito-chat-button"
                onClick={handleNewIncognitoChat}
                disabled={!canCreateIncognito || createIncognito.isPending}
                className={cn(
                  'flex w-full items-center gap-2 rounded-sm px-2 py-1.5 text-[13px] font-medium text-db-navy-800 transition-colors',
                  canCreateIncognito ? 'hover:bg-db-oat-medium' : 'cursor-not-allowed opacity-50',
                )}
              >
                <EyeOffIcon className="h-4 w-4" />
                <span>New Incognito Chat</span>
                {!canCreateIncognito && (
                  <span className="ml-auto text-xs text-db-gray-text">
                    ({sessionStatus?.maxChats}/{sessionStatus?.maxChats})
                  </span>
                )}
              </button>
            </div>
          )}
        </div>
        {onToggleCollapsed && (
          <button
            type="button"
            onClick={onToggleCollapsed}
            aria-label="Collapse sidebar"
            title="Collapse sidebar"
            className="rounded p-1.5 text-db-gray-text hover:bg-db-oat-medium hover:text-db-navy-800"
          >
            <ChevLeftSmall className="h-3.5 w-3.5" />
          </button>
        )}
      </div>

      {/* Search */}
      <div className="px-2.5 pb-2">
        <ChatSearchInput
          value={searchQuery}
          onChange={onSearchQueryChange}
          placeholder="Search chats"
        />
      </div>

      {/* Status filter tabs */}
      <div className="flex gap-0.5 px-2.5 pb-1.5">
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

      {/* Chat list */}
      <div data-testid="chat-list" className="flex-1 overflow-y-auto px-1.5 py-1">
        {showChatListLoading ? (
          <div data-testid="chat-list-loading" className="p-4 text-center text-xs text-db-gray-text">
            Loading chats...
          </div>
        ) : filteredChats.length === 0 ? (
          <div data-testid="chat-list-empty" className="p-4 text-center text-xs text-db-gray-text">
            {searchQuery
              ? 'No matching chats'
              : statusFilter === 'archived'
              ? 'No archived chats'
              : 'No chats yet'}
          </div>
        ) : (
          filteredChats.map((chat) => (
            <ChatListItem
              key={chat.id}
              chat={chat}
              isSelected={chat.id === currentChatId}
              onClick={() => onSelectChat(chat.id)}
              onHover={!chat.isDraft && onHoverChat ? () => onHoverChat(chat.id) : undefined}
              onRename={
                !chat.isDraft && onRenameChat ? (title) => onRenameChat(chat.id, title) : undefined
              }
              onArchive={
                !chat.isDraft && onArchiveChat ? () => onArchiveChat(chat.id) : undefined
              }
              onRestore={
                !chat.isDraft && onRestoreChat ? () => onRestoreChat(chat.id) : undefined
              }
              onDelete={!chat.isDraft && onDeleteChat ? () => onDeleteChat(chat.id) : undefined}
              onExport={!chat.isDraft && onExportChat ? () => onExportChat(chat.id) : undefined}
            />
          ))
        )}
      </div>

      {/* Incognito chats section */}
      <IncognitoSection
        currentChatId={currentChatId}
        onSelectChat={onSelectChat}
        onHoverChat={onHoverChat}
        enabled={enableAuxQueries}
      />

      {/* Active Jobs Indicator */}
      <div className="border-t border-db-gray-lines px-2 py-1.5">
        <ActiveJobsIndicator onNavigateToChat={onSelectChat} enabled={enableAuxQueries} />
      </div>

      {/* Navigation Links */}
      {(showAgentsLink || showTemplatesLink) && (
        <div className="flex flex-col gap-px border-t border-db-gray-lines px-1.5 py-1.5">
          {showAgentsLink && (
            <SidebarNavButton
              icon={<AgentIcon className="h-3.5 w-3.5" />}
              label="Agents"
              active={onAgentsRoute}
              onClick={() => navigate('/agents')}
            />
          )}
          {showTemplatesLink && (
            <SidebarNavButton
              icon={<TemplateIcon className="h-3.5 w-3.5" />}
              label="Templates"
              active={onTemplatesRoute}
              onClick={() => navigate('/templates')}
            />
          )}
        </div>
      )}

      {/* User Profile - shown at bottom of sidebar */}
      <UserProfile />
    </aside>
  );
}

interface SidebarNavButtonProps {
  icon: React.ReactNode;
  label: string;
  active: boolean;
  onClick: () => void;
}

function SidebarNavButton({ icon, label, active, onClick }: SidebarNavButtonProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        'flex w-full items-center gap-2.5 rounded px-2.5 py-2 text-[13px] font-medium transition-colors',
        active
          ? 'bg-db-oat-medium text-db-navy-800'
          : 'text-db-gray-text hover:bg-db-oat-medium hover:text-db-navy-800',
      )}
    >
      {icon}
      {label}
    </button>
  );
}

function CollapsedAvatar() {
  return (
    <div className="mb-3 flex h-7 w-7 items-center justify-center rounded-full bg-db-lava-600 text-[10px] font-bold text-white">
      U
    </div>
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
        'flex-1 rounded px-1 py-1 text-[11px] font-medium capitalize transition-colors',
        isActive
          ? 'bg-db-oat-medium text-db-navy-800'
          : 'bg-transparent text-db-gray-text hover:text-db-navy-800',
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
  onHover?: () => void;
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
  onHover,
  onRename,
  onArchive,
  onRestore,
  onDelete,
  onExport,
}: ChatListItemProps) {
  const [showMenu, setShowMenu] = React.useState(false);
  const menuRef = React.useRef<HTMLDivElement>(null);
  const buttonRef = React.useRef<HTMLButtonElement>(null);

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

  React.useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && showMenu) setShowMenu(false);
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

  // Compact relative time for the right-aligned mono timestamp
  const timestamp = chat.updatedAt ? formatRelativeShort(chat.updatedAt) : '';

  return (
    <div className="group relative">
      <button
        data-testid={`chat-item-${chat.id}`}
        onClick={onClick}
        onMouseEnter={onHover}
        className={cn(
          'flex w-full items-center gap-2 rounded px-2 py-1.5 pr-8 text-left transition-colors',
          isSelected ? 'bg-db-oat-medium' : 'hover:bg-db-oat-light',
        )}
      >
        <div className="min-w-0 flex-1">
          <div
            className={cn(
              'truncate text-[12px] text-db-navy-800',
              isSelected ? 'font-medium' : 'font-normal',
              !chat.title && 'italic text-db-gray-text',
            )}
          >
            {chat.title || 'New chat...'}
          </div>
          {(isDraft || isArchived) && (
            <div className="mt-0.5 flex items-center gap-1.5">
              {isDraft && (
                <span className="rounded bg-db-oat-medium px-1.5 py-px text-[10px] text-db-gray-text">
                  Draft
                </span>
              )}
              {isArchived && (
                <span className="rounded bg-db-oat-medium px-1.5 py-px text-[10px] text-db-gray-text">
                  Archived
                </span>
              )}
            </div>
          )}
        </div>
        {timestamp && (
          <span className="shrink-0 font-db-mono text-[10px] text-db-gray-text">
            {timestamp}
          </span>
        )}
      </button>

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
            'absolute right-1 top-1/2 -translate-y-1/2 rounded p-1 text-db-gray-text opacity-0 transition-opacity group-hover:opacity-100 focus:opacity-100',
            'hover:bg-db-oat-medium hover:text-db-navy-800',
          )}
          aria-label="Chat options"
          aria-haspopup="menu"
          aria-expanded={showMenu}
          aria-controls={`chat-menu-${chat.id}`}
        >
          <MoreIcon className="h-3.5 w-3.5" />
        </button>
      )}

      {showMenu && (
        <div
          ref={menuRef}
          id={`chat-menu-${chat.id}`}
          role="menu"
          aria-label="Chat actions"
          className="absolute right-0 top-full z-50 mt-1 w-48 rounded-db-md border border-db-gray-lines bg-white p-1 shadow-db-md animate-in fade-in-0 zoom-in-95"
        >
          {onRename && (
            <ContextMenuItem
              data-testid="chat-action-rename"
              icon={<EditIcon className="h-3.5 w-3.5" />}
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
              icon={<ExportIcon className="h-3.5 w-3.5" />}
              label="Export"
              onClick={() => handleMenuAction(onExport)}
            />
          )}
          {isArchived
            ? onRestore && (
                <ContextMenuItem
                  data-testid="chat-action-restore"
                  icon={<RestoreIcon className="h-3.5 w-3.5" />}
                  label="Unarchive"
                  onClick={() => handleMenuAction(onRestore)}
                />
              )
            : onArchive && (
                <ContextMenuItem
                  data-testid="chat-action-archive"
                  icon={<ArchiveIcon className="h-3.5 w-3.5" />}
                  label="Archive"
                  onClick={() => handleMenuAction(onArchive)}
                />
              )}
          {onDelete && (
            <>
              <div className="my-1 h-px bg-db-gray-lines" />
              <ContextMenuItem
                data-testid="chat-action-delete"
                icon={<TrashIcon className="h-3.5 w-3.5" />}
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

function ContextMenuItem({
  icon,
  label,
  onClick,
  variant = 'default',
  'data-testid': dataTestId,
}: ContextMenuItemProps) {
  return (
    <button
      type="button"
      data-testid={dataTestId}
      onClick={onClick}
      className={cn(
        'flex w-full cursor-pointer items-center gap-2 rounded-sm px-2 py-1.5 text-[13px] font-medium transition-colors',
        variant === 'destructive'
          ? 'text-db-lava-700 hover:bg-db-lava-100'
          : 'text-db-navy-800 hover:bg-db-oat-medium',
      )}
    >
      {icon}
      {label}
    </button>
  );
}

// Format absolute timestamp -> compact "2m" / "1h" / "Yesterday" / "2d" / "MMM D"
function formatRelativeShort(input: string | Date): string {
  const date = typeof input === 'string' ? new Date(input) : input;
  if (Number.isNaN(date.getTime())) return '';
  const diff = Date.now() - date.getTime();
  const min = Math.round(diff / 60_000);
  if (min < 1) return 'now';
  if (min < 60) return `${min}m`;
  const hr = Math.round(min / 60);
  if (hr < 24) return `${hr}h`;
  const day = Math.round(hr / 24);
  if (day === 1) return 'Yesterday';
  if (day < 7) return `${day}d`;
  return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

// =====================================================================
// Icons
// =====================================================================

function PlusIcon({ className }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <path d="M5 12h14" />
      <path d="M12 5v14" />
    </svg>
  );
}

function MoreIcon({ className }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <circle cx="12" cy="12" r="1" />
      <circle cx="12" cy="5" r="1" />
      <circle cx="12" cy="19" r="1" />
    </svg>
  );
}

function EditIcon({ className }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
      <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
    </svg>
  );
}

function ArchiveIcon({ className }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <rect width="20" height="5" x="2" y="3" rx="1" />
      <path d="M4 8v11a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8" />
      <path d="M10 12h4" />
    </svg>
  );
}

function RestoreIcon({ className }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
      <path d="M3 3v5h5" />
    </svg>
  );
}

function ExportIcon({ className }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <polyline points="7 10 12 15 17 10" />
      <line x1="12" x2="12" y1="15" y2="3" />
    </svg>
  );
}

function TrashIcon({ className }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <path d="M3 6h18" />
      <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6" />
      <path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2" />
      <line x1="10" x2="10" y1="11" y2="17" />
      <line x1="14" x2="14" y1="11" y2="17" />
    </svg>
  );
}

function ChevronDownIcon({ className }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <path d="m6 9 6 6 6-6" />
    </svg>
  );
}

function ChevLeftSmall({ className }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <path d="M15 6l-6 6 6 6" />
    </svg>
  );
}

function ChevRightSmall({ className }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <path d="M9 6l6 6-6 6" />
    </svg>
  );
}

function EyeOffIcon({ className }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94" />
      <path d="M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19" />
      <path d="m1 1 22 22" />
      <path d="M14.12 14.12a3 3 0 1 1-4.24-4.24" />
    </svg>
  );
}

function AgentIcon({ className }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <path d="M12 8V4H8" />
      <rect width="16" height="12" x="4" y="8" rx="2" />
      <path d="M2 14h2M20 14h2M15 13v2M9 13v2" />
    </svg>
  );
}

function TemplateIcon({ className }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <polyline points="14 2 14 8 20 8" />
      <line x1="16" x2="8" y1="13" y2="13" />
      <line x1="16" x2="8" y1="17" y2="17" />
      <polyline points="10 9 9 9 8 9" />
    </svg>
  );
}
