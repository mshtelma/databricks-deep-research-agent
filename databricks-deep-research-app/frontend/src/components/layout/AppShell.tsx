/**
 * AppShell — left sidebar (chat list + Agents/Templates nav) + main content.
 *
 * Used by `/agents`, `/designer/:id`, and `/templates` routes so they share
 * the same chrome as `/chat`. ChatPage manages its own sidebar inline because
 * it owns more local state (streaming, dialogs); pages that don't need that
 * state can wrap their content in <AppShell>{children}</AppShell>.
 */

import * as React from 'react';
import { useNavigate } from 'react-router-dom';
import { ChatSidebar } from '@/components/chat/ChatSidebar';
import { useChats } from '@/hooks';
import {
  useUpdateChat,
  useDeleteChat,
  useRestoreChat,
  useExportChat,
} from '@/hooks/useChats';
import type { Chat, ChatStatus } from '@/types';

const COLLAPSED_KEY = 'chatSidebarCollapsed';

interface AppShellProps {
  children: React.ReactNode;
}

export function AppShell({ children }: AppShellProps): React.ReactElement {
  const navigate = useNavigate();

  const [collapsed, setCollapsed] = React.useState<boolean>(() => {
    if (typeof window === 'undefined') return false;
    try {
      return window.localStorage?.getItem?.(COLLAPSED_KEY) === '1';
    } catch {
      return false;
    }
  });
  const toggleCollapsed = React.useCallback(() => {
    setCollapsed((prev) => {
      const next = !prev;
      try {
        window.localStorage.setItem(COLLAPSED_KEY, next ? '1' : '0');
      } catch {
        /* ignore quota / privacy mode */
      }
      return next;
    });
  }, []);

  const [searchQuery, setSearchQuery] = React.useState('');
  const [statusFilter, setStatusFilter] = React.useState<'active' | 'archived' | 'all'>('active');

  const { data: chatsData, isLoading } = useChats();
  const updateChat = useUpdateChat();
  const deleteChat = useDeleteChat();
  const restoreChat = useRestoreChat();
  const exportChat = useExportChat();

  const chats: Chat[] = React.useMemo(() => chatsData?.items ?? [], [chatsData]);

  const handleSelect = React.useCallback(
    (chatId: string) => navigate(`/chat/${chatId}`),
    [navigate],
  );
  const handleNew = React.useCallback(() => navigate('/chat'), [navigate]);

  const handleRename = React.useCallback(
    (chatId: string, newTitle: string) =>
      updateChat.mutate({ chatId, data: { title: newTitle } }),
    [updateChat],
  );
  const handleArchive = React.useCallback(
    (chatId: string) =>
      updateChat.mutate({ chatId, data: { status: 'archived' as ChatStatus } }),
    [updateChat],
  );
  const handleRestore = React.useCallback(
    (chatId: string) => restoreChat.mutate(chatId),
    [restoreChat],
  );
  const handleDelete = React.useCallback(
    (chatId: string) => {
      if (window.confirm('Delete this chat? This cannot be undone.')) {
        deleteChat.mutate(chatId);
      }
    },
    [deleteChat],
  );
  const handleExport = React.useCallback(
    (chatId: string) => {
      exportChat.mutate({ chatId, format: 'markdown' });
    },
    [exportChat],
  );

  return (
    <div className="db-root flex h-screen overflow-hidden bg-db-oat-light font-db-sans text-db-navy-800">
      <ChatSidebar
        chats={chats}
        onSelectChat={handleSelect}
        onNewChat={handleNew}
        onRenameChat={handleRename}
        onArchiveChat={handleArchive}
        onRestoreChat={handleRestore}
        onDeleteChat={handleDelete}
        onExportChat={handleExport}
        searchQuery={searchQuery}
        onSearchQueryChange={setSearchQuery}
        statusFilter={statusFilter}
        onStatusFilterChange={setStatusFilter}
        isLoading={isLoading}
        collapsed={collapsed}
        onToggleCollapsed={toggleCollapsed}
      />
      <main className="flex min-h-0 min-w-0 flex-1 flex-col">{children}</main>
    </div>
  );
}

export default AppShell;
