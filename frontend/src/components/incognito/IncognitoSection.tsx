/**
 * Incognito chats section for the sidebar.
 *
 * Displays incognito chats separately from regular chats,
 * with session expiry countdown and distinct styling.
 */

import * as React from 'react'
import { cn } from '@/lib/utils'
import { useIncognitoChats } from '@/hooks/useIncognitoChats'
import type { Chat } from '@/types'

interface IncognitoSectionProps {
  currentChatId?: string
  onSelectChat: (chatId: string) => void
  onHoverChat?: (chatId: string) => void
  className?: string
}

export function IncognitoSection({
  currentChatId,
  onSelectChat,
  onHoverChat,
  className,
}: IncognitoSectionProps) {
  const { data, isLoading } = useIncognitoChats()

  // Don't render if no incognito chats
  if (!isLoading && (!data || data.items.length === 0)) {
    return null
  }

  return (
    <div className={cn('border-t', className)}>
      {/* Section header */}
      <div className="flex items-center gap-2 px-3 py-2 text-xs text-muted-foreground">
        <EyeOffIcon className="w-3.5 h-3.5" />
        <span className="font-medium">Incognito</span>
        {data?.sessionExpiresAt && (
          <ExpiryCountdown expiresAt={data.sessionExpiresAt} />
        )}
      </div>

      {/* Incognito chat list */}
      <div className="px-2 pb-2 space-y-1">
        {isLoading ? (
          <div className="px-3 py-2 text-xs text-muted-foreground">
            Loading...
          </div>
        ) : (
          data?.items.map((chat) => (
            <IncognitoChatItem
              key={chat.id}
              chat={chat}
              isSelected={chat.id === currentChatId}
              onClick={() => onSelectChat(chat.id)}
              onHover={onHoverChat ? () => onHoverChat(chat.id) : undefined}
            />
          ))
        )}
      </div>
    </div>
  )
}

interface IncognitoChatItemProps {
  chat: Chat
  isSelected: boolean
  onClick: () => void
  onHover?: () => void
}

function IncognitoChatItem({
  chat,
  isSelected,
  onClick,
  onHover,
}: IncognitoChatItemProps) {
  return (
    <button
      type="button"
      data-testid={`incognito-chat-${chat.id}`}
      onClick={onClick}
      onMouseEnter={onHover}
      className={cn(
        'w-full text-left p-2.5 rounded-lg transition-colors',
        'hover:bg-amber-100/50 dark:hover:bg-amber-900/30',
        isSelected && 'bg-amber-100 dark:bg-amber-900/40',
        'border border-transparent',
        isSelected && 'border-amber-300 dark:border-amber-700'
      )}
    >
      <div className="flex items-center gap-2">
        <EyeOffIcon className="w-3.5 h-3.5 text-amber-600 dark:text-amber-400 shrink-0" />
        <p
          className={cn(
            'font-medium truncate text-sm',
            !chat.title && 'italic text-muted-foreground'
          )}
        >
          {chat.title || 'New incognito chat...'}
        </p>
      </div>
    </button>
  )
}

interface ExpiryCountdownProps {
  expiresAt: string
}

function ExpiryCountdown({ expiresAt }: ExpiryCountdownProps) {
  const [timeLeft, setTimeLeft] = React.useState(() =>
    formatTimeLeft(new Date(expiresAt))
  )

  React.useEffect(() => {
    const timer = setInterval(() => {
      setTimeLeft(formatTimeLeft(new Date(expiresAt)))
    }, 60000) // Update every minute

    return () => clearInterval(timer)
  }, [expiresAt])

  return (
    <span className="ml-auto text-amber-600 dark:text-amber-400">
      {timeLeft}
    </span>
  )
}

function formatTimeLeft(expiresAt: Date): string {
  const now = new Date()
  const diffMs = expiresAt.getTime() - now.getTime()

  if (diffMs <= 0) {
    return 'expiring...'
  }

  const diffMinutes = Math.floor(diffMs / (1000 * 60))

  if (diffMinutes < 1) {
    return '<1m'
  } else if (diffMinutes < 60) {
    return `${diffMinutes}m`
  } else {
    const hours = Math.floor(diffMinutes / 60)
    const mins = diffMinutes % 60
    return mins > 0 ? `${hours}h ${mins}m` : `${hours}h`
  }
}

function EyeOffIcon({ className }: { className?: string }) {
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
      <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94" />
      <path d="M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19" />
      <path d="m1 1 22 22" />
      <path d="M14.12 14.12a3 3 0 1 1-4.24-4.24" />
    </svg>
  )
}

export default IncognitoSection
