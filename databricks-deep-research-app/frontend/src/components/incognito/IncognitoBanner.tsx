/**
 * Banner component explaining incognito mode.
 *
 * Shown at the top of incognito chats to inform users
 * about the temporary nature of the chat.
 */

import * as React from 'react'
import { cn } from '@/lib/utils'

interface IncognitoBannerProps {
  expiresAt?: string | null
  className?: string
  /** Called when user dismisses the banner */
  onDismiss?: () => void
}

export function IncognitoBanner({
  expiresAt,
  className,
  onDismiss,
}: IncognitoBannerProps) {
  const [isDismissed, setIsDismissed] = React.useState(false)

  const handleDismiss = () => {
    setIsDismissed(true)
    onDismiss?.()
  }

  if (isDismissed) {
    return null
  }

  const expiryText = expiresAt
    ? formatTimeRemaining(new Date(expiresAt))
    : 'when you close your browser'

  return (
    <div
      className={cn(
        'flex items-center gap-3 px-4 py-3',
        'bg-amber-50 dark:bg-amber-900/20',
        'border-b border-amber-200 dark:border-amber-800/50',
        className
      )}
    >
      <EyeOffIcon className="w-5 h-5 text-amber-600 dark:text-amber-400 shrink-0" />

      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium text-amber-800 dark:text-amber-300">
          Incognito Mode
        </p>
        <p className="text-xs text-amber-700 dark:text-amber-400/80">
          This chat will be deleted {expiryText}. Click &quot;Keep Chat&quot; to save it.
        </p>
      </div>

      <button
        type="button"
        onClick={handleDismiss}
        className={cn(
          'p-1 rounded hover:bg-amber-200/50 dark:hover:bg-amber-800/50',
          'text-amber-600 dark:text-amber-400',
          'transition-colors'
        )}
        aria-label="Dismiss banner"
      >
        <CloseIcon className="w-4 h-4" />
      </button>
    </div>
  )
}

/**
 * Format time remaining in human-readable format.
 */
function formatTimeRemaining(expiresAt: Date): string {
  const now = new Date()
  const diffMs = expiresAt.getTime() - now.getTime()

  if (diffMs <= 0) {
    return 'soon'
  }

  const diffMinutes = Math.floor(diffMs / (1000 * 60))

  if (diffMinutes < 1) {
    return 'in less than a minute'
  } else if (diffMinutes === 1) {
    return 'in 1 minute'
  } else if (diffMinutes < 60) {
    return `in ${diffMinutes} minutes`
  } else {
    const hours = Math.floor(diffMinutes / 60)
    return hours === 1 ? 'in 1 hour' : `in ${hours} hours`
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

function CloseIcon({ className }: { className?: string }) {
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
      <path d="M18 6 6 18" />
      <path d="m6 6 12 12" />
    </svg>
  )
}

export default IncognitoBanner
