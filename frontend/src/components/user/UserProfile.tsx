/**
 * User profile component for sidebar display.
 * Shows avatar, name, and expandable dropdown with full details.
 */

import { useState } from 'react'
import { cn } from '@/lib/utils'
import { useUserProfile } from '@/hooks/useUserProfile'
import { UserAvatar } from './UserAvatar'
import { UserDropdown } from './UserDropdown'

interface UserProfileProps {
  /** Additional CSS classes */
  className?: string
}

/**
 * Skeleton loader for the profile component.
 */
function UserProfileSkeleton() {
  return (
    <div className="p-3 flex items-center gap-3">
      <div className="w-8 h-8 rounded-full bg-muted animate-pulse" />
      <div className="flex-1 min-w-0">
        <div className="h-4 bg-muted rounded animate-pulse w-24" />
      </div>
    </div>
  )
}

/**
 * Error fallback for the profile component.
 */
function UserProfileError() {
  return (
    <div className="p-3 text-sm text-muted-foreground">
      <span className="flex items-center gap-2">
        <span className="w-8 h-8 rounded-full bg-muted flex items-center justify-center text-xs">
          ?
        </span>
        <span>Signed in</span>
      </span>
    </div>
  )
}

/**
 * User profile display component for the sidebar footer.
 * Shows the authenticated user's avatar, name, and expandable details.
 */
export function UserProfile({ className }: UserProfileProps) {
  const { data: profile, isLoading, error } = useUserProfile()
  const [isOpen, setIsOpen] = useState(false)

  if (isLoading) {
    return (
      <div className={cn('border-t', className)}>
        <UserProfileSkeleton />
      </div>
    )
  }

  if (error || !profile) {
    return (
      <div className={cn('border-t', className)}>
        <UserProfileError />
      </div>
    )
  }

  return (
    <div className={cn('border-t', className)}>
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className={cn(
          'w-full p-3 flex items-center gap-3',
          'hover:bg-accent/50 transition-colors duration-150',
          'focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2'
        )}
        aria-expanded={isOpen}
        aria-controls="user-profile-dropdown"
      >
        <UserAvatar
          userId={profile.userId}
          displayName={profile.displayName}
          email={profile.email}
        />
        <div className="flex-1 text-left min-w-0">
          <p className="text-sm font-medium truncate">
            {profile.displayName || profile.email}
          </p>
        </div>
        <ChevronIcon direction={isOpen ? 'up' : 'down'} />
      </button>

      {isOpen && (
        <div id="user-profile-dropdown">
          <UserDropdown profile={profile} />
        </div>
      )}
    </div>
  )
}

/**
 * Chevron icon for expand/collapse indication.
 */
function ChevronIcon({ direction }: { direction: 'up' | 'down' }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={cn(
        'w-4 h-4 text-muted-foreground transition-transform duration-200',
        direction === 'up' && 'rotate-180'
      )}
    >
      <path d="m6 9 6 6 6-6" />
    </svg>
  )
}
