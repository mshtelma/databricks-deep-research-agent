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
    <div className="flex items-center gap-2.5 p-2.5">
      <div className="h-7 w-7 animate-pulse rounded-full bg-db-oat-medium" />
      <div className="min-w-0 flex-1">
        <div className="h-3.5 w-24 animate-pulse rounded bg-db-oat-medium" />
      </div>
    </div>
  )
}

/**
 * Error fallback for the profile component.
 */
function UserProfileError() {
  return (
    <div className="flex items-center gap-2.5 p-2.5 text-[12px] text-db-gray-text">
      <span className="flex h-7 w-7 items-center justify-center rounded-full bg-db-oat-medium text-[10px] font-bold text-db-gray-text">
        ?
      </span>
      <span>Signed in</span>
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
      <div className={cn('border-t border-db-gray-lines', className)}>
        <UserProfileSkeleton />
      </div>
    )
  }

  if (error || !profile) {
    return (
      <div className={cn('border-t border-db-gray-lines', className)}>
        <UserProfileError />
      </div>
    )
  }

  return (
    <div className={cn('border-t border-db-gray-lines', className)}>
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className={cn(
          'flex w-full items-center gap-2.5 px-3 py-2.5 transition-colors',
          'hover:bg-db-oat-light',
          'focus:outline-none focus-visible:ring-2 focus-visible:ring-db-lava-600 focus-visible:ring-offset-1',
        )}
        aria-expanded={isOpen}
        aria-controls="user-profile-dropdown"
      >
        <UserAvatar
          userId={profile.userId}
          displayName={profile.displayName}
          email={profile.email}
        />
        <div className="min-w-0 flex-1 text-left">
          <p className="truncate text-[12px] font-medium text-db-navy-800">
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
        'h-3 w-3 text-db-gray-text transition-transform duration-200',
        direction === 'up' && 'rotate-180',
      )}
    >
      <path d="m6 9 6 6 6-6" />
    </svg>
  )
}
