/**
 * User dropdown component showing expanded profile details.
 */

import { cn } from '@/lib/utils'
import type { UserProfile } from '@/types'

interface UserDropdownProps {
  /** User profile data */
  profile: UserProfile
  /** Additional CSS classes */
  className?: string
}

/**
 * Expandable dropdown showing full user profile details.
 * Animated entry with smooth transitions.
 */
export function UserDropdown({ profile, className }: UserDropdownProps) {
  return (
    <div
      className={cn(
        'px-3 pb-3 pt-1 text-sm text-muted-foreground',
        'animate-in fade-in-0 slide-in-from-top-1 duration-200',
        className
      )}
    >
      <p className="truncate" title={profile.email}>
        {profile.email}
      </p>
      {profile.workspace && (
        <p className="text-xs mt-1 text-muted-foreground/80">
          Workspace: {profile.workspace}
        </p>
      )}
      <p className="text-xs mt-1 text-muted-foreground/60">
        ID: {profile.userId.substring(0, 8)}...
      </p>
    </div>
  )
}
