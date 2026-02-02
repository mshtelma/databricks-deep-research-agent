/**
 * User avatar component displaying initials with a deterministic background color.
 */

import { cn } from '@/lib/utils'
import { getAvatarColor, getInitials } from '@/utils/avatarColors'

interface UserAvatarProps {
  /** User's unique identifier (used for color generation) */
  userId: string
  /** User's display name */
  displayName: string
  /** User's email address */
  email: string
  /** Avatar size variant */
  size?: 'sm' | 'md' | 'lg'
  /** Additional CSS classes */
  className?: string
}

const sizeClasses = {
  sm: 'w-6 h-6 text-xs',
  md: 'w-8 h-8 text-sm',
  lg: 'w-10 h-10 text-base',
} as const

/**
 * Circular avatar showing user initials with a deterministic background color
 * based on the user's ID.
 */
export function UserAvatar({
  userId,
  displayName,
  email,
  size = 'md',
  className,
}: UserAvatarProps) {
  const backgroundColor = getAvatarColor(userId)
  const initials = getInitials(displayName, email)

  return (
    <div
      className={cn(
        'rounded-full flex items-center justify-center font-medium text-white select-none',
        'transition-transform hover:scale-105',
        sizeClasses[size],
        className
      )}
      style={{ backgroundColor }}
      aria-label={`Avatar for ${displayName || email}`}
    >
      {initials}
    </div>
  )
}
