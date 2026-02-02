/**
 * Avatar color utilities for deterministic color generation
 * based on user identity.
 */

// Curated color palette for avatar backgrounds
// Colors are chosen for readability with white text
const AVATAR_COLORS = [
  '#3B82F6', // Blue
  '#10B981', // Emerald
  '#F59E0B', // Amber
  '#EF4444', // Red
  '#8B5CF6', // Violet
  '#EC4899', // Pink
  '#06B6D4', // Cyan
  '#F97316', // Orange
] as const

/**
 * Generate a deterministic color based on user ID.
 * Same user always gets the same color across sessions.
 *
 * @param userId - The user's unique identifier
 * @returns A hex color string from the palette
 */
export function getAvatarColor(userId: string): string {
  // Simple hash function to convert string to number
  const hash = userId.split('').reduce((acc, char) => {
    return char.charCodeAt(0) + ((acc << 5) - acc)
  }, 0)

  const index = Math.abs(hash) % AVATAR_COLORS.length
  return AVATAR_COLORS[index] ?? AVATAR_COLORS[0]
}

/**
 * Extract initials from a display name or email.
 *
 * Priority:
 * 1. First and last initials from display name (e.g., "John Doe" -> "JD")
 * 2. First two characters of display name if single word (e.g., "Admin" -> "AD")
 * 3. First two characters of email (e.g., "john@example.com" -> "JO")
 *
 * @param displayName - The user's display name
 * @param email - The user's email address
 * @returns Uppercase initials (1-2 characters)
 */
export function getInitials(displayName: string, email: string): string {
  // Try to extract from display name first
  if (displayName && displayName.trim()) {
    const parts = displayName.trim().split(/\s+/)
    if (parts.length >= 2) {
      // First and last name initials
      const firstPart = parts[0]
      const lastPart = parts[parts.length - 1]
      const firstInitial = firstPart?.[0] ?? ''
      const lastInitial = lastPart?.[0] ?? ''
      return (firstInitial + lastInitial).toUpperCase()
    }
    // Single name - take first two characters
    return displayName.trim().substring(0, 2).toUpperCase()
  }

  // Fall back to email
  if (email && email.trim()) {
    // Take first two characters of email (before @)
    const localPart = email.split('@')[0] ?? ''
    return localPart.substring(0, 2).toUpperCase()
  }

  // Ultimate fallback
  return '??'
}
