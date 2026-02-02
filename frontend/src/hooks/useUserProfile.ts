/**
 * Hook for fetching and caching the current user's profile.
 */

import { useQuery } from '@tanstack/react-query'
import { userApi } from '@/api/client'
import type { UserProfile } from '@/types'

/**
 * Fetch and cache the current user's profile information.
 *
 * Profile data is cached for 5 minutes and persisted indefinitely
 * since it rarely changes during a session.
 *
 * @returns Query result with user profile data, loading state, and error
 */
export function useUserProfile() {
  return useQuery<UserProfile>({
    queryKey: ['user', 'profile'],
    queryFn: userApi.getProfile,
    staleTime: 5 * 60 * 1000, // 5 minutes
    gcTime: Infinity, // Never garbage collect
    retry: 1, // Only retry once on failure
    refetchOnWindowFocus: false, // Don't refetch on window focus
  })
}
