/**
 * Hook for managing draft chats in localStorage.
 *
 * Draft chats exist only locally until the first message is successfully
 * processed and persisted to the database. This enables instant "new chat"
 * creation without waiting for database operations.
 */

import { useState, useEffect, useCallback } from 'react';

const DRAFT_STORAGE_KEY = 'deep_research_draft_chats';
const DRAFT_MAX_AGE_MS = 24 * 60 * 60 * 1000; // 24 hours

/**
 * Draft chat that exists only in local storage.
 */
export interface DraftChat {
  id: string;
  title: string | null;
  status: 'active';
  createdAt: string;
  updatedAt: string;
  messageCount: number;
  isDraft: true;
  pendingContent?: string; // Unsent message content for beforeunload warning
}

interface DraftChatState {
  drafts: Record<string, DraftChat>;
}

/**
 * Load drafts from localStorage, filtering out stale entries (>24h old).
 */
function loadDrafts(): Record<string, DraftChat> {
  try {
    const stored = localStorage.getItem(DRAFT_STORAGE_KEY);
    if (!stored) return {};

    const state: DraftChatState = JSON.parse(stored);
    const now = Date.now();

    // Filter out stale drafts (older than 24 hours)
    const validDrafts: Record<string, DraftChat> = {};
    for (const [id, draft] of Object.entries(state.drafts || {})) {
      const createdAt = new Date(draft.createdAt).getTime();
      if (now - createdAt < DRAFT_MAX_AGE_MS) {
        validDrafts[id] = draft;
      }
    }

    return validDrafts;
  } catch {
    return {};
  }
}

/**
 * Save drafts to localStorage.
 */
function saveDrafts(drafts: Record<string, DraftChat>): void {
  try {
    const state: DraftChatState = { drafts };
    localStorage.setItem(DRAFT_STORAGE_KEY, JSON.stringify(state));
  } catch {
    // localStorage may be full or unavailable
    console.warn('Failed to save draft chats to localStorage');
  }
}

/**
 * Hook for managing draft chats.
 *
 * @returns Object with draft management functions and state.
 */
export function useDraftChats() {
  const [drafts, setDrafts] = useState<Record<string, DraftChat>>(() => loadDrafts());

  // Persist to localStorage on change
  useEffect(() => {
    saveDrafts(drafts);
  }, [drafts]);

  /**
   * Create a new draft chat.
   *
   * @returns The newly created draft chat.
   */
  const createDraft = useCallback((): DraftChat => {
    const id = crypto.randomUUID();
    const now = new Date().toISOString();
    const draft: DraftChat = {
      id,
      title: null,
      status: 'active',
      createdAt: now,
      updatedAt: now,
      messageCount: 0,
      isDraft: true,
    };

    setDrafts(prev => ({ ...prev, [id]: draft }));
    return draft;
  }, []);

  /**
   * Remove a draft from local storage.
   * Called when the draft has been persisted to the database.
   *
   * @param id - Draft chat ID to remove.
   */
  const removeDraft = useCallback((id: string): void => {
    setDrafts(prev => {
      const next = { ...prev };
      delete next[id];
      return next;
    });
  }, []);

  /**
   * Update pending content for a draft (used for beforeunload warning).
   *
   * @param id - Draft chat ID.
   * @param pendingContent - Content being typed but not sent.
   */
  const updateDraftContent = useCallback((id: string, pendingContent: string): void => {
    setDrafts(prev => {
      const draft = prev[id];
      if (!draft) return prev;
      return {
        ...prev,
        [id]: { ...draft, pendingContent, updatedAt: new Date().toISOString() },
      };
    });
  }, []);

  /**
   * Check if a chat ID belongs to an active draft.
   *
   * Only considers drafts created within the last 60 seconds as "active".
   * Older drafts are considered stale (likely persisted to DB in a previous session)
   * and should not prevent message fetching.
   *
   * @param id - Chat ID to check.
   * @returns True if the chat is an active draft.
   */
  const isDraft = useCallback((id: string): boolean => {
    const draft = drafts[id];
    if (!draft) return false;

    // Only consider recent drafts as active
    // Older drafts are stale and shouldn't block message fetching
    const IN_FLIGHT_WINDOW_MS = 60000; // 60 seconds
    const createdAt = new Date(draft.createdAt).getTime();
    const isRecent = Date.now() - createdAt < IN_FLIGHT_WINDOW_MS;

    return isRecent;
  }, [drafts]);

  /**
   * Get all drafts as a sorted list (most recent first).
   *
   * @returns Array of draft chats sorted by updatedAt descending.
   */
  const getDraftList = useCallback((): DraftChat[] => {
    return Object.values(drafts).sort(
      (a, b) => new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime()
    );
  }, [drafts]);

  /**
   * Get a specific draft by ID.
   *
   * @param id - Draft chat ID.
   * @returns The draft chat or undefined if not found.
   */
  const getDraft = useCallback((id: string): DraftChat | undefined => {
    return drafts[id];
  }, [drafts]);

  /**
   * Clear stale drafts that should no longer exist.
   * Called when API chats are loaded to sync localStorage with backend.
   *
   * Keeps drafts only if:
   * - Created within the last 60 seconds (in-flight, waiting for persistence)
   *
   * Removes drafts if:
   * - Older than 60 seconds AND not in the API response
   *   (indicates failed persistence or DB was cleaned)
   *
   * @param apiChatIds - Set of chat IDs from the API response.
   */
  const clearStaleDrafts = useCallback((apiChatIds: Set<string>): void => {
    const IN_FLIGHT_WINDOW_MS = 60000; // 60 seconds

    setDrafts(prev => {
      const now = Date.now();
      const cleaned: Record<string, DraftChat> = {};

      for (const [id, draft] of Object.entries(prev)) {
        const createdAt = new Date(draft.createdAt).getTime();
        const isRecent = now - createdAt < IN_FLIGHT_WINDOW_MS;
        const existsInApi = apiChatIds.has(id);

        // Keep draft only if:
        // 1. It's in API (already persisted, will be filtered out by deduplication anyway)
        // 2. OR it's recent (likely in-flight, waiting for persistence)
        if (existsInApi || isRecent) {
          cleaned[id] = draft;
        } else {
          // Remove stale draft (older than 60s and not in API = failed persistence or DB cleaned)
          console.log(`[Drafts] Removing stale draft: ${id} (age: ${Math.round((now - createdAt) / 1000)}s)`);
        }
      }

      return cleaned;
    });
  }, []);

  /**
   * Clear all drafts from localStorage.
   * Useful for manual cleanup or debugging.
   */
  const clearAllDrafts = useCallback((): void => {
    setDrafts({});
  }, []);

  return {
    drafts,
    createDraft,
    removeDraft,
    updateDraftContent,
    isDraft,
    getDraftList,
    getDraft,
    clearStaleDrafts,
    clearAllDrafts,
  };
}
