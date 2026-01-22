/**
 * Module-level singleton store for chat streaming state.
 * This preserves research panel state when switching between chats,
 * preventing the flash/reset that occurs when state is lost during navigation.
 *
 * Design decisions:
 * - Module-level Map, not React state/context -> survives component unmount
 * - LRU eviction with max 10 chats -> bounded memory
 * - Timestamp tracking -> evict truly unused chats
 * - TypeScript interface -> type safety for all state fields
 */

import type { StreamEvent, QueryMode } from '@/types';
import type { VerificationSummary } from '@/types/citation';
import type { StreamingClaim, ToolActivity } from '@/hooks/useStreamingQuery';

type AgentStatus =
  | 'idle'
  | 'classifying'
  | 'planning'
  | 'researching'
  | 'reflecting'
  | 'synthesizing'
  | 'verifying'
  | 'complete'
  | 'error';

interface Plan {
  title?: string;
  reasoning?: string;
  steps: Array<{
    index: number;
    title: string;
    description?: string;
    status: 'pending' | 'in_progress' | 'completed' | 'skipped';
  }>;
}

export interface ChatStreamingSnapshot {
  events: StreamEvent[];
  streamingContent: string;
  currentPlan: Plan | null;
  currentStepIndex: number;
  agentStatus: AgentStatus;
  streamingClaims: StreamingClaim[];
  streamingVerificationSummary: VerificationSummary | null;
  currentQueryMode: QueryMode | null;
  startTime: number | null;
  currentAgent: string | null;
  agentMessageId: string | null;
  toolActivity: ToolActivity | null;
  // Timestamp for LRU eviction
  savedAt: number;
}

// Module-level state (singleton)
const MAX_STORED_CHATS = 10;
const chatStates = new Map<string, ChatStreamingSnapshot>();

/**
 * Save streaming state for a chat.
 * Uses LRU eviction to prevent unbounded memory growth.
 */
export function saveStreamingState(chatId: string, state: ChatStreamingSnapshot): void {
  chatStates.set(chatId, { ...state, savedAt: Date.now() });

  // LRU eviction: remove oldest if over limit
  if (chatStates.size > MAX_STORED_CHATS) {
    let oldestKey: string | null = null;
    let oldestTime = Infinity;

    for (const [key, value] of chatStates) {
      if (value.savedAt < oldestTime) {
        oldestTime = value.savedAt;
        oldestKey = key;
      }
    }

    if (oldestKey) {
      chatStates.delete(oldestKey);
      console.log('[chatStreamingState] Evicted oldest chat:', oldestKey);
    }
  }
}

/**
 * Get saved streaming state for a chat.
 */
export function getStreamingState(chatId: string): ChatStreamingSnapshot | undefined {
  return chatStates.get(chatId);
}

/**
 * Clear streaming state for a chat (e.g., when chat is deleted).
 */
export function clearStreamingState(chatId: string): void {
  const deleted = chatStates.delete(chatId);
  if (deleted) {
    console.log('[chatStreamingState] Cleared state for chat:', chatId);
  }
}

/**
 * Check if we have saved streaming state for a chat.
 */
export function hasStreamingState(chatId: string): boolean {
  return chatStates.has(chatId);
}

/**
 * Get the number of chats with saved state (for debugging).
 */
export function getStoredChatCount(): number {
  return chatStates.size;
}
