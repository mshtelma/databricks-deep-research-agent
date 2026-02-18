/**
 * usePlanReview - Hook for plan review workflow via SSE.
 *
 * Features (T045):
 * - Listen for PlanReviewEvent via SSE
 * - handleApprove() - approve plan as-is
 * - handleApproveWithEdits(editedPlan) - approve with modifications
 * - handleReject(reason) - reject plan
 * - Timer state for countdown
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import type {
  PlanWithSources,
  PlanReviewEvent,
  EditedPlan,
  PlanReviewAction,
} from '@/types/dataSources';

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api/v1';

interface UsePlanReviewOptions {
  /** Session ID for the research session */
  sessionId: string | null;
  /** Whether plan review is enabled */
  enabled?: boolean;
  /** Callback when plan review is needed */
  onPlanReviewNeeded?: (plan: PlanWithSources) => void;
  /** Callback when plan review is resolved */
  onPlanReviewResolved?: () => void;
}

export interface UsePlanReviewReturn {
  /** Current plan awaiting review */
  planForReview: PlanWithSources | null;
  /** Whether we're waiting for user review */
  isReviewPending: boolean;
  /** Remaining time before auto-approve */
  remainingTime: number;
  /** Timeout duration from server */
  timeoutSeconds: number;
  /** Approve plan as-is */
  handleApprove: () => Promise<void>;
  /** Approve plan with edits */
  handleApproveWithEdits: (editedPlan: EditedPlan) => Promise<void>;
  /** Reject plan */
  handleReject: (reason?: string) => Promise<void>;
  /** Clear current review state */
  clearReview: () => void;
  /** Handle incoming plan review event */
  handlePlanReviewEvent: (event: PlanReviewEvent) => void;
  /** Error if any */
  error: string | null;
  /** Whether an action is in progress */
  isLoading: boolean;
}

export function usePlanReview({
  sessionId,
  enabled = true,
  onPlanReviewNeeded,
  onPlanReviewResolved,
}: UsePlanReviewOptions): UsePlanReviewReturn {
  const [planForReview, setPlanForReview] = useState<PlanWithSources | null>(null);
  const [timeoutSeconds, setTimeoutSeconds] = useState(30);
  const [remainingTime, setRemainingTime] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);

  const isReviewPending = planForReview !== null;

  // Clear timer
  const clearTimer = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  // Start countdown timer
  const startTimer = useCallback(
    (seconds: number) => {
      clearTimer();
      setRemainingTime(seconds);

      timerRef.current = setInterval(() => {
        setRemainingTime((prev) => {
          if (prev <= 1) {
            clearTimer();
            return 0;
          }
          return prev - 1;
        });
      }, 1000);
    },
    [clearTimer]
  );

  // Clear review state
  const clearReview = useCallback(() => {
    clearTimer();
    setPlanForReview(null);
    setRemainingTime(0);
    setError(null);
    onPlanReviewResolved?.();
  }, [clearTimer, onPlanReviewResolved]);

  // Send plan review response to server
  const sendResponse = useCallback(
    async (action: PlanReviewAction, data?: { editedPlan?: EditedPlan; reason?: string }) => {
      if (!sessionId) {
        setError('No session ID available');
        return;
      }

      setIsLoading(true);
      setError(null);

      try {
        const response = await fetch(
          `${API_BASE_URL}/research/${sessionId}/plan-review`,
          {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              action,
              edited_plan: data?.editedPlan,
              reason: data?.reason,
            }),
          }
        );

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          throw new Error(
            errorData.message || `Failed to submit plan review: ${response.status}`
          );
        }

        clearReview();
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
        throw err;
      } finally {
        setIsLoading(false);
      }
    },
    [sessionId, clearReview]
  );

  // Approve plan as-is
  const handleApprove = useCallback(async () => {
    await sendResponse('approve');
  }, [sendResponse]);

  // Approve plan with edits
  const handleApproveWithEdits = useCallback(
    async (editedPlan: EditedPlan) => {
      await sendResponse('approve_with_edits', { editedPlan });
    },
    [sendResponse]
  );

  // Reject plan
  const handleReject = useCallback(
    async (reason?: string) => {
      await sendResponse('reject', { reason });
    },
    [sendResponse]
  );

  // Handle incoming plan review event
  const handlePlanReviewEvent = useCallback(
    (event: PlanReviewEvent) => {
      setPlanForReview(event.plan);
      setTimeoutSeconds(event.timeoutSeconds);
      startTimer(event.timeoutSeconds);
      onPlanReviewNeeded?.(event.plan);
    },
    [startTimer, onPlanReviewNeeded]
  );

  // Connect to SSE for plan review events
  useEffect(() => {
    if (!sessionId || !enabled) {
      return;
    }

    // Note: This assumes the main research stream handles plan_review events
    // and calls handlePlanReviewEvent. In a real implementation, you might
    // need to connect to a dedicated SSE endpoint or parse events from the
    // main research stream.

    // For now, we expose handlePlanReviewEvent via the hook's interface
    // so it can be called from the streaming query handler.

    return () => {
      clearTimer();
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
    };
  }, [sessionId, enabled, clearTimer]);

  // Clean up on unmount
  useEffect(() => {
    return () => {
      clearTimer();
    };
  }, [clearTimer]);

  return {
    planForReview,
    isReviewPending,
    remainingTime,
    timeoutSeconds,
    handleApprove,
    handleApproveWithEdits,
    handleReject,
    clearReview,
    handlePlanReviewEvent,
    error,
    isLoading,
  };
}

/**
 * Helper to check if an event is a plan review event.
 */
export function isPlanReviewEvent(event: unknown): event is PlanReviewEvent {
  return (
    typeof event === 'object' &&
    event !== null &&
    'eventType' in event &&
    (event as { eventType: unknown }).eventType === 'plan_review'
  );
}

/**
 * Parse plan review event from SSE data.
 */
export function parsePlanReviewEvent(data: unknown): PlanReviewEvent | null {
  if (!isPlanReviewEvent(data)) {
    return null;
  }
  return data;
}

export default usePlanReview;
