/**
 * useCitations hook - Manages claim and citation state for messages.
 *
 * Provides:
 * - Fetching claims/citations for a message
 * - Real-time updates via SSE events
 * - Active citation popover state
 * - Citation lookup by position in text
 */

import { useState, useCallback, useMemo, useEffect, useRef } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { snakeToCamel } from '@/utils/caseConversion';
import type {
  Claim,
  Citation,
  VerificationSummary,
  CorrectionMetrics,
  MessageClaimsResponse,
  ClaimVerifiedEvent,
  CitationCorrectedEvent,
  VerificationSummaryEvent,
  VerificationVerdict,
} from '@/types/citation';

// Max retries for polling empty claims (prevents infinite polling)
// Increased from 10 to 30 to handle long-running ReAct synthesis (~5-7 min)
const MAX_CLAIM_POLL_RETRIES = 30;
// Polling interval in milliseconds
const CLAIM_POLL_INTERVAL_MS = 3000;

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api/v1';

interface UseCitationsOptions {
  /** Whether to enable real-time updates */
  enableRealtime?: boolean;
  /** Callback when verification summary updates */
  onVerificationUpdate?: (summary: VerificationSummary) => void;
}

interface UseCitationsReturn {
  /** All claims for the message */
  claims: Claim[];
  /** Verification summary */
  verificationSummary: VerificationSummary | null;
  /** Correction metrics */
  correctionMetrics: CorrectionMetrics | null;
  /** Loading state */
  isLoading: boolean;
  /** Error state */
  error: Error | null;
  /** Currently active citation (for popover) */
  activeCitationId: string | null;
  /** Set the active citation */
  setActiveCitationId: (id: string | null) => void;
  /** Get claim at a specific position in text */
  getClaimAtPosition: (position: number) => Claim | null;
  /** Get all claims within a position range */
  getClaimsInRange: (start: number, end: number) => Claim[];
  /** Get primary citation for a claim */
  getPrimaryCitation: (claimId: string) => Citation | null;
  /** Refetch claims data */
  refetch: () => void;
  /** Handle claim verified SSE event */
  handleClaimVerified: (event: ClaimVerifiedEvent) => void;
  /** Handle citation corrected SSE event */
  handleCitationCorrected: (event: CitationCorrectedEvent) => void;
  /** Handle verification summary SSE event */
  handleVerificationSummary: (event: VerificationSummaryEvent) => void;
}

/**
 * Fetch claims for a message from the API.
 * Converts snake_case keys from Python API to camelCase for TypeScript.
 */
async function fetchMessageClaims(
  messageId: string
): Promise<MessageClaimsResponse> {
  const response = await fetch(`${API_BASE_URL}/messages/${messageId}/claims`);
  if (!response.ok) {
    throw new Error(`Failed to fetch claims: ${response.statusText}`);
  }
  const data = await response.json();
  return snakeToCamel(data) as MessageClaimsResponse;
}

export function useCitations(
  messageId: string | null,
  options: UseCitationsOptions = {}
): UseCitationsReturn {
  const { onVerificationUpdate } = options;
  const queryClient = useQueryClient();

  // Active citation state for popover
  const [activeCitationId, setActiveCitationId] = useState<string | null>(null);

  // Track retry count for polling empty claims
  // Uses ref instead of state to avoid unnecessary re-renders
  const pollRetryCountRef = useRef(0);

  // Reset retry count when messageId changes
  useEffect(() => {
    pollRetryCountRef.current = 0;
  }, [messageId]);

  // Query for claims data with polling support for race condition
  // When claims are empty (persistence not complete), poll every 3s until claims arrive
  const {
    data,
    isLoading,
    error,
    refetch,
  } = useQuery({
    queryKey: ['messageClaims', messageId],
    queryFn: async () => {
      const result = await fetchMessageClaims(messageId!);
      return result;
    },
    enabled: !!messageId,
    staleTime: 30000, // 30 seconds
    // Poll every 3s while claims are empty, stop when claims arrive or max retries reached
    refetchInterval: (query) => {
      const queryData = query.state.data;
      // If we got data but claims array is empty, keep polling
      // This handles race condition where message renders before claims are persisted
      if (queryData && queryData.claims.length === 0 && pollRetryCountRef.current < MAX_CLAIM_POLL_RETRIES) {
        pollRetryCountRef.current += 1;
        return CLAIM_POLL_INTERVAL_MS;
      }
      return false; // Stop polling
    },
  });

  const claims = data?.claims ?? [];
  const verificationSummary = data?.verificationSummary ?? null;
  const correctionMetrics = data?.correctionMetrics ?? null;

  // Create position-based lookup map
  const positionMap = useMemo(() => {
    const map = new Map<number, Claim>();
    for (const claim of claims) {
      // Map each position in the claim's range to the claim
      for (let pos = claim.positionStart; pos < claim.positionEnd; pos++) {
        map.set(pos, claim);
      }
    }
    return map;
  }, [claims]);

  // Create claim ID lookup map
  const claimMap = useMemo(() => {
    return new Map(claims.map((claim) => [claim.id, claim]));
  }, [claims]);

  /**
   * Get claim at a specific position in text
   */
  const getClaimAtPosition = useCallback(
    (position: number): Claim | null => {
      return positionMap.get(position) ?? null;
    },
    [positionMap]
  );

  /**
   * Get all claims within a position range
   */
  const getClaimsInRange = useCallback(
    (start: number, end: number): Claim[] => {
      const found = new Set<Claim>();
      for (let pos = start; pos < end; pos++) {
        const claim = positionMap.get(pos);
        if (claim) {
          found.add(claim);
        }
      }
      return Array.from(found);
    },
    [positionMap]
  );

  /**
   * Get primary citation for a claim
   */
  const getPrimaryCitation = useCallback(
    (claimId: string): Citation | null => {
      const claim = claimMap.get(claimId);
      if (!claim) return null;
      return claim.citations.find((c) => c.isPrimary) ?? claim.citations[0] ?? null;
    },
    [claimMap]
  );

  /**
   * Handle claim_verified SSE event
   * Updates the claim in the cache with the new verification status
   */
  const handleClaimVerified = useCallback(
    (event: ClaimVerifiedEvent) => {
      if (!messageId) return;

      queryClient.setQueryData<MessageClaimsResponse>(
        ['messageClaims', messageId],
        (old) => {
          if (!old) return old;

          const updatedClaims = old.claims.map((claim) => {
            if (claim.id === event.claimId) {
              return {
                ...claim,
                verificationVerdict: event.verdict,
                confidenceLevel: event.confidenceLevel,
                verificationReasoning: event.reasoning,
              };
            }
            return claim;
          });

          // Recalculate summary
          const summary = calculateVerificationSummary(updatedClaims);

          return {
            ...old,
            claims: updatedClaims,
            verificationSummary: summary,
          };
        }
      );
    },
    [messageId, queryClient]
  );

  /**
   * Handle citation_corrected SSE event
   */
  const handleCitationCorrected = useCallback(
    (event: CitationCorrectedEvent) => {
      if (!messageId) return;

      queryClient.setQueryData<MessageClaimsResponse>(
        ['messageClaims', messageId],
        (old) => {
          if (!old) return old;

          const updatedClaims = old.claims.map((claim) => {
            if (claim.id === event.claimId) {
              // Add the correction to the claim's corrections array
              return {
                ...claim,
                corrections: [
                  ...claim.corrections,
                  {
                    id: `correction-${Date.now()}`,
                    correctionType: event.correctionType,
                    originalEvidence: null, // Will be filled on full refetch
                    correctedEvidence: null,
                    reasoning: event.reasoning,
                  },
                ],
              };
            }
            return claim;
          });

          // Recalculate correction metrics
          const metrics = calculateCorrectionMetrics(updatedClaims);

          return {
            ...old,
            claims: updatedClaims,
            correctionMetrics: metrics,
          };
        }
      );
    },
    [messageId, queryClient]
  );

  /**
   * Handle verification_summary SSE event
   */
  const handleVerificationSummary = useCallback(
    (event: VerificationSummaryEvent) => {
      if (!messageId) return;

      const summary: VerificationSummary = {
        totalClaims: event.totalClaims,
        supportedCount: event.supported,
        partialCount: event.partial,
        unsupportedCount: event.unsupported,
        contradictedCount: event.contradicted,
        abstainedCount: event.abstainedCount,
        unsupportedRate: event.totalClaims > 0
          ? event.unsupported / event.totalClaims
          : 0,
        contradictedRate: event.totalClaims > 0
          ? event.contradicted / event.totalClaims
          : 0,
        warning: event.warning,
      };

      queryClient.setQueryData<MessageClaimsResponse>(
        ['messageClaims', messageId],
        (old) => {
          if (!old) return old;
          return {
            ...old,
            verificationSummary: summary,
          };
        }
      );

      onVerificationUpdate?.(summary);
    },
    [messageId, queryClient, onVerificationUpdate]
  );

  // Notify on verification updates
  useEffect(() => {
    if (verificationSummary) {
      onVerificationUpdate?.(verificationSummary);
    }
  }, [verificationSummary, onVerificationUpdate]);

  return {
    claims,
    verificationSummary,
    correctionMetrics,
    isLoading,
    error: error as Error | null,
    activeCitationId,
    setActiveCitationId,
    getClaimAtPosition,
    getClaimsInRange,
    getPrimaryCitation,
    refetch,
    handleClaimVerified,
    handleCitationCorrected,
    handleVerificationSummary,
  };
}

/**
 * Calculate verification summary from claims
 */
function calculateVerificationSummary(claims: Claim[]): VerificationSummary {
  const verdictCounts: Record<VerificationVerdict, number> = {
    supported: 0,
    partial: 0,
    unsupported: 0,
    contradicted: 0,
  };

  let abstainedCount = 0;

  for (const claim of claims) {
    if (claim.abstained) {
      abstainedCount++;
    } else if (claim.verificationVerdict) {
      verdictCounts[claim.verificationVerdict]++;
    }
  }

  const totalClaims = claims.length;

  return {
    totalClaims,
    supportedCount: verdictCounts.supported,
    partialCount: verdictCounts.partial,
    unsupportedCount: verdictCounts.unsupported,
    contradictedCount: verdictCounts.contradicted,
    abstainedCount,
    unsupportedRate: totalClaims > 0
      ? verdictCounts.unsupported / totalClaims
      : 0,
    contradictedRate: totalClaims > 0
      ? verdictCounts.contradicted / totalClaims
      : 0,
    warning:
      verdictCounts.unsupported > 0 || verdictCounts.contradicted > 0,
  };
}

/**
 * Calculate correction metrics from claims
 */
function calculateCorrectionMetrics(claims: Claim[]): CorrectionMetrics {
  const typeCounts = {
    keep: 0,
    replace: 0,
    remove: 0,
    add_alternate: 0,
  };

  let totalCorrections = 0;

  for (const claim of claims) {
    for (const correction of claim.corrections) {
      totalCorrections++;
      typeCounts[correction.correctionType]++;
    }
  }

  const totalClaims = claims.length;

  return {
    totalCorrections,
    keepCount: typeCounts.keep,
    replaceCount: typeCounts.replace,
    removeCount: typeCounts.remove,
    addAlternateCount: typeCounts.add_alternate,
    correctionRate: totalClaims > 0 ? totalCorrections / totalClaims : 0,
  };
}

export default useCitations;
