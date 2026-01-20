/**
 * AgentMessageWithCitations - Wrapper component that fetches and provides citation data.
 *
 * This component uses the useCitations hook to fetch claims and verification data,
 * then passes them to AgentMessage for rendering.
 */

import * as React from 'react';
import { Message, Source, ResearchPlan, SourceType } from '@/types';
import { useCitations } from '@/hooks/useCitations';
import { AgentMessage } from './AgentMessage';

interface ReasoningSummary {
  planTitle?: string;
  stepsCompleted: number;
  totalSteps: number;
  totalSources: number;
  planIterations?: number;
  observations?: string[];
}

interface AgentMessageWithCitationsProps {
  message: Message;
  sources?: Source[];
  reasoning?: ReasoningSummary;
  plan?: ResearchPlan | null;
  isStreaming?: boolean;
  onRegenerate?: () => void;
  className?: string;
  /** Hide the Sources & Citations section (when shown in ResearchPanel) */
  hideSourcesSection?: boolean;
}

// Helper to validate UUID format (8-4-4-4-12 hex characters)
const isValidUUID = (id: string): boolean =>
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(id);

export function AgentMessageWithCitations({
  message,
  sources: propSources,
  reasoning,
  plan,
  isStreaming = false,
  onRegenerate,
  className,
  hideSourcesSection = false,
}: AgentMessageWithCitationsProps) {
  // Fetch citations for this message
  // Only fetch if message has a valid UUID (not a placeholder like 'streaming-*', 'session-*', etc.)
  const shouldFetchCitations = message.id && isValidUUID(message.id);

  const {
    claims,
    verificationSummary,
  } = useCitations(shouldFetchCitations ? message.id : null);

  // Always enable citation parsing so [Key] markers render as interactive elements
  // Even when claims haven't loaded yet, markers should be clickable (shown as "unresolved")
  const enableCitations = true;

  // Extract unique sources from claims' evidence spans
  // This populates the sources accordion with cited sources
  const extractedSources = React.useMemo(() => {
    if (!claims || claims.length === 0) return [];

    const urlMap = new Map<string, Source>();
    claims.forEach(claim => {
      claim.citations?.forEach(citation => {
        const span = citation.evidenceSpan;
        // Try both paths: source.url (denormalized) and direct sourceUrl property
        const sourceUrl = span?.source?.url ||
                          (span as unknown as { sourceUrl?: string })?.sourceUrl;
        if (sourceUrl && !urlMap.has(sourceUrl)) {
          urlMap.set(sourceUrl, {
            id: span?.source?.id || span?.sourceId || sourceUrl,
            url: sourceUrl,
            title: span?.source?.title || span?.sectionHeading || 'Unknown Source',
            snippet: span?.quoteText?.slice(0, 150) ?? null,
            relevanceScore: span?.relevanceScore ?? null,
            sourceType: ((span?.source as { sourceType?: string })?.sourceType ?? 'web') as SourceType,
            sourceMetadata: (span?.source as { sourceMetadata?: Record<string, unknown> })?.sourceMetadata ?? null,
          });
        }
      });
    });
    return Array.from(urlMap.values());
  }, [claims]);

  // Use prop sources if provided, otherwise use extracted sources from claims
  const sources = propSources && propSources.length > 0 ? propSources : extractedSources;

  return (
    <AgentMessage
      message={message}
      sources={sources}
      reasoning={reasoning}
      plan={plan}
      isStreaming={isStreaming}
      onRegenerate={onRegenerate}
      className={className}
      claims={claims}
      verificationSummary={verificationSummary}
      enableCitations={enableCitations}
      hideSourcesSection={hideSourcesSection}
    />
  );
}

export default AgentMessageWithCitations;
