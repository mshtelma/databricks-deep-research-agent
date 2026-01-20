import * as React from 'react';
import {
  useFloating,
  autoUpdate,
  offset,
  flip,
  shift,
  useDismiss,
  useInteractions,
  FloatingPortal,
  useTransitionStyles,
} from '@floating-ui/react';
import { Message, Source, ResearchPlan } from '@/types';
import { cn } from '@/lib/utils';
import { Card, CardContent } from '@/components/ui/card';
import { MarkdownRenderer, CitationContext } from '@/components/common';
import { EvidenceCard, SourceGroupedCitations } from '@/components/citations';
import { MessageExportMenu } from './MessageExportMenu';
import type { Claim, VerificationSummary } from '@/types/citation';

interface ReasoningSummary {
  planTitle?: string;
  stepsCompleted: number;
  totalSteps: number;
  totalSources: number;
  planIterations?: number;
  observations?: string[];
}

interface AgentMessageProps {
  message: Message;
  sources?: Source[];
  reasoning?: ReasoningSummary;
  plan?: ResearchPlan | null;
  isStreaming?: boolean;
  onRegenerate?: () => void;
  className?: string;
  /** Claims with citations for this message */
  claims?: Claim[];
  /** Verification summary for this message */
  verificationSummary?: VerificationSummary | null;
  /** Enable claim-level citation display */
  enableCitations?: boolean;
  /** Hide the Sources & Citations section (when shown in ResearchPanel) */
  hideSourcesSection?: boolean;
}

export function AgentMessage({
  message,
  sources = [],
  reasoning,
  plan,
  isStreaming = false,
  onRegenerate,
  className,
  claims = [],
  verificationSummary,
  enableCitations = false,
  hideSourcesSection = false,
}: AgentMessageProps) {
  const [showSources, setShowSources] = React.useState(false);
  const [showReasoning, setShowReasoning] = React.useState(false);
  const [showVerification, setShowVerification] = React.useState(false);
  const [activeCitationKey, setActiveCitationKey] = React.useState<string | null>(null);
  const [popoverClaim, setPopoverClaim] = React.useState<Claim | null>(null);

  // Ref to track popover hide timeout for proper cleanup
  const popoverTimeoutRef = React.useRef<ReturnType<typeof setTimeout> | null>(null);

  // Cleanup popover timeout on unmount to prevent setState on unmounted component
  React.useEffect(() => {
    return () => {
      if (popoverTimeoutRef.current) {
        clearTimeout(popoverTimeoutRef.current);
      }
    };
  }, []);

  const hasReasoning = reasoning || (plan && plan.steps && plan.steps.length > 0);
  const hasCitations = enableCitations && claims.length > 0;

  // Floating UI setup for smart popover positioning
  const isPopoverOpen = popoverClaim !== null;

  const { refs, floatingStyles, context } = useFloating({
    open: isPopoverOpen,
    onOpenChange: (open) => {
      if (!open) {
        setPopoverClaim(null);
        setActiveCitationKey(null);
      }
    },
    placement: 'bottom-start',
    middleware: [
      offset(8),
      flip({ padding: 8, fallbackAxisSideDirection: 'end' }),
      shift({ padding: 8, crossAxis: true }),
    ],
    whileElementsMounted: autoUpdate,
  });

  const dismiss = useDismiss(context, {
    escapeKey: true,
    outsidePress: true,
  });

  const { getFloatingProps } = useInteractions([dismiss]);

  const { isMounted, styles: transitionStyles } = useTransitionStyles(context, {
    duration: 150,
    initial: { opacity: 0, transform: 'scale(0.95)' },
  });

  // Build citation data map for MarkdownRenderer using ALL citation keys
  // This allows multi-marker sentences like "[Arxiv][Arxiv-2]" to resolve correctly
  const citationData = React.useMemo(() => {
    if (!enableCitations || claims.length === 0) return undefined;

    const map = new Map<string, CitationContext>();
    claims.forEach((claim) => {
      // Get all keys: prefer citationKeys array, fallback to single citationKey
      const keys = claim.citationKeys || (claim.citationKey ? [claim.citationKey] : []);
      if (keys.length === 0) return; // Skip claims without citation keys

      // Extract URL from the primary citation's evidence span
      const primaryCitation = claim.citations[0];
      const url = primaryCitation?.evidenceSpan?.source?.url ||
                  (primaryCitation?.evidenceSpan as { sourceUrl?: string })?.sourceUrl;

      // Map ALL keys to the same claim context
      for (const key of keys) {
        map.set(key, {
          claim,
          verdict: claim.verificationVerdict,
          url,
        });
      }
    });
    return map;
  }, [claims, enableCitations]);

  // Handle citation click - open source URL in new tab
  // Uses citationKey to look up the claim
  const handleCitationClick = React.useCallback((citationKey: string) => {
    const context = citationData?.get(citationKey);
    const claim = context?.claim;
    if (!claim) return;

    // Try both paths: source.url (denormalized) and sourceUrl (direct)
    const sourceUrl = claim.citations[0]?.evidenceSpan?.source?.url ||
                      (claim.citations[0]?.evidenceSpan as { sourceUrl?: string })?.sourceUrl;
    if (sourceUrl) {
      // Open source URL in new tab
      window.open(sourceUrl, '_blank', 'noopener,noreferrer');
    }
  }, [citationData]);

  // Handle citation hover - show evidence card popover
  // Uses citationKey to look up the claim, element to position popover
  const handleCitationHover = React.useCallback((
    citationKey: string | null,
    element?: HTMLElement | null
  ) => {
    // Clear any pending hide timeout
    if (popoverTimeoutRef.current) {
      clearTimeout(popoverTimeoutRef.current);
      popoverTimeoutRef.current = null;
    }

    if (citationKey === null) {
      // Mouse left - hide popover (after delay to allow moving to popover)
      popoverTimeoutRef.current = setTimeout(() => {
        setPopoverClaim(null);
        setActiveCitationKey(null);
        popoverTimeoutRef.current = null;
      }, 100);
    } else {
      // Mouse entered - show popover anchored to the citation marker element
      const context = citationData?.get(citationKey);
      const claim = context?.claim;
      if (claim) {
        setActiveCitationKey(citationKey);
        setPopoverClaim(claim);
        // Use the marker element as the floating reference
        if (element) {
          refs.setReference(element);
        }
      }
    }
  }, [citationData, refs]);

  // Note: click-outside and escape key are handled by useDismiss from floating-ui

  // Helper to validate UUID format (8-4-4-4-12 hex characters)
  const isValidMessageId = React.useMemo(() =>
    /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(message.id),
    [message.id]
  );

  return (
    <div data-testid="agent-response" className={cn('flex justify-start', className)}>
      <Card className="max-w-[90%] bg-muted">
        <CardContent className="p-4">
          {/* Export menu in top-right corner - only show when not streaming and has valid ID */}
          {!isStreaming && isValidMessageId && (
            <div className="flex justify-end -mt-1 -mr-1 mb-2">
              <MessageExportMenu
                messageId={message.id}
                hasClaims={claims.length > 0}
              />
            </div>
          )}

          {/* Message content with markdown rendering and citation support */}
          <div className="relative">
            <MarkdownRenderer
              content={message.content}
              enableCitations={enableCitations || !isStreaming}
              // IMPORTANT:
              // - When we have verified claims, force numeric/key parsing so [Arxiv] / [1] markers become interactive.
              // - When we DON'T have claims yet (e.g., deferred persistence / slow DB), use 'auto' so we still
              //   parse numeric markers OR link citations depending on what the model produced.
              citationMode={claims.length > 0 ? 'numeric' : 'auto'}
              citationData={citationData}
              onCitationClick={handleCitationClick}
              onCitationHover={handleCitationHover}
              activeCitationKey={activeCitationKey}
            />
            {isStreaming && (
              <span className="inline-block w-2 h-4 bg-primary animate-pulse ml-1 align-text-bottom" />
            )}

            {/* Evidence card popover with smart positioning */}
            {isMounted && popoverClaim && (
              <FloatingPortal>
                <div
                  ref={refs.setFloating}
                  style={{ ...floatingStyles, ...transitionStyles }}
                  {...getFloatingProps()}
                  className="z-50"
                >
                  <EvidenceCard
                    citation={popoverClaim.citations[0]}
                    claimText={popoverClaim.claimText}
                    verdict={popoverClaim.verificationVerdict}
                    isPopover={true}
                    onClose={() => {
                      setPopoverClaim(null);
                      setActiveCitationKey(null);
                    }}
                  />
                </div>
              </FloatingPortal>
            )}
          </div>

          {/* Sources & Citations - Unified source-centric display */}
          {/* Hidden when hideSourcesSection is true (ResearchPanel shows citations instead) */}
          {hasCitations && !isStreaming && !hideSourcesSection && (
            <div data-testid="sources-citations-section" className="mt-4 pt-4 border-t">
              <button
                data-testid="sources-citations-toggle"
                onClick={() => setShowVerification(!showVerification)}
                aria-expanded={showVerification}
                aria-label="View sources and citations"
                className="text-sm text-muted-foreground hover:text-foreground flex items-center gap-2 w-full"
              >
                <ShieldIcon className="w-4 h-4" />
                <span>Sources & Citations</span>
                {verificationSummary && (
                  <VerificationSummaryBadges summary={verificationSummary} />
                )}
                <ChevronIcon
                  className={cn(
                    'w-4 h-4 transition-transform ml-auto',
                    showVerification && 'rotate-180'
                  )}
                />
              </button>

              {showVerification && (
                <div className="mt-3">
                  {/* Warning if high unsupported/contradicted rate */}
                  {verificationSummary?.warning && (
                    <div data-testid="verification-summary-warning" className="bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-lg p-2 text-xs text-amber-800 dark:text-amber-200 mb-3">
                      Some claims could not be fully verified. Please check the citations for details.
                    </div>
                  )}
                  <SourceGroupedCitations
                    claims={claims}
                    verificationSummary={verificationSummary}
                    onClaimClick={handleCitationClick}
                  />
                </div>
              )}
            </div>
          )}

          {/* Reasoning section */}
          {hasReasoning && !isStreaming && (
            <div className="mt-4 pt-4 border-t">
              <button
                onClick={() => setShowReasoning(!showReasoning)}
                aria-expanded={showReasoning}
                aria-label="View reasoning details"
                className="text-sm text-muted-foreground hover:text-foreground flex items-center gap-1"
              >
                <BrainIcon className="w-4 h-4" />
                View reasoning
                {reasoning && (
                  <span className="text-xs text-muted-foreground ml-1">
                    ({reasoning.stepsCompleted}/{reasoning.totalSteps} steps)
                  </span>
                )}
                <ChevronIcon
                  className={cn(
                    'w-4 h-4 transition-transform',
                    showReasoning && 'rotate-180'
                  )}
                />
              </button>

              {showReasoning && (
                <div className="mt-3 space-y-3">
                  {/* Plan overview */}
                  {plan && (
                    <div className="bg-background/50 p-3 rounded-lg">
                      <h4 className="text-xs font-semibold text-muted-foreground mb-2 flex items-center gap-1">
                        <PlanIcon className="w-3.5 h-3.5" />
                        Research Plan
                        {plan.iteration && plan.iteration > 1 && (
                          <span className="font-normal">(Iteration {plan.iteration})</span>
                        )}
                      </h4>
                      {plan.title && (
                        <p className="text-sm font-medium mb-2">{plan.title}</p>
                      )}
                      {plan.thought && (
                        <p className="text-xs text-muted-foreground mb-3">{plan.thought}</p>
                      )}

                      {/* Steps summary */}
                      <div className="space-y-1.5">
                        {plan.steps.map((step, index) => (
                          <PlanStepItem key={step.id || index} step={step} index={index} />
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Reasoning summary stats */}
                  {reasoning && (
                    <div className="flex flex-wrap gap-3 text-xs">
                      <div className="flex items-center gap-1 text-muted-foreground">
                        <StepsIcon className="w-3.5 h-3.5" />
                        <span>{reasoning.stepsCompleted} steps completed</span>
                      </div>
                      <div className="flex items-center gap-1 text-muted-foreground">
                        <SourcesIcon className="w-3.5 h-3.5" />
                        <span>{reasoning.totalSources} sources used</span>
                      </div>
                      {reasoning.planIterations && reasoning.planIterations > 1 && (
                        <div className="flex items-center gap-1 text-muted-foreground">
                          <RefreshIcon className="w-3.5 h-3.5" />
                          <span>{reasoning.planIterations} plan iterations</span>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Observations */}
                  {reasoning?.observations && reasoning.observations.length > 0 && (
                    <div className="space-y-2">
                      <h4 className="text-xs font-semibold text-muted-foreground flex items-center gap-1">
                        <NotesIcon className="w-3.5 h-3.5" />
                        Key Observations
                      </h4>
                      <ul className="space-y-1">
                        {reasoning.observations.slice(0, 5).map((obs, i) => (
                          <li
                            key={i}
                            className="text-xs text-muted-foreground bg-background/50 p-2 rounded"
                          >
                            {obs.length > 200 ? obs.slice(0, 200) + '...' : obs}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Sources section - only show when no claim-level citations (fallback) */}
          {sources.length > 0 && !hasCitations && (
            <div className={cn('mt-4 pt-4 border-t', hasReasoning && !showReasoning && 'pt-4')}>
              <button
                onClick={() => setShowSources(!showSources)}
                aria-expanded={showSources}
                aria-label={`View ${sources.length} source${sources.length !== 1 ? 's' : ''}`}
                className="text-sm text-muted-foreground hover:text-foreground flex items-center gap-1"
              >
                <SourcesIcon className="w-4 h-4" />
                {sources.length} source{sources.length !== 1 ? 's' : ''}
                <ChevronIcon
                  className={cn(
                    'w-4 h-4 transition-transform',
                    showSources && 'rotate-180'
                  )}
                />
              </button>

              {showSources && (
                <div className="mt-2 space-y-2">
                  {sources.map((source, index) => (
                    <SourceCard key={source.url || index} source={source} index={index + 1} />
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Timestamp and regenerate */}
          {!isStreaming && (
            <div className="flex items-center justify-between mt-2">
              <span className="text-xs text-muted-foreground">
                {message.createdAt ? new Date(message.createdAt).toLocaleTimeString() : ''}
              </span>
              {onRegenerate && (
                <button
                  data-testid="regenerate-response"
                  onClick={onRegenerate}
                  className="text-xs text-muted-foreground hover:text-foreground transition-colors"
                >
                  Regenerate
                </button>
              )}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

interface VerificationSummaryBadgesProps {
  summary: VerificationSummary;
}

function VerificationSummaryBadges({ summary }: VerificationSummaryBadgesProps) {
  return (
    <div data-testid="verification-summary-badges" className="flex items-center gap-1 text-xs">
      {summary.supportedCount > 0 && (
        <span className="text-green-600 bg-green-50 dark:bg-green-900/20 px-1.5 py-0.5 rounded">
          {summary.supportedCount}
        </span>
      )}
      {summary.partialCount > 0 && (
        <span className="text-amber-600 bg-amber-50 dark:bg-amber-900/20 px-1.5 py-0.5 rounded">
          {summary.partialCount}
        </span>
      )}
      {summary.unsupportedCount > 0 && (
        <span className="text-red-600 bg-red-50 dark:bg-red-900/20 px-1.5 py-0.5 rounded">
          {summary.unsupportedCount}
        </span>
      )}
      {summary.contradictedCount > 0 && (
        <span className="text-purple-600 bg-purple-50 dark:bg-purple-900/20 px-1.5 py-0.5 rounded">
          {summary.contradictedCount}
        </span>
      )}
    </div>
  );
}

interface PlanStepItemProps {
  step: ResearchPlan['steps'][0];
  index: number;
}

function PlanStepItem({ step }: PlanStepItemProps) {
  const statusIcons = {
    pending: <CircleIcon className="w-3 h-3 text-muted-foreground" />,
    in_progress: <SpinnerIcon className="w-3 h-3 text-primary animate-spin" />,
    completed: <CheckIcon className="w-3 h-3 text-green-500" />,
    skipped: <MinusIcon className="w-3 h-3 text-muted-foreground" />,
  };

  return (
    <div className="flex items-start gap-2 text-xs">
      <span className="mt-0.5">{statusIcons[step.status]}</span>
      <div className="flex-1 min-w-0">
        <span
          className={cn(
            step.status === 'completed' && 'text-muted-foreground',
            step.status === 'skipped' && 'text-muted-foreground line-through'
          )}
        >
          {step.title}
        </span>
        {step.needsSearch && (
          <span className="ml-1 text-[10px] text-blue-600 dark:text-blue-400">[search]</span>
        )}
      </div>
    </div>
  );
}

function SourceCard({ source, index }: { source: Source; index: number }) {
  return (
    <a
      data-testid="citation"
      href={source.url}
      target="_blank"
      rel="noopener noreferrer"
      className="block p-2 rounded border hover:bg-background transition-colors"
    >
      <div className="flex items-start gap-2">
        <span className="text-xs bg-primary/10 text-primary px-1.5 py-0.5 rounded">
          [{index}]
        </span>
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium truncate">{source.title || source.url}</p>
          {source.snippet && (
            <p className="text-xs text-muted-foreground line-clamp-2 mt-1">
              {source.snippet}
            </p>
          )}
        </div>
      </div>
    </a>
  );
}

// Icons

function ShieldIcon({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10" />
      <path d="m9 12 2 2 4-4" />
    </svg>
  );
}

function BrainIcon({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 4.44-1.54" />
      <path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-4.44-1.54" />
    </svg>
  );
}

function PlanIcon({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <path d="M9 11V6a2 2 0 1 0-4 0v5a2 2 0 0 0 4 0Z" />
      <path d="M9 8h3" />
      <path d="M12 8h3" />
      <path d="M15 8V5a2 2 0 1 1 4 0v6a2 2 0 0 1-4 0v-3" />
      <path d="M5 16v1" />
      <path d="M19 16v1" />
      <path d="M5 20h14" />
    </svg>
  );
}

function SourcesIcon({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20" />
    </svg>
  );
}

function StepsIcon({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <path d="M12 22v-4" />
      <path d="M12 8V4" />
      <path d="M12 18a2 2 0 1 0 0-4 2 2 0 0 0 0 4Z" />
      <path d="M12 8a2 2 0 1 0 0-4 2 2 0 0 0 0 4Z" />
    </svg>
  );
}

function NotesIcon({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <path d="M16 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V8Z" />
      <path d="M15 3v4a2 2 0 0 0 2 2h4" />
      <path d="M10 12h4" />
      <path d="M10 16h4" />
    </svg>
  );
}

function RefreshIcon({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
      <path d="M3 3v5h5" />
      <path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16" />
      <path d="M16 16h5v5" />
    </svg>
  );
}

function ChevronIcon({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <path d="m6 9 6 6 6-6" />
    </svg>
  );
}

function CircleIcon({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <circle cx="12" cy="12" r="10" />
    </svg>
  );
}

function SpinnerIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none">
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="4"
      />
      <path
        className="opacity-75"
        fill="currentColor"
        d="m4 12a8 8 0 0 1 8-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 0 1 4 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
      />
    </svg>
  );
}

function CheckIcon({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <path d="M20 6 9 17l-5-5" />
    </svg>
  );
}

function MinusIcon({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <path d="M5 12h14" />
    </svg>
  );
}
