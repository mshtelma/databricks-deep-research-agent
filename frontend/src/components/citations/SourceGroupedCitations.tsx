/**
 * SourceGroupedCitations - Source-centric citation display
 *
 * Groups claims by their source for a unified view that shows:
 * - Each source with its title and URL
 * - Claims supported by that source with verdict indicators
 * - Per-source verification statistics
 */

import * as React from 'react';
import { cn } from '@/lib/utils';
import type { Claim, VerificationSummary } from '@/types/citation';
import { VerificationBadge } from './VerificationBadge';
import { FloatingEvidenceCard } from './FloatingEvidenceCard';
import {
  groupClaimsBySource,
  getDomainFromUrl,
  type SourceWithClaims,
  type SourceStats,
} from '@/lib/groupClaimsBySource';

interface SourceGroupedCitationsProps {
  claims: Claim[];
  verificationSummary?: VerificationSummary | null;
  /** Handler when clicking a claim citation (by citationKey) */
  onClaimClick?: (citationKey: string) => void;
  className?: string;
}

export const SourceGroupedCitations: React.FC<SourceGroupedCitationsProps> = ({
  claims,
  verificationSummary,
  onClaimClick,
  className,
}) => {
  const [expandedSources, setExpandedSources] = React.useState<Set<string>>(
    () => new Set()
  );

  const groupedSources = React.useMemo(
    () => groupClaimsBySource(claims),
    [claims]
  );

  // Auto-expand first source if there's only one
  React.useEffect(() => {
    if (groupedSources.length === 1) {
      const firstSource = groupedSources[0];
      if (firstSource) {
        const firstKey = firstSource.source.url ?? firstSource.source.id;
        setExpandedSources(new Set([firstKey]));
      }
    }
  }, [groupedSources]);

  const toggleSource = (sourceKey: string) => {
    setExpandedSources((prev) => {
      const next = new Set(prev);
      if (next.has(sourceKey)) {
        next.delete(sourceKey);
      } else {
        next.add(sourceKey);
      }
      return next;
    });
  };

  const handleSourceHeaderClick = (url: string | null, e: React.MouseEvent) => {
    e.stopPropagation();
    if (url) {
      window.open(url, '_blank', 'noopener,noreferrer');
    }
  };

  if (groupedSources.length === 0) {
    return null;
  }

  return (
    <div className={cn('space-y-3', className)}>
      {/* Header with summary */}
      <div className="flex items-center justify-between">
        <h4 className="text-xs font-semibold text-muted-foreground flex items-center gap-2">
          <DocumentStackIcon className="w-4 h-4" />
          Sources & Citations ({groupedSources.length} source
          {groupedSources.length !== 1 ? 's' : ''})
        </h4>
        {verificationSummary && (
          <VerificationSummaryMini summary={verificationSummary} />
        )}
      </div>

      {/* Grouped sources */}
      <div className="space-y-2">
        {groupedSources.map((group) => {
          const sourceKey = group.source.url ?? group.source.id;
          return (
            <SourceGroup
              key={sourceKey}
              group={group}
              isExpanded={expandedSources.has(sourceKey)}
              onToggle={() => toggleSource(sourceKey)}
              onSourceClick={handleSourceHeaderClick}
              onClaimClick={onClaimClick}
            />
          );
        })}
      </div>
    </div>
  );
};

interface SourceGroupProps {
  group: SourceWithClaims;
  isExpanded: boolean;
  onToggle: () => void;
  onSourceClick: (url: string | null, e: React.MouseEvent) => void;
  onClaimClick?: (citationKey: string) => void;
}

function SourceGroup({
  group,
  isExpanded,
  onToggle,
  onSourceClick,
  onClaimClick,
}: SourceGroupProps) {
  const { source, claims, stats } = group;
  const domain = getDomainFromUrl(source.url);

  return (
    <div className="border rounded-lg bg-background/50 overflow-hidden">
      {/* Source header - clickable to expand/collapse */}
      <button
        onClick={onToggle}
        className="w-full flex items-center gap-2 p-3 text-left hover:bg-muted/50 transition-colors"
      >
        <DocumentIcon className="w-4 h-4 text-muted-foreground shrink-0" />
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span
              className="text-sm font-medium truncate hover:underline cursor-pointer text-foreground"
              onClick={(e) => onSourceClick(source.url, e)}
              title={source.url ?? undefined}
            >
              {source.title || domain}
            </span>
            <span className="text-xs text-muted-foreground shrink-0">
              ({claims.length} claim{claims.length !== 1 ? 's' : ''})
            </span>
          </div>
          {source.url && source.title && (
            <span className="text-[10px] text-muted-foreground truncate block">
              {domain}
            </span>
          )}
        </div>
        {/* Source-level verification indicator */}
        <SourceStatsIndicator stats={stats} />
        <ChevronIcon
          className={cn(
            'w-4 h-4 text-muted-foreground transition-transform shrink-0',
            isExpanded && 'rotate-180'
          )}
        />
      </button>

      {/* Expanded claims list */}
      {isExpanded && (
        <div className="px-3 pb-3 pt-1 space-y-1.5 border-t border-border/50">
          {claims.map(({ claim, index }, i) => (
            <ClaimRow
              key={claim.id}
              claim={claim}
              index={index}
              isLast={i === claims.length - 1}
              onClaimClick={onClaimClick}
            />
          ))}
        </div>
      )}
    </div>
  );
}

interface ClaimRowProps {
  claim: Claim;
  index: number;
  isLast: boolean;
  onClaimClick?: (citationKey: string) => void;
}

function ClaimRow({ claim, index, isLast, onClaimClick }: ClaimRowProps) {
  const primaryCitation =
    claim.citations.find((c) => c.isPrimary) ?? claim.citations[0];

  if (!primaryCitation) return null;

  // Use citationKey if available, fallback to index
  const displayKey = claim.citationKey || String(index);

  return (
    <div className="flex items-start gap-2 text-sm">
      {/* Tree connector */}
      <span className="text-muted-foreground text-xs mt-1.5 font-mono shrink-0">
        {isLast ? '\u2514' : '\u251C'}{'\u2500'}
      </span>

      {/* Citation marker with floating evidence card */}
      <FloatingEvidenceCard
        citation={primaryCitation}
        claimText={claim.claimText}
        verdict={claim.verificationVerdict}
        placement="right-start"
      >
        <button
          className={cn(
            'text-xs px-1.5 py-0.5 rounded font-medium shrink-0',
            'bg-primary/10 text-primary hover:bg-primary/20 transition-colors'
          )}
          onClick={(e) => {
            e.stopPropagation();
            if (claim.citationKey) {
              onClaimClick?.(claim.citationKey);
            }
          }}
        >
          [{displayKey}]
        </button>
      </FloatingEvidenceCard>

      {/* Claim text */}
      <span className="flex-1 line-clamp-2 text-muted-foreground">
        &ldquo;{claim.claimText}&rdquo;
      </span>

      {/* Verdict badge */}
      {claim.verificationVerdict && (
        <VerificationBadge
          verdict={claim.verificationVerdict}
          size="sm"
          showLabel={false}
        />
      )}
    </div>
  );
}

function SourceStatsIndicator({ stats }: { stats: SourceStats }) {
  if (stats.total === 0) return null;

  return (
    <div className="flex items-center gap-0.5 text-[10px]">
      {stats.supported > 0 && (
        <span className="text-green-600 dark:text-green-400 bg-green-50 dark:bg-green-900/30 px-1 rounded">
          {stats.supported}
        </span>
      )}
      {stats.partial > 0 && (
        <span className="text-amber-600 dark:text-amber-400 bg-amber-50 dark:bg-amber-900/30 px-1 rounded">
          {stats.partial}
        </span>
      )}
      {stats.unsupported > 0 && (
        <span className="text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-900/30 px-1 rounded">
          {stats.unsupported}
        </span>
      )}
      {stats.contradicted > 0 && (
        <span className="text-purple-600 dark:text-purple-400 bg-purple-50 dark:bg-purple-900/30 px-1 rounded">
          {stats.contradicted}
        </span>
      )}
    </div>
  );
}

function VerificationSummaryMini({
  summary,
}: {
  summary: VerificationSummary;
}) {
  return (
    <div className="flex items-center gap-1.5 text-[10px]">
      <span className="text-muted-foreground">{summary.totalClaims} claims:</span>
      {summary.supportedCount > 0 && (
        <span className="text-green-600 dark:text-green-400">
          {summary.supportedCount} verified
        </span>
      )}
      {summary.warning && (
        <span className="text-amber-600 dark:text-amber-400">needs review</span>
      )}
    </div>
  );
}

// Icons
function DocumentStackIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20" />
    </svg>
  );
}

function DocumentIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z" />
      <polyline points="14,2 14,8 20,8" />
      <line x1="16" y1="13" x2="8" y2="13" />
      <line x1="16" y1="17" x2="8" y2="17" />
    </svg>
  );
}

function ChevronIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="m6 9 6 6 6-6" />
    </svg>
  );
}

SourceGroupedCitations.displayName = 'SourceGroupedCitations';

export default SourceGroupedCitations;
