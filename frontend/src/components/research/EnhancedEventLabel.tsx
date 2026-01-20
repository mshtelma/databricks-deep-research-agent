import { cn } from '@/lib/utils';
import type { StreamEvent, ClaimVerifiedEvent, VerificationSummaryEvent } from '@/types';
import { formatActivityLabel, getActivityColor } from '@/utils/activityLabels';

interface EnhancedEventLabelProps {
  event: StreamEvent;
  className?: string;
}

/**
 * Enhanced event label component with contextual information and badges.
 * Extends the basic activity label formatting with icons and verdict badges.
 */
export function EnhancedEventLabel({ event, className }: EnhancedEventLabelProps) {
  const baseLabel = formatActivityLabel(event);
  const colorClass = getActivityColor(event);

  // Special handling for claim verification events
  if (event.eventType === 'claim_verified') {
    return <ClaimVerifiedLabel event={event as ClaimVerifiedEvent} className={className} />;
  }

  if (event.eventType === 'verification_summary') {
    return <VerificationSummaryLabel event={event as VerificationSummaryEvent} className={className} />;
  }

  return (
    <div className={cn('flex items-center gap-2 text-sm', colorClass, className)}>
      {getEventIcon(event.eventType)}
      <span>{baseLabel}</span>
    </div>
  );
}

interface ClaimVerifiedLabelProps {
  event: ClaimVerifiedEvent;
  className?: string;
}

function ClaimVerifiedLabel({ event, className }: ClaimVerifiedLabelProps) {
  // Handle both camelCase (from by_alias) and snake_case (fallback)
  // Zod validates both cases but doesn't transform, so runtime data keeps original casing
  const rawEvent = event as unknown as Record<string, unknown>;
  const claimText = (event.claimText ?? rawEvent.claim_text ?? '') as string;
  const verdict = (event.verdict ?? rawEvent.verdict ?? 'unknown') as string;
  const truncatedClaim = claimText ? truncate(claimText, 60) : '';

  const verdictInfo = getVerdictInfo(verdict);

  return (
    <div className={cn('flex items-center gap-2 text-sm', className)}>
      <span aria-hidden="true">{verdictInfo.icon}</span>
      <span className="truncate text-muted-foreground">{truncatedClaim}</span>
      <span
        className={cn(
          'px-1.5 py-0.5 rounded text-xs font-medium',
          verdictInfo.badgeClass
        )}
      >
        {verdictInfo.label}
      </span>
    </div>
  );
}

interface VerificationSummaryLabelProps {
  event: VerificationSummaryEvent;
  className?: string;
}

function VerificationSummaryLabel({ event, className }: VerificationSummaryLabelProps) {
  // Handle both camelCase (from by_alias) and snake_case (fallback)
  const rawEvent = event as unknown as Record<string, unknown>;
  const totalClaims = (event.totalClaims ?? rawEvent.total_claims ?? 0) as number;
  const supported = (event.supported ?? rawEvent.supported ?? 0) as number;
  const partial = (event.partial ?? rawEvent.partial ?? 0) as number;
  const unsupported = (event.unsupported ?? rawEvent.unsupported ?? 0) as number;

  return (
    <div className={cn('flex items-center gap-2 text-sm', className)}>
      <span aria-hidden="true">📊</span>
      <span className="text-muted-foreground">
        Verified {totalClaims} claims:
      </span>
      {supported > 0 && (
        <span className="px-1.5 py-0.5 rounded text-xs bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200">
          {supported} supported
        </span>
      )}
      {partial > 0 && (
        <span className="px-1.5 py-0.5 rounded text-xs bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-200">
          {partial} partial
        </span>
      )}
      {unsupported > 0 && (
        <span className="px-1.5 py-0.5 rounded text-xs bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200">
          {unsupported} unsupported
        </span>
      )}
    </div>
  );
}

/** Get verdict styling info */
function getVerdictInfo(verdict: string | undefined): { icon: string; label: string; badgeClass: string } {
  switch (verdict) {
    case 'supported':
      return {
        icon: '\u2705', // checkmark
        label: 'Supported',
        badgeClass: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200',
      };
    case 'partial':
      return {
        icon: '\u26A0\uFE0F', // warning
        label: 'Partial',
        badgeClass: 'bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-200',
      };
    case 'unsupported':
      return {
        icon: '\u274C', // x mark
        label: 'Unsupported',
        badgeClass: 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200',
      };
    case 'contradicted':
      return {
        icon: '\u274E', // cross mark
        label: 'Contradicted',
        badgeClass: 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200',
      };
    default:
      return {
        icon: '\u2753', // question mark
        label: verdict || 'Unknown',
        badgeClass: 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200',
      };
  }
}

/** Get icon for event type */
function getEventIcon(eventType: string): React.ReactNode {
  const iconMap: Record<string, string> = {
    agent_started: '\uD83D\uDE80', // rocket
    agent_completed: '\u2705', // checkmark
    plan_created: '\uD83D\uDCCB', // clipboard
    step_started: '\u25B6\uFE0F', // play
    step_completed: '\u2714\uFE0F', // check
    tool_call: '\uD83D\uDD0D', // search
    tool_result: '\uD83D\uDCE5', // inbox
    reflection_decision: '\uD83E\uDD14', // thinking
    synthesis_started: '\u270D\uFE0F', // writing hand
    synthesis_progress: '\u270D\uFE0F', // writing hand
    research_completed: '\uD83C\uDF89', // party
    error: '\u274C', // x
    claim_verified: '\u2705', // check
    verification_summary: '\uD83D\uDCCA', // chart
    research_started: '\uD83D\uDE80', // rocket
    claim_generated: '\uD83D\uDCA1', // light bulb
    citation_corrected: '\uD83D\uDD27', // wrench
    numeric_claim_detected: '\uD83D\uDD22', // input numbers
    content_revised: '\u270F\uFE0F', // pencil
    persistence_completed: '\uD83D\uDCBE', // floppy disk
  };

  const icon = iconMap[eventType] || '\u2022';
  return <span aria-hidden="true">{icon}</span>;
}

/** Truncate text to maxLength */
function truncate(str: string, maxLength: number): string {
  if (!str) return '';
  if (str.length <= maxLength) return str;
  return str.slice(0, maxLength - 1) + '\u2026';
}
