/**
 * Shared event filtering for display.
 * Used by EventFeed and CenteredActivityPanel.
 */

import type { StreamEvent } from '@/types';
import { inferSourceType } from './eventStats';

const ENTERPRISE_SOURCE_TYPES = new Set(['genie', 'vector_search', 'knowledge_assistant']);

/**
 * Filter to keep only interesting events for display.
 */
export function filterInterestingEvents(events: StreamEvent[]): StreamEvent[] {
  return events.filter((event) => {
    // Always show errors
    if (event.eventType === 'error') return true;

    // Skip synthesis_progress (too noisy during writing)
    if (event.eventType === 'synthesis_progress') return false;

    // Conditionally show tool_result events
    if (event.eventType === 'tool_result') {
      const result = event as unknown as {
        sourcesCrawled?: number;
        sources_crawled?: number;
        toolName?: string;
        tool_name?: string;
        sourceType?: string;
        source_type?: string;
      };
      const sourcesCrawled = result.sourcesCrawled ?? result.sources_crawled;

      // Always show if pages were crawled
      if (sourcesCrawled != null && sourcesCrawled > 0) return true;

      // Show enterprise tool results (always have meaningful content)
      const sourceType = inferSourceType(event);
      if (ENTERPRISE_SOURCE_TYPES.has(sourceType)) return true;

      // Fallback: show non-web tool results
      const toolName = result.toolName ?? result.tool_name;
      if (toolName && toolName !== 'web_search' && toolName !== 'web_crawl') return true;

      return false;
    }

    // Keep meaningful milestone events
    return [
      'agent_started',
      'agent_completed',
      'plan_created',
      'step_started',
      'step_completed',
      'tool_call',
      'reflection_decision',
      'synthesis_started',
      'research_completed',
      'claim_verified',
      'verification_summary',
    ].includes(event.eventType);
  });
}
