/**
 * Shared event filtering for display.
 * Used by EventFeed and CenteredActivityPanel.
 */

import type { StreamEvent } from '@/types';

/** Result data from a tool_result merged into its matching tool_call. */
export interface MergedResultData {
  sourcesAdded: number;
  sourcesCrawled: number;
  resultPreview: string;
}

/**
 * Pair tool_result data with matching tool_call events (FIFO by toolName).
 *
 * The framework's ReactLoop reassembles results in original tool_call order,
 * so the Nth ToolResultEvent for a given toolName always corresponds to the
 * Nth ToolCallEvent for that toolName.
 *
 * Returns a new array — original events are not mutated.
 */
function mergeToolEvents(events: StreamEvent[]): StreamEvent[] {
  // Phase 1: Build FIFO queues of tool_result data keyed by toolName
  const resultQueues = new Map<string, MergedResultData[]>();
  for (const event of events) {
    if (event.eventType !== 'tool_result') continue;
    const tr = event as unknown as {
      toolName?: string;
      tool_name?: string;
      sourcesAdded?: number;
      sources_added?: number;
      sourcesCrawled?: number;
      sources_crawled?: number;
      resultPreview?: string;
      result_preview?: string;
    };
    const name = tr.toolName ?? tr.tool_name ?? '';
    if (!resultQueues.has(name)) resultQueues.set(name, []);
    resultQueues.get(name)!.push({
      sourcesAdded: tr.sourcesAdded ?? tr.sources_added ?? 0,
      sourcesCrawled: tr.sourcesCrawled ?? tr.sources_crawled ?? 0,
      resultPreview: tr.resultPreview ?? tr.result_preview ?? '',
    });
  }

  // Phase 2: Create annotated copies of tool_call events
  const callCounters = new Map<string, number>();
  return events.map((event) => {
    if (event.eventType !== 'tool_call') return event;
    const tc = event as unknown as { toolName?: string; tool_name?: string };
    const name = tc.toolName ?? tc.tool_name ?? '';
    const idx = callCounters.get(name) ?? 0;
    callCounters.set(name, idx + 1);

    const queue = resultQueues.get(name);
    if (queue && idx < queue.length) {
      return { ...event, _mergedResult: queue[idx] };
    }
    return event; // No result yet — event is "in-flight"
  });
}

/**
 * Filter to keep only interesting events for display.
 * tool_result data is merged into its matching tool_call before filtering.
 */
export function filterInterestingEvents(events: StreamEvent[]): StreamEvent[] {
  const merged = mergeToolEvents(events);
  return merged.filter((event) => {
    // Always show errors
    if (event.eventType === 'error') return true;

    // Skip synthesis_progress (too noisy during writing)
    if (event.eventType === 'synthesis_progress') return false;

    // tool_result: suppressed — data is merged into the matching tool_call
    if (event.eventType === 'tool_result') return false;

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
