/**
 * Shared event statistics computation.
 * Used by ActivityTabContent and CenteredActivityPanel.
 */

import type { StreamEvent } from '@/types';

export interface EventStats {
  searchQueries: number;
  sourcesFound: number;
  claimsVerified: number;
  claimsSupported: number;
  // Per-source-type breakdown
  webQueries: number;
  enterpriseQueries: number; // genie + vector_search + KA
  genieQueries: number;
  vectorSearchQueries: number;
  kaQueries: number;
  fileQueries: number;
}

/**
 * Infer the source type from a stream event.
 * Prefers explicit sourceType field, falls back to tool name matching.
 */
export function inferSourceType(event: StreamEvent): string {
  const e = event as unknown as Record<string, unknown>;
  // Prefer explicit sourceType from backend
  if (e.sourceType || e.source_type) return (e.sourceType || e.source_type) as string;
  // Fallback: match by tool name pattern (backward compat)
  const toolName = (e.toolName ?? e.tool_name ?? '') as string;
  if (toolName === 'web_search') return 'web_search';
  if (toolName === 'web_crawl') return 'web_crawl';
  if (toolName.startsWith('query_genie_')) return 'genie';
  if (toolName.startsWith('search_')) return 'vector_search';
  if (toolName.startsWith('ask_')) return 'knowledge_assistant';
  if (toolName === 'file_search') return 'file_search';
  return 'unknown';
}

const ENTERPRISE_TYPES = new Set(['genie', 'vector_search', 'knowledge_assistant']);

/**
 * Compute stats from stream events for display in stats bar.
 * Classifies by source type for web/enterprise breakdown.
 */
export function computeEventStats(events: StreamEvent[]): EventStats {
  let searchQueries = 0;
  let sourcesFound = 0;
  let claimsVerified = 0;
  let claimsSupported = 0;
  let webQueries = 0;
  let enterpriseQueries = 0;
  let genieQueries = 0;
  let vectorSearchQueries = 0;
  let kaQueries = 0;
  let fileQueries = 0;

  for (const event of events) {
    if (event.eventType === 'tool_call') {
      searchQueries++;
      const st = inferSourceType(event);
      if (st === 'web_search' || st === 'web_crawl') {
        webQueries++;
      } else if (st === 'genie') {
        enterpriseQueries++;
        genieQueries++;
      } else if (st === 'vector_search') {
        enterpriseQueries++;
        vectorSearchQueries++;
      } else if (st === 'knowledge_assistant') {
        enterpriseQueries++;
        kaQueries++;
      } else if (st === 'file_search') {
        fileQueries++;
      } else if (ENTERPRISE_TYPES.has(st)) {
        enterpriseQueries++;
      }
    }
    if (event.eventType === 'tool_result') {
      const result = event as unknown as {
        sourcesCrawled?: number;
        sources_crawled?: number;
      };
      sourcesFound += result.sourcesCrawled ?? result.sources_crawled ?? 0;
    }
    if (event.eventType === 'claim_verified') {
      claimsVerified++;
      const claim = event as unknown as { verdict?: string };
      if (claim.verdict === 'supported') claimsSupported++;
    }
  }

  return {
    searchQueries,
    sourcesFound,
    claimsVerified,
    claimsSupported,
    webQueries,
    enterpriseQueries,
    genieQueries,
    vectorSearchQueries,
    kaQueries,
    fileQueries,
  };
}
