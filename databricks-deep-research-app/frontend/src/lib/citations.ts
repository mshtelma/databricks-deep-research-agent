/**
 * Shared citation-data map construction.
 *
 * One claim may carry several citation keys (multi-marker sentences like
 * "[Arxiv][Arxiv-2]"); every key maps to the same CitationContext so
 * MarkdownRenderer resolves each marker. Extracted from AgentMessage so the
 * agent-surface structured-output cells reuse the exact same construction.
 */

import type { CitationContext } from '@/components/common/MarkdownRenderer';
import type { Claim } from '@/types/citation';

export function buildCitationDataMap(
  claims: Claim[],
): Map<string, CitationContext> | undefined {
  if (claims.length === 0) return undefined;

  const map = new Map<string, CitationContext>();
  claims.forEach((claim) => {
    // Prefer the citationKeys array, fall back to the single citationKey.
    const keys =
      claim.citationKeys || (claim.citationKey ? [claim.citationKey] : []);
    if (keys.length === 0) return;

    // URL from the primary citation's evidence span.
    const primaryCitation = claim.citations[0];
    const url =
      primaryCitation?.evidenceSpan?.source?.url ||
      (primaryCitation?.evidenceSpan as { sourceUrl?: string } | undefined)
        ?.sourceUrl;

    for (const key of keys) {
      map.set(key, {
        claim,
        verdict: claim.verificationVerdict,
        url,
      });
    }
  });
  return map;
}
