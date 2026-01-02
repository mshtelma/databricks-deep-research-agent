/**
 * Utility for grouping claims by their source for source-centric display.
 */
import type { Claim, SourceMetadata, VerificationVerdict } from '@/types/citation';

/** Verification stats for a source */
export interface SourceStats {
  total: number;
  supported: number;
  partial: number;
  unsupported: number;
  contradicted: number;
}

/** A source with its associated claims */
export interface SourceWithClaims {
  source: SourceMetadata;
  claims: Array<{
    claim: Claim;
    index: number; // Original index for citation markers [N]
  }>;
  /** Aggregate verification stats for this source */
  stats: SourceStats;
}

/**
 * Groups claims by their primary source URL.
 * A claim may have multiple citations, but we group by the primary citation's source.
 *
 * @param claims - Array of claims to group
 * @returns Array of sources with their associated claims, sorted by claim count (most first)
 */
export function groupClaimsBySource(claims: Claim[]): SourceWithClaims[] {
  const sourceMap = new Map<string, SourceWithClaims>();

  claims.forEach((claim, index) => {
    // Get primary citation's source (prefer isPrimary, fall back to first)
    const primaryCitation = claim.citations.find(c => c.isPrimary) ?? claim.citations[0];
    if (!primaryCitation?.evidenceSpan?.source) return;

    const source = primaryCitation.evidenceSpan.source;
    const key = source.url ?? source.id;
    if (!key) return;

    if (!sourceMap.has(key)) {
      sourceMap.set(key, {
        source,
        claims: [],
        stats: { total: 0, supported: 0, partial: 0, unsupported: 0, contradicted: 0 },
      });
    }

    const group = sourceMap.get(key)!;
    group.claims.push({ claim, index });
    group.stats.total++;

    // Update verdict counts
    if (claim.verificationVerdict) {
      const verdict = claim.verificationVerdict as keyof Omit<SourceStats, 'total'>;
      if (verdict in group.stats) {
        group.stats[verdict]++;
      }
    }
  });

  // Sort by number of claims (most claims first)
  return Array.from(sourceMap.values()).sort(
    (a, b) => b.claims.length - a.claims.length
  );
}

/**
 * Get a display-friendly domain name from a URL.
 *
 * @param url - The URL to extract domain from
 * @returns Domain name or truncated URL if parsing fails
 */
export function getDomainFromUrl(url: string | null): string {
  if (!url) return 'Unknown Source';
  try {
    return new URL(url).hostname.replace('www.', '');
  } catch {
    return url.length > 50 ? url.slice(0, 47) + '...' : url;
  }
}

/**
 * Calculate the dominant verdict for a source based on its claims.
 *
 * @param stats - Source verification stats
 * @returns The most common verdict, or null if no verdicts
 */
export function getDominantVerdict(stats: SourceStats): VerificationVerdict | null {
  if (stats.total === 0) return null;

  const verdicts: Array<{ verdict: VerificationVerdict; count: number }> = [
    { verdict: 'supported', count: stats.supported },
    { verdict: 'partial', count: stats.partial },
    { verdict: 'unsupported', count: stats.unsupported },
    { verdict: 'contradicted', count: stats.contradicted },
  ];

  const nonZero = verdicts.filter(v => v.count > 0);
  if (nonZero.length === 0) return null;

  return nonZero.reduce((max, curr) =>
    curr.count > max.count ? curr : max
  ).verdict;
}
