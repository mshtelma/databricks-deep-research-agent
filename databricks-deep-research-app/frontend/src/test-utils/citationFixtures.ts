import type { Claim } from '@/types/citation';
import type { CitationContext } from '@/components/common/MarkdownRenderer';

/**
 * Minimal, domain-agnostic {@link Claim} carrying a single citation key.
 * Shared across citation/popover tests so the fixture shape lives in one place.
 */
export function makeClaim(key: string, overrides: Partial<Claim> = {}): Claim {
  return {
    id: `claim-${key}`,
    claimText: 'A verifiable statement.',
    claimType: 'general',
    confidenceLevel: 'high',
    positionStart: 0,
    positionEnd: 0,
    verificationVerdict: 'supported',
    verificationReasoning: null,
    abstained: false,
    citations: [
      {
        evidenceSpan: {
          id: `span-${key}`,
          sourceId: `src-${key}`,
          quoteText: 'Supporting evidence quote.',
          startOffset: null,
          endOffset: null,
          sectionHeading: null,
          relevanceScore: null,
          hasNumericContent: false,
          source: {
            id: `src-${key}`,
            title: 'Example Source',
            url: 'https://example.com/doc',
            author: null,
            publishedDate: null,
            contentType: null,
          },
        },
        confidenceScore: 0.9,
        isPrimary: true,
      },
    ],
    corrections: [],
    numericDetail: null,
    citationKey: key,
    citationKeys: [key],
    ...overrides,
  };
}

/** A {@link CitationContext} (MarkdownRenderer's per-key map value) for one key. */
export function makeCitationContext(key: string): CitationContext {
  const claim = makeClaim(key);
  return {
    claim,
    verdict: claim.verificationVerdict,
    url: claim.citations[0]?.evidenceSpan.source.url ?? undefined,
  };
}
