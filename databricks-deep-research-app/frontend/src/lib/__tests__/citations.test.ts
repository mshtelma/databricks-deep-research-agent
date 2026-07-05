import { describe, expect, it } from 'vitest';

import { buildCitationDataMap } from '../citations';
import type { Claim } from '@/types/citation';

function makeClaim(overrides: Partial<Claim>): Claim {
  return {
    id: 'c1',
    claimText: 'text',
    claimType: 'general',
    confidenceLevel: null,
    positionStart: 0,
    positionEnd: 4,
    verificationVerdict: 'supported',
    verificationReasoning: null,
    abstained: false,
    citations: [],
    corrections: [],
    numericDetail: null,
    citationKey: null,
    citationKeys: null,
    ...overrides,
  };
}

describe('buildCitationDataMap', () => {
  it('maps every citation key of a claim to the same context', () => {
    const claim = makeClaim({
      citationKey: 'Arxiv',
      citationKeys: ['Arxiv', 'Arxiv-2'],
    });
    const map = buildCitationDataMap([claim]);
    expect(map?.get('Arxiv')?.claim).toBe(claim);
    expect(map?.get('Arxiv-2')?.claim).toBe(claim);
    expect(map?.get('Arxiv')?.verdict).toBe('supported');
  });

  it('falls back to the single citationKey', () => {
    const map = buildCitationDataMap([makeClaim({ citationKey: 'K1' })]);
    expect(map?.has('K1')).toBe(true);
  });

  it('skips keyless claims and returns undefined for empty input', () => {
    expect(buildCitationDataMap([])).toBeUndefined();
    const map = buildCitationDataMap([makeClaim({})]);
    expect(map?.size ?? 0).toBe(0);
  });

  it('extracts the primary evidence URL', () => {
    const claim = makeClaim({
      citationKey: 'K1',
      citations: [
        {
          evidenceSpan: { source: { url: 'https://example.com/a' } },
        } as unknown as Claim['citations'][number],
      ],
    });
    const map = buildCitationDataMap([claim]);
    expect(map?.get('K1')?.url).toBe('https://example.com/a');
  });
});
