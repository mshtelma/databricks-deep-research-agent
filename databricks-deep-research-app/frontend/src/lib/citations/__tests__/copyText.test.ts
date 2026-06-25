import { describe, it, expect } from 'vitest';
import { buildCitationCopyText } from '../copyText';
import type { Citation, EvidenceSpan, SourceMetadata } from '@/types/citation';

function makeSource(over: Partial<SourceMetadata> = {}): SourceMetadata {
  return {
    id: 's1',
    title: null,
    url: null,
    author: null,
    publishedDate: null,
    contentType: null,
    ...over,
  };
}

function makeCitation(
  span: Partial<EvidenceSpan>,
  source: Partial<SourceMetadata> = {}
): Citation {
  return {
    evidenceSpan: {
      id: 'e1',
      sourceId: 's1',
      quoteText: '',
      startOffset: null,
      endOffset: null,
      sectionHeading: null,
      relevanceScore: null,
      hasNumericContent: false,
      source: makeSource(source),
      ...span,
    },
    confidenceScore: null,
    isPrimary: true,
  };
}

describe('buildCitationCopyText', () => {
  it('combines quote, title and url', () => {
    const citation = makeCitation(
      { quoteText: 'Revenue grew 12%.' },
      { title: 'Acme 10-K', url: 'https://example.com/10k' }
    );
    expect(buildCitationCopyText(citation, 'Acme revenue grew.')).toBe(
      '"Revenue grew 12%."\n\nSource: Acme 10-K — https://example.com/10k'
    );
  });

  it('falls back to claim text when there is no quote', () => {
    const citation = makeCitation({ quoteText: '' }, { url: 'https://example.com' });
    expect(buildCitationCopyText(citation, 'A claim without a quote.')).toBe(
      'A claim without a quote.\n\nSource: https://example.com'
    );
  });

  it('omits the source line when neither title nor url is present', () => {
    const citation = makeCitation({ quoteText: 'Just a quote.' });
    expect(buildCitationCopyText(citation)).toBe('"Just a quote."');
  });

  it('returns claim text alone when citation is null', () => {
    expect(buildCitationCopyText(null, 'Only the claim.')).toBe('Only the claim.');
  });

  it('returns an empty string when nothing is available', () => {
    expect(buildCitationCopyText(null)).toBe('');
    expect(buildCitationCopyText(undefined, '   ')).toBe('');
  });
});
