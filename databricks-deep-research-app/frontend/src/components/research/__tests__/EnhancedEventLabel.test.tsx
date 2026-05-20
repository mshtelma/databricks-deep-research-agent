/**
 * EnhancedEventLabel tests for the 3 new citation-pipeline cards (#7 fix).
 *
 * Before this fix the activity log rendered claim_generated, citation_corrected,
 * and numeric_claim_detected as the generic icon+text fallback. These tests
 * verify each event type renders a dedicated card with its semantic badges.
 */

import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import { EnhancedEventLabel } from '../EnhancedEventLabel';
import type {
  ClaimGeneratedEvent,
  CitationCorrectedEvent,
  NumericClaimDetectedEvent,
} from '@/types';

const TS = '2026-05-19T10:00:00Z';

describe('EnhancedEventLabel — claim_generated', () => {
  it('renders the truncated claim text and a "Generated" badge', () => {
    const event: ClaimGeneratedEvent = {
      eventType: 'claim_generated',
      timestamp: TS,
      claimText: 'Snowflake product revenue reached $2.67 billion in FY24.',
      positionStart: 0,
      positionEnd: 60,
      evidencePreview: 'Snowflake 10-K',
      confidenceLevel: 'high',
    };
    render(<EnhancedEventLabel event={event} />);
    expect(screen.getByText(/Snowflake product revenue/i)).toBeInTheDocument();
    expect(screen.getByText('Generated')).toBeInTheDocument();
  });
});

describe('EnhancedEventLabel — citation_corrected', () => {
  it('renders the correction action with action-coloured badge', () => {
    const event: CitationCorrectedEvent = {
      eventType: 'citation_corrected',
      timestamp: TS,
      claimId: 'claim-42',
      correctionType: 'remove',
      reasoning: 'No matching source span found',
    };
    render(<EnhancedEventLabel event={event} />);
    expect(screen.getByText('Removed')).toBeInTheDocument();
    expect(screen.getByText(/Citation/i)).toBeInTheDocument();
    expect(screen.getByText(/No matching source span/i)).toBeInTheDocument();
  });

  it('renders the "Kept" label for correctionType=keep', () => {
    const event: CitationCorrectedEvent = {
      eventType: 'citation_corrected',
      timestamp: TS,
      claimId: 'claim-7',
      correctionType: 'keep',
      reasoning: null,
    };
    render(<EnhancedEventLabel event={event} />);
    expect(screen.getByText('Kept')).toBeInTheDocument();
  });
});

describe('EnhancedEventLabel — numeric_claim_detected', () => {
  it('shows the numeric value, derivation type, and verification status', () => {
    const event: NumericClaimDetectedEvent = {
      eventType: 'numeric_claim_detected',
      timestamp: TS,
      claimId: 'claim-3',
      rawValue: '$111.2B',
      normalizedValue: '111200000000',
      unit: 'USD',
      derivationType: 'direct',
      qaVerified: true,
    };
    render(<EnhancedEventLabel event={event} />);
    expect(screen.getByText(/111200000000/)).toBeInTheDocument();
    expect(screen.getByText('direct')).toBeInTheDocument();
    expect(screen.getByText('verified')).toBeInTheDocument();
  });

  it('falls back to rawValue when normalizedValue is null', () => {
    const event: NumericClaimDetectedEvent = {
      eventType: 'numeric_claim_detected',
      timestamp: TS,
      claimId: 'claim-9',
      rawValue: '~50K customers',
      normalizedValue: null,
      unit: null,
      derivationType: 'computed',
      qaVerified: false,
    };
    render(<EnhancedEventLabel event={event} />);
    expect(screen.getByText(/~50K customers/)).toBeInTheDocument();
    expect(screen.getByText('computed')).toBeInTheDocument();
    expect(screen.getByText('unverified')).toBeInTheDocument();
  });
});
