import { describe, expect, it } from 'vitest';

import {
  buildQuerySubmission,
  mergeRunContext,
  runContextActiveCount,
} from '../runContext';

describe('runContext helpers', () => {
  it('mergeRunContext preserves explicit false values', () => {
    expect(
      mergeRunContext(
        { verifySources: true, enablePlanReview: true },
        { verifySources: false, enablePlanReview: false },
      ),
    ).toMatchObject({
      verifySources: false,
      enablePlanReview: false,
    });
  });

  it('buildQuerySubmission includes false run options', () => {
    const submission = buildQuerySubmission({
      message: 'research AI',
      runContext: {
        queryMode: 'deep_research',
        verifySources: false,
        enablePlanReview: false,
        enableCrossSessionMemory: false,
        allowLiveSearch: false,
      },
      surfaceInputs: { ticker: 'NVDA' },
      surfaceAction: 'run',
    });

    expect(submission).toMatchObject({
      message: 'research AI',
      queryMode: 'deep_research',
      verifySources: false,
      enablePlanReview: false,
      enableCrossSessionMemory: false,
      allowLiveSearch: false,
      surfaceInputs: { ticker: 'NVDA' },
      surfaceAction: 'run',
    });
  });

  it('counts values that differ from defaults', () => {
    expect(
      runContextActiveCount(
        { researchDepth: 'extended', verifySources: false, tone: '' },
        { researchDepth: 'auto', verifySources: true, tone: '' },
      ),
    ).toBe(2);
  });
});
