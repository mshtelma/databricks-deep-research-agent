import { describe, expect, it } from 'vitest';

import {
  deriveShortAgentNameFromPrompt,
  isPromptLikeAgentName,
} from '../agentNaming';

describe('agentNaming', () => {
  it('derives a short agent name from a bootstrap prompt', () => {
    expect(
      deriveShortAgentNameFromPrompt(
        'Use main OfficeQA benchmark treasury chunks vector search create deep treseaury documetns',
      ),
    ).toBe('OfficeQA Treasury Documents Agent');
  });

  it('keeps docs as a meaningful domain term', () => {
    expect(deriveShortAgentNameFromPrompt('Build a treasury docs workflow')).toBe(
      'Treasury Documents Agent',
    );
  });

  it('prefers explicit quoted names in prompts', () => {
    expect(
      deriveShortAgentNameFromPrompt(
        'Create "APAC Revenue Sentinel" with SQL and web research tools',
      ),
    ).toBe('APAC Revenue Sentinel');
  });

  it('detects default or prompt-like generated names', () => {
    expect(isPromptLikeAgentName('Untitled Agent')).toBe(true);
    expect(isPromptLikeAgentName('Create a treasury research agent')).toBe(true);
    expect(isPromptLikeAgentName('Treasury QA Agent')).toBe(false);
  });
});
