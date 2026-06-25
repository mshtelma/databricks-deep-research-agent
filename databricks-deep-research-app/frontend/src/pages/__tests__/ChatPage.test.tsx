import { describe, expect, it } from 'vitest';

import { shouldFetchChatFullForChat } from '../chatPageUtils';

describe('ChatPage draft fetch eligibility', () => {
  it('allows chatFull fetches for a draft once the API chat exists', () => {
    expect(shouldFetchChatFullForChat('chat-1', true, true)).toBe(true);
  });

  it('continues to block fresh local drafts that are not in the API', () => {
    expect(shouldFetchChatFullForChat('chat-1', true, false)).toBe(false);
  });

  it('fetches normal persisted chats', () => {
    expect(shouldFetchChatFullForChat('chat-1', false, false)).toBe(true);
  });
});
