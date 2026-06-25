import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, renderHook, waitFor } from '@testing-library/react';
import type { ReactNode } from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { jobsApi } from '@/api/client';
import { useStreamingQuery } from '../useStreamingQuery';

class FakeEventSource {
  static instances: FakeEventSource[] = [];

  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  readyState = 1;
  close = vi.fn(() => {
    this.readyState = 2;
  });

  constructor(public readonly url: string) {
    FakeEventSource.instances.push(this);
  }
}

function makeWrapper(client: QueryClient) {
  return function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
  };
}

describe('useStreamingQuery job completion fallback', () => {
  let client: QueryClient;

  beforeEach(() => {
    FakeEventSource.instances = [];
    vi.stubGlobal('EventSource', FakeEventSource);
    client = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it('invalidates chat content when a completed job closes without persistence_completed', async () => {
    const invalidateSpy = vi.spyOn(client, 'invalidateQueries');
    vi.spyOn(jobsApi, 'submit').mockResolvedValue({
      sessionId: 'session-1',
      status: 'in_progress',
    } as Awaited<ReturnType<typeof jobsApi.submit>>);
    vi.spyOn(jobsApi, 'streamUrl').mockReturnValue('/stream/session-1');

    const onStreamComplete = vi.fn();
    const { result } = renderHook(
      () => useStreamingQuery('chat-1', { onStreamComplete }),
      { wrapper: makeWrapper(client) },
    );

    await act(async () => {
      await result.current.sendQuery({
        message: 'write the report',
        queryMode: 'deep_research',
      });
    });

    expect(FakeEventSource.instances).toHaveLength(1);
    const source = FakeEventSource.instances[0];
    if (!source) throw new Error('Expected EventSource instance');

    act(() => {
      source.onmessage?.({
        data: JSON.stringify({ eventType: 'job_completed', status: 'completed' }),
      } as MessageEvent);
    });

    await waitFor(() => {
      expect(onStreamComplete).toHaveBeenCalledTimes(1);
      expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: ['jobs'] });
      expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: ['messages', 'chat-1'] });
      expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: ['chatFull', 'chat-1'] });
      expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: ['chats'] });
    });
    expect(source.close).toHaveBeenCalledTimes(1);
  });
});
