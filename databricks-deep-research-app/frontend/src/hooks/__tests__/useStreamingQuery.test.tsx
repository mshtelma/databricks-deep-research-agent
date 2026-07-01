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

describe('useStreamingQuery handleSseError terminal-status check', () => {
  let client: QueryClient;

  beforeEach(() => {
    FakeEventSource.instances = [];
    vi.stubGlobal('EventSource', FakeEventSource);
    client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    vi.spyOn(jobsApi, 'submit').mockResolvedValue({
      sessionId: 'session-1',
      status: 'in_progress',
    } as Awaited<ReturnType<typeof jobsApi.submit>>);
    vi.spyOn(jobsApi, 'streamUrl').mockReturnValue('/stream/session-1');
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  // Drive a stream to the >=3-rapid-error trip in handleSseError.
  async function startStreamAndFailThrice() {
    const { result } = renderHook(() => useStreamingQuery('chat-1'), {
      wrapper: makeWrapper(client),
    });
    await act(async () => {
      await result.current.sendQuery({ message: 'q', queryMode: 'deep_research' });
    });
    const source = FakeEventSource.instances[0];
    if (!source) throw new Error('Expected EventSource instance');
    act(() => {
      source.onerror?.(new Event('error'));
      source.onerror?.(new Event('error'));
      source.onerror?.(new Event('error'));
    });
    return result;
  }

  it('suppresses the failure banner when the job actually completed', async () => {
    const getSpy = vi.spyOn(jobsApi, 'get').mockResolvedValue({
      status: 'completed',
    } as Awaited<ReturnType<typeof jobsApi.get>>);
    const invalidateSpy = vi.spyOn(client, 'invalidateQueries');

    const result = await startStreamAndFailThrice();

    await waitFor(() => {
      expect(getSpy).toHaveBeenCalledWith('chat-1', 'session-1');
      expect(result.current.agentStatus).toBe('complete');
    });
    expect(result.current.errorDetails).toBeNull();
    expect(result.current.error).toBeNull();
    expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: ['messages', 'chat-1'] });
  });

  it('shows the recoverable banner when the job is still in progress', async () => {
    vi.spyOn(jobsApi, 'get').mockResolvedValue({
      status: 'in_progress',
    } as Awaited<ReturnType<typeof jobsApi.get>>);

    const result = await startStreamAndFailThrice();

    await waitFor(() => {
      expect(result.current.errorDetails?.errorCode).toBe('CONNECTION_FAILED');
    });
    expect(result.current.errorDetails?.recoverable).toBe(true);
  });

  it('reports a job failure when the job failed', async () => {
    vi.spyOn(jobsApi, 'get').mockResolvedValue({
      status: 'failed',
    } as Awaited<ReturnType<typeof jobsApi.get>>);

    const result = await startStreamAndFailThrice();

    await waitFor(() => {
      expect(result.current.errorDetails?.errorCode).toBe('JOB_FAILED');
    });
    expect(result.current.errorDetails?.recoverable).toBe(false);
  });

  it('falls back to the recoverable banner when status cannot be verified', async () => {
    vi.spyOn(jobsApi, 'get').mockRejectedValue(new Error('network'));

    const result = await startStreamAndFailThrice();

    await waitFor(() => {
      expect(result.current.errorDetails?.errorCode).toBe('CONNECTION_FAILED');
    });
  });
});
