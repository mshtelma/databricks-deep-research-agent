import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, renderHook, waitFor } from '@testing-library/react';
import type { ReactNode } from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ApiError, chatsApi, jobsApi } from '@/api/client';
import type { CompiledSubmission } from '@/lib/surfaceCompile';
import { useSurfacePreviewRun } from '../useSurfacePreviewRun';

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

const COMPILED: CompiledSubmission = {
  query: 'What changed last quarter?',
  surfaceInputs: { region: 'emea' },
  researchDepth: 'light',
  verifySources: true,
};

const PREVIEW_CHAT = {
  id: 'preview-chat-1',
  title: 'Preview run — Test Agent',
} as Awaited<ReturnType<typeof chatsApi.create>>;

describe('useSurfacePreviewRun', () => {
  let client: QueryClient;
  let ensureSavedAgentId: ReturnType<typeof vi.fn>;

  function renderRun() {
    return renderHook(
      () =>
        useSurfacePreviewRun({
          agentId: 'agent-1',
          agentName: 'Test Agent',
          ensureSavedAgentId:
            ensureSavedAgentId as unknown as () => Promise<string | null>,
        }),
      { wrapper: makeWrapper(client) },
    );
  }

  async function startAndAwaitSubmit(result: {
    current: ReturnType<typeof useSurfacePreviewRun>;
  }) {
    act(() => {
      result.current.start('run', COMPILED);
    });
    await waitFor(() => {
      expect(jobsApi.submit).toHaveBeenCalledTimes(1);
    });
  }

  beforeEach(() => {
    FakeEventSource.instances = [];
    vi.stubGlobal('EventSource', FakeEventSource);
    client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    ensureSavedAgentId = vi.fn().mockResolvedValue('agent-1');
    vi.spyOn(chatsApi, 'create').mockResolvedValue(PREVIEW_CHAT);
    vi.spyOn(chatsApi, 'getFull').mockResolvedValue(
      null as unknown as Awaited<ReturnType<typeof chatsApi.getFull>>,
    );
    vi.spyOn(jobsApi, 'submit').mockResolvedValue({
      sessionId: 'session-1',
      status: 'in_progress',
    } as Awaited<ReturnType<typeof jobsApi.submit>>);
    vi.spyOn(jobsApi, 'streamUrl').mockReturnValue('/stream/session-1');
    vi.spyOn(jobsApi, 'cancel').mockResolvedValue({
      sessionId: 'session-1',
      status: 'cancelled',
    } as Awaited<ReturnType<typeof jobsApi.cancel>>);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it('start() saves, creates the preview chat, then submits with the compiled payload', async () => {
    const { result } = renderRun();
    await startAndAwaitSubmit(result);

    // Order: save-first → chat create → job submit.
    const saveOrder = ensureSavedAgentId.mock.invocationCallOrder[0];
    const createOrder = vi.mocked(chatsApi.create).mock.invocationCallOrder[0];
    const submitOrder = vi.mocked(jobsApi.submit).mock.invocationCallOrder[0];
    expect(saveOrder).toBeLessThan(createOrder!);
    expect(createOrder).toBeLessThan(submitOrder!);

    expect(chatsApi.create).toHaveBeenCalledWith({
      title: 'Preview run — Test Agent',
    });
    expect(jobsApi.submit).toHaveBeenCalledWith(
      expect.objectContaining({
        chatId: 'preview-chat-1',
        query: 'What changed last quarter?',
        queryMode: 'deep_research',
        agentId: 'agent-1',
        surfaceInputs: { region: 'emea' },
        researchDepth: 'light',
        verifySources: true,
        turnIntent: 'research',
      }),
    );
    expect(result.current.previewChatId).toBe('preview-chat-1');
    expect(result.current.runState['run']).toMatchObject({
      status: 'running',
      preview: 'real',
      action: 'run',
    });
    expect(result.current.isActive).toBe(true);
  });

  it('reuses the preview chat on a second run', async () => {
    const { result } = renderRun();
    await startAndAwaitSubmit(result);

    // Finish the first run so isActive drops.
    const source = FakeEventSource.instances[0];
    act(() => {
      source?.onmessage?.({
        data: JSON.stringify({
          eventType: 'persistence_completed',
          chatId: 'preview-chat-1',
          messageId: 'm1',
          researchSessionId: 'rs1',
          wasDraft: false,
        }),
      } as MessageEvent);
      source?.onmessage?.({
        data: JSON.stringify({ eventType: 'job_completed', status: 'completed' }),
      } as MessageEvent);
    });
    await waitFor(() => {
      expect(result.current.isActive).toBe(false);
    });

    act(() => {
      result.current.start('run', COMPILED);
    });
    await waitFor(() => {
      expect(jobsApi.submit).toHaveBeenCalledTimes(2);
    });
    expect(chatsApi.create).toHaveBeenCalledTimes(1);
  });

  it('does nothing when the save-first gate fails', async () => {
    ensureSavedAgentId.mockResolvedValue(null);
    const { result } = renderRun();

    act(() => {
      result.current.start('run', COMPILED);
    });
    await waitFor(() => {
      expect(ensureSavedAgentId).toHaveBeenCalledTimes(1);
    });
    expect(chatsApi.create).not.toHaveBeenCalled();
    expect(jobsApi.submit).not.toHaveBeenCalled();
    expect(result.current.runState['run']).toBeUndefined();
    expect(result.current.isActive).toBe(false);
  });

  it('persistence_completed flips the ref to completed with ids and invalidates chatFull', async () => {
    const invalidateSpy = vi.spyOn(client, 'invalidateQueries');
    const { result } = renderRun();
    await startAndAwaitSubmit(result);

    const source = FakeEventSource.instances[0];
    act(() => {
      source?.onmessage?.({
        data: JSON.stringify({
          eventType: 'persistence_completed',
          chatId: 'preview-chat-1',
          messageId: 'm1',
          researchSessionId: 'rs1',
          wasDraft: false,
        }),
      } as MessageEvent);
    });

    await waitFor(() => {
      expect(result.current.runState['run']).toMatchObject({
        status: 'completed',
        preview: 'real',
        action: 'run',
        message_id: 'm1',
        session_id: 'rs1',
      });
    });
    expect(invalidateSpy).toHaveBeenCalledWith({
      queryKey: ['chatFull', 'preview-chat-1'],
    });
    expect(invalidateSpy).toHaveBeenCalledWith({
      queryKey: ['messages', 'preview-chat-1'],
    });
  });

  it('marks the run failed when submission is rejected with 429', async () => {
    vi.mocked(jobsApi.submit).mockRejectedValue(
      new ApiError(429, 'MAX_CONCURRENT_JOBS', 'too many jobs'),
    );
    const { result } = renderRun();

    act(() => {
      result.current.start('run', COMPILED);
    });

    await waitFor(() => {
      expect(result.current.runState['run']).toMatchObject({
        status: 'failed',
        preview: 'real',
      });
    });
    expect(result.current.errorDetails?.errorCode).toBe('MAX_CONCURRENT_JOBS');
    expect(result.current.isActive).toBe(false);
  });

  it('stop() cancels the job and marks the run cancelled', async () => {
    const { result } = renderRun();
    await startAndAwaitSubmit(result);

    act(() => {
      result.current.stop();
    });

    await waitFor(() => {
      expect(result.current.runState['run']).toMatchObject({
        status: 'cancelled',
        preview: 'real',
      });
    });
    expect(jobsApi.cancel).toHaveBeenCalledWith('preview-chat-1', 'session-1');
  });

  it('ignores start() while a run is active', async () => {
    const { result } = renderRun();
    await startAndAwaitSubmit(result);

    act(() => {
      result.current.start('run', COMPILED);
    });
    // Still exactly one submission and one chat.
    await waitFor(() => {
      expect(result.current.runState['run']?.status).toBe('running');
    });
    expect(jobsApi.submit).toHaveBeenCalledTimes(1);
    expect(chatsApi.create).toHaveBeenCalledTimes(1);
  });

  it('job_completed without persistence_completed yields completed without ids', async () => {
    const { result } = renderRun();
    await startAndAwaitSubmit(result);

    const source = FakeEventSource.instances[0];
    act(() => {
      source?.onmessage?.({
        data: JSON.stringify({ eventType: 'job_completed', status: 'completed' }),
      } as MessageEvent);
    });

    await waitFor(() => {
      expect(result.current.runState['run']).toMatchObject({
        status: 'completed',
        preview: 'real',
      });
    });
    expect(result.current.runState['run']?.message_id).toBeUndefined();
    expect(result.current.isActive).toBe(false);
  });
});
