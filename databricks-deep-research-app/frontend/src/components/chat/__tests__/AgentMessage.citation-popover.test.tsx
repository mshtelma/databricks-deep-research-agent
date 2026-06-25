import '@testing-library/jest-dom/vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, beforeAll } from 'vitest';
import { AgentMessage } from '../AgentMessage';
import type { Message } from '@/types';
import { makeClaim } from '@/test-utils/citationFixtures';

// floating-ui's autoUpdate uses ResizeObserver, which jsdom does not implement.
// Stub it so mounting the floating EvidenceCard exercises the real popover path
// (this test must fail on the React #185 loop, not on a missing global).
beforeAll(() => {
  if (!('ResizeObserver' in globalThis)) {
    (globalThis as unknown as { ResizeObserver: unknown }).ResizeObserver = class {
      observe() {}
      unobserve() {}
      disconnect() {}
    };
  }
});

/** Non-UUID id skips MessageExportMenu; hideSourcesSection skips SourceGroupedCitations. */
function makeMessage(content: string): Message {
  return {
    id: 'agent-msg-1',
    chatId: 'chat-1',
    role: 'agent',
    content,
    createdAt: '2026-01-01T00:00:00.000Z',
    isEdited: false,
  };
}

function renderAgentMessage(content = 'Alpha [1] beta.') {
  return render(
    <AgentMessage
      message={makeMessage(content)}
      enableCitations
      claims={[makeClaim('1')]}
      hideSourcesSection
    />
  );
}

describe('AgentMessage citation popover (regression: React #185 hover crash)', () => {
  it('opens the evidence card on hover without an infinite render loop', async () => {
    renderAgentMessage();
    const marker = screen.getByTestId('citation-marker-1');

    // Pre-fix this throws "Maximum update depth exceeded" (React #185) during the
    // hover-triggered re-render/remount loop. Post-fix the card mounts cleanly.
    fireEvent.mouseEnter(marker);

    expect(await screen.findByTestId('evidence-card')).toBeInTheDocument();
  });

  it('pins the card on click so it survives mouse-leave', async () => {
    renderAgentMessage();
    const marker = screen.getByTestId('citation-marker-1');

    fireEvent.click(marker);
    expect(await screen.findByTestId('evidence-card')).toBeInTheDocument();

    // Leaving the marker must NOT close a pinned card.
    fireEvent.mouseLeave(marker);
    expect(screen.getByTestId('evidence-card')).toBeInTheDocument();
  });

  it('closes the card on Escape', async () => {
    renderAgentMessage();
    fireEvent.click(screen.getByTestId('citation-marker-1'));
    await screen.findByTestId('evidence-card');

    fireEvent.keyDown(document.body, { key: 'Escape' });
    await waitFor(() =>
      expect(screen.queryByTestId('evidence-card')).not.toBeInTheDocument()
    );
  });
});
