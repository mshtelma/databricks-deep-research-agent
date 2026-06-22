import '@testing-library/jest-dom/vitest';
import * as React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { MarkdownRenderer } from '../MarkdownRenderer';
import { ActiveCitationContext } from '@/components/citations';
import { makeCitationContext } from '@/test-utils/citationFixtures';

const citationData = new Map([['1', makeCitationContext('1')]]);

/**
 * Mimics AgentMessage's previously-buggy wiring: re-renders on demand and passes
 * FRESH inline citation callbacks each render (the original footgun). The renderer
 * must keep the marker DOM node stable (no remount) regardless.
 */
function Harness({ activeKey }: { activeKey: string | null }) {
  const [, force] = React.useReducer((x: number) => x + 1, 0);
  return (
    <ActiveCitationContext.Provider value={activeKey}>
      <button data-testid="force" onClick={force}>
        force
      </button>
      <MarkdownRenderer
        content="Alpha [1] beta."
        enableCitations
        citationMode="numeric"
        citationData={citationData}
        onCitationClick={() => {}}
        onCitationHover={() => {}}
      />
    </ActiveCitationContext.Provider>
  );
}

describe('MarkdownRenderer citation marker stability', () => {
  it('does not remount the marker when the parent re-renders with new callbacks', () => {
    render(<Harness activeKey={null} />);
    const before = screen.getByTestId('citation-marker-1');

    fireEvent.click(screen.getByTestId('force')); // re-render with fresh inline callbacks
    fireEvent.click(screen.getByTestId('force'));

    // Same DOM node ⇒ no remount. (Pre-fix this was a new node every render — the
    // churn that, with the popover's anchor effect, produced the React #185 loop.)
    expect(screen.getByTestId('citation-marker-1')).toBe(before);
  });

  it('toggles the active highlight via context without remounting the marker', () => {
    const { rerender } = render(<Harness activeKey={null} />);
    const before = screen.getByTestId('citation-marker-1');
    expect(before.className).not.toContain('ring-1');

    rerender(<Harness activeKey="1" />);
    const after = screen.getByTestId('citation-marker-1');
    expect(after).toBe(before); // same node
    expect(after.className).toContain('ring-1'); // now highlighted
  });
});
