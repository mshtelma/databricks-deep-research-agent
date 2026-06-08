/**
 * NormalizationFixPill component tests.
 *
 * Covers:
 *   - Zero fixes → component renders nothing (the spec's hard requirement).
 *   - 1+ fixes → renders pill with exact count.
 *   - Clicking pill toggles the expanded panel with one row per fix.
 *   - `compact={true}` collapses to "!" indicator.
 *   - Unknown `kind` values render with a generic label (forward-compat).
 *   - aria-label format matches the spec.
 */

import { describe, expect, it, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import { NormalizationFixPill } from '../NormalizationFixPill';
import type { NormalizationFix } from '@/types/agentDesigner';

const FIX_SUBTYPE: NormalizationFix = {
  kind: 'subtype_rewrite',
  path: 'root.children.0.config.subtype',
  before: 'lane_researcher',
  after: 'researcher',
  rationale: 'Closest framework subtype.',
};

const FIX_TIER: NormalizationFix = {
  kind: 'tier_rewrite',
  path: 'root.children.0.config.model_tier',
  before: 'standard',
  after: 'analytical',
  rationale: 'Default for researcher-class agents.',
};

describe('NormalizationFixPill', () => {
  it('renders nothing when fixes is empty', () => {
    const { container } = render(<NormalizationFixPill fixes={[]} />);
    expect(container.firstChild).toBeNull();
  });

  it('renders nothing when fixes is undefined', () => {
    // @ts-expect-error — exercise the runtime defensive guard
    const { container } = render(<NormalizationFixPill fixes={undefined} />);
    expect(container.firstChild).toBeNull();
  });

  it('shows count for a single fix', () => {
    render(<NormalizationFixPill fixes={[FIX_SUBTYPE]} />);
    expect(
      screen.getByRole('button', { name: /Auto-repair: 1 issue fixed/i }),
    ).toBeInTheDocument();
    expect(screen.getByText(/auto-fixed 1 issue/i)).toBeInTheDocument();
  });

  it('shows pluralized count for multiple fixes', () => {
    render(<NormalizationFixPill fixes={[FIX_SUBTYPE, FIX_TIER]} />);
    expect(
      screen.getByRole('button', { name: /Auto-repair: 2 issues fixed/i }),
    ).toBeInTheDocument();
    expect(screen.getByText(/auto-fixed 2 issues/i)).toBeInTheDocument();
  });

  it('expands and collapses the detail panel when clicked', () => {
    render(<NormalizationFixPill fixes={[FIX_SUBTYPE, FIX_TIER]} />);
    const toggle = screen.getByRole('button', { name: /Auto-repair/i });
    expect(toggle).toHaveAttribute('aria-expanded', 'false');
    // Detail panel not rendered yet.
    expect(screen.queryByLabelText('Designer auto-repair details')).toBeNull();

    fireEvent.click(toggle);
    expect(toggle).toHaveAttribute('aria-expanded', 'true');
    const panel = screen.getByLabelText('Designer auto-repair details');
    expect(panel).toBeInTheDocument();
    // One row per fix.
    expect(panel.querySelectorAll('li').length).toBe(2);
    // Rows show the labels.
    expect(panel.textContent).toContain('Subtype rewritten');
    expect(panel.textContent).toContain('Model tier rewritten');
    // Before/after values.
    expect(panel.textContent).toContain('lane_researcher');
    expect(panel.textContent).toContain('researcher');
    expect(panel.textContent).toContain('standard');
    expect(panel.textContent).toContain('analytical');

    fireEvent.click(toggle);
    expect(toggle).toHaveAttribute('aria-expanded', 'false');
    expect(screen.queryByLabelText('Designer auto-repair details')).toBeNull();
  });

  it('renders compact "!" indicator when compact=true', () => {
    render(
      <NormalizationFixPill fixes={[FIX_SUBTYPE, FIX_TIER]} compact />,
    );
    const indicator = screen.getByRole('status', {
      name: /Auto-repair: 2 issues fixed/i,
    });
    expect(indicator).toBeInTheDocument();
    expect(indicator.textContent).toBe('!');
    // No expand button when compact.
    expect(screen.queryByRole('button', { name: /Auto-repair/i })).toBeNull();
  });

  it('renders unknown kind with raw kind label', () => {
    const exoticFix: NormalizationFix = {
      kind: 'future_rewrite_kind',
      path: 'root.something',
      before: 1,
      after: 2,
      rationale: 'A future kind.',
    };
    render(<NormalizationFixPill fixes={[exoticFix]} />);
    const toggle = screen.getByRole('button', { name: /Auto-repair/i });
    fireEvent.click(toggle);
    // Unknown kind falls back to the raw kind string as label.
    expect(screen.getByText('future_rewrite_kind')).toBeInTheDocument();
    // No JS error means render succeeded.
  });

  it('renders all known fix kinds with friendly labels', () => {
    // The 8 fix kinds currently emitted by the Layer 2 normalizer + write-
    // time escapers. Any new kind added here should also be mapped in
    // NormalizationFixPill's KIND_META so the UI shows a label instead of
    // the raw kind string.
    const everyKind: NormalizationFix[] = [
      { kind: 'subtype_rewrite', path: 'a', before: 'x', after: 'y', rationale: '' },
      { kind: 'tier_rewrite', path: 'a', before: 'x', after: 'y', rationale: '' },
      { kind: 'tool_kind_rewrite', path: 'a', before: 'x', after: 'y', rationale: '' },
      { kind: 'auto_bind_retrieval', path: 'a', before: [], after: [], rationale: '' },
      { kind: 'auto_declare_pool', path: 'a', before: null, after: {}, rationale: '' },
      { kind: 'set_minimum_max_tool_calls', path: 'a', before: 0, after: 6, rationale: '' },
      { kind: 'tool_consolidation', path: 'a', before: [], after: [], rationale: '' },
      { kind: 'brace_escape', path: 'a', before: 'x', after: 'y', rationale: '' },
    ];
    render(<NormalizationFixPill fixes={everyKind} />);
    fireEvent.click(screen.getByRole('button', { name: /Auto-repair/i }));
    const panel = screen.getByLabelText('Designer auto-repair details');
    // Each fix gets a friendly label — no raw kind strings should appear.
    const expectedLabels = [
      'Subtype rewritten',
      'Model tier rewritten',
      'Tool kind rewritten',
      'Retrieval tools auto-bound',
      'Pool auto-declared',
      'Tool-call budget raised',
      'Tools consolidated',
      'Prompt braces escaped',
    ];
    for (const label of expectedLabels) {
      expect(panel.textContent).toContain(label);
    }
    // And no raw kind strings leak through.
    for (const fix of everyKind) {
      expect(panel.textContent).not.toContain(fix.kind);
    }
  });

  it('copies the path to clipboard when the path button is clicked', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, 'clipboard', {
      configurable: true,
      value: { writeText },
    });

    render(<NormalizationFixPill fixes={[FIX_SUBTYPE]} />);
    fireEvent.click(screen.getByRole('button', { name: /Auto-repair/i }));
    const pathBtn = screen.getByText(FIX_SUBTYPE.path);
    fireEvent.click(pathBtn);
    // Microtask flush so the async writeText resolves.
    await Promise.resolve();
    expect(writeText).toHaveBeenCalledWith(FIX_SUBTYPE.path);
  });

  it('formats null/undefined before-values as ∅', () => {
    const nullFix: NormalizationFix = {
      kind: 'auto_declare_pool',
      path: 'pools.0',
      before: null,
      after: { name: 'sources' },
      rationale: 'Pool was referenced but never declared.',
    };
    render(<NormalizationFixPill fixes={[nullFix]} />);
    fireEvent.click(screen.getByRole('button', { name: /Auto-repair/i }));
    const panel = screen.getByLabelText('Designer auto-repair details');
    expect(panel.textContent).toContain('∅');
  });
});
