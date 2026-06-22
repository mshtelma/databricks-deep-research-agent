import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useCitationPopover } from '../useCitationPopover';
import type { Claim } from '@/types/citation';

// The hook treats the claim opaquely, so a thin stand-in is sufficient.
const CLAIM = { id: 'c1', claimText: 'A claim.' } as unknown as Claim;
const marker = () => document.createElement('sup');

describe('useCitationPopover', () => {
  it('does nothing when the key does not resolve to a claim', () => {
    const { result } = renderHook(() => useCitationPopover(() => null));
    act(() => result.current.onMarkerEnter('1', marker()));
    expect(result.current.open).toBe(false);
    expect(result.current.claim).toBeNull();
    act(() => result.current.onMarkerClick('1'));
    expect(result.current.open).toBe(false);
  });

  it('opens on hover (unpinned) and pins on click', () => {
    const { result } = renderHook(() => useCitationPopover(() => CLAIM));

    act(() => result.current.onMarkerEnter('1', marker()));
    expect(result.current.open).toBe(true);
    expect(result.current.pinned).toBe(false);
    expect(result.current.claim).toBe(CLAIM);
    expect(result.current.activeKey).toBe('1');

    act(() => result.current.onMarkerClick('1'));
    expect(result.current.open).toBe(true);
    expect(result.current.pinned).toBe(true);
  });

  it('close() resets all state', () => {
    const { result } = renderHook(() => useCitationPopover(() => CLAIM));
    act(() => result.current.onMarkerClick('1'));
    act(() => result.current.close());
    expect(result.current.open).toBe(false);
    expect(result.current.pinned).toBe(false);
    expect(result.current.claim).toBeNull();
    expect(result.current.activeKey).toBeNull();
  });

  describe('with fake timers', () => {
    beforeEach(() => {
      vi.useFakeTimers();
    });
    afterEach(() => {
      vi.useRealTimers();
    });

    it('closes after the delay when unpinned and the marker is left', () => {
      const { result } = renderHook(() => useCitationPopover(() => CLAIM));
      act(() => result.current.onMarkerEnter('1', marker()));
      act(() => {
        result.current.onMarkerLeave();
        vi.advanceTimersByTime(300);
      });
      expect(result.current.open).toBe(false);
    });

    it('stays open when pinned and the marker is left', () => {
      const { result } = renderHook(() => useCitationPopover(() => CLAIM));
      act(() => result.current.onMarkerClick('1'));
      act(() => {
        result.current.onMarkerLeave();
        vi.advanceTimersByTime(300);
      });
      expect(result.current.open).toBe(true);
    });

    it('hover bridge: entering the card cancels the pending close', () => {
      const { result } = renderHook(() => useCitationPopover(() => CLAIM));
      act(() => result.current.onMarkerEnter('1', marker()));
      act(() => {
        result.current.onMarkerLeave();
        vi.advanceTimersByTime(100); // partway through the close delay
        result.current.onCardEnter(); // moving onto the card cancels it
        vi.advanceTimersByTime(300);
      });
      expect(result.current.open).toBe(true);
    });
  });
});

describe('useCitationPopover anchor loop-proofing', () => {
  it('caps re-resolution under marker churn (warns once, never loops)', () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    // Simulate a marker that remounts on every render: each DOM lookup returns a
    // fresh, detached node, so the anchor is never "connected". The guard must
    // stop after a single re-resolve instead of looping (React #185).
    const qs = vi
      .spyOn(document, 'querySelector')
      .mockImplementation(() => document.createElement('sup'));
    try {
      const { result } = renderHook(() => useCitationPopover(() => CLAIM));
      act(() => result.current.onMarkerEnter('1', document.createElement('sup')));
      // Reaching this assertion at all proves there was no infinite update loop;
      // the guard detected the churn and logged exactly the diagnostic once.
      expect(warn).toHaveBeenCalledWith(
        expect.stringContaining('citation marker keeps remounting')
      );
    } finally {
      qs.mockRestore();
      warn.mockRestore();
    }
  });
});
