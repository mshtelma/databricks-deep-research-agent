import { renderHook } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { useEventCallback } from '../useEventCallback';

describe('useEventCallback', () => {
  it('keeps a stable identity across renders but invokes the latest fn', () => {
    const fn1 = vi.fn();
    const fn2 = vi.fn();
    const { result, rerender } = renderHook(({ fn }) => useEventCallback(fn), {
      initialProps: { fn: fn1 as (...a: [string, number]) => void },
    });
    const stable = result.current;

    rerender({ fn: fn2 as (...a: [string, number]) => void });
    expect(result.current).toBe(stable); // identity unchanged across renders

    result.current('x', 1);
    expect(fn2).toHaveBeenCalledWith('x', 1); // latest fn is invoked
    expect(fn1).not.toHaveBeenCalled(); // the stale one is not
  });

  it('is a no-op (no throw) when fn is undefined', () => {
    const { result } = renderHook(() => useEventCallback<[], void>(undefined));
    expect(() => result.current()).not.toThrow();
  });
});
