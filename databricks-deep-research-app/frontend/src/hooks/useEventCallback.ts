import * as React from 'react';

// useLayoutEffect on the client, useEffect on the server (avoids the SSR warning).
const useIsomorphicLayoutEffect =
  typeof window !== 'undefined' ? React.useLayoutEffect : React.useEffect;

/**
 * Returns a function whose identity is STABLE for the lifetime of the component
 * but which always invokes the latest `fn`.
 *
 * Use this for event handlers that are passed to memoized children (e.g.
 * `React.memo` components or values listed in a `useMemo`/`useEffect` dependency
 * array). Passing a fresh inline arrow each render busts that memoization; if the
 * child uses the callback to build React element `type`s (as `MarkdownRenderer`
 * does for citation markers), the churn can remount the subtree on every render
 * and, combined with a re-resolving effect, produce an infinite update loop
 * (React error #185).
 *
 * Constraint: the returned function reads the latest `fn` via a ref that is
 * synchronised in a layout effect, so it must only be called from event handlers
 * or effects — never during render.
 */
export function useEventCallback<A extends unknown[], R>(
  fn: ((...args: A) => R) | undefined
): (...args: A) => R | undefined {
  const ref = React.useRef(fn);
  useIsomorphicLayoutEffect(() => {
    ref.current = fn;
  });
  return React.useCallback((...args: A) => ref.current?.(...args), []);
}

export default useEventCallback;
