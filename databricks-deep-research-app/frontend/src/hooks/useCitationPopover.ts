/**
 * useCitationPopover - state machine + floating-ui wiring for the inline citation
 * evidence card.
 *
 * Behavior (per product decision):
 *   - Hover a citation marker  → open the card (anchored next to the marker).
 *   - Move onto the card        → stays open (hover bridge cancels the close timer).
 *   - Click a citation marker   → PIN the card open until dismissed.
 *   - Esc / click-outside       → close.
 *
 * Positioning fix: the floating reference is held in React state (`elements.reference`)
 * and re-resolved from the live DOM by the marker's stable `data-testid` if the node
 * detaches across a markdown re-render. This avoids the "top-left (0,0)" flash caused
 * by imperatively calling refs.setReference() after a state update.
 */
import * as React from 'react';
import {
  useFloating,
  autoUpdate,
  offset,
  flip,
  shift,
  useDismiss,
  useInteractions,
  useTransitionStyles,
} from '@floating-ui/react';
import type { Claim } from '@/types/citation';

/** Delay before closing on mouse-leave, long enough to cross the marker→card gap. */
const CLOSE_DELAY_MS = 180;
/** Gap between the marker and the card (also keeps the card from covering the marker). */
const OFFSET_PX = 8;

export interface CitationPopoverApi {
  /** Whether a claim is currently selected (card mounted). */
  open: boolean;
  /** Whether the card is pinned (click) and ignores mouse-leave. */
  pinned: boolean;
  /** The claim backing the open card, if any. */
  claim: Claim | null;
  /** The citation key the card is anchored to, if any. */
  activeKey: string | null;
  refs: ReturnType<typeof useFloating>['refs'];
  floatingStyles: React.CSSProperties;
  getFloatingProps: (
    userProps?: React.HTMLProps<HTMLElement>
  ) => Record<string, unknown>;
  isMounted: boolean;
  transitionStyles: React.CSSProperties;
  /** Marker mouse-enter: open (non-pinned) anchored to `el`. */
  onMarkerEnter: (key: string, el: HTMLElement) => void;
  /** Marker mouse-leave: schedule close unless pinned. */
  onMarkerLeave: () => void;
  /** Marker click: pin the card open. */
  onMarkerClick: (key: string) => void;
  /** Card mouse-enter: cancel pending close (hover bridge). */
  onCardEnter: () => void;
  /** Card mouse-leave: schedule close unless pinned. */
  onCardLeave: () => void;
  /** Force-close and unpin. */
  close: () => void;
  /**
   * Stable, ready-to-spread citation callbacks for `MarkdownRenderer`. Spreading
   * these (rather than passing inline arrows) keeps the renderer's `components`
   * memo stable, which prevents the marker-remount loop (React #185) at the source.
   */
  markdownCitationProps: {
    onCitationClick: (key: string) => void;
    onCitationHover: (key: string | null, el?: HTMLElement | null) => void;
  };
}

function findMarkerElement(key: string): HTMLElement | null {
  if (typeof document === 'undefined') return null;
  return document.querySelector<HTMLElement>(
    `[data-testid="citation-marker-${key}"]`
  );
}

export function useCitationPopover(
  resolveClaim: (key: string) => Claim | null
): CitationPopoverApi {
  const [activeKey, setActiveKey] = React.useState<string | null>(null);
  const [claim, setClaim] = React.useState<Claim | null>(null);
  const [pinned, setPinned] = React.useState(false);
  const [referenceEl, setReferenceEl] = React.useState<HTMLElement | null>(null);

  // Mirror `pinned` in a ref so the deferred close timer reads the latest value
  // without being re-created on every pin/unpin.
  const pinnedRef = React.useRef(false);
  React.useEffect(() => {
    pinnedRef.current = pinned;
  }, [pinned]);

  const closeTimer = React.useRef<ReturnType<typeof setTimeout> | null>(null);
  const clearCloseTimer = React.useCallback(() => {
    if (closeTimer.current) {
      clearTimeout(closeTimer.current);
      closeTimer.current = null;
    }
  }, []);

  // Cleanup on unmount.
  React.useEffect(() => clearCloseTimer, [clearCloseTimer]);

  const open = claim !== null;

  const close = React.useCallback(() => {
    clearCloseTimer();
    setClaim(null);
    setActiveKey(null);
    setPinned(false);
    setReferenceEl(null);
  }, [clearCloseTimer]);

  const { refs, floatingStyles, context } = useFloating({
    open,
    onOpenChange: (next) => {
      if (!next) close();
    },
    placement: 'bottom-start',
    middleware: [
      offset(OFFSET_PX),
      flip({ padding: 8, fallbackAxisSideDirection: 'end' }),
      shift({ padding: 8, crossAxis: true }),
    ],
    elements: { reference: referenceEl },
    whileElementsMounted: autoUpdate,
  });

  const dismiss = useDismiss(context, { escapeKey: true, outsidePress: true });
  const { getFloatingProps } = useInteractions([dismiss]);
  const { isMounted, styles: transitionStyles } = useTransitionStyles(context, {
    duration: 150,
    initial: { opacity: 0, transform: 'scale(0.95)' },
  });

  const scheduleClose = React.useCallback(() => {
    clearCloseTimer();
    closeTimer.current = setTimeout(() => {
      closeTimer.current = null;
      if (!pinnedRef.current) close();
    }, CLOSE_DELAY_MS);
  }, [clearCloseTimer, close]);

  const onMarkerEnter = React.useCallback(
    (key: string, el: HTMLElement) => {
      clearCloseTimer();
      // While pinned, leave the pinned card in place rather than chasing hovers.
      if (pinnedRef.current) return;
      const resolved = resolveClaim(key);
      if (!resolved) return;
      setActiveKey(key);
      setClaim(resolved);
      setReferenceEl(el);
    },
    [clearCloseTimer, resolveClaim]
  );

  const onMarkerLeave = React.useCallback(() => {
    if (!pinnedRef.current) scheduleClose();
  }, [scheduleClose]);

  const onMarkerClick = React.useCallback(
    (key: string) => {
      clearCloseTimer();
      const resolved = resolveClaim(key);
      if (!resolved) return;
      // Resolve the anchor from the live DOM so click works without a prior hover
      // (e.g. touch) and survives markdown re-renders.
      const el = findMarkerElement(key);
      setActiveKey(key);
      setClaim(resolved);
      if (el) setReferenceEl(el);
      setPinned(true);
    },
    [clearCloseTimer, resolveClaim]
  );

  const onCardEnter = React.useCallback(() => {
    clearCloseTimer();
  }, [clearCloseTimer]);

  const onCardLeave = React.useCallback(() => {
    if (!pinnedRef.current) scheduleClose();
  }, [scheduleClose]);

  // Keep the anchor valid: if the referenced node detaches (e.g. a markdown
  // re-render remounted the marker), re-resolve it from the stable data-testid.
  //
  // Loop-proofing: re-resolve at most ONCE per activeKey while the anchor is
  // detached, and re-arm only after observing a connected node. This guarantees
  // the effect cannot drive an infinite update loop even if a future change
  // reintroduces per-render marker remounts (the original React #185 cause).
  const resolvedKeyRef = React.useRef<string | null>(null);
  const churnWarnedRef = React.useRef(false);
  React.useLayoutEffect(() => {
    if (!activeKey) {
      resolvedKeyRef.current = null;
      return;
    }
    if (referenceEl && referenceEl.isConnected) {
      resolvedKeyRef.current = null; // healthy anchor → allow future re-resolves
      return;
    }
    if (resolvedKeyRef.current === activeKey) {
      if (!churnWarnedRef.current) {
        churnWarnedRef.current = true;
        // eslint-disable-next-line no-console
        console.warn(
          '[useCitationPopover] citation marker keeps remounting; the popover anchor ' +
            'cannot stabilize. Ensure MarkdownRenderer citation callbacks are stable.'
        );
      }
      return; // already re-resolved for this key while detached → stop (cannot loop)
    }
    const live = findMarkerElement(activeKey);
    if (live && live !== referenceEl) {
      resolvedKeyRef.current = activeKey;
      setReferenceEl(live);
    }
  }, [activeKey, referenceEl]);

  // Stable, ready-to-spread citation callbacks for MarkdownRenderer (see the
  // CitationPopoverApi doc). Deps are the hook's own useCallback-stable handlers,
  // so this object identity changes only if those do (effectively never).
  const markdownCitationProps = React.useMemo(
    () => ({
      onCitationClick: (key: string) => onMarkerClick(key),
      onCitationHover: (key: string | null, el?: HTMLElement | null) => {
        if (key && el) onMarkerEnter(key, el);
        else onMarkerLeave();
      },
    }),
    [onMarkerClick, onMarkerEnter, onMarkerLeave]
  );

  return {
    open,
    pinned,
    claim,
    activeKey,
    refs,
    floatingStyles,
    getFloatingProps,
    isMounted,
    transitionStyles,
    onMarkerEnter,
    onMarkerLeave,
    onMarkerClick,
    onCardEnter,
    onCardLeave,
    close,
    markdownCitationProps,
  };
}

export default useCitationPopover;
