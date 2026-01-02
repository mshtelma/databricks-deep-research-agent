/**
 * FloatingEvidenceCard - Smart positioning wrapper for EvidenceCard
 *
 * Uses @floating-ui/react for viewport-aware positioning:
 * - Automatically flips to opposite side if not enough space
 * - Shifts along axis to stay within viewport boundaries
 * - Smooth hover interactions with configurable delays
 */

import * as React from 'react';
import {
  useFloating,
  autoUpdate,
  offset,
  flip,
  shift,
  useHover,
  useFocus,
  useDismiss,
  useInteractions,
  FloatingPortal,
  useTransitionStyles,
  type Placement,
} from '@floating-ui/react';
import { EvidenceCard } from './EvidenceCard';
import type { Citation, VerificationVerdict } from '@/types/citation';

interface FloatingEvidenceCardProps {
  /** Citation data to display */
  citation: Citation;
  /** Claim text for context */
  claimText?: string;
  /** Verification verdict */
  verdict?: VerificationVerdict | null;
  /** Whether the popover is open (controlled mode) */
  open?: boolean;
  /** Callback when open state changes (controlled mode) */
  onOpenChange?: (open: boolean) => void;
  /** The trigger element - must accept ref */
  children: React.ReactElement;
  /** Initial preferred placement */
  placement?: Placement;
  /** Keywords to highlight in the quote */
  highlightKeywords?: string[];
  /** Custom z-index for the floating element */
  zIndex?: number;
}

export const FloatingEvidenceCard: React.FC<FloatingEvidenceCardProps> = ({
  citation,
  claimText,
  verdict,
  open: controlledOpen,
  onOpenChange: controlledOnOpenChange,
  children,
  placement = 'bottom-start',
  highlightKeywords = [],
  zIndex = 50,
}) => {
  // Internal state for uncontrolled mode
  const [uncontrolledOpen, setUncontrolledOpen] = React.useState(false);

  // Determine if controlled or uncontrolled
  const isControlled = controlledOpen !== undefined;
  const open = isControlled ? controlledOpen : uncontrolledOpen;
  const onOpenChange = isControlled ? controlledOnOpenChange : setUncontrolledOpen;

  const { refs, floatingStyles, context } = useFloating({
    open,
    onOpenChange,
    placement,
    middleware: [
      // 8px gap from trigger element
      offset(8),
      // Flip to opposite side if not enough space
      flip({
        fallbackAxisSideDirection: 'end',
        padding: 8, // Stay 8px from viewport edges
      }),
      // Shift along axis to stay in viewport
      shift({
        padding: 8,
        crossAxis: true,
      }),
    ],
    whileElementsMounted: autoUpdate,
  });

  // Interaction hooks
  const hover = useHover(context, {
    delay: { open: 200, close: 100 },
    restMs: 100, // Wait 100ms of stable hover before opening
  });

  const focus = useFocus(context);

  const dismiss = useDismiss(context, {
    escapeKey: true,
    outsidePress: true,
  });

  const { getReferenceProps, getFloatingProps } = useInteractions([
    hover,
    focus,
    dismiss,
  ]);

  // Smooth transition styles
  const { isMounted, styles: transitionStyles } = useTransitionStyles(context, {
    duration: 150,
    initial: { opacity: 0, transform: 'scale(0.95)' },
  });

  // Clone children with ref and props
  const triggerElement = React.cloneElement(children, {
    ref: refs.setReference,
    ...getReferenceProps(),
  });

  return (
    <>
      {triggerElement}
      {isMounted && (
        <FloatingPortal>
          <div
            ref={refs.setFloating}
            style={{
              ...floatingStyles,
              ...transitionStyles,
              zIndex,
            }}
            {...getFloatingProps()}
          >
            <EvidenceCard
              citation={citation}
              claimText={claimText}
              verdict={verdict}
              highlightKeywords={highlightKeywords}
              isPopover={true}
              onClose={() => onOpenChange?.(false)}
            />
          </div>
        </FloatingPortal>
      )}
    </>
  );
};

FloatingEvidenceCard.displayName = 'FloatingEvidenceCard';

export default FloatingEvidenceCard;
