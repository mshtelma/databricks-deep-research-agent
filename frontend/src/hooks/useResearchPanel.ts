import { useState, useEffect, useCallback, useRef } from 'react';

export type ResearchPanelTab = 'activity' | 'cited' | 'all';

interface UseResearchPanelOptions {
  /** Auto-collapse delay in ms after streaming ends (default: 800ms) */
  autoCollapseDelay?: number;
  /** Default tab to show (default: 'activity') */
  defaultTab?: ResearchPanelTab;
  /** Whether to auto-switch to 'cited' tab after collapse (default: true) */
  switchToCitedOnCollapse?: boolean;
}

interface UseResearchPanelReturn {
  /** Currently active tab */
  activeTab: ResearchPanelTab;
  /** Set the active tab */
  setActiveTab: (tab: ResearchPanelTab) => void;
  /** Whether the panel content is expanded */
  isExpanded: boolean;
  /** Toggle panel expansion */
  toggleExpanded: () => void;
  /** Manually set expansion state */
  setIsExpanded: (expanded: boolean) => void;
  /** Whether to show the live indicator */
  showLiveIndicator: boolean;
}

/**
 * Hook for managing Research Panel state.
 * Handles auto-collapse behavior when streaming ends and tab switching.
 */
export function useResearchPanel(
  isStreaming: boolean,
  hasContent: boolean,
  options: UseResearchPanelOptions = {}
): UseResearchPanelReturn {
  const {
    autoCollapseDelay = 800,
    defaultTab = 'activity',
    switchToCitedOnCollapse = true,
  } = options;

  const [activeTab, setActiveTab] = useState<ResearchPanelTab>(defaultTab);
  const [isExpanded, setIsExpanded] = useState(true);

  // Track if user has manually interacted with the panel
  const userInteractedRef = useRef(false);
  // Track previous streaming state to detect transitions
  const wasStreamingRef = useRef(false);
  // Track if we've already handled this streaming session's completion
  const handledCompletionRef = useRef(false);

  // Detect streaming start - reset states
  useEffect(() => {
    if (isStreaming && !wasStreamingRef.current) {
      // Streaming just started
      setIsExpanded(true);
      setActiveTab('activity');
      userInteractedRef.current = false;
      handledCompletionRef.current = false;
    }
    wasStreamingRef.current = isStreaming;
  }, [isStreaming]);

  // Auto-collapse when streaming ends (only if user hasn't interacted)
  useEffect(() => {
    if (!isStreaming && wasStreamingRef.current && hasContent && !handledCompletionRef.current) {
      // Streaming just ended
      handledCompletionRef.current = true;

      // Only auto-collapse if user hasn't manually interacted
      if (!userInteractedRef.current) {
        const timer = setTimeout(() => {
          setIsExpanded(false);
          if (switchToCitedOnCollapse) {
            setActiveTab('cited');
          }
        }, autoCollapseDelay);

        return () => clearTimeout(timer);
      }
    }
  }, [isStreaming, hasContent, autoCollapseDelay, switchToCitedOnCollapse]);

  // Wrapper to track user interaction
  const handleSetActiveTab = useCallback((tab: ResearchPanelTab) => {
    userInteractedRef.current = true;
    setActiveTab(tab);
  }, []);

  const handleSetIsExpanded = useCallback((expanded: boolean) => {
    userInteractedRef.current = true;
    setIsExpanded(expanded);
  }, []);

  const toggleExpanded = useCallback(() => {
    userInteractedRef.current = true;
    setIsExpanded((prev) => !prev);
  }, []);

  return {
    activeTab,
    setActiveTab: handleSetActiveTab,
    isExpanded,
    setIsExpanded: handleSetIsExpanded,
    toggleExpanded,
    showLiveIndicator: isStreaming,
  };
}

export default useResearchPanel;
