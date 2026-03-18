import * as React from 'react';

type KeyboardShortcut = {
  key: string;
  ctrl?: boolean;
  meta?: boolean;
  shift?: boolean;
  handler: () => void;
  preventDefault?: boolean;
};

interface UseKeyboardShortcutsOptions {
  shortcuts: KeyboardShortcut[];
  enabled?: boolean;
}

/**
 * Hook for registering global keyboard shortcuts.
 *
 * @param options - The shortcuts configuration
 * @param options.shortcuts - Array of keyboard shortcuts to register
 * @param options.enabled - Whether shortcuts are enabled (default: true)
 *
 * @example
 * ```tsx
 * useKeyboardShortcuts({
 *   shortcuts: [
 *     {
 *       key: 'n',
 *       meta: true,
 *       handler: () => createNewChat(),
 *       preventDefault: true,
 *     },
 *     {
 *       key: 'Enter',
 *       meta: true,
 *       handler: () => sendMessage(),
 *       preventDefault: true,
 *     },
 *   ],
 * });
 * ```
 */
export function useKeyboardShortcuts({
  shortcuts,
  enabled = true,
}: UseKeyboardShortcutsOptions) {
  React.useEffect(() => {
    if (!enabled) return;

    const handleKeyDown = (event: KeyboardEvent) => {
      // Don't trigger shortcuts when typing in inputs (except for specific ones)
      const target = event.target as HTMLElement;
      const isInput =
        target.tagName === 'INPUT' ||
        target.tagName === 'TEXTAREA' ||
        target.isContentEditable;

      for (const shortcut of shortcuts) {
        const keyMatch = event.key.toLowerCase() === shortcut.key.toLowerCase();
        const ctrlMatch = shortcut.ctrl ? event.ctrlKey : !event.ctrlKey;
        const metaMatch = shortcut.meta ? event.metaKey : !event.metaKey;
        const shiftMatch = shortcut.shift ? event.shiftKey : !event.shiftKey;

        // For Enter key in inputs, we want to allow the shortcut
        const allowInInput =
          shortcut.key === 'Enter' && (shortcut.meta || shortcut.ctrl);

        if (keyMatch && ctrlMatch && metaMatch && shiftMatch) {
          // Skip if in input unless explicitly allowed
          if (isInput && !allowInInput) continue;

          if (shortcut.preventDefault !== false) {
            event.preventDefault();
          }
          shortcut.handler();
          break;
        }
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [shortcuts, enabled]);
}

/**
 * Predefined shortcuts for common chat actions.
 *
 * @param handlers - Object containing handler functions
 * @returns Array of keyboard shortcuts
 */
export function useChatKeyboardShortcuts(handlers: {
  onNewChat?: () => void;
  onSendMessage?: () => void;
  onFocusSearch?: () => void;
  onStopResearch?: () => void;
}) {
  const isMac =
    typeof navigator !== 'undefined' && navigator.platform.includes('Mac');

  const shortcuts = React.useMemo(() => {
    const result: KeyboardShortcut[] = [];

    // Cmd/Ctrl + N - New chat
    if (handlers.onNewChat) {
      result.push({
        key: 'n',
        meta: isMac,
        ctrl: !isMac,
        handler: handlers.onNewChat,
        preventDefault: true,
      });
    }

    // Cmd/Ctrl + Enter - Send message (handled in MessageInput usually)
    if (handlers.onSendMessage) {
      result.push({
        key: 'Enter',
        meta: isMac,
        ctrl: !isMac,
        handler: handlers.onSendMessage,
        preventDefault: true,
      });
    }

    // Cmd/Ctrl + K - Focus search
    if (handlers.onFocusSearch) {
      result.push({
        key: 'k',
        meta: isMac,
        ctrl: !isMac,
        handler: handlers.onFocusSearch,
        preventDefault: true,
      });
    }

    // Escape - Stop research (only when appropriate)
    if (handlers.onStopResearch) {
      result.push({
        key: 'Escape',
        handler: handlers.onStopResearch,
        preventDefault: false, // Let dialogs also handle escape
      });
    }

    return result;
  }, [handlers, isMac]);

  useKeyboardShortcuts({ shortcuts });
}

/**
 * Get a human-readable shortcut string for display.
 *
 * @param shortcut - Shortcut configuration
 * @returns Formatted string like "Cmd+N" or "Ctrl+N"
 */
export function formatShortcut(shortcut: Omit<KeyboardShortcut, 'handler'>): string {
  const isMac =
    typeof navigator !== 'undefined' && navigator.platform.includes('Mac');
  const parts: string[] = [];

  if (shortcut.ctrl) {
    parts.push(isMac ? '⌃' : 'Ctrl');
  }
  if (shortcut.meta) {
    parts.push(isMac ? '⌘' : 'Ctrl');
  }
  if (shortcut.shift) {
    parts.push(isMac ? '⇧' : 'Shift');
  }

  // Format special keys
  let keyName = shortcut.key;
  if (keyName === 'Enter') keyName = '↵';
  if (keyName === 'Escape') keyName = 'Esc';
  if (keyName.length === 1) keyName = keyName.toUpperCase();

  parts.push(keyName);

  return parts.join(isMac ? '' : '+');
}
