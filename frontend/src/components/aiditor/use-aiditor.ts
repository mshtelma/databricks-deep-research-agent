/**
 * Core state management hook for AIditor.
 */

import { useCallback, useEffect, useReducer } from 'react';
import type {
  AIditorState,
  EditType,
  Highlight,
  HighlightType,
  MCPShortcut,
  StoredSettings,
} from './types';
import { CUSTOM_SHORTCUT_COLORS, DEFAULT_BUILTIN_SHORTCUTS, HIGHLIGHT_COLORS } from './types';

// =============================================================================
// Initial State
// =============================================================================

const STORAGE_KEY_MARKDOWN = 'aiditor-markdown';
const STORAGE_KEY_SETTINGS = 'aiditor-settings';
const AUTO_SAVE_INTERVAL = 2000;

const MAX_UNDO_STACK = 50;

/**
 * Default MCP shortcuts shipped with the app.
 * Users can customize these in Settings > Shortcuts.
 */
const DEFAULT_MCP_SHORTCUTS: MCPShortcut[] = [
  {
    key: 'G',
    type: 'MCP',
    endpointType: 'genie',
    endpointId: '01f105957f5013769e4c712c8da7dd64',
    endpointName: 'Gas Station Sales Analytics',
    instruction: 'Query sales data: fuel transactions, daily revenue, payment methods, and store sales trends',
    color: CUSTOM_SHORTCUT_COLORS.genie,
    resultBehavior: 'insert_below',
  },
  {
    key: 'B',
    type: 'MCP',
    endpointType: 'genie',
    endpointId: '01f10595808d113eba3e605b21a52991',
    endpointName: 'Gas Station Marketing Analytics',
    instruction: 'Query marketing data: campaigns, loyalty programs, promotions, and member tiers',
    color: CUSTOM_SHORTCUT_COLORS.genie,
    resultBehavior: 'insert_below',
  },
  {
    key: 'R',
    type: 'MCP',
    endpointType: 'knowledge_assistant',
    endpointId: '112212ba',
    endpointName: 'Retail Operations KA',
    instruction: 'Ask about fuel operations, store management, loyalty programs, safety compliance, and financial KPIs',
    color: CUSTOM_SHORTCUT_COLORS.knowledge_assistant,
    resultBehavior: 'insert_below',
  },
  {
    key: 'T',
    type: 'MCP',
    endpointType: 'external',
    endpointId: 'mcp-tavily-can',
    endpointName: 'Web Search (Tavily)',
    instruction: 'Search the web for current information, news, and references',
    color: CUSTOM_SHORTCUT_COLORS.external,
    resultBehavior: 'insert_below',
  },
];

const DEFAULT_MARKDOWN = `# Gas Station Retail Operations Report

AIditor lets you enrich documents with live data from Databricks. Toggle **AIditor Mode** (Cmd+E), press a shortcut key, select text, and click the action button to insert results inline.

## Getting Started

### Mouse Flow (default)
1. Press **Cmd+E** to enable AIditor Mode
2. Press a shortcut key (e.g. **G**) to activate a data source
3. Select text with your mouse — it becomes the query
4. Click the action button that appears

### Keyboard Flow (press / to toggle)
1. Press **/** to enter keyboard selection mode
2. Use **Shift+Arrow** to select characters, **Shift+⌥Arrow** for words, **Shift+⌘Arrow** for lines
3. Press **Enter** to confirm your selection
4. Press a command key (A/S/D/F/G/B/R/T) to apply it

| Key | Source | What it does |
|-----|--------|--------------|
| **G** | Sales Genie | Query fuel & store sales data |
| **B** | Marketing Genie | Query campaigns & loyalty data |
| **R** | Retail Ops KA | Ask the operations knowledge base |
| **T** | Web Search | Search the web via Tavily |
| A / S / D / F | LLM | Remove / Shorten / Expand / Custom edit |

---

## Try It: Sales Data (press G)

Select the question below, then click **Query Tools**.

What are the total gas sales by fuel type for the last month?

## Try It: Marketing Data (press B)

Select the question below, then click **Query Tools**.

What are the loyalty member counts by tier and their average annual spend?

## Try It: Operations Knowledge Base (press R)

Select the question below, then click **Query Tools**.

What are the EPA compliance requirements for underground storage tanks?

## Try It: Web Search (press T)

Select the question below, then click **Query Tools**.

What are the latest trends in gas station convenience store retail?

---

## Combining Data + Edits

You can mix tool queries and LLM edits in a single pass. For example:

1. Press **G**, select a sales question — it gets highlighted for a Genie query.
2. Press **D**, select a paragraph — it gets highlighted for LLM expansion.
3. Click **Process Edits** — AIditor first fetches the tool data, inserts it into the document, then sends the enriched document to the LLM for editing. One click, both done.

You can also queue multiple tool queries (e.g., one **G** + one **R**) and they all run before the LLM step.
`;


const initialState: AIditorState = {
  markdown: DEFAULT_MARKDOWN,
  isAIditorMode: false,
  activeHighlighter: null,
  pendingInstruction: null,
  pendingEndpointId: null,
  pendingEndpointName: null,
  highlights: [],
  highlightUndoStack: [],
  isKeyboardSelecting: false,
  pendingSelectionText: null,
  selectedModel: null,
  isProcessing: false,
  processedResult: null,
  mcpShortcuts: DEFAULT_MCP_SHORTCUTS,
  settingsOpen: false,
  builtinShortcuts: DEFAULT_BUILTIN_SHORTCUTS,
};

// =============================================================================
// Actions
// =============================================================================

export type Action =
  | { type: 'SET_MARKDOWN'; payload: string }
  | { type: 'TOGGLE_AIDITOR_MODE' }
  | { type: 'SET_ACTIVE_HIGHLIGHTER'; payload: { highlighter: HighlightType | null; instruction?: string | null; endpointId?: string | null; endpointName?: string | null } }
  | { type: 'ADD_HIGHLIGHT'; payload: Highlight }
  | { type: 'REMOVE_HIGHLIGHT'; payload: string }
  | { type: 'CLEAR_HIGHLIGHTS' }
  | { type: 'UNDO_HIGHLIGHT' }
  | { type: 'TOGGLE_KEYBOARD_SELECTING' }
  | { type: 'SET_PENDING_SELECTION'; payload: string | null }
  | { type: 'SET_SELECTED_MODEL'; payload: string | null }
  | { type: 'SET_PROCESSING'; payload: boolean }
  | { type: 'SET_PROCESSED_RESULT'; payload: string | null }
  | { type: 'SET_MCP_SHORTCUTS'; payload: MCPShortcut[] }
  | { type: 'SET_SETTINGS_OPEN'; payload: boolean }
  | { type: 'SET_BUILTIN_SHORTCUTS'; payload: typeof DEFAULT_BUILTIN_SHORTCUTS }
  | { type: 'ACCEPT_RESULT' }
  | { type: 'DISCARD_RESULT' }
  | { type: 'LOAD_STATE'; payload: Partial<AIditorState> };

export function reducer(state: AIditorState, action: Action): AIditorState {
  switch (action.type) {
    case 'SET_MARKDOWN':
      return { ...state, markdown: action.payload };

    case 'TOGGLE_AIDITOR_MODE':
      return {
        ...state,
        isAIditorMode: !state.isAIditorMode,
        activeHighlighter: null,
        pendingInstruction: null,
        pendingEndpointId: null,
        pendingEndpointName: null,
        isKeyboardSelecting: false,
        pendingSelectionText: null,
      };

    case 'SET_ACTIVE_HIGHLIGHTER':
      return {
        ...state,
        activeHighlighter: action.payload.highlighter,
        pendingInstruction: action.payload.instruction ?? null,
        pendingEndpointId: action.payload.endpointId ?? null,
        pendingEndpointName: action.payload.endpointName ?? null,
      };

    case 'ADD_HIGHLIGHT':
      return {
        ...state,
        highlights: [...state.highlights, action.payload],
        highlightUndoStack: [...state.highlightUndoStack, state.highlights].slice(-MAX_UNDO_STACK),
      };

    case 'REMOVE_HIGHLIGHT':
      return {
        ...state,
        highlights: state.highlights.filter((h) => h.id !== action.payload),
        highlightUndoStack: [...state.highlightUndoStack, state.highlights].slice(-MAX_UNDO_STACK),
      };

    case 'CLEAR_HIGHLIGHTS':
      return {
        ...state,
        highlights: [],
        highlightUndoStack: state.highlights.length > 0
          ? [...state.highlightUndoStack, state.highlights].slice(-MAX_UNDO_STACK)
          : state.highlightUndoStack,
      };

    case 'UNDO_HIGHLIGHT': {
      if (state.highlightUndoStack.length === 0) return state;
      const newStack = [...state.highlightUndoStack];
      const previousHighlights = newStack.pop()!;
      return {
        ...state,
        highlights: previousHighlights,
        highlightUndoStack: newStack,
      };
    }

    case 'TOGGLE_KEYBOARD_SELECTING':
      return {
        ...state,
        isKeyboardSelecting: !state.isKeyboardSelecting,
        pendingSelectionText: null,
        // Clear the active highlighter when entering keyboard selection mode
        // (user selects first, then picks command)
        activeHighlighter: !state.isKeyboardSelecting ? null : state.activeHighlighter,
      };

    case 'SET_PENDING_SELECTION':
      return { ...state, pendingSelectionText: action.payload };

    case 'SET_SELECTED_MODEL':
      return { ...state, selectedModel: action.payload };

    case 'SET_PROCESSING':
      return { ...state, isProcessing: action.payload };

    case 'SET_PROCESSED_RESULT':
      return { ...state, processedResult: action.payload };

    case 'SET_MCP_SHORTCUTS':
      return { ...state, mcpShortcuts: action.payload };

    case 'SET_SETTINGS_OPEN':
      return { ...state, settingsOpen: action.payload };

    case 'SET_BUILTIN_SHORTCUTS':
      return { ...state, builtinShortcuts: action.payload };

    case 'ACCEPT_RESULT':
      return {
        ...state,
        markdown: state.processedResult || state.markdown,
        processedResult: null,
        highlights: [],
        highlightUndoStack: [],
      };

    case 'DISCARD_RESULT':
      return { ...state, processedResult: null };

    case 'LOAD_STATE':
      return { ...state, ...action.payload };

    default:
      return state;
  }
}

// =============================================================================
// Hook
// =============================================================================

export function useAIditor() {
  const [state, dispatch] = useReducer(reducer, initialState);

  // Load saved state on mount
  useEffect(() => {
    const savedMarkdown = localStorage.getItem(STORAGE_KEY_MARKDOWN);
    const savedSettings = localStorage.getItem(STORAGE_KEY_SETTINGS);

    const loadedState: Partial<AIditorState> = {};

    if (savedMarkdown) {
      loadedState.markdown = savedMarkdown;
    }

    if (savedSettings) {
      try {
        const settings: StoredSettings = JSON.parse(savedSettings);
        if (settings.selectedModel) {
          loadedState.selectedModel = settings.selectedModel;
        }
        if (settings.mcpShortcuts && settings.mcpShortcuts.length > 0) {
          loadedState.mcpShortcuts = settings.mcpShortcuts;
        }
        // If user has no saved MCP shortcuts, keep the defaults from initialState
        if (settings.builtinShortcuts) {
          loadedState.builtinShortcuts = settings.builtinShortcuts;
        }
      } catch {
        console.warn('Failed to parse stored settings');
      }
    }

    if (Object.keys(loadedState).length > 0) {
      dispatch({ type: 'LOAD_STATE', payload: loadedState });
    }
  }, []);

  // Auto-save markdown
  useEffect(() => {
    const timer = setInterval(() => {
      if (state.markdown) {
        localStorage.setItem(STORAGE_KEY_MARKDOWN, state.markdown);
      }
    }, AUTO_SAVE_INTERVAL);

    return () => clearInterval(timer);
  }, [state.markdown]);

  // Save settings when they change
  useEffect(() => {
    const settings: StoredSettings = {
      selectedModel: state.selectedModel,
      mcpShortcuts: state.mcpShortcuts,
      llmShortcuts: [],
      builtinShortcuts: state.builtinShortcuts,
    };
    localStorage.setItem(STORAGE_KEY_SETTINGS, JSON.stringify(settings));
  }, [state.selectedModel, state.mcpShortcuts, state.builtinShortcuts]);

  // =============================================================================
  // Actions
  // =============================================================================

  const setMarkdown = useCallback((markdown: string) => {
    dispatch({ type: 'SET_MARKDOWN', payload: markdown });
  }, []);

  const toggleAIditorMode = useCallback(() => {
    dispatch({ type: 'TOGGLE_AIDITOR_MODE' });
  }, []);

  const setActiveHighlighter = useCallback((highlighter: HighlightType | null, instruction?: string, endpointId?: string, endpointName?: string) => {
    dispatch({ type: 'SET_ACTIVE_HIGHLIGHTER', payload: { highlighter, instruction, endpointId, endpointName } });
  }, []);

  const addHighlight = useCallback(
    (type: HighlightType, text: string, startOffset: number, endOffset: number, instruction?: string, endpointId?: string, endpointName?: string) => {
      const highlight: Highlight = {
        id: crypto.randomUUID(),
        type,
        text,
        startOffset,
        endOffset,
        instruction,
        endpointId,
        endpointName,
        color: HIGHLIGHT_COLORS[type],
      };
      dispatch({ type: 'ADD_HIGHLIGHT', payload: highlight });
      return highlight;
    },
    []
  );

  const removeHighlight = useCallback((id: string) => {
    dispatch({ type: 'REMOVE_HIGHLIGHT', payload: id });
  }, []);

  const clearHighlights = useCallback(() => {
    dispatch({ type: 'CLEAR_HIGHLIGHTS' });
  }, []);

  const undoHighlight = useCallback(() => {
    dispatch({ type: 'UNDO_HIGHLIGHT' });
  }, []);

  const setSelectedModel = useCallback((model: string | null) => {
    dispatch({ type: 'SET_SELECTED_MODEL', payload: model });
  }, []);

  const setProcessing = useCallback((processing: boolean) => {
    dispatch({ type: 'SET_PROCESSING', payload: processing });
  }, []);

  const setProcessedResult = useCallback((result: string | null) => {
    dispatch({ type: 'SET_PROCESSED_RESULT', payload: result });
  }, []);

  const setMcpShortcuts = useCallback((shortcuts: MCPShortcut[]) => {
    dispatch({ type: 'SET_MCP_SHORTCUTS', payload: shortcuts });
  }, []);

  const setSettingsOpen = useCallback((open: boolean) => {
    dispatch({ type: 'SET_SETTINGS_OPEN', payload: open });
  }, []);

  const setBuiltinShortcuts = useCallback((shortcuts: typeof DEFAULT_BUILTIN_SHORTCUTS) => {
    dispatch({ type: 'SET_BUILTIN_SHORTCUTS', payload: shortcuts });
  }, []);

  const toggleKeyboardSelecting = useCallback(() => {
    dispatch({ type: 'TOGGLE_KEYBOARD_SELECTING' });
  }, []);

  const setPendingSelection = useCallback((text: string | null) => {
    dispatch({ type: 'SET_PENDING_SELECTION', payload: text });
  }, []);

  const acceptResult = useCallback(() => {
    dispatch({ type: 'ACCEPT_RESULT' });
  }, []);

  const discardResult = useCallback(() => {
    dispatch({ type: 'DISCARD_RESULT' });
  }, []);

  // =============================================================================
  // Keyboard Handler
  // =============================================================================

  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      const isInput = target.tagName === 'INPUT' || target.tagName === 'TEXTAREA';
      const isContentEditable = target.isContentEditable;
      const isTypingContext = isInput || isContentEditable;

      const normalizeKey = (value: string) => value.toLowerCase().trim();

      const parseShortcut = (value: string) => {
        const cleaned = value.replace('⌘', 'cmd+').replace('⌥', 'alt+').replace('⌃', 'ctrl+');
        const parts = cleaned.split('+').map((part) => part.trim().toLowerCase()).filter(Boolean);
        return {
          key: parts[parts.length - 1] || '',
          meta: parts.includes('cmd') || parts.includes('meta'),
          ctrl: parts.includes('ctrl'),
          alt: parts.includes('alt'),
          shift: parts.includes('shift'),
        };
      };

      const matchesShortcut = (shortcut: string) => {
        const parsed = parseShortcut(shortcut);
        const keyMatches = normalizeKey(e.key) === normalizeKey(parsed.key);
        if (!keyMatches) return false;
        if (parsed.meta && !e.metaKey) return false;
        if (parsed.ctrl && !e.ctrlKey) return false;
        if (parsed.alt && !e.altKey) return false;
        if (parsed.shift && !e.shiftKey) return false;
        return true;
      };

      const shortcutById = (id: string) =>
        state.builtinShortcuts.find((shortcut) => shortcut.id === id)?.key;

      // FIX #4: Toggle mode (Cmd+E) must work EVERYWHERE, including inside textareas
      const toggleKey = shortcutById('toggle-mode');
      if (toggleKey && matchesShortcut(toggleKey)) {
        e.preventDefault();
        toggleAIditorMode();
        return;
      }

      // Undo highlight: Cmd+Z / Ctrl+Z in AIditor mode
      // This intercepts before the textarea's native undo so highlights
      // are undone first; once the undo stack is empty, native undo resumes.
      if (state.isAIditorMode && state.highlightUndoStack.length > 0) {
        const isUndo = (e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'z' && !e.shiftKey;
        if (isUndo) {
          e.preventDefault();
          undoHighlight();
          return;
        }
      }

      // Only handle other shortcuts in AIditor mode
      if (!state.isAIditorMode) {
        return;
      }

      // FIX #5: When typing in any input/textarea, suppress single-key shortcuts
      // so the user can type freely (e.g. in custom instruction text fields).
      // Escape is still allowed since it's a control key, not a character.
      // EXCEPTION: In keyboard selection mode the content div is contentEditable
      // (for the blinking caret), but we still need to handle command keys.
      if (isTypingContext && e.key !== 'Escape' && !state.isKeyboardSelecting) {
        return;
      }

      const key = e.key.toLowerCase();

      // Escape: Deactivate highlighter OR exit keyboard selection mode
      const escapeKey = shortcutById('escape');
      if ((escapeKey && matchesShortcut(escapeKey)) || key === 'escape') {
        e.preventDefault();
        if (state.isKeyboardSelecting) {
          toggleKeyboardSelecting();
        } else {
          setActiveHighlighter(null);
        }
        return;
      }

      // "/" toggles keyboard selection mode
      if (key === '/') {
        e.preventDefault();
        toggleKeyboardSelecting();
        return;
      }

      // --- Keyboard selection mode: Enter confirms selection ---
      if (state.isKeyboardSelecting && !state.pendingSelectionText) {
        // In keyboard selection mode, allow Shift+Arrow / Shift+Opt+Arrow /
        // Shift+Cmd+Arrow to pass through for native text selection.
        // Only intercept Enter to confirm the selection.
        if (key === 'enter') {
          e.preventDefault();
          const sel = window.getSelection();
          const text = sel?.toString().trim() || '';
          if (text) {
            setPendingSelection(text);
          }
        }
        // Let all other keys pass through for native selection behavior
        return;
      }

      // --- Pending selection mode: next key picks the command ---
      if (state.pendingSelectionText) {
        // Build maps of all command keys
        const editTypeMap: Record<string, EditType> = {};
        const removeKey = shortcutById('remove');
        const lessKey = shortcutById('less');
        const moreKey = shortcutById('more');
        const customKey = shortcutById('custom');
        if (removeKey) editTypeMap[normalizeKey(parseShortcut(removeKey).key)] = 'REMOVE';
        if (lessKey) editTypeMap[normalizeKey(parseShortcut(lessKey).key)] = 'LESS';
        if (moreKey) editTypeMap[normalizeKey(parseShortcut(moreKey).key)] = 'MORE';
        if (customKey) editTypeMap[normalizeKey(parseShortcut(customKey).key)] = 'CUSTOM';

        const pendingText = state.pendingSelectionText;

        if (editTypeMap[key]) {
          e.preventDefault();
          addHighlight(editTypeMap[key], pendingText, 0, pendingText.length);
          setPendingSelection(null);
          // Stay in keyboard selection mode for more selections
          return;
        }

        const mcpShortcut = state.mcpShortcuts.find(
          (s) => s.key.toLowerCase() === key
        );
        if (mcpShortcut) {
          e.preventDefault();
          if (mcpShortcut.endpointType === 'llm') {
            addHighlight('CUSTOM', pendingText, 0, pendingText.length, mcpShortcut.instruction);
          } else {
            addHighlight(
              mcpShortcut.endpointType, pendingText, 0, pendingText.length,
              undefined, mcpShortcut.endpointId, mcpShortcut.endpointName,
            );
          }
          setPendingSelection(null);
          return;
        }
        // Unknown key — ignore, keep pending selection
        return;
      }

      // --- Normal mouse-first mode (existing flow) ---

      // Built-in edit shortcuts
      const editTypeMap: Record<string, EditType> = {};
      const removeKey = shortcutById('remove');
      const lessKey = shortcutById('less');
      const moreKey = shortcutById('more');
      const customKey = shortcutById('custom');
      if (removeKey) editTypeMap[normalizeKey(parseShortcut(removeKey).key)] = 'REMOVE';
      if (lessKey) editTypeMap[normalizeKey(parseShortcut(lessKey).key)] = 'LESS';
      if (moreKey) editTypeMap[normalizeKey(parseShortcut(moreKey).key)] = 'MORE';
      if (customKey) editTypeMap[normalizeKey(parseShortcut(customKey).key)] = 'CUSTOM';

      if (editTypeMap[key]) {
        e.preventDefault();
        setActiveHighlighter(editTypeMap[key]);
        return;
      }

      // Custom shortcuts (MCP and LLM)
      const mcpShortcut = state.mcpShortcuts.find(
        (s) => s.key.toLowerCase() === key
      );
      if (mcpShortcut) {
        e.preventDefault();
        if (mcpShortcut.endpointType === 'llm') {
          // LLM shortcut: activate CUSTOM highlighter with pre-filled instruction
          setActiveHighlighter('CUSTOM', mcpShortcut.instruction);
        } else {
          // MCP shortcut: activate with endpoint info so highlights carry the ID
          setActiveHighlighter(
            mcpShortcut.endpointType,
            mcpShortcut.instruction,
            mcpShortcut.endpointId,
            mcpShortcut.endpointName,
          );
        }
        return;
      }
    },
    [state.isAIditorMode, state.isKeyboardSelecting, state.pendingSelectionText, state.highlightUndoStack.length, state.mcpShortcuts, state.builtinShortcuts, toggleAIditorMode, setActiveHighlighter, undoHighlight, toggleKeyboardSelecting, setPendingSelection, addHighlight]
  );

  // Attach keyboard listener
  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);

  return {
    state,
    actions: {
      setMarkdown,
      toggleAIditorMode,
      setActiveHighlighter,
      addHighlight,
      removeHighlight,
      clearHighlights,
      undoHighlight,
      toggleKeyboardSelecting,
      setPendingSelection,
      setSelectedModel,
      setProcessing,
      setProcessedResult,
      setMcpShortcuts,
      setSettingsOpen,
      setBuiltinShortcuts,
      acceptResult,
      discardResult,
    },
  };
}
