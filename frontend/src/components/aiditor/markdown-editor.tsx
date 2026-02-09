/**
 * Markdown Editor component with edit mode and AIditor (highlight) mode.
 */

import { useCallback, useEffect, useRef } from 'react';
import { marked } from 'marked';
import { Textarea } from '@/components/ui/textarea';
import { ScrollArea } from '@/components/ui/scroll-area';
import type { Highlight, HighlightType, MCPShortcut } from './types';

interface MarkdownEditorProps {
  markdown: string;
  onChange: (value: string) => void;
  isAIditorMode: boolean;
  activeHighlighter: HighlightType | null;
  isKeyboardSelecting: boolean;
  pendingSelectionText: string | null;
  pendingInstruction?: string | null;
  pendingEndpointId?: string | null;
  pendingEndpointName?: string | null;
  highlights: Highlight[];
  mcpShortcuts: MCPShortcut[];
  onAddHighlight: (
    type: HighlightType,
    text: string,
    startOffset: number,
    endOffset: number,
    instruction?: string,
    endpointId?: string,
    endpointName?: string,
  ) => void;
  onRemoveHighlight: (id: string) => void;
  onRequestCustomInstruction?: (text: string, callback: (instruction: string) => void) => void;
}

// =============================================================================
// DOM-based Highlight Utilities
// =============================================================================

/**
 * Walk all text nodes under `root` in document order,
 * building a flat "text position → node" mapping.
 */
function getTextNodes(root: Node): { node: Text; offset: number }[] {
  const result: { node: Text; offset: number }[] = [];
  let currentOffset = 0;

  function walk(node: Node) {
    if (node.nodeType === Node.TEXT_NODE) {
      result.push({ node: node as Text, offset: currentOffset });
      currentOffset += (node as Text).textContent?.length || 0;
    } else {
      for (const child of Array.from(node.childNodes)) {
        walk(child);
      }
    }
  }

  walk(root);
  return result;
}

/**
 * Find the start offset within the full text of a container,
 * where `searchText` begins. Uses a normalized comparison that
 * collapses whitespace, so cross-element selections match.
 */
function findTextRange(
  fullText: string,
  searchText: string,
): { start: number; end: number } | null {
  // Normalize: collapse whitespace runs to single spaces
  const normalizedFull = fullText.replace(/\s+/g, ' ');
  const normalizedSearch = searchText.replace(/\s+/g, ' ');

  const idx = normalizedFull.indexOf(normalizedSearch);
  if (idx === -1) return null;

  // Map normalized index back to original string positions
  let origIdx = 0;
  let normIdx = 0;
  let start = -1;
  let end = -1;

  while (origIdx <= fullText.length && normIdx <= normalizedFull.length) {
    if (normIdx === idx && start === -1) {
      start = origIdx;
    }
    if (normIdx === idx + normalizedSearch.length) {
      end = origIdx;
      break;
    }
    if (origIdx < fullText.length) {
      const origChar = fullText[origIdx]!;
      const normChar = normalizedFull[normIdx]!;
      if (origChar === normChar) {
        origIdx++;
        normIdx++;
      } else if (/\s/.test(origChar) && normIdx < normalizedFull.length && /\s/.test(normChar)) {
        // Both are whitespace - this is part of a normalized whitespace run
        origIdx++;
        // Only advance normIdx if we're moving to the normalized single space
        if (origIdx >= fullText.length || !/\s/.test(fullText[origIdx]!)) {
          normIdx++;
        }
      } else if (/\s/.test(origChar)) {
        // Extra whitespace in original (not in normalized), skip without advancing normIdx
        origIdx++;
      } else {
        // Characters don't match but neither is whitespace - advance both
        origIdx++;
        normIdx++;
      }
    } else {
      break;
    }
  }

  if (start === -1 || end === -1) return null;
  return { start, end };
}

/**
 * Apply highlight spans to the DOM by wrapping text nodes.
 * This handles cross-element selections because it operates on text nodes directly.
 */
function applyHighlightToDOM(
  container: HTMLElement,
  highlight: Highlight,
) {
  const textNodes = getTextNodes(container);
  if (textNodes.length === 0) return;

  // Build the full text from all text nodes
  const fullText = textNodes.map(({ node }) => node.textContent || '').join('');

  // Find where the highlight text appears in the full text
  const range = findTextRange(fullText, highlight.text);
  if (!range) return;

  const { start, end } = range;

  // Find which text nodes to wrap
  const nodesToWrap: { node: Text; startInNode: number; endInNode: number }[] = [];

  for (const { node, offset } of textNodes) {
    const nodeLen = node.textContent?.length || 0;
    const nodeEnd = offset + nodeLen;

    // Does this node overlap with our highlight range?
    if (nodeEnd <= start || offset >= end) continue;

    const startInNode = Math.max(0, start - offset);
    const endInNode = Math.min(nodeLen, end - offset);
    nodesToWrap.push({ node, startInNode, endInNode });
  }

  // Wrap text portions in highlight spans (reverse order to avoid offset shifts)
  for (const { node, startInNode, endInNode } of nodesToWrap.reverse()) {
    const text = node.textContent || '';
    if (startInNode === 0 && endInNode === text.length) {
      // Wrap entire text node
      const span = createHighlightSpan(highlight);
      span.textContent = text;
      node.parentNode?.replaceChild(span, node);
    } else {
      // Split the text node and wrap the middle part
      const before = text.slice(0, startInNode);
      const middle = text.slice(startInNode, endInNode);
      const after = text.slice(endInNode);

      const parent = node.parentNode;
      if (!parent) continue;

      const fragment = document.createDocumentFragment();
      if (before) fragment.appendChild(document.createTextNode(before));

      const span = createHighlightSpan(highlight);
      span.textContent = middle;
      fragment.appendChild(span);

      if (after) fragment.appendChild(document.createTextNode(after));

      parent.replaceChild(fragment, node);
    }
  }
}

function createHighlightSpan(highlight: Highlight): HTMLSpanElement {
  const span = document.createElement('span');
  span.className = 'aiditor-highlight';
  span.dataset.highlightId = highlight.id;
  span.style.backgroundColor = `${highlight.color}40`;
  span.style.borderBottom = `2px solid ${highlight.color}`;
  span.style.cursor = 'pointer';
  span.title = highlight.instruction
    ? `${highlight.type}: ${highlight.instruction}`
    : highlight.type;
  return span;
}

// =============================================================================
// Component
// =============================================================================

// Built-in edit type badges for the command palette
const EDIT_TYPE_BADGES: { type: string; key: string; label: string; color: string }[] = [
  { type: 'REMOVE', key: 'A', label: 'Remove', color: '#FF6B6B' },
  { type: 'LESS', key: 'S', label: 'Less', color: '#FFB347' },
  { type: 'MORE', key: 'D', label: 'More', color: '#7EC8E3' },
  { type: 'CUSTOM', key: 'F', label: 'Custom', color: '#77DD77' },
];

export function MarkdownEditor({
  markdown,
  onChange,
  isAIditorMode,
  activeHighlighter,
  isKeyboardSelecting,
  pendingSelectionText,
  pendingInstruction,
  pendingEndpointId,
  pendingEndpointName,
  highlights,
  mcpShortcuts,
  onAddHighlight,
  onRemoveHighlight,
  onRequestCustomInstruction,
}: MarkdownEditorProps) {
  const contentRef = useRef<HTMLDivElement>(null);

  // Handle text selection in AIditor mode
  const handleMouseUp = useCallback(() => {
    if (!isAIditorMode || !activeHighlighter) return;

    const selection = window.getSelection();
    if (!selection || selection.isCollapsed) return;

    const text = selection.toString().trim();
    if (!text) return;

    // For CUSTOM type with a pre-filled instruction (from LLM shortcut), use it directly
    if (activeHighlighter === 'CUSTOM' && pendingInstruction) {
      onAddHighlight(activeHighlighter, text, 0, text.length, pendingInstruction);
    } else if (activeHighlighter === 'CUSTOM') {
      // For CUSTOM type without pre-filled instruction, request via modal
      if (onRequestCustomInstruction) {
        onRequestCustomInstruction(text, (instruction) => {
          onAddHighlight(activeHighlighter, text, 0, text.length, instruction);
        });
      }
    } else {
      // Add highlight directly, threading through MCP endpoint info if present
      onAddHighlight(
        activeHighlighter,
        text,
        0,
        text.length,
        undefined,
        pendingEndpointId ?? undefined,
        pendingEndpointName ?? undefined,
      );
    }

    // Clear selection
    selection.removeAllRanges();
  }, [isAIditorMode, activeHighlighter, pendingInstruction, pendingEndpointId, pendingEndpointName, onAddHighlight, onRequestCustomInstruction]);

  // Render base markdown to HTML (without highlights)
  const renderBaseHTML = useCallback(() => {
    return marked.parse(markdown, { async: false }) as string;
  }, [markdown]);

  // Apply highlights via DOM manipulation after render
  useEffect(() => {
    if (!contentRef.current || !isAIditorMode) return;

    // Reset to base HTML first
    contentRef.current.innerHTML = renderBaseHTML();

    // Apply each highlight via DOM traversal
    for (const highlight of highlights) {
      applyHighlightToDOM(contentRef.current, highlight);
    }
  }, [markdown, highlights, isAIditorMode, renderBaseHTML]);

  // Handle click on highlights to remove them
  const handleContentClick = useCallback(
    (e: React.MouseEvent) => {
      const target = e.target as HTMLElement;
      if (target.classList.contains('aiditor-highlight')) {
        const highlightId = target.dataset.highlightId;
        if (highlightId) {
          onRemoveHighlight(highlightId);
        }
      }
    },
    [onRemoveHighlight]
  );

  // Focus the content div when entering keyboard selection mode
  useEffect(() => {
    if (isKeyboardSelecting && contentRef.current) {
      contentRef.current.focus();
      // Place the caret at the start so the blinking cursor is visible
      const sel = window.getSelection();
      if (sel && contentRef.current.firstChild) {
        sel.collapse(contentRef.current.firstChild, 0);
      }
    }
  }, [isKeyboardSelecting]);

  // Block typing in keyboard selection mode — allow only navigation & selection keys.
  // contentEditable is required for the visible blinking caret, but we must
  // prevent the user from actually modifying the rendered markdown.
  const handleKeyDownInContent = useCallback(
    (e: React.KeyboardEvent) => {
      if (!isKeyboardSelecting) return;

      // ALWAYS block Enter in the contentEditable — it must NOT insert a newline.
      // The global keydown handler in use-aiditor.ts captures Enter for selection
      // confirmation; here we just prevent the DOM mutation.
      if (e.key === 'Enter') {
        e.preventDefault();
        return;
      }

      // When there's a pending selection, block ALL keys in the contentEditable.
      // The global handler will route command keys (A/S/D/F/G/B/R/T etc.)
      // to the appropriate action. We must not let them modify the DOM.
      if (pendingSelectionText) {
        e.preventDefault();
        return;
      }

      // Allow: arrows, shift, alt, meta, ctrl, Home, End, Page Up/Down, Escape, Tab
      const allowed = new Set([
        'ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown',
        'Home', 'End', 'PageUp', 'PageDown',
        'Shift', 'Alt', 'Control', 'Meta',
        'Escape', 'Tab',
      ]);
      if (!allowed.has(e.key) && !e.metaKey && !e.ctrlKey) {
        e.preventDefault();
      }
    },
    [isKeyboardSelecting, pendingSelectionText]
  );

  // Block paste and input events to prevent modification of the rendered content
  const blockMutation = useCallback((e: React.SyntheticEvent) => {
    if (isKeyboardSelecting) e.preventDefault();
  }, [isKeyboardSelecting]);

  // Edit mode: textarea
  if (!isAIditorMode) {
    return (
      <Textarea
        value={markdown}
        onChange={(e) => onChange(e.target.value)}
        placeholder="# Start writing your markdown here...

Paste or type your content, then toggle AIditor mode to mark sections for AI editing."
        className="min-h-[300px] font-mono text-sm border-0 focus-visible:ring-0 p-6"
        style={{ resize: 'vertical' }}
      />
    );
  }

  // AIditor mode: rendered markdown with DOM-applied highlights
  return (
    <div
      className="relative"
      style={{ resize: 'vertical', overflow: 'auto', minHeight: '300px', height: '500px' }}
    >
      <ScrollArea className="h-full">
        <div
          ref={contentRef}
          className={`p-6 markdown-preview min-h-[300px] outline-none ${
            isKeyboardSelecting ? 'ring-2 ring-blue-400 ring-inset' : ''
          }`}
          onMouseUp={handleMouseUp}
          onClick={handleContentClick}
          onKeyDown={handleKeyDownInContent}
          onPaste={blockMutation}
          onCut={blockMutation}
          onDrop={blockMutation}
          // contentEditable gives us a visible blinking caret + native
          // Shift+Arrow selection. We block all mutations via onKeyDown/onPaste.
          contentEditable={isKeyboardSelecting}
          suppressContentEditableWarning
          tabIndex={isKeyboardSelecting ? 0 : -1}
          style={{
            cursor: isKeyboardSelecting
              ? 'text'
              : activeHighlighter
                ? 'crosshair'
                : 'default',
            userSelect: isKeyboardSelecting || activeHighlighter ? 'text' : 'auto',
            caretColor: isKeyboardSelecting ? '#3b82f6' : 'transparent',
          }}
        />
      </ScrollArea>
      {/* Keyboard selection / pending selection indicator */}
      {isKeyboardSelecting && !pendingSelectionText && (
        <div className="absolute bottom-0 left-0 right-0 px-3 py-1.5 bg-blue-500/90 text-white text-xs font-medium backdrop-blur-sm">
          Keyboard Selection Mode — use Shift+Arrow to select text, then press Enter
        </div>
      )}
      {/* Command palette: shows styled shortcut badges after selection is confirmed */}
      {isKeyboardSelecting && pendingSelectionText && (
        <div className="absolute bottom-0 left-0 right-0 bg-gray-900/95 text-white backdrop-blur-sm border-t border-white/10">
          <div className="px-3 py-1.5 text-[11px] text-gray-300 border-b border-white/5 flex items-center gap-2">
            <span className="truncate max-w-[60%]">
              &ldquo;{pendingSelectionText.length > 80 ? pendingSelectionText.slice(0, 80) + '…' : pendingSelectionText}&rdquo;
            </span>
            <span className="text-gray-500 ml-auto shrink-0">Press a key to apply</span>
          </div>
          <div className="px-3 py-2 flex items-center gap-1.5 flex-wrap">
            {/* Built-in edit type keys */}
            {EDIT_TYPE_BADGES.map(({ key, label, color }) => (
              <span
                key={key}
                className="inline-flex items-center gap-1 px-2 py-1 rounded text-xs font-medium border"
                style={{
                  backgroundColor: `${color}20`,
                  borderColor: `${color}60`,
                  color: color,
                }}
              >
                <kbd className="px-1 py-0.5 bg-white/10 rounded text-[10px] font-bold min-w-[18px] text-center">
                  {key}
                </kbd>
                {label}
              </span>
            ))}

            {/* Separator */}
            {mcpShortcuts.length > 0 && (
              <span className="w-px h-5 bg-white/20 mx-0.5" />
            )}

            {/* MCP shortcut keys */}
            {mcpShortcuts.map((shortcut) => (
              <span
                key={shortcut.key}
                className="inline-flex items-center gap-1 px-2 py-1 rounded text-xs font-medium border"
                style={{
                  backgroundColor: `${shortcut.color}20`,
                  borderColor: `${shortcut.color}60`,
                  color: shortcut.color,
                }}
              >
                <kbd className="px-1 py-0.5 bg-white/10 rounded text-[10px] font-bold min-w-[18px] text-center">
                  {shortcut.key}
                </kbd>
                {shortcut.endpointName.length > 12 ? shortcut.endpointName.slice(0, 12) + '…' : shortcut.endpointName}
              </span>
            ))}

            {/* Escape hint */}
            <span className="text-gray-500 text-[10px] ml-auto">
              <kbd className="px-1 py-0.5 bg-white/5 rounded text-[10px]">Esc</kbd> cancel
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
