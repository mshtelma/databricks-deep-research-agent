/**
 * Editor Toolbar with mode toggle, highlight badges, and actions.
 */

import { Button } from '@/components/ui/button';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip';
import { Sparkles, FilePlus, Trash2, Undo2, Search, Keyboard } from 'lucide-react';
import type { EditType, Highlight, HighlightType, MCPShortcut } from './types';

const LLM_TYPES = ['REMOVE', 'LESS', 'MORE', 'CUSTOM'];
const MCP_TYPES = ['genie', 'vector', 'external', 'knowledge_assistant'];

interface EditorToolbarProps {
  isAIditorMode: boolean;
  activeHighlighter: HighlightType | null;
  activeEndpointId: string | null;
  isKeyboardSelecting: boolean;
  pendingSelectionText: string | null;
  highlights: Highlight[];
  mcpShortcuts: MCPShortcut[];
  isProcessing: boolean;
  hasResult: boolean;
  canUndo: boolean;
  onToggleMode: () => void;
  onSetHighlighter: (type: HighlightType | null, instruction?: string) => void;
  onToggleKeyboardSelecting: () => void;
  onProcess: () => void;
  onQueryTools: () => void;
  onClearHighlights: () => void;
  onUndoHighlight: () => void;
  onNewDocument: () => void;
}

// Edit type configuration with colors matching the reference design
const EDIT_TYPES: { type: EditType; key: string; label: string; color: string; bgColor: string }[] = [
  { type: 'REMOVE', key: 'A', label: 'Remove', color: '#FF6B6B', bgColor: 'rgba(255, 107, 107, 0.15)' },
  { type: 'LESS', key: 'S', label: 'Less', color: '#FFB347', bgColor: 'rgba(255, 179, 71, 0.15)' },
  { type: 'MORE', key: 'D', label: 'More', color: '#7EC8E3', bgColor: 'rgba(126, 200, 227, 0.15)' },
  { type: 'CUSTOM', key: 'F', label: 'Custom', color: '#77DD77', bgColor: 'rgba(119, 221, 119, 0.15)' },
];

export function EditorToolbar({
  isAIditorMode,
  activeHighlighter,
  activeEndpointId,
  isKeyboardSelecting,
  pendingSelectionText,
  highlights,
  mcpShortcuts,
  isProcessing,
  hasResult,
  canUndo,
  onToggleMode,
  onSetHighlighter,
  onToggleKeyboardSelecting,
  onProcess,
  onQueryTools,
  onClearHighlights,
  onUndoHighlight,
  onNewDocument,
}: EditorToolbarProps) {
  const llmHighlightCount = highlights.filter((h) => LLM_TYPES.includes(h.type)).length;
  const mcpHighlightCount = highlights.filter((h) => MCP_TYPES.includes(h.type)).length;
  return (
    <TooltipProvider>
      <div className="space-y-3 mb-4">
        {/* Main toolbar row */}
        <div className="flex items-center gap-4 flex-wrap">
          {/* AIditor Mode Toggle */}
          <div className="flex items-center gap-2">
            <Switch
              id="aiditor-mode"
              checked={isAIditorMode}
              onCheckedChange={onToggleMode}
            />
            <Label htmlFor="aiditor-mode" className="text-sm font-medium cursor-pointer">
              AIditor Mode
            </Label>
          </div>

          {/* Keyboard Selection Mode Toggle */}
          {isAIditorMode && (
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant={isKeyboardSelecting ? 'default' : 'outline'}
                  size="sm"
                  onClick={onToggleKeyboardSelecting}
                  className={isKeyboardSelecting ? 'bg-blue-500 hover:bg-blue-600 text-white' : ''}
                >
                  <Keyboard className="h-4 w-4 mr-1" />
                  <kbd className="px-1 py-0.5 bg-black/10 dark:bg-white/10 rounded text-[10px]">/</kbd>
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                <p>
                  {isKeyboardSelecting
                    ? 'Keyboard selection active — Shift+Arrow to select, Enter to confirm'
                    : 'Toggle keyboard selection mode (/)'}
                </p>
              </TooltipContent>
            </Tooltip>
          )}

          {/* Highlight Mode Badges */}
          {isAIditorMode && !isKeyboardSelecting && (
            <div className="flex items-center gap-1">
              <span className="text-xs text-muted-foreground mr-1">Highlight mode:</span>
              {EDIT_TYPES.map(({ type, key, label, color, bgColor }) => (
                <Tooltip key={type}>
                  <TooltipTrigger asChild>
                    <button
                      onClick={() => onSetHighlighter(activeHighlighter === type ? null : type)}
                      className={`
                        inline-flex items-center gap-1 px-2 py-1 rounded-md text-xs font-medium
                        transition-all duration-150 border
                        ${activeHighlighter === type 
                          ? 'ring-2 ring-offset-1 ring-offset-background' 
                          : 'hover:opacity-80'}
                      `}
                      style={{
                        backgroundColor: activeHighlighter === type ? color : bgColor,
                        borderColor: color,
                        color: activeHighlighter === type ? '#fff' : color,
                      }}
                    >
                      <span
                        className="w-2.5 h-2.5 rounded-full"
                        style={{ backgroundColor: color }}
                      />
                      {label}
                      <kbd className="ml-0.5 px-1 py-0.5 bg-black/10 dark:bg-white/10 rounded text-[10px]">
                        {key}
                      </kbd>
                    </button>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>{label} - Press {key} to activate</p>
                  </TooltipContent>
                </Tooltip>
              ))}

              {/* Custom Shortcut Badges (MCP + LLM) */}
              {mcpShortcuts.length > 0 && (
                <>
                  <span className="w-px h-4 bg-border mx-1" />
                  {mcpShortcuts.map((shortcut) => {
                    const isLLM = shortcut.endpointType === 'llm';
                    const displayLabel = isLLM
                      ? (shortcut.instruction?.substring(0, 12) || 'LLM')
                      : shortcut.endpointName.substring(0, 10);
                    // A shortcut is "active" when the highlighter type matches
                    // AND the endpoint ID matches (so G and B don't both light up)
                    const isActive = isLLM
                      ? activeHighlighter === 'CUSTOM'
                      : activeHighlighter === shortcut.endpointType
                        && activeEndpointId === shortcut.endpointId;
                    return (
                      <Tooltip key={shortcut.key}>
                        <TooltipTrigger asChild>
                          <button
                            onClick={() => {
                              if (isLLM) {
                                onSetHighlighter('CUSTOM', shortcut.instruction);
                              } else {
                                const epType = shortcut.endpointType as HighlightType;
                                onSetHighlighter(
                                  isActive ? null : epType
                                );
                              }
                            }}
                            className={`
                              inline-flex items-center gap-1 px-2 py-1 rounded-md text-xs font-medium
                              transition-all duration-150 border
                              ${isActive 
                                ? 'ring-2 ring-offset-1 ring-offset-background' 
                                : 'hover:opacity-80'}
                            `}
                            style={{
                              backgroundColor: isActive 
                                ? shortcut.color 
                                : `${shortcut.color}26`,
                              borderColor: shortcut.color,
                              color: isActive ? '#fff' : shortcut.color,
                            }}
                          >
                            <span
                              className="w-2.5 h-2.5 rounded-full"
                              style={{ backgroundColor: shortcut.color }}
                            />
                            {displayLabel}
                            <kbd className="ml-0.5 px-1 py-0.5 bg-black/10 dark:bg-white/10 rounded text-[10px]">
                              {shortcut.key}
                            </kbd>
                          </button>
                        </TooltipTrigger>
                        <TooltipContent>
                          <p>
                            {isLLM ? `LLM: ${shortcut.instruction || 'Custom instruction'}` : shortcut.endpointName}
                            {' - Press '}{shortcut.key}{' to activate'}
                          </p>
                        </TooltipContent>
                      </Tooltip>
                    );
                  })}
                </>
              )}
            </div>
          )}

          {/* Spacer */}
          <div className="flex-1" />

          {/* Action buttons */}
          <div className="flex items-center gap-2">
            {/* New Document */}
            <Tooltip>
              <TooltipTrigger asChild>
                <Button variant="outline" size="sm" onClick={onNewDocument}>
                  <FilePlus className="h-4 w-4 mr-1" />
                  New
                </Button>
              </TooltipTrigger>
              <TooltipContent>Create new document</TooltipContent>
            </Tooltip>

            {/* Undo Highlight */}
            {canUndo && (
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button variant="outline" size="sm" onClick={onUndoHighlight}>
                    <Undo2 className="h-4 w-4 mr-1" />
                    Undo
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Undo last highlight change (⌘Z)</TooltipContent>
              </Tooltip>
            )}

            {/* Clear Highlights */}
            {highlights.length > 0 && (
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button variant="outline" size="sm" onClick={onClearHighlights}>
                    <Trash2 className="h-4 w-4 mr-1" />
                    Clear ({highlights.length})
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Clear all highlights</TooltipContent>
              </Tooltip>
            )}

            {/* Query Tools (MCP only — no LLM edits queued) */}
            {mcpHighlightCount > 0 && llmHighlightCount === 0 && (
              <Button
                variant="default"
                size="sm"
                onClick={onQueryTools}
                disabled={isProcessing || hasResult}
                className="bg-primary hover:bg-primary/90 gap-1"
              >
                <Search className="h-4 w-4" />
                {isProcessing ? 'Querying...' : `Query Tools (${mcpHighlightCount})`}
              </Button>
            )}

            {/* Process Edits — LLM only, or LLM + MCP combined */}
            {llmHighlightCount > 0 && (
              <Button
                variant="default"
                size="sm"
                onClick={onProcess}
                disabled={isProcessing || hasResult}
                className="bg-primary hover:bg-primary/90"
              >
                <Sparkles className="h-4 w-4 mr-1" />
                {isProcessing
                  ? (mcpHighlightCount > 0 ? 'Fetching data & editing...' : 'Processing...')
                  : mcpHighlightCount > 0
                    ? `Process Edits (${mcpHighlightCount} queries + ${llmHighlightCount} edits)`
                    : `Process Edits (${llmHighlightCount})`
                }
              </Button>
            )}
          </div>
        </div>

        {/* Instruction text */}
        {isAIditorMode && (
          <p className="text-xs text-muted-foreground">
            {isKeyboardSelecting
              ? pendingSelectionText
                ? 'Press a command key: A (remove) · S (shorten) · D (expand) · F (custom) · G/B/R/T (MCP tools)'
                : 'Use Shift+Arrow to select text, Shift+⌥Arrow for words, Shift+⌘Arrow for lines. Press Enter to confirm.'
              : 'Press a shortcut key to activate, then select text. Press / for keyboard selection mode.'
            }
          </p>
        )}
      </div>
    </TooltipProvider>
  );
}
