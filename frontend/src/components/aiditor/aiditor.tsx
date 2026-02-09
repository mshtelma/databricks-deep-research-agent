/**
 * Main AIditor component - AI-powered Markdown Editor.
 */

import { useState, useCallback, useEffect } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Card, CardContent } from '@/components/ui/card';
import { useAIditor } from './use-aiditor';
import { Header } from './header';
import { MarkdownEditor } from './markdown-editor';
import { EditorToolbar } from './editor-toolbar';
import { ComparisonView } from './comparison-view';
import { SettingsContent } from './settings-content';
import { CustomInstructionModal } from './custom-instruction-modal';
import type {
  ChatRequest,
  ChatResponse,
  DebugInfo,
  EditInstruction,
  EditType,
  ExternalQueryRequest,
  ExternalQueryResponse,
  GenieQueryRequest,
  GenieQueryResponse,
  Highlight,
  KnowledgeAssistantQueryRequest,
  KnowledgeAssistantQueryResponse,
  VectorSearchRequest,
  VectorSearchResponse,
  MCPEndpoints,
  ModelsResponse,
} from './types';

// API base URL — must match the backend prefix in main.py
const API_BASE = '/api/aiditor';

interface AIditorProps {
  /** Content from chat to load into editor */
  initialContent?: string | null;
  /** Called after initialContent has been consumed (prevents re-load) */
  onContentConsumed?: () => void;
  /** Callback to export current markdown back to chat */
  onExportToChat?: (content: string) => void;
}

export function AIditor({ initialContent, onContentConsumed, onExportToChat }: AIditorProps = {}) {
  const { state, actions } = useAIditor();

  // Load content from chat when initialContent changes
  useEffect(() => {
    if (initialContent) {
      actions.setMarkdown(initialContent);
      onContentConsumed?.();
    }
  }, [initialContent]); // eslint-disable-line react-hooks/exhaustive-deps

  // Modal state for custom instructions
  const [customModalOpen, setCustomModalOpen] = useState(false);
  const [pendingCustomText, setPendingCustomText] = useState('');
  const [customCallback, setCustomCallback] = useState<((instruction: string) => void) | null>(null);

  // API data
  const [models, setModels] = useState<ModelsResponse | null>(null);
  const [modelsLoading, setModelsLoading] = useState(false);
  const [mcpEndpoints, setMcpEndpoints] = useState<MCPEndpoints | null>(null);
  const [mcpLoading, setMcpLoading] = useState(false);
  const [apiError, setApiError] = useState<string | null>(null);
  const [debugInfo, setDebugInfo] = useState<DebugInfo | null>(null);

  // Tab state
  const [activeTab, setActiveTab] = useState<'editor' | 'compare' | 'settings'>('editor');

  // Fetch models on mount (with timeout to avoid hanging on large workspaces)
  const fetchModels = useCallback(async () => {
    setModelsLoading(true);
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 20000); // 20s timeout
    try {
      const response = await fetch(`${API_BASE}/models`, { signal: controller.signal });
      if (response.ok) {
        const data: ModelsResponse = await response.json();
        setModels(data);
        if (!state.selectedModel && data.default_model) {
          actions.setSelectedModel(data.default_model);
        }
      } else {
        const detail = await response.text().catch(() => response.statusText);
        console.warn('Failed to load models:', detail);
      }
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') {
        console.warn('Models fetch timed out — setting default model');
        // Set a sensible default even if the fetch timed out
        if (!state.selectedModel) {
          actions.setSelectedModel('databricks-claude-sonnet-4');
        }
      } else {
        console.error('Failed to fetch models:', error);
      }
    } finally {
      clearTimeout(timeout);
      setModelsLoading(false);
    }
  }, [state.selectedModel, actions]);

  // Fetch MCP endpoints on mount (with timeout)
  const fetchMcpEndpoints = useCallback(async () => {
    setMcpLoading(true);
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 20000); // 20s timeout
    try {
      const response = await fetch(`${API_BASE}/mcp/endpoints`, { signal: controller.signal });
      if (response.ok) {
        const data: MCPEndpoints = await response.json();
        setMcpEndpoints(data);
      } else {
        const detail = await response.text().catch(() => response.statusText);
        console.warn('Failed to load MCP endpoints:', detail);
      }
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') {
        console.warn('MCP endpoints fetch timed out');
      } else {
        console.error('Failed to fetch MCP endpoints:', error);
      }
    } finally {
      clearTimeout(timeout);
      setMcpLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchModels();
    fetchMcpEndpoints();
  }, []);

  // Handle custom instruction request
  const handleRequestCustomInstruction = useCallback(
    (text: string, callback: (instruction: string) => void) => {
      setPendingCustomText(text);
      setCustomCallback(() => callback);
      setCustomModalOpen(true);
    },
    []
  );

  // Handle custom instruction confirm
  const handleCustomConfirm = useCallback(
    (instruction: string) => {
      if (customCallback) {
        customCallback(instruction);
      }
      setCustomModalOpen(false);
      setPendingCustomText('');
      setCustomCallback(null);
    },
    [customCallback]
  );

  // Handle custom instruction cancel
  const handleCustomCancel = useCallback(() => {
    setCustomModalOpen(false);
    setPendingCustomText('');
    setCustomCallback(null);
  }, []);

  // Handle new document
  const handleNewDocument = useCallback(() => {
    if (state.markdown && !confirm('Are you sure you want to create a new document? Current content will be cleared.')) {
      return;
    }
    actions.setMarkdown('');
    actions.clearHighlights();
  }, [state.markdown, actions]);

  // -------------------------------------------------------------------------
  // Shared helper: execute a single MCP highlight and return markdown result
  // -------------------------------------------------------------------------
  const executeMcpHighlight = useCallback(
    async (highlight: Highlight): Promise<string> => {
      if (highlight.type === 'external') {
        const req: ExternalQueryRequest = {
          connection_name: highlight.endpointId || 'tavily-search',
          query: highlight.text,
          max_results: 3,
        };
        const resp = await fetch(`${API_BASE}/mcp/external`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(req),
        });
        if (resp.ok) {
          const data: ExternalQueryResponse = await resp.json();
          return data.markdown_summary;
        }
        const detail = await resp.text().catch(() => resp.statusText);
        return `> **Error querying ${highlight.endpointName || 'external tool'}**: ${detail}`;
      }

      if (highlight.type === 'genie') {
        const req: GenieQueryRequest = {
          space_id: highlight.endpointId || '',
          query: highlight.text,
        };
        const resp = await fetch(`${API_BASE}/mcp/genie`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(req),
        });
        if (resp.ok) {
          const data: GenieQueryResponse = await resp.json();
          return data.markdown_table;
        }
        const detail = await resp.text().catch(() => resp.statusText);
        return `> **Error querying ${highlight.endpointName || 'Genie'}**: ${detail}`;
      }

      if (highlight.type === 'vector') {
        const req: VectorSearchRequest = {
          index_name: highlight.endpointId || '',
          query: highlight.text,
          num_results: 5,
        };
        const resp = await fetch(`${API_BASE}/mcp/vector-search`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(req),
        });
        if (resp.ok) {
          const data: VectorSearchResponse = await resp.json();
          return data.markdown_list;
        }
        const detail = await resp.text().catch(() => resp.statusText);
        return `> **Error querying ${highlight.endpointName || 'Vector Search'}**: ${detail}`;
      }

      if (highlight.type === 'knowledge_assistant') {
        const req: KnowledgeAssistantQueryRequest = {
          tile_id: highlight.endpointId || '',
          query: highlight.text,
        };
        const resp = await fetch(`${API_BASE}/mcp/knowledge-assistant`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(req),
        });
        if (resp.ok) {
          const data: KnowledgeAssistantQueryResponse = await resp.json();
          return data.answer;
        }
        const detail = await resp.text().catch(() => resp.statusText);
        return `> **Error querying ${highlight.endpointName || 'Knowledge Assistant'}**: ${detail}`;
      }

      return '';
    },
    []
  );

  // -------------------------------------------------------------------------
  // Shared helper: run all MCP highlights, inserting results into markdown
  // Returns the enriched markdown string.
  // -------------------------------------------------------------------------
  const runMcpQueries = useCallback(
    async (markdown: string, mcpHighlights: Highlight[]): Promise<string> => {
      let updated = markdown;
      for (const highlight of mcpHighlights) {
        try {
          const resultMarkdown = await executeMcpHighlight(highlight);
          if (resultMarkdown) {
            const idx = updated.indexOf(highlight.text);
            if (idx !== -1) {
              const insertPos = idx + highlight.text.length;
              updated =
                updated.slice(0, insertPos) +
                '\n\n' + resultMarkdown + '\n' +
                updated.slice(insertPos);
            } else {
              updated += '\n\n' + resultMarkdown;
            }
          }
        } catch (error) {
          console.error(`Failed to query ${highlight.type}:`, error);
          setApiError(
            `Failed to query ${highlight.endpointName || highlight.type}: ${error instanceof Error ? error.message : 'Unknown error'}`
          );
        }
      }
      return updated;
    },
    [executeMcpHighlight]
  );

  // -------------------------------------------------------------------------
  // Process edits with LLM (no MCP highlights queued)
  // -------------------------------------------------------------------------
  const handleProcess = useCallback(async () => {
    if (state.highlights.length === 0 || !state.selectedModel) return;

    const MCP_TYPES = ['genie', 'vector', 'external', 'knowledge_assistant'];
    const mcpHighlights = state.highlights.filter((h) => MCP_TYPES.includes(h.type));
    const llmHighlights = state.highlights.filter((h) => !MCP_TYPES.includes(h.type));

    if (llmHighlights.length === 0) return;

    actions.setProcessing(true);

    try {
      // --- Phase 1: If MCP highlights are also queued, fetch data first ---
      let enrichedMarkdown = state.markdown;
      if (mcpHighlights.length > 0) {
        enrichedMarkdown = await runMcpQueries(state.markdown, mcpHighlights);
      }

      // --- Phase 2: Send enriched markdown + LLM edit instructions to LLM ---
      const edits: EditInstruction[] = llmHighlights.map((h) => ({
        type: h.type as EditType,
        text: h.text,
        instruction: h.instruction || null,
      }));

      const request: ChatRequest = {
        model: state.selectedModel,
        original_markdown: enrichedMarkdown,
        edits,
      };

      const response = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request),
      });

      if (response.ok) {
        const data: ChatResponse = await response.json();
        // If we enriched the markdown with MCP data, update the base so
        // "accept" applies on top of the enriched version.
        if (mcpHighlights.length > 0) {
          actions.setMarkdown(enrichedMarkdown);
        }
        actions.setProcessedResult(data.content);
        setDebugInfo(data.debug ?? null);
        setActiveTab('compare');
      } else {
        const detail = await response.text().catch(() => response.statusText);
        setApiError(`Failed to process edits: ${detail}`);
      }
    } catch (error) {
      console.error('Failed to process edits:', error);
      setApiError(`Failed to process edits: ${error instanceof Error ? error.message : 'Network error'}`);
    } finally {
      actions.setProcessing(false);
    }
  }, [state.highlights, state.selectedModel, state.markdown, actions, runMcpQueries]);

  // -------------------------------------------------------------------------
  // Process MCP tool highlights only (no LLM edits queued)
  // -------------------------------------------------------------------------
  const handleQueryTools = useCallback(async () => {
    const mcpHighlights = state.highlights.filter((h) =>
      ['genie', 'vector', 'external', 'knowledge_assistant'].includes(h.type)
    );
    if (mcpHighlights.length === 0) return;

    actions.setProcessing(true);
    try {
      const updatedMarkdown = await runMcpQueries(state.markdown, mcpHighlights);
      actions.setMarkdown(updatedMarkdown);
      actions.clearHighlights();
    } finally {
      actions.setProcessing(false);
    }
  }, [state.highlights, state.markdown, actions, runMcpQueries]);

  // Handle accept result
  const handleAccept = useCallback(() => {
    actions.acceptResult();
    setDebugInfo(null);
    setActiveTab('editor');
  }, [actions]);

  // Handle discard result
  const handleDiscard = useCallback(() => {
    actions.discardResult();
    setDebugInfo(null);
    setActiveTab('editor');
  }, [actions]);

  return (
    <div className="flex-1 flex flex-col min-h-0 bg-background">
      {/* Header */}
      <Header
        models={models?.models || []}
        modelsLoading={modelsLoading}
        selectedModel={state.selectedModel}
        defaultModel={models?.default_model || null}
        onSelectModel={actions.setSelectedModel}
        onRefreshModels={fetchModels}
        onOpenSettings={() => setActiveTab('settings')}
        onExportToChat={onExportToChat ? () => onExportToChat(state.markdown) : undefined}
      />

      {/* Error Banner */}
      {apiError && (
        <div className="bg-destructive/10 border-b border-destructive/20 px-4 py-2 text-sm text-destructive flex items-center justify-between">
          <span>{apiError}</span>
          <button
            onClick={() => setApiError(null)}
            className="ml-4 text-destructive hover:text-destructive/80 font-medium"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Main Content */}
      <main className="flex-1 min-h-0 overflow-y-auto container max-w-6xl mx-auto px-4 py-4">
        <Tabs
          value={activeTab}
          onValueChange={(v) => setActiveTab(v as 'editor' | 'compare' | 'settings')}
        >
          <TabsList className="mb-4">
            <TabsTrigger value="editor">Editor</TabsTrigger>
            <TabsTrigger value="compare" disabled={!state.processedResult}>
              Compare
            </TabsTrigger>
            <TabsTrigger value="settings">Settings</TabsTrigger>
          </TabsList>

          {/* Editor Tab */}
          <TabsContent value="editor" className="mt-0">
            <EditorToolbar
              isAIditorMode={state.isAIditorMode}
              activeHighlighter={state.activeHighlighter}
              activeEndpointId={state.pendingEndpointId}
              isKeyboardSelecting={state.isKeyboardSelecting}
              pendingSelectionText={state.pendingSelectionText}
              highlights={state.highlights}
              mcpShortcuts={state.mcpShortcuts}
              isProcessing={state.isProcessing}
              hasResult={!!state.processedResult}
              canUndo={state.highlightUndoStack.length > 0}
              onToggleMode={actions.toggleAIditorMode}
              onSetHighlighter={actions.setActiveHighlighter}
              onToggleKeyboardSelecting={actions.toggleKeyboardSelecting}
              onProcess={handleProcess}
              onQueryTools={handleQueryTools}
              onClearHighlights={actions.clearHighlights}
              onUndoHighlight={actions.undoHighlight}
              onNewDocument={handleNewDocument}
            />

            <Card className="border bg-card">
              <CardContent className="p-0">
                <MarkdownEditor
                  markdown={state.markdown}
                  onChange={actions.setMarkdown}
                  isAIditorMode={state.isAIditorMode}
                  activeHighlighter={state.activeHighlighter}
                  isKeyboardSelecting={state.isKeyboardSelecting}
                  pendingSelectionText={state.pendingSelectionText}
                  pendingInstruction={state.pendingInstruction}
                  pendingEndpointId={state.pendingEndpointId}
                  pendingEndpointName={state.pendingEndpointName}
                  highlights={state.highlights}
                  mcpShortcuts={state.mcpShortcuts}
                  onAddHighlight={actions.addHighlight}
                  onRemoveHighlight={actions.removeHighlight}
                  onRequestCustomInstruction={handleRequestCustomInstruction}
                />
              </CardContent>
            </Card>

            {/* Status Bar */}
            {state.isAIditorMode && state.activeHighlighter && (() => {
              // Determine color and label for the active highlighter
              const colorMap: Record<string, string> = {
                REMOVE: '#FF6B6B', LESS: '#FFB347', MORE: '#7EC8E3', CUSTOM: '#77DD77',
                genie: '#9B59B6', vector: '#8E44AD', external: '#6C3483', knowledge_assistant: '#2ECC71',
              };
              let label = state.activeHighlighter.toUpperCase();
              let bgColor = colorMap[state.activeHighlighter] || '#9B59B6';
              // If an MCP endpoint is active, show its name and use its shortcut color
              if (state.pendingEndpointName) {
                const matchedShortcut = state.mcpShortcuts.find(
                  (s) => s.endpointId === state.pendingEndpointId
                );
                label = state.pendingEndpointName;
                if (matchedShortcut) bgColor = matchedShortcut.color;
              }
              return (
                <div
                  className="mt-2 px-3 py-2 rounded-md text-sm text-white"
                  style={{ backgroundColor: bgColor }}
                >
                  <strong>{label}</strong> mode active — select text to highlight, then click <strong>Query Tools</strong>
                </div>
              );
            })()}
          </TabsContent>

          {/* Compare Tab */}
          <TabsContent value="compare" className="mt-0">
            {state.processedResult && (
              <ComparisonView
                original={state.markdown}
                processed={state.processedResult}
                highlights={state.highlights}
                debugInfo={debugInfo}
                onAccept={handleAccept}
                onDiscard={handleDiscard}
              />
            )}
          </TabsContent>

          {/* Settings Tab */}
          <TabsContent value="settings" className="mt-0">
            <SettingsContent
              models={models?.models || []}
              modelsLoading={modelsLoading}
              selectedModel={state.selectedModel}
              defaultModel={models?.default_model || null}
              onSelectModel={actions.setSelectedModel}
              onRefreshModels={fetchModels}
              genieSpaces={mcpEndpoints?.genie_spaces || []}
              vectorIndexes={mcpEndpoints?.vector_indexes || []}
              externalConnections={mcpEndpoints?.external_connections || []}
              knowledgeAssistants={mcpEndpoints?.knowledge_assistants || []}
              mcpLoading={mcpLoading}
              onRefreshMcp={fetchMcpEndpoints}
              mcpShortcuts={state.mcpShortcuts}
              onUpdateShortcuts={actions.setMcpShortcuts}
              builtinShortcuts={state.builtinShortcuts}
              onUpdateBuiltinShortcuts={actions.setBuiltinShortcuts}
            />
          </TabsContent>
        </Tabs>
      </main>

      {/* Custom Instruction Modal */}
      <CustomInstructionModal
        open={customModalOpen}
        selectedText={pendingCustomText}
        onConfirm={handleCustomConfirm}
        onCancel={handleCustomCancel}
      />
    </div>
  );
}
