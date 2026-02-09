/**
 * Type definitions for AIditor components.
 */

// =============================================================================
// Edit Types
// =============================================================================

export type EditType = 'REMOVE' | 'LESS' | 'MORE' | 'CUSTOM';

export type MCPEndpointType = 'genie' | 'vector' | 'external' | 'knowledge_assistant';

export type CustomShortcutEndpointType = MCPEndpointType | 'llm';

export type HighlightType = EditType | MCPEndpointType;

// =============================================================================
// Highlight
// =============================================================================

export interface Highlight {
  id: string;
  type: HighlightType;
  text: string;
  startOffset: number;
  endOffset: number;
  instruction?: string;
  endpointId?: string;   // MCP endpoint ID (e.g., "tavily-search", genie space ID)
  endpointName?: string; // MCP endpoint display name
  color: string;
}

export interface EditInstruction {
  type: EditType;
  text: string;
  instruction?: string | null;
}

// =============================================================================
// API Types
// =============================================================================

export interface ModelInfo {
  name: string;
  display_name: string;
  status: 'READY' | 'NOT_READY' | 'PENDING';
  task: string;
}

export interface ModelsResponse {
  models: ModelInfo[];
  default_model: string | null;
}

export interface ChatRequest {
  model: string;
  original_markdown: string;
  edits: EditInstruction[];
}

export interface UsageInfo {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens?: number;
}

export interface DebugInfo {
  system_prompt: string;
  user_message: string;
  raw_response: string;
}

export interface ChatResponse {
  content: string;
  model: string;
  usage?: UsageInfo;
  debug?: DebugInfo;
}

// =============================================================================
// MCP Types
// =============================================================================

export interface GenieSpaceInfo {
  id: string;
  name: string;
  tables: string[];
  status: string;
}

export interface VectorIndexInfo {
  name: string;
  endpoint: string;
  num_docs: number;
}

export interface ExternalConnectionInfo {
  name: string;
  type: string;
  status: string;
}

export interface KnowledgeAssistantInfo {
  tile_id: string;
  name: string;
  endpoint_name: string;
  status: string;
}

export interface KnowledgeAssistantQueryRequest {
  tile_id: string;
  query: string;
}

export interface KnowledgeAssistantQueryResponse {
  status: string;
  answer: string;
  sources: string[];
  error?: string;
}

export interface MCPEndpoints {
  genie_spaces: GenieSpaceInfo[];
  vector_indexes: VectorIndexInfo[];
  external_connections: ExternalConnectionInfo[];
  knowledge_assistants: KnowledgeAssistantInfo[];
}

export interface GenieQueryRequest {
  space_id: string;
  query: string;
}

export interface GenieQueryResponse {
  status: string;
  sql?: string;
  columns: string[];
  data: unknown[][];
  markdown_table: string;
  error?: string;
}

export interface VectorSearchRequest {
  index_name: string;
  query: string;
  num_results?: number;
}

export interface VectorSearchResult {
  text: string;
  source?: string;
  score: number;
}

export interface VectorSearchResponse {
  results: VectorSearchResult[];
  markdown_list: string;
  error?: string;
}

export interface ExternalQueryRequest {
  connection_name: string;
  query: string;
  max_results?: number;
}

export interface ExternalSearchResult {
  title: string;
  url: string;
  snippet: string;
}

export interface ExternalQueryResponse {
  results: ExternalSearchResult[];
  markdown_summary: string;
  error?: string;
}

// =============================================================================
// Shortcut Types
// =============================================================================

export interface MCPShortcut {
  key: string;
  type: 'MCP' | 'LLM';
  endpointType: CustomShortcutEndpointType;
  endpointId: string;
  endpointName: string;
  instruction: string;
  color: string;
  resultBehavior: 'insert_below';
}

export interface LLMShortcut {
  key: string;
  type: 'LLM';
  editType: EditType;
  instruction?: string;
  color: string;
}

export interface BuiltinShortcut {
  id: string;
  action: string;
  key: string;
  description: string;
}

// =============================================================================
// State Types
// =============================================================================

export interface AIditorState {
  // Editor
  markdown: string;
  isAIditorMode: boolean;

  // Highlighting
  activeHighlighter: HighlightType | null;
  pendingInstruction: string | null; // Pre-filled instruction from LLM shortcuts
  pendingEndpointId: string | null;  // MCP endpoint ID when MCP shortcut is active
  pendingEndpointName: string | null; // MCP endpoint display name
  highlights: Highlight[];
  highlightUndoStack: Highlight[][]; // Stack of previous highlight states for undo

  // Keyboard selection mode (/ to toggle)
  isKeyboardSelecting: boolean;
  pendingSelectionText: string | null; // Text selected via keyboard, waiting for command

  // Processing
  selectedModel: string | null;
  isProcessing: boolean;
  processedResult: string | null;

  // MCP
  mcpShortcuts: MCPShortcut[];

  // Settings
  settingsOpen: boolean;
  builtinShortcuts: BuiltinShortcut[];
}

export interface StoredSettings {
  selectedModel: string | null;
  mcpShortcuts: MCPShortcut[];
  llmShortcuts: LLMShortcut[];
  builtinShortcuts?: BuiltinShortcut[];
}

// =============================================================================
// Constants
// =============================================================================

export const HIGHLIGHT_COLORS: Record<HighlightType, string> = {
  REMOVE: '#FF6B6B',
  LESS: '#FFB347',
  MORE: '#7EC8E3',
  CUSTOM: '#77DD77',
  genie: '#9B59B6',
  vector: '#8E44AD',
  external: '#6C3483',
  knowledge_assistant: '#2ECC71',
};

export const CUSTOM_SHORTCUT_COLORS: Record<CustomShortcutEndpointType, string> = {
  genie: '#9B59B6',
  vector: '#8E44AD',
  external: '#6C3483',
  knowledge_assistant: '#2ECC71',
  llm: '#E67E22',
};

export const EDIT_TYPE_LABELS: Record<EditType, string> = {
  REMOVE: 'Remove',
  LESS: 'Shorten',
  MORE: 'Expand',
  CUSTOM: 'Custom',
};

export const RESERVED_KEYS = ['a', 's', 'd', 'f', 'escape', '/', 'e'];

export const DEFAULT_BUILTIN_SHORTCUTS: BuiltinShortcut[] = [
  {
    id: 'toggle-mode',
    action: 'Toggle AIditor Mode',
    key: '⌘E',
    description: 'Switch between edit and AIditor mode',
  },
  {
    id: 'toggle-selection',
    action: 'Toggle Selection Mode',
    key: '/',
    description: 'Enable keyboard text selection',
  },
  {
    id: 'remove',
    action: 'Highlight: Remove (Red)',
    key: 'A',
    description: 'Mark text for removal',
  },
  {
    id: 'less',
    action: 'Highlight: Less (Orange)',
    key: 'S',
    description: 'Mark text to shorten',
  },
  {
    id: 'more',
    action: 'Highlight: More (Blue)',
    key: 'D',
    description: 'Mark text to expand',
  },
  {
    id: 'custom',
    action: 'Highlight: Custom (Green)',
    key: 'F',
    description: 'Mark with custom instruction',
  },
  {
    id: 'escape',
    action: 'Deactivate Highlight',
    key: 'Esc',
    description: 'Cancel current highlight mode',
  },
];
