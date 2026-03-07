/**
 * Custom Agents types for managing user-defined research agents.
 *
 * This module defines types for:
 * - Custom agent definitions and configurations
 * - Agent visibility and permissions
 * - Preset steps for manual research workflows
 * - Prompt templates
 *
 * NOTE: All field names use camelCase to match the backend BaseSchema
 * serialization (alias_generator=to_camel, serialize_by_alias=True).
 */

import type { SourceScope, DataSourceType } from './dataSources';

// =============================================================================
// Agent Visibility and Core Types
// =============================================================================

/** Agent visibility options */
export type AgentVisibility = 'private' | 'workspace';

/** Agent workflow mode */
export type WorkflowMode = 'planner' | 'manual' | 'hybrid';

/** Research depth options */
export type ResearchDepth = 'light' | 'medium' | 'extended';

/** Output format options */
export type OutputFormat = 'markdown' | 'json';

// =============================================================================
// Custom Agent Definition
// =============================================================================

/** Source configuration for a custom agent */
export interface AgentSourceConfig {
  scope: SourceScope;
  enabledSources: string[];
  disabledSources: string[];
}

/** Workflow configuration for a custom agent */
export interface AgentWorkflowConfig {
  usePlanner: boolean;
  defaultDepth: ResearchDepth;
  workflowMode: WorkflowMode;
  enableClarification: boolean;
  maxSteps?: number;
  maxToolCalls?: number;
}

/** Output configuration for a custom agent */
export interface AgentOutputConfig {
  format: OutputFormat;
  jsonSchema?: string | null;
}

/** Custom agent definition */
export interface CustomAgent {
  id: string;
  ownerId: string;
  name: string;
  description: string | null;
  avatarUrl: string | null;
  visibility: AgentVisibility;
  systemPromptTemplateId: string | null;
  synthesisTemplateId: string | null;
  inlineSystemPrompt: string | null;
  inlineSynthesisPrompt: string | null;
  sourceConfig: AgentSourceConfig;
  workflowConfig: AgentWorkflowConfig;
  outputConfig: AgentOutputConfig;
  /** Per-agent model tier overrides: tier name -> endpoint ID */
  modelOverrides: Record<string, string> | null;
  /** Domain filter mode: 'include' | 'exclude' | 'both' | null */
  domainFilterMode: string | null;
  /** Whitelist domain patterns */
  includeDomains: string[] | null;
  /** Blacklist domain patterns */
  excludeDomains: string[] | null;
  isActive: boolean;
  createdAt: string;
  updatedAt: string;
}

/** Simplified agent for list display */
export interface CustomAgentSummary {
  id: string;
  name: string;
  description: string | null;
  avatarUrl: string | null;
  visibility: AgentVisibility;
  /** Source scope: 'all' | 'enterprise_only' | 'web_only' */
  sourceScope?: SourceScope;
  isActive?: boolean;
  ownerId: string;
  ownerName?: string;
  capabilities?: AgentCapability[];
  /** Whether the agent has model tier overrides configured */
  hasModelOverrides?: boolean;
  /** Whether the agent has domain filtering configured */
  hasDomainFilter?: boolean;
  /** Whether the agent defines source scope or enabled sources */
  hasSourceConfig?: boolean;
}

/** Agent capability for display */
export type AgentCapability =
  | 'web_search'
  | 'enterprise_sources'
  | 'structured_output'
  | 'manual_workflow'
  | 'custom_prompts';

// =============================================================================
// Preset Steps
// =============================================================================

/** Source hint for a preset step */
export interface PresetStepSourceHint {
  sourceId: string;
  sourceName: string;
  sourceType: DataSourceType;
  priority: 1 | 2 | 3;
  queryHint?: string | null;
}

/** Preset step definition */
export interface PresetStep {
  id: string;
  agentId: string;
  title: string;
  description: string | null;
  order: number;
  isRequired: boolean;
  /** Per-step source scope override: 'enterprise_only' | 'web_only' | 'all' | null (inherit) */
  sourceScope: string | null;
  sourceHints?: PresetStepSourceHint[];
  createdAt: string;
  updatedAt: string;
}

/** Create preset step request */
export interface CreatePresetStepRequest {
  title: string;
  description?: string;
  order?: number;
  isRequired?: boolean;
  sourceScope?: string | null;
  sourceHints?: PresetStepSourceHint[];
}

/** Update preset step request */
export interface UpdatePresetStepRequest {
  title?: string;
  description?: string | null;
  order?: number;
  isRequired?: boolean;
  sourceScope?: string | null;
  sourceHints?: PresetStepSourceHint[];
}

/** Reorder preset steps request */
export interface ReorderPresetStepsRequest {
  stepIds: string[];
}

// =============================================================================
// Prompt Templates
// =============================================================================

/** Prompt template type */
export type PromptTemplateType = 'system' | 'synthesis';

/** Prompt template visibility */
export type PromptTemplateVisibility = 'system' | 'workspace' | 'private';

/** Prompt template definition */
export interface PromptTemplate {
  id: string;
  name: string;
  description: string | null;
  templateType: PromptTemplateType;
  content: string;
  visibility: PromptTemplateVisibility;
  ownerId: string | null;
  createdAt: string;
  updatedAt: string;
}

// =============================================================================
// API Request/Response Types
// =============================================================================

/** Request to create a custom agent */
export interface CreateCustomAgentRequest {
  name: string;
  description?: string;
  avatarUrl?: string;
  visibility?: AgentVisibility;
  systemPromptTemplateId?: string;
  synthesisTemplateId?: string;
  inlineSystemPrompt?: string;
  inlineSynthesisPrompt?: string;
  sourceConfig?: Partial<AgentSourceConfig>;
  workflowConfig?: Partial<AgentWorkflowConfig>;
  outputConfig?: Partial<AgentOutputConfig>;
  modelOverrides?: Record<string, string>;
  domainFilterMode?: string;
  includeDomains?: string[];
  excludeDomains?: string[];
}

/** Request to update a custom agent */
export interface UpdateCustomAgentRequest {
  name?: string;
  description?: string | null;
  avatarUrl?: string | null;
  visibility?: AgentVisibility;
  systemPromptTemplateId?: string | null;
  synthesisTemplateId?: string | null;
  inlineSystemPrompt?: string | null;
  inlineSynthesisPrompt?: string | null;
  sourceConfig?: Partial<AgentSourceConfig>;
  workflowConfig?: Partial<AgentWorkflowConfig>;
  outputConfig?: Partial<AgentOutputConfig>;
  modelOverrides?: Record<string, string> | null;
  domainFilterMode?: string | null;
  includeDomains?: string[] | null;
  excludeDomains?: string[] | null;
  isActive?: boolean;
}

/** List agents response */
export interface CustomAgentListResponse {
  agents: CustomAgentSummary[];
  total: number;
  userAgents: number;
  workspaceAgents: number;
  systemAgents: number;
}

/** List agents query params */
export interface ListAgentsParams {
  visibility?: AgentVisibility;
  include_system?: boolean;
  search?: string;
  limit?: number;
  offset?: number;
}

/** List preset steps response */
export interface PresetStepsResponse {
  steps: PresetStep[];
  total: number;
}

/** List prompt templates response */
export interface PromptTemplatesResponse {
  templates: PromptTemplate[];
  total: number;
}

// =============================================================================
// Display Utilities
// =============================================================================

/** Human-readable labels for agent visibility */
export const AGENT_VISIBILITY_LABELS: Record<AgentVisibility, string> = {
  private: 'Private',
  workspace: 'Workspace',
};

/** Human-readable labels for workflow modes */
export const WORKFLOW_MODE_LABELS: Record<WorkflowMode, string> = {
  planner: 'Auto (Planner)',
  manual: 'Manual Steps',
  hybrid: 'Hybrid',
};

/** Human-readable labels for research depths */
export const RESEARCH_DEPTH_LABELS: Record<ResearchDepth, string> = {
  light: 'Light (1-3 steps)',
  medium: 'Medium (3-6 steps)',
  extended: 'Extended (5-10 steps)',
};

/** Human-readable labels for output formats */
export const OUTPUT_FORMAT_LABELS: Record<OutputFormat, string> = {
  markdown: 'Markdown',
  json: 'JSON',
};

/** Human-readable labels for agent capabilities */
export const AGENT_CAPABILITY_LABELS: Record<AgentCapability, string> = {
  web_search: 'Web Search',
  enterprise_sources: 'Enterprise Sources',
  structured_output: 'Structured Output',
  manual_workflow: 'Manual Workflow',
  custom_prompts: 'Custom Prompts',
};

// =============================================================================
// Helper Functions
// =============================================================================

/** Get capabilities from agent config */
export function getAgentCapabilities(agent: CustomAgent): AgentCapability[] {
  const caps: AgentCapability[] = [];

  // Check source scope
  if (agent.sourceConfig.scope === 'all' || agent.sourceConfig.scope === 'web_only') {
    caps.push('web_search');
  }
  if (agent.sourceConfig.scope === 'all' || agent.sourceConfig.scope === 'enterprise_only') {
    caps.push('enterprise_sources');
  }

  // Check workflow config
  if (agent.workflowConfig.workflowMode !== 'planner') {
    caps.push('manual_workflow');
  }

  // Check output config
  if (agent.outputConfig.format === 'json') {
    caps.push('structured_output');
  }

  // Check for custom prompts
  if (
    agent.inlineSystemPrompt ||
    agent.inlineSynthesisPrompt ||
    agent.systemPromptTemplateId ||
    agent.synthesisTemplateId
  ) {
    caps.push('custom_prompts');
  }

  return caps;
}

/** Get default agent source config */
export function getDefaultSourceConfig(): AgentSourceConfig {
  return {
    scope: 'all',
    enabledSources: [],
    disabledSources: [],
  };
}

/** Get default agent workflow config */
export function getDefaultWorkflowConfig(): AgentWorkflowConfig {
  return {
    usePlanner: true,
    defaultDepth: 'medium',
    workflowMode: 'planner',
    enableClarification: true,
  };
}

/** Get default agent output config */
export function getDefaultOutputConfig(): AgentOutputConfig {
  return {
    format: 'markdown',
    jsonSchema: null,
  };
}
