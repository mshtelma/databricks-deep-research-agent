/**
 * Types, constants, and factory functions for custom agent E2E tests.
 *
 * Enterprise source IDs are config-driven via environment variables
 * with defaults matching the integration test values.
 */

import { generateTestId } from './test-data';

// ---------------------------------------------------------------------------
// Types (mirror docs/custom-agents.md API contract)
// ---------------------------------------------------------------------------

export type SourceScope = 'all' | 'enterprise_only' | 'web_only';
export type Visibility = 'private' | 'workspace' | 'system';
export type WorkflowMode = 'planner' | 'manual' | 'hybrid';
export type ResearchDepth = 'light' | 'medium' | 'extended';
export type DomainFilterMode = 'include' | 'exclude' | 'both';

export interface CustomAgentConfig {
  name: string;
  description?: string;
  visibility?: Visibility;
  default_depth?: ResearchDepth;
  default_mode?: WorkflowMode;
  use_planner?: boolean;
  enable_clarification?: boolean;
  source_scope?: SourceScope;
  enabled_sources?: string[];
  disabled_sources?: string[];
  domain_filter_mode?: DomainFilterMode;
  include_domains?: string[];
  exclude_domains?: string[];
  model_overrides?: Record<string, string>;
}

export interface PresetStepConfig {
  title: string;
  description?: string;
  order: number;
  is_required?: boolean;
  source_scope?: SourceScope;
  source_hints?: {
    preferred_sources?: string[];
    search_queries?: string[];
    filters?: Record<string, string>;
  };
}

export interface AgentResponse {
  id: string;
  name: string;
  description: string | null;
  visibility: Visibility;
  defaultDepth: ResearchDepth;
  defaultMode: WorkflowMode;
  usePlanner: boolean;
  sourceScope: SourceScope;
  enabledSources: string[] | null;
  disabledSources: string[];
  domainFilterMode: DomainFilterMode | null;
  includeDomains: string[] | null;
  excludeDomains: string[] | null;
  modelOverrides: Record<string, string> | null;
  modelOverrideWarnings?: Array<{
    tier: string;
    endpoint: string;
    message: string;
  }>;
  ownerId: string;
  createdAt: string;
  updatedAt: string;
}

export interface AgentListResponse {
  agents: AgentResponse[];
  total: number;
  userAgents: number;
  workspaceAgents: number;
  systemAgents: number;
}

export interface StepResponse {
  id: string;
  agentId: string;
  title: string;
  description: string | null;
  order: number;
  isRequired: boolean;
  sourceScope: SourceScope | null;
  sourceHints: Record<string, unknown> | null;
  createdAt: string;
  updatedAt: string;
}

// ---------------------------------------------------------------------------
// Enterprise Source IDs (env-driven with integration test defaults)
// ---------------------------------------------------------------------------

export const KA_SOURCE_ID =
  process.env.E2E_KA_SOURCE_ID ?? 'assistant:ka-99a12b9d-endpoint';

export const GENIE_SOURCE_ID =
  process.env.E2E_GENIE_SOURCE_ID ?? 'genie:01f0b5ab5b841281858ae25da3f58125';

export const VS_SOURCE_ID =
  process.env.E2E_VS_SOURCE_ID ?? 'vs:anthony_ivan.demo-toolsapp.pdf_chunks_index';

export const ALL_ENTERPRISE_SOURCES = [KA_SOURCE_ID, GENIE_SOURCE_ID, VS_SOURCE_ID];

// ---------------------------------------------------------------------------
// Timeout Constants (milliseconds)
// ---------------------------------------------------------------------------

export const AGENT_TIMEOUTS = {
  /** API CRUD operations */
  api: 10_000,
  /** UI interactions (dropdown, selection) */
  ui: 5_000,
  /** Light-depth research (~1-3 steps) */
  lightResearch: 180_000,
  /** Standard research (~3-5 steps) */
  research: 360_000,
  /** Full scenario with multiple research rounds */
  fullScenario: 600_000,
} as const;

// ---------------------------------------------------------------------------
// Agent Config Factory Functions
// ---------------------------------------------------------------------------

/** Minimal agent with only required fields. */
export function makeMinimalAgent(namePrefix = 'E2E Minimal'): CustomAgentConfig {
  return {
    name: `${namePrefix} ${generateTestId()}`,
  };
}

/** Agent restricted to enterprise sources only. */
export function makeEnterpriseAgent(namePrefix = 'E2E Enterprise'): CustomAgentConfig {
  return {
    name: `${namePrefix} ${generateTestId()}`,
    description: 'Enterprise-only research agent for E2E testing',
    default_depth: 'light',
    source_scope: 'enterprise_only',
    enabled_sources: ALL_ENTERPRISE_SOURCES,
  };
}

/** Agent restricted to a single Knowledge Assistant source. */
export function makeKAOnlyAgent(namePrefix = 'E2E KA-Only'): CustomAgentConfig {
  return {
    name: `${namePrefix} ${generateTestId()}`,
    description: 'KA-only research agent for E2E testing',
    default_depth: 'light',
    source_scope: 'enterprise_only',
    enabled_sources: [KA_SOURCE_ID],
  };
}

/** Agent with web-only scope and domain include filter. */
export function makeWebFilteredAgent(namePrefix = 'E2E WebFiltered'): CustomAgentConfig {
  return {
    name: `${namePrefix} ${generateTestId()}`,
    description: 'Web-only with domain filter for E2E testing',
    default_depth: 'light',
    source_scope: 'web_only',
    domain_filter_mode: 'include',
    include_domains: ['*.gov', '*.edu'],
  };
}

/** Agent with domain exclude filter. */
export function makeExcludeDomainAgent(namePrefix = 'E2E ExcludeDomain'): CustomAgentConfig {
  return {
    name: `${namePrefix} ${generateTestId()}`,
    description: 'Web-only with domain exclusion for E2E testing',
    default_depth: 'light',
    source_scope: 'web_only',
    domain_filter_mode: 'exclude',
    exclude_domains: ['reddit.com', '*.pinterest.com'],
  };
}

/** Agent with model overrides (may include a stale endpoint). */
export function makeModelOverrideAgent(namePrefix = 'E2E ModelOverride'): CustomAgentConfig {
  return {
    name: `${namePrefix} ${generateTestId()}`,
    description: 'Agent with model overrides for E2E testing',
    model_overrides: {
      complex: 'databricks-meta-llama-3-1-70b-instruct',
    },
  };
}

/** Agent with a stale (nonexistent) endpoint override for fallback testing. */
export function makeStaleEndpointAgent(namePrefix = 'E2E StaleEndpoint'): CustomAgentConfig {
  return {
    name: `${namePrefix} ${generateTestId()}`,
    description: 'Agent with stale endpoint for fallback testing',
    default_depth: 'light',
    model_overrides: {
      complex: 'nonexistent-endpoint-for-testing',
    },
  };
}

/** Agent with workspace visibility. */
export function makeWorkspaceAgent(namePrefix = 'E2E Workspace'): CustomAgentConfig {
  return {
    name: `${namePrefix} ${generateTestId()}`,
    description: 'Workspace-visible agent for E2E testing',
    visibility: 'workspace',
  };
}

/** Agent configured for manual workflow mode. */
export function makeManualWorkflowAgent(namePrefix = 'E2E Manual'): CustomAgentConfig {
  return {
    name: `${namePrefix} ${generateTestId()}`,
    description: 'Manual workflow agent for E2E testing',
    default_depth: 'light',
    default_mode: 'manual',
    use_planner: false,
  };
}

/** Agent configured for hybrid workflow mode. */
export function makeHybridWorkflowAgent(namePrefix = 'E2E Hybrid'): CustomAgentConfig {
  return {
    name: `${namePrefix} ${generateTestId()}`,
    description: 'Hybrid workflow agent for E2E testing',
    default_depth: 'light',
    default_mode: 'hybrid',
    use_planner: true,
  };
}

// ---------------------------------------------------------------------------
// Preset Step Factories
// ---------------------------------------------------------------------------

/** A set of 3 ordered preset steps for workflow testing. */
export function makePresetSteps(): PresetStepConfig[] {
  return [
    {
      title: 'Gather background context',
      description: 'Collect foundational information on the topic',
      order: 1,
      is_required: true,
    },
    {
      title: 'Analyze key findings',
      description: 'Examine and compare the gathered information',
      order: 2,
      is_required: true,
    },
    {
      title: 'Identify gaps and open questions',
      description: 'Find areas that need further investigation',
      order: 3,
      is_required: false,
    },
  ];
}

/** A preset step with per-step source scope override. */
export function makeSourceScopedStep(
  order: number,
  scope: SourceScope,
): PresetStepConfig {
  return {
    title: `Step ${order}: ${scope} search`,
    description: `Search using ${scope} sources`,
    order,
    is_required: true,
    source_scope: scope,
    source_hints: {
      search_queries: ['general information'],
    },
  };
}

// ---------------------------------------------------------------------------
// Research Queries (generic, domain-agnostic)
// ---------------------------------------------------------------------------

export const AGENT_RESEARCH_QUERIES = {
  /** Generic query suitable for any enterprise source. */
  enterprise: 'What are the main topics covered in the available knowledge base?',
  /** Generic query for web-only agents. */
  web: 'What are recent developments in renewable energy technology?',
  /** Short query for light-depth testing. */
  light: 'Summarize the key trends in cloud computing',
  /** Query for domain-filtered agents. */
  domainFiltered: 'What are the latest government regulations on data privacy?',
} as const;
