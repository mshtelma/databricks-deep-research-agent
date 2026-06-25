/**
 * QuerySubmission - unified interface for research query submission.
 *
 * Replaces the 8+ positional parameters that were passed through
 * MessageInput → ChatPage → useStreamingQuery → jobsApi.submit.
 */

import type { QueryMode } from './index';
import type { ResearchDepth } from '@/components/chat/ResearchDepthSelector';
import type { SourceScope } from './dataSources';

export interface QuerySubmission {
  message: string;
  queryMode?: QueryMode;
  researchDepth?: ResearchDepth;
  verifySources?: boolean;
  outputType?: string;
  sourceScope?: SourceScope;
  enabledSources?: string[];
  disabledSources?: string[];
  /**
   * Optional report writing tone (lowercase framework Tone member name, e.g.
   * 'objective'). Undefined => server default (unchanged synthesis).
   */
  tone?: string;
  /**
   * Optional report output language (free-form language name, e.g. 'Spanish').
   * Undefined => server default (unchanged synthesis).
   */
  outputLanguage?: string;
  // Feature extensions
  fileIds?: string[];
  agentId?: string;
  enablePlanReview?: boolean;
  /**
   * Per-turn routing for custom-agent chats:
   * - 'auto' (default): classify intent — answer a follow-up from gathered data, else re-run.
   * - 'chat': answer from already-gathered data (no workflow re-run).
   * - 'research': force a fresh agent run.
   * Ignored unless an agent is selected and the chat has prior research.
   */
  turnIntent?: 'auto' | 'chat' | 'research';
  /**
   * MCP server names attached to this query via the chat data-source selector.
   * Merged server-side into the run's mcp_servers (Feature 4.3 / E1).
   */
  enabledMcpServers?: string[];
  /**
   * Skill names attached to this query via the chat data-source selector.
   * Merged server-side into the workflow's agent skills (Feature 2.2 / E1).
   */
  enabledSkills?: string[];
  /**
   * Per-run override: recall facts from the user's prior chats (P2). undefined =>
   * inherit the global cross_session_memory flag.
   */
  enableCrossSessionMemory?: boolean;
  /**
   * Per-run override: allow the live-web-search follow-up escape hatch (P2).
   * undefined => inherit the global followup_live_search flag.
   */
  allowLiveSearch?: boolean;
}
