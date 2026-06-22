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
}
