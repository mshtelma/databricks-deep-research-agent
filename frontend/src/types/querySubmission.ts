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
}
