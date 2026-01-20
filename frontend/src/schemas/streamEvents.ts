/**
 * Zod schemas for SSE event validation.
 *
 * These schemas provide runtime validation for events received from the backend.
 * Malformed events are logged and gracefully ignored rather than crashing the app.
 */

import { z } from 'zod';

// Base event schema - all events must have eventType and timestamp
const BaseEventSchema = z.object({
  eventType: z.string(),
  timestamp: z.string().optional(),
});

// Agent events
export const AgentStartedEventSchema = BaseEventSchema.extend({
  eventType: z.literal('agent_started'),
  agent: z.string(),
  model_tier: z.string().optional(),
});

export const AgentCompletedEventSchema = BaseEventSchema.extend({
  eventType: z.literal('agent_completed'),
  agent: z.string(),
  duration_ms: z.number().optional(),
  durationMs: z.number().optional(), // camelCase variant from by_alias=True
});

// Research lifecycle events
export const ResearchStartedEventSchema = BaseEventSchema.extend({
  eventType: z.literal('research_started'),
  message_id: z.string(),
  research_session_id: z.string().nullable().optional(),
});

export const ResearchCompletedEventSchema = BaseEventSchema.extend({
  eventType: z.literal('research_completed'),
  session_id: z.string().optional(),
  sessionId: z.string().optional(), // camelCase variant
  total_steps_executed: z.number().optional(),
  totalStepsExecuted: z.number().optional(), // camelCase variant
  total_steps_skipped: z.number().optional(),
  totalStepsSkipped: z.number().optional(), // camelCase variant
  plan_iterations: z.number().optional(),
  planIterations: z.number().optional(), // camelCase variant
  total_duration_ms: z.number().optional(),
  totalDurationMs: z.number().optional(), // camelCase variant
});

// Plan events
export const PlanStepSchema = z.object({
  step_id: z.string().optional(),
  title: z.string(),
  description: z.string().optional(),
  status: z.string().optional(),
});

export const PlanCreatedEventSchema = BaseEventSchema.extend({
  eventType: z.literal('plan_created'),
  plan_id: z.string().optional(),
  title: z.string(),
  thought: z.string().optional(),
  steps: z.array(PlanStepSchema),
  iteration: z.number().optional(),
});

// Step events
export const StepStartedEventSchema = BaseEventSchema.extend({
  eventType: z.literal('step_started'),
  step_index: z.number().optional(),
  stepIndex: z.number().optional(), // camelCase variant
  step_id: z.string().optional(),
  stepId: z.string().optional(), // camelCase variant
  step_title: z.string().optional(),
  stepTitle: z.string().optional(), // camelCase variant
  step_type: z.string().optional(),
  stepType: z.string().optional(), // camelCase variant
});

export const StepCompletedEventSchema = BaseEventSchema.extend({
  eventType: z.literal('step_completed'),
  step_index: z.number().optional(),
  stepIndex: z.number().optional(), // camelCase variant
  step_id: z.string().optional(),
  stepId: z.string().optional(), // camelCase variant
  observation_summary: z.string().optional(),
  observationSummary: z.string().optional(), // camelCase variant
  sources_found: z.number().optional(),
  sourcesFound: z.number().optional(), // camelCase variant
});

// Tool events
export const ToolCallEventSchema = BaseEventSchema.extend({
  eventType: z.literal('tool_call'),
  tool_name: z.string().optional(),
  toolName: z.string().optional(), // camelCase variant
  tool_args: z.record(z.string(), z.unknown()).optional(),
  toolArgs: z.record(z.string(), z.unknown()).optional(), // camelCase variant
  call_number: z.number().optional(),
  callNumber: z.number().optional(), // camelCase variant
});

export const ToolResultEventSchema = BaseEventSchema.extend({
  eventType: z.literal('tool_result'),
  tool_name: z.string().optional(),
  toolName: z.string().optional(), // camelCase variant
  result_preview: z.string().optional(),
  resultPreview: z.string().optional(), // camelCase variant
  sources_crawled: z.number().optional(),
  sourcesCrawled: z.number().optional(), // camelCase variant
});

// Reflection event
export const ReflectionDecisionEventSchema = BaseEventSchema.extend({
  eventType: z.literal('reflection_decision'),
  decision: z.string(),
  reasoning: z.string().optional(),
  suggested_changes: z.array(z.string()).optional(),
});

// Synthesis events
export const SynthesisStartedEventSchema = BaseEventSchema.extend({
  eventType: z.literal('synthesis_started'),
  total_observations: z.number().optional(),
  totalObservations: z.number().optional(), // camelCase variant
  total_sources: z.number().optional(),
  totalSources: z.number().optional(), // camelCase variant
});

export const SynthesisProgressEventSchema = BaseEventSchema.extend({
  eventType: z.literal('synthesis_progress'),
  content_chunk: z.string().optional(),
  contentChunk: z.string().optional(), // camelCase variant
});

// Citation verification events
export const ClaimVerifiedEventSchema = BaseEventSchema.extend({
  eventType: z.literal('claim_verified'),
  claim_id: z.string().optional(),
  claimId: z.string().optional(), // camelCase variant
  claim_text: z.string().optional(),
  claimText: z.string().optional(), // camelCase variant
  position_start: z.number().optional(),
  positionStart: z.number().optional(), // camelCase variant
  position_end: z.number().optional(),
  positionEnd: z.number().optional(), // camelCase variant
  verdict: z.string().optional(),
  confidence_level: z.string().optional(),
  confidenceLevel: z.string().optional(), // camelCase variant
  evidence_preview: z.string().optional(),
  evidencePreview: z.string().optional(), // camelCase variant
  reasoning: z.string().nullable().optional(),
});

export const CitationCorrectedEventSchema = BaseEventSchema.extend({
  eventType: z.literal('citation_corrected'),
  claim_id: z.string().optional(),
  correction_type: z.string().optional(),
  reasoning: z.string().nullable().optional(),
});

export const VerificationSummaryEventSchema = BaseEventSchema.extend({
  eventType: z.literal('verification_summary'),
  message_id: z.string().optional(),
  total_claims: z.number().optional(),
  totalClaims: z.number().optional(), // camelCase variant
  supported: z.number().optional(),
  partial: z.number().optional(),
  unsupported: z.number().optional(),
  contradicted: z.number().optional(),
  abstained_count: z.number().optional(),
  abstainedCount: z.number().optional(), // camelCase variant
  citation_corrections: z.number().optional(),
  warning: z.boolean().optional(),
});

// Content revised event (Stage 7)
export const ContentRevisedEventSchema = BaseEventSchema.extend({
  eventType: z.literal('content_revised'),
  content: z.string().optional(),
  revision_count: z.number().optional(),
  revisionCount: z.number().optional(), // camelCase variant
});

// Persistence event
export const PersistenceCompletedEventSchema = BaseEventSchema.extend({
  eventType: z.literal('persistence_completed'),
  chat_id: z.string(),
  message_id: z.string(),
  research_session_id: z.string().nullable().optional(),
  chat_title: z.string().optional(),
  was_draft: z.boolean().optional(),
  counts: z.record(z.string(), z.number()).optional(),
});

// Error event
export const StreamErrorEventSchema = BaseEventSchema.extend({
  eventType: z.literal('error'),
  error_code: z.string().optional(),
  error_message: z.string().optional(),
  errorMessage: z.string().optional(), // camelCase variant
  recoverable: z.boolean().optional(),
});

// Clarification event
export const ClarificationNeededEventSchema = BaseEventSchema.extend({
  eventType: z.literal('clarification_needed'),
  questions: z.array(z.string()),
  round: z.number().optional(),
});

// Discriminated union of all known event types
export const StreamEventSchema = z.discriminatedUnion('eventType', [
  AgentStartedEventSchema,
  AgentCompletedEventSchema,
  ResearchStartedEventSchema,
  ResearchCompletedEventSchema,
  PlanCreatedEventSchema,
  StepStartedEventSchema,
  StepCompletedEventSchema,
  ToolCallEventSchema,
  ToolResultEventSchema,
  ReflectionDecisionEventSchema,
  SynthesisStartedEventSchema,
  SynthesisProgressEventSchema,
  ClaimVerifiedEventSchema,
  CitationCorrectedEventSchema,
  VerificationSummaryEventSchema,
  ContentRevisedEventSchema,
  PersistenceCompletedEventSchema,
  StreamErrorEventSchema,
  ClarificationNeededEventSchema,
]);

// Type inferred from schema
export type ValidatedStreamEvent = z.infer<typeof StreamEventSchema>;

/**
 * Safely parse and validate an SSE event.
 * Returns the validated event or null if validation fails.
 * Logs warnings for malformed events but doesn't throw.
 */
export function parseStreamEvent(jsonString: string): ValidatedStreamEvent | null {
  try {
    const parsed = JSON.parse(jsonString);

    // First check if it has the required eventType field
    if (!parsed || typeof parsed.eventType !== 'string') {
      console.warn('[SSE Validation] Event missing eventType:', parsed);
      return null;
    }

    // Try to validate with schema
    const result = StreamEventSchema.safeParse(parsed);

    if (result.success) {
      return result.data;
    }

    // Log validation error but return the raw parsed data as unknown event
    // This allows forward compatibility with new event types
    console.warn(
      '[SSE Validation] Event failed schema validation:',
      result.error.format(),
      'Raw event:',
      parsed
    );

    // Return raw parsed as a base event for unknown types
    // The switch statement will ignore unknown eventTypes anyway
    return parsed as ValidatedStreamEvent;
  } catch (e) {
    console.error('[SSE Validation] Failed to parse JSON:', e, jsonString);
    return null;
  }
}
