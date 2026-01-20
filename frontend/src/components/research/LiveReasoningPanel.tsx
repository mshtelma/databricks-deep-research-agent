import * as React from 'react';
import { cn } from '@/lib/utils';
import { ReasoningStep, ReasoningStepData, StepType } from './ReasoningStep';
import type { StreamEvent, ResearchPlan } from '@/types';

interface LiveReasoningPanelProps {
  plan: ResearchPlan | null;
  currentStepIndex: number | null;
  events: StreamEvent[];
  isStreaming?: boolean;
  className?: string;
}

export function LiveReasoningPanel({
  plan,
  currentStepIndex,
  events,
  isStreaming = false,
  className,
}: LiveReasoningPanelProps) {
  const [expandedSteps, setExpandedSteps] = React.useState<Set<string>>(new Set());

  // Build reasoning steps from plan and events
  const reasoningSteps = React.useMemo(() => {
    if (!plan?.steps) return [];

    return plan.steps.map((step, index): ReasoningStepData => {
      const stepId = step.id || `step-${index}`;

      // Get step-related events
      const stepStartEvent = events.find(
        (e) => e.eventType === 'step_started' && 'stepIndex' in e && (e as { stepIndex?: number }).stepIndex === index
      );
      const stepCompleteEvent = events.find(
        (e) => e.eventType === 'step_completed' && 'stepIndex' in e && (e as { stepIndex?: number }).stepIndex === index
      );

      // Determine status
      let status: ReasoningStepData['status'] = 'pending';
      if (step.status === 'completed') {
        status = 'completed';
      } else if (step.status === 'skipped') {
        status = 'pending'; // Show as pending for skipped
      } else if (index === currentStepIndex) {
        status = 'in_progress';
      }

      // Determine step type based on step properties
      let type: StepType = 'analyze';
      if (step.needsSearch) {
        type = 'search';
      } else if (step.stepType === 'analysis') {
        type = 'analyze';
      }

      // Extract details from events
      const details: ReasoningStepData['details'] = {};

      if (stepCompleteEvent && 'observationSummary' in stepCompleteEvent) {
        const eventData = stepCompleteEvent as unknown as Record<string, unknown>;
        details.observation = eventData.observationSummary as string | undefined;
        details.sourcesFound = eventData.sourcesFound as number | undefined;
      }

      // Note: search queries and URLs would come from additional events
      // that we'd need to add to the backend streaming
      if (step.observation) {
        details.observation = step.observation;
      }

      return {
        id: stepId,
        type,
        title: step.title,
        description: step.description,
        status,
        timestamp: stepStartEvent?.timestamp,
        details: Object.keys(details).length > 0 ? details : undefined,
      };
    });
  }, [plan?.steps, currentStepIndex, events]);

  const toggleStep = (stepId: string) => {
    setExpandedSteps((prev) => {
      const next = new Set(prev);
      if (next.has(stepId)) {
        next.delete(stepId);
      } else {
        next.add(stepId);
      }
      return next;
    });
  };

  if (!plan || reasoningSteps.length === 0) {
    return null;
  }

  // Calculate progress
  const completedCount = reasoningSteps.filter((s) => s.status === 'completed').length;
  const progressPercent = (completedCount / reasoningSteps.length) * 100;

  return (
    <div className={cn('rounded-lg border bg-card', className)}>
      {/* Header */}
      <div className="p-4 border-b">
        <div className="flex items-center justify-between mb-2">
          <h3 className="font-semibold text-sm flex items-center gap-2">
            <BrainIcon className="w-4 h-4 text-primary" />
            Research Progress
            {isStreaming && (
              <span className="inline-flex items-center gap-1 text-xs font-normal text-muted-foreground">
                <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
                Live
              </span>
            )}
          </h3>
          <span className="text-xs text-muted-foreground">
            {completedCount} / {reasoningSteps.length} steps
          </span>
        </div>

        {/* Progress bar */}
        <div className="h-1.5 bg-muted rounded-full overflow-hidden">
          <div
            className="h-full bg-primary transition-all duration-300 ease-out"
            style={{ width: `${progressPercent}%` }}
          />
        </div>

        {/* Plan title and thought */}
        {(plan.title || plan.thought) && (
          <div className="mt-3 text-xs text-muted-foreground">
            {plan.title && <p className="font-medium">{plan.title}</p>}
            {plan.thought && <p className="mt-1 line-clamp-2">{plan.thought}</p>}
          </div>
        )}
      </div>

      {/* Steps */}
      <div className="p-3 space-y-2 max-h-[400px] overflow-y-auto">
        {reasoningSteps.map((step) => (
          <ReasoningStep
            key={step.id}
            step={step}
            isExpanded={expandedSteps.has(step.id)}
            onToggle={() => toggleStep(step.id)}
          />
        ))}
      </div>

      {/* Reflection events */}
      <ReflectionSummary events={events} />
    </div>
  );
}

interface ReflectionSummaryProps {
  events: StreamEvent[];
}

function ReflectionSummary({ events }: ReflectionSummaryProps) {
  const reflectionEvents = events.filter((e) => e.eventType === 'reflection_decision');

  if (reflectionEvents.length === 0) {
    return null;
  }

  const latestReflection = reflectionEvents[reflectionEvents.length - 1];
  if (!latestReflection || !('decision' in latestReflection)) return null;

  const decisionColors = {
    continue: 'text-green-600 bg-green-50 dark:bg-green-950/30',
    adjust: 'text-amber-600 bg-amber-50 dark:bg-amber-950/30',
    complete: 'text-blue-600 bg-blue-50 dark:bg-blue-950/30',
  };

  return (
    <div className="p-3 border-t bg-muted/30">
      <div className="flex items-center gap-2 mb-1">
        <ReflectIcon className="w-3.5 h-3.5 text-muted-foreground" />
        <span className="text-xs font-medium text-muted-foreground">Latest Reflection</span>
      </div>
      <div className="flex items-center gap-2">
        <span
          className={cn(
            'px-2 py-0.5 rounded text-xs font-medium capitalize',
            decisionColors[latestReflection.decision as keyof typeof decisionColors]
          )}
        >
          {latestReflection.decision}
        </span>
        {latestReflection.reasoning && (
          <span className="text-xs text-muted-foreground truncate">
            {latestReflection.reasoning}
          </span>
        )}
      </div>
    </div>
  );
}

// Icons

function BrainIcon({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 4.44-1.54" />
      <path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-4.44-1.54" />
    </svg>
  );
}

function ReflectIcon({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <circle cx="12" cy="12" r="10" />
      <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" />
      <path d="M12 17h.01" />
    </svg>
  );
}
