import * as React from 'react';
import { Message, Source, ResearchPlan } from '@/types';
import { cn } from '@/lib/utils';
import { Card, CardContent } from '@/components/ui/card';
import { MarkdownRenderer } from '@/components/common';

interface ReasoningSummary {
  planTitle?: string;
  stepsCompleted: number;
  totalSteps: number;
  totalSources: number;
  planIterations?: number;
  observations?: string[];
}

interface AgentMessageProps {
  message: Message;
  sources?: Source[];
  reasoning?: ReasoningSummary;
  plan?: ResearchPlan | null;
  isStreaming?: boolean;
  onRegenerate?: () => void;
  className?: string;
}

export function AgentMessage({
  message,
  sources = [],
  reasoning,
  plan,
  isStreaming = false,
  onRegenerate,
  className,
}: AgentMessageProps) {
  const [showSources, setShowSources] = React.useState(false);
  const [showReasoning, setShowReasoning] = React.useState(false);

  const hasReasoning = reasoning || (plan && plan.steps && plan.steps.length > 0);

  return (
    <div data-testid="agent-response" className={cn('flex justify-start', className)}>
      <Card className="max-w-[90%] bg-muted">
        <CardContent className="p-4">
          {/* Message content with markdown rendering */}
          <div className="relative">
            <MarkdownRenderer content={message.content} />
            {isStreaming && (
              <span className="inline-block w-2 h-4 bg-primary animate-pulse ml-1 align-text-bottom" />
            )}
          </div>

          {/* Reasoning section */}
          {hasReasoning && !isStreaming && (
            <div className="mt-4 pt-4 border-t">
              <button
                onClick={() => setShowReasoning(!showReasoning)}
                aria-expanded={showReasoning}
                aria-label="View reasoning details"
                className="text-sm text-muted-foreground hover:text-foreground flex items-center gap-1"
              >
                <BrainIcon className="w-4 h-4" />
                View reasoning
                {reasoning && (
                  <span className="text-xs text-muted-foreground ml-1">
                    ({reasoning.stepsCompleted}/{reasoning.totalSteps} steps)
                  </span>
                )}
                <ChevronIcon
                  className={cn(
                    'w-4 h-4 transition-transform',
                    showReasoning && 'rotate-180'
                  )}
                />
              </button>

              {showReasoning && (
                <div className="mt-3 space-y-3">
                  {/* Plan overview */}
                  {plan && (
                    <div className="bg-background/50 p-3 rounded-lg">
                      <h4 className="text-xs font-semibold text-muted-foreground mb-2 flex items-center gap-1">
                        <PlanIcon className="w-3.5 h-3.5" />
                        Research Plan
                        {plan.iteration && plan.iteration > 1 && (
                          <span className="font-normal">(Iteration {plan.iteration})</span>
                        )}
                      </h4>
                      {plan.title && (
                        <p className="text-sm font-medium mb-2">{plan.title}</p>
                      )}
                      {plan.thought && (
                        <p className="text-xs text-muted-foreground mb-3">{plan.thought}</p>
                      )}

                      {/* Steps summary */}
                      <div className="space-y-1.5">
                        {plan.steps.map((step, index) => (
                          <PlanStepItem key={step.id || index} step={step} index={index} />
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Reasoning summary stats */}
                  {reasoning && (
                    <div className="flex flex-wrap gap-3 text-xs">
                      <div className="flex items-center gap-1 text-muted-foreground">
                        <StepsIcon className="w-3.5 h-3.5" />
                        <span>{reasoning.stepsCompleted} steps completed</span>
                      </div>
                      <div className="flex items-center gap-1 text-muted-foreground">
                        <SourcesIcon className="w-3.5 h-3.5" />
                        <span>{reasoning.totalSources} sources used</span>
                      </div>
                      {reasoning.planIterations && reasoning.planIterations > 1 && (
                        <div className="flex items-center gap-1 text-muted-foreground">
                          <RefreshIcon className="w-3.5 h-3.5" />
                          <span>{reasoning.planIterations} plan iterations</span>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Observations */}
                  {reasoning?.observations && reasoning.observations.length > 0 && (
                    <div className="space-y-2">
                      <h4 className="text-xs font-semibold text-muted-foreground flex items-center gap-1">
                        <NotesIcon className="w-3.5 h-3.5" />
                        Key Observations
                      </h4>
                      <ul className="space-y-1">
                        {reasoning.observations.slice(0, 5).map((obs, i) => (
                          <li
                            key={i}
                            className="text-xs text-muted-foreground bg-background/50 p-2 rounded"
                          >
                            {obs.length > 200 ? obs.slice(0, 200) + '...' : obs}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Sources section */}
          {sources.length > 0 && (
            <div className={cn('mt-4 pt-4 border-t', hasReasoning && !showReasoning && 'pt-4')}>
              <button
                onClick={() => setShowSources(!showSources)}
                aria-expanded={showSources}
                aria-label={`View ${sources.length} source${sources.length !== 1 ? 's' : ''}`}
                className="text-sm text-muted-foreground hover:text-foreground flex items-center gap-1"
              >
                <SourcesIcon className="w-4 h-4" />
                {sources.length} source{sources.length !== 1 ? 's' : ''}
                <ChevronIcon
                  className={cn(
                    'w-4 h-4 transition-transform',
                    showSources && 'rotate-180'
                  )}
                />
              </button>

              {showSources && (
                <div className="mt-2 space-y-2">
                  {sources.map((source, index) => (
                    <SourceCard key={source.url || index} source={source} index={index + 1} />
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Timestamp and regenerate */}
          {!isStreaming && (
            <div className="flex items-center justify-between mt-2">
              <span className="text-xs text-muted-foreground">
                {new Date(message.created_at).toLocaleTimeString()}
              </span>
              {onRegenerate && (
                <button
                  data-testid="regenerate-response"
                  onClick={onRegenerate}
                  className="text-xs text-muted-foreground hover:text-foreground transition-colors"
                >
                  Regenerate
                </button>
              )}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

interface PlanStepItemProps {
  step: ResearchPlan['steps'][0];
  index: number;
}

function PlanStepItem({ step }: PlanStepItemProps) {
  const statusIcons = {
    pending: <CircleIcon className="w-3 h-3 text-muted-foreground" />,
    in_progress: <SpinnerIcon className="w-3 h-3 text-primary animate-spin" />,
    completed: <CheckIcon className="w-3 h-3 text-green-500" />,
    skipped: <MinusIcon className="w-3 h-3 text-muted-foreground" />,
  };

  return (
    <div className="flex items-start gap-2 text-xs">
      <span className="mt-0.5">{statusIcons[step.status]}</span>
      <div className="flex-1 min-w-0">
        <span
          className={cn(
            step.status === 'completed' && 'text-muted-foreground',
            step.status === 'skipped' && 'text-muted-foreground line-through'
          )}
        >
          {step.title}
        </span>
        {step.needs_search && (
          <span className="ml-1 text-[10px] text-blue-600 dark:text-blue-400">[search]</span>
        )}
      </div>
    </div>
  );
}

function SourceCard({ source, index }: { source: Source; index: number }) {
  return (
    <a
      data-testid="citation"
      href={source.url}
      target="_blank"
      rel="noopener noreferrer"
      className="block p-2 rounded border hover:bg-background transition-colors"
    >
      <div className="flex items-start gap-2">
        <span className="text-xs bg-primary/10 text-primary px-1.5 py-0.5 rounded">
          [{index}]
        </span>
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium truncate">{source.title || source.url}</p>
          {source.snippet && (
            <p className="text-xs text-muted-foreground line-clamp-2 mt-1">
              {source.snippet}
            </p>
          )}
        </div>
      </div>
    </a>
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

function PlanIcon({ className }: { className?: string }) {
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
      <path d="M9 11V6a2 2 0 1 0-4 0v5a2 2 0 0 0 4 0Z" />
      <path d="M9 8h3" />
      <path d="M12 8h3" />
      <path d="M15 8V5a2 2 0 1 1 4 0v6a2 2 0 0 1-4 0v-3" />
      <path d="M5 16v1" />
      <path d="M19 16v1" />
      <path d="M5 20h14" />
    </svg>
  );
}

function SourcesIcon({ className }: { className?: string }) {
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
      <path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20" />
    </svg>
  );
}

function StepsIcon({ className }: { className?: string }) {
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
      <path d="M12 22v-4" />
      <path d="M12 8V4" />
      <path d="M12 18a2 2 0 1 0 0-4 2 2 0 0 0 0 4Z" />
      <path d="M12 8a2 2 0 1 0 0-4 2 2 0 0 0 0 4Z" />
    </svg>
  );
}

function NotesIcon({ className }: { className?: string }) {
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
      <path d="M16 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V8Z" />
      <path d="M15 3v4a2 2 0 0 0 2 2h4" />
      <path d="M10 12h4" />
      <path d="M10 16h4" />
    </svg>
  );
}

function RefreshIcon({ className }: { className?: string }) {
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
      <path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
      <path d="M3 3v5h5" />
      <path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16" />
      <path d="M16 16h5v5" />
    </svg>
  );
}

function ChevronIcon({ className }: { className?: string }) {
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
      <path d="m6 9 6 6 6-6" />
    </svg>
  );
}

function CircleIcon({ className }: { className?: string }) {
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
    </svg>
  );
}

function SpinnerIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none">
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth="4"
      />
      <path
        className="opacity-75"
        fill="currentColor"
        d="m4 12a8 8 0 0 1 8-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 0 1 4 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
      />
    </svg>
  );
}

function CheckIcon({ className }: { className?: string }) {
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
      <path d="M20 6 9 17l-5-5" />
    </svg>
  );
}

function MinusIcon({ className }: { className?: string }) {
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
      <path d="M5 12h14" />
    </svg>
  );
}
