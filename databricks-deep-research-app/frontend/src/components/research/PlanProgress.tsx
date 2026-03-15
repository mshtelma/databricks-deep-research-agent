import * as React from 'react';
import { cn } from '@/lib/utils';

interface PlanStep {
  index: number;
  title: string;
  description?: string;
  status: 'pending' | 'in_progress' | 'completed' | 'skipped';
  step_type?: 'research' | 'analysis';
  needs_search?: boolean;
  observation?: string | null;
}

interface Plan {
  title?: string;
  reasoning?: string;
  thought?: string;
  steps: PlanStep[];
  iteration?: number;
}

interface PlanProgressProps {
  plan: Plan | null;
  currentStepIndex?: number;
  showDetails?: boolean;
  className?: string;
}

export function PlanProgress({
  plan,
  currentStepIndex = -1,
  showDetails = false,
  className,
}: PlanProgressProps) {
  const [expandedSteps, setExpandedSteps] = React.useState<Set<number>>(new Set());

  if (!plan || plan.steps.length === 0) {
    return null;
  }

  const toggleStep = (index: number) => {
    setExpandedSteps((prev) => {
      const next = new Set(prev);
      if (next.has(index)) {
        next.delete(index);
      } else {
        next.add(index);
      }
      return next;
    });
  };

  // Calculate progress
  const completedCount = plan.steps.filter((s) => s.status === 'completed').length;
  const progressPercent = (completedCount / plan.steps.length) * 100;

  return (
    <div data-testid="reasoning-panel" className={cn('rounded-lg border bg-card p-3', className)}>
      {/* Header */}
      <div className="flex items-center justify-between mb-2">
        <h3 data-testid="research-status" className="font-semibold text-xs flex items-center gap-1.5">
          <PlanIcon className="w-3.5 h-3.5" />
          Research Plan
          {plan.iteration && plan.iteration > 1 && (
            <span className="text-[10px] font-normal text-muted-foreground">
              (Iteration {plan.iteration})
            </span>
          )}
        </h3>
        <span className="text-[10px] text-muted-foreground">
          {completedCount}/{plan.steps.length}
        </span>
      </div>

      {/* Progress bar */}
      <div className="h-1 bg-muted rounded-full overflow-hidden mb-2">
        <div
          className="h-full bg-primary transition-all duration-300"
          style={{ width: `${progressPercent}%` }}
        />
      </div>

      {/* Plan title and reasoning */}
      {(plan.title || plan.reasoning || plan.thought) && showDetails && (
        <div className="mb-2 p-1.5 bg-muted/50 rounded text-[11px]">
          {plan.title && <p className="font-medium">{plan.title}</p>}
          {(plan.reasoning || plan.thought) && (
            <p className="text-muted-foreground mt-0.5 line-clamp-2">
              {plan.reasoning || plan.thought}
            </p>
          )}
        </div>
      )}

      {/* Steps - scrollable container */}
      <div className="space-y-1 max-h-[240px] overflow-y-auto pr-0.5">
        {plan.steps.map((step, index) => (
          <StepItem
            key={step.index ?? index}
            step={step}
            stepIndex={index}
            isActive={index === currentStepIndex}
            isExpanded={expandedSteps.has(index)}
            onToggle={() => toggleStep(index)}
            showDetails={showDetails}
          />
        ))}
      </div>
    </div>
  );
}

interface StepItemProps {
  step: PlanStep;
  stepIndex: number;
  isActive: boolean;
  isExpanded: boolean;
  onToggle: () => void;
  showDetails: boolean;
}

function StepItem({ step, stepIndex, isActive, isExpanded, onToggle, showDetails }: StepItemProps) {
  const hasExpandableContent = showDetails && (step.description || step.observation);

  return (
    <div
      data-testid={`reasoning-step-${stepIndex}`}
      className={cn(
        'rounded transition-colors',
        isActive && 'bg-primary/5 border border-primary/20',
        !isActive && 'hover:bg-muted/50'
      )}
    >
      <button
        type="button"
        onClick={onToggle}
        disabled={!hasExpandableContent}
        className={cn(
          'w-full flex items-start gap-2 p-1.5 text-left',
          hasExpandableContent && 'cursor-pointer'
        )}
      >
        <StepStatusIcon status={step.status} isActive={isActive} stepType={step.step_type} needsSearch={step.needs_search} />
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-1.5">
            <p
              className={cn(
                'text-xs font-medium',
                step.status === 'completed' && 'text-muted-foreground',
                step.status === 'skipped' && 'text-muted-foreground'
              )}
            >
              {step.title}
            </p>
            {step.needs_search && (
              <span className="text-[10px] px-1 py-0.5 rounded bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300">
                Search
              </span>
            )}
          </div>
          {!isExpanded && step.description && showDetails && (
            <p className="text-[10px] text-muted-foreground mt-0.5 line-clamp-1">
              {step.description}
            </p>
          )}
        </div>
        {hasExpandableContent && (
          <ChevronIcon
            className={cn(
              'w-3.5 h-3.5 text-muted-foreground transition-transform flex-shrink-0 mt-0.5',
              isExpanded && 'rotate-180'
            )}
          />
        )}
      </button>

      {/* Expanded content */}
      {isExpanded && hasExpandableContent && (
        <div className="px-1.5 pb-1.5 pt-0 ml-6 border-l-2 border-muted">
          {step.description && (
            <p className="text-[10px] text-muted-foreground mb-1">{step.description}</p>
          )}
          {step.observation && (
            <div className="bg-muted/50 p-1.5 rounded text-[10px]">
              <p className="font-medium text-muted-foreground mb-0.5">Observation:</p>
              <p className="text-foreground">{step.observation}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

interface StepStatusIconProps {
  status: PlanStep['status'];
  isActive: boolean;
  stepType?: string;
  needsSearch?: boolean;
}

function StepStatusIcon({ status, isActive, needsSearch }: StepStatusIconProps) {
  if (isActive && status !== 'completed') {
    return (
      <div className="w-4 h-4 rounded-full border-2 border-primary flex items-center justify-center flex-shrink-0">
        <div className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
      </div>
    );
  }

  switch (status) {
    case 'completed':
      return (
        <div className="w-4 h-4 rounded-full bg-green-500 flex items-center justify-center flex-shrink-0">
          <CheckIcon className="w-2.5 h-2.5 text-white" />
        </div>
      );
    case 'skipped':
      return (
        <div className="w-4 h-4 rounded-full bg-muted flex items-center justify-center flex-shrink-0">
          <MinusIcon className="w-2.5 h-2.5 text-muted-foreground" />
        </div>
      );
    default:
      if (needsSearch) {
        return (
          <div className="w-4 h-4 rounded-full border-2 border-blue-300 flex items-center justify-center flex-shrink-0">
            <SearchIcon className="w-2.5 h-2.5 text-blue-400" />
          </div>
        );
      }
      return (
        <div className="w-4 h-4 rounded-full border-2 border-muted-foreground/30 flex-shrink-0" />
      );
  }
}

// Icons

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

function CheckIcon({ className }: { className?: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="3"
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

function SearchIcon({ className }: { className?: string }) {
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
      <circle cx="11" cy="11" r="8" />
      <path d="m21 21-4.3-4.3" />
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
