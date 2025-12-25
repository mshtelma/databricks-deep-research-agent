import { cn } from '@/lib/utils';

interface PlanStep {
  index: number;
  title: string;
  description?: string;
  status: 'pending' | 'in_progress' | 'completed' | 'skipped';
}

interface Plan {
  title?: string;
  reasoning?: string;
  steps: PlanStep[];
}

interface PlanProgressProps {
  plan: Plan | null;
  currentStepIndex?: number;
  className?: string;
}

export function PlanProgress({
  plan,
  currentStepIndex = -1,
  className,
}: PlanProgressProps) {
  if (!plan || plan.steps.length === 0) {
    return null;
  }

  return (
    <div data-testid="reasoning-panel" className={cn('rounded-lg border bg-card p-4', className)}>
      <h3 data-testid="research-status" className="font-semibold text-sm mb-3 flex items-center gap-2">
        <PlanIcon className="w-4 h-4" />
        Research Plan
        {plan.title && <span className="font-normal text-muted-foreground">- {plan.title}</span>}
      </h3>

      <div className="space-y-2">
        {plan.steps.map((step, index) => (
          <StepItem
            key={step.index ?? index}
            step={step}
            stepIndex={index}
            isActive={index === currentStepIndex}
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
}

function StepItem({ step, stepIndex, isActive }: StepItemProps) {
  return (
    <div
      data-testid={`reasoning-step-${stepIndex}`}
      className={cn(
        'flex items-start gap-3 p-2 rounded-md transition-colors',
        isActive && 'bg-primary/5 border border-primary/20'
      )}
    >
      <StepStatusIcon status={step.status} isActive={isActive} />
      <div className="flex-1 min-w-0">
        <p
          className={cn(
            'text-sm font-medium',
            step.status === 'completed' && 'text-muted-foreground line-through',
            step.status === 'skipped' && 'text-muted-foreground'
          )}
        >
          {step.title}
        </p>
        {step.description && (
          <p className="text-xs text-muted-foreground mt-0.5 line-clamp-2">
            {step.description}
          </p>
        )}
      </div>
    </div>
  );
}

interface StepStatusIconProps {
  status: PlanStep['status'];
  isActive: boolean;
}

function StepStatusIcon({ status, isActive }: StepStatusIconProps) {
  if (isActive && status !== 'completed') {
    return (
      <div className="w-5 h-5 rounded-full border-2 border-primary flex items-center justify-center">
        <div className="w-2 h-2 rounded-full bg-primary animate-pulse" />
      </div>
    );
  }

  switch (status) {
    case 'completed':
      return (
        <div className="w-5 h-5 rounded-full bg-green-500 flex items-center justify-center">
          <CheckIcon className="w-3 h-3 text-white" />
        </div>
      );
    case 'skipped':
      return (
        <div className="w-5 h-5 rounded-full bg-muted flex items-center justify-center">
          <MinusIcon className="w-3 h-3 text-muted-foreground" />
        </div>
      );
    default:
      return (
        <div className="w-5 h-5 rounded-full border-2 border-muted-foreground/30" />
      );
  }
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
