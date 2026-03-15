/**
 * PlanReviewModal - Modal for reviewing and editing research plans.
 *
 * Features (T043):
 * - Display plan steps with source hints per step
 * - Countdown timer showing time until auto-proceed
 * - Approve, Edit, and Cancel buttons
 * - If editing: show PlanEditor component
 */

import * as React from 'react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { PlanEditor } from './PlanEditor';
import type {
  PlanWithSources,
  PlanStepWithSources,
  AvailableSource,
  EditedPlan,
} from '@/types/dataSources';

interface PlanReviewModalProps {
  isOpen: boolean;
  plan: PlanWithSources | null;
  availableSources: AvailableSource[];
  timeoutSeconds: number;
  onApprove: () => void;
  onApproveWithEdits: (editedPlan: EditedPlan) => void;
  onReject: (reason?: string) => void;
  onClose: () => void;
  className?: string;
}

export function PlanReviewModal({
  isOpen,
  plan,
  availableSources,
  timeoutSeconds,
  onApprove,
  onApproveWithEdits,
  onReject,
  onClose,
  className,
}: PlanReviewModalProps) {
  const dialogRef = React.useRef<HTMLDivElement>(null);
  const [isEditing, setIsEditing] = React.useState(false);
  const [editedSteps, setEditedSteps] = React.useState<PlanStepWithSources[]>([]);
  const [remainingTime, setRemainingTime] = React.useState(timeoutSeconds);
  const [rejectReason, setRejectReason] = React.useState('');
  const [showRejectInput, setShowRejectInput] = React.useState(false);

  // Initialize/reset state when modal opens or plan changes
  React.useEffect(() => {
    if (isOpen && plan) {
      setIsEditing(false);
      setEditedSteps(plan.steps);
      setRemainingTime(timeoutSeconds);
      setRejectReason('');
      setShowRejectInput(false);
    }
  }, [isOpen, plan, timeoutSeconds]);

  // Countdown timer
  React.useEffect(() => {
    if (!isOpen || remainingTime <= 0) return;

    const timer = setInterval(() => {
      setRemainingTime((prev) => {
        if (prev <= 1) {
          // Auto-approve when timer reaches 0
          onApprove();
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(timer);
  }, [isOpen, remainingTime, onApprove]);

  // Close on escape key (if not editing)
  React.useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen && !isEditing) {
        onClose();
      }
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, isEditing, onClose]);

  // Close on click outside (if not editing)
  React.useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (
        dialogRef.current &&
        !dialogRef.current.contains(e.target as Node) &&
        isOpen &&
        !isEditing
      ) {
        onClose();
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [isOpen, isEditing, onClose]);

  const handleApproveWithEdits = () => {
    onApproveWithEdits({ steps: editedSteps });
  };

  const handleReject = () => {
    if (showRejectInput) {
      onReject(rejectReason || undefined);
    } else {
      setShowRejectInput(true);
    }
  };

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return mins > 0 ? `${mins}:${secs.toString().padStart(2, '0')}` : `${secs}s`;
  };

  if (!isOpen || !plan) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop */}
      <div className="fixed inset-0 bg-black/50" aria-hidden="true" />

      {/* Dialog */}
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="plan-review-title"
        className={cn(
          'relative z-50 w-full max-w-3xl rounded-lg bg-background shadow-lg',
          'max-h-[90vh] overflow-hidden flex flex-col',
          'animate-in fade-in-0 zoom-in-95',
          className
        )}
      >
        {/* Header with countdown */}
        <div className="flex items-center justify-between px-6 py-4 border-b">
          <div>
            <h3 id="plan-review-title" className="text-lg font-semibold">
              Review Research Plan
            </h3>
            <p className="text-sm text-muted-foreground mt-0.5">{plan.title}</p>
          </div>
          <div className="flex items-center gap-4">
            {/* Countdown timer */}
            <div className="flex items-center gap-2">
              <ClockIcon className="h-4 w-4 text-muted-foreground" />
              <span
                className={cn(
                  'text-sm font-mono',
                  remainingTime <= 10 ? 'text-amber-600 font-semibold' : 'text-muted-foreground'
                )}
              >
                {formatTime(remainingTime)}
              </span>
              <span className="text-xs text-muted-foreground">until auto-approve</span>
            </div>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6">
          {/* Plan thought/reasoning */}
          {plan.thought && (
            <div className="mb-4 p-4 rounded-lg bg-muted/50 border">
              <div className="flex items-center gap-2 mb-2">
                <LightbulbIcon className="h-4 w-4 text-amber-600" />
                <span className="text-sm font-medium">Reasoning</span>
              </div>
              <p className="text-sm text-muted-foreground">{plan.thought}</p>
            </div>
          )}

          {/* Steps display or editor */}
          {isEditing ? (
            <PlanEditor
              steps={editedSteps}
              availableSources={availableSources}
              onStepsChange={setEditedSteps}
            />
          ) : (
            <div className="space-y-3">
              {plan.steps.map((step, index) => (
                <PlanStepCard key={step.id} step={step} index={index} />
              ))}
            </div>
          )}

          {/* Reject reason input */}
          {showRejectInput && (
            <div className="mt-4 p-4 rounded-lg border bg-destructive/5">
              <label className="text-sm font-medium block mb-2">
                Reason for rejection (optional)
              </label>
              <textarea
                value={rejectReason}
                onChange={(e) => setRejectReason(e.target.value)}
                placeholder="Explain why this plan should be rejected..."
                rows={2}
                className={cn(
                  'w-full resize-none rounded-md border border-input bg-background px-3 py-2 text-sm',
                  'placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring'
                )}
              />
            </div>
          )}
        </div>

        {/* Footer actions */}
        <div className="flex items-center justify-between px-6 py-4 border-t bg-muted/30">
          <div>
            {!showRejectInput ? (
              <Button variant="outline" onClick={handleReject}>
                Reject Plan
              </Button>
            ) : (
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  onClick={() => setShowRejectInput(false)}
                >
                  Back
                </Button>
                <Button variant="destructive" onClick={handleReject}>
                  Confirm Reject
                </Button>
              </div>
            )}
          </div>
          <div className="flex gap-2">
            {isEditing ? (
              <>
                <Button variant="outline" onClick={() => setIsEditing(false)}>
                  Cancel Edit
                </Button>
                <Button onClick={handleApproveWithEdits}>
                  <CheckIcon className="h-4 w-4 mr-2" />
                  Approve with Changes
                </Button>
              </>
            ) : (
              <>
                <Button variant="outline" onClick={() => setIsEditing(true)}>
                  <EditIcon className="h-4 w-4 mr-2" />
                  Edit Plan
                </Button>
                <Button onClick={onApprove}>
                  <CheckIcon className="h-4 w-4 mr-2" />
                  Approve
                </Button>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

interface PlanStepCardProps {
  step: PlanStepWithSources;
  index: number;
}

function PlanStepCard({ step, index }: PlanStepCardProps) {
  const [isExpanded, setIsExpanded] = React.useState(false);

  return (
    <div className="rounded-lg border bg-card p-4">
      <div className="flex items-start gap-3">
        <div
          className={cn(
            'flex items-center justify-center w-6 h-6 rounded-full text-xs font-medium',
            'bg-primary/10 text-primary'
          )}
        >
          {index + 1}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between gap-2">
            <h4 className="font-medium text-sm">{step.title}</h4>
            <div className="flex items-center gap-2">
              <span
                className={cn(
                  'px-2 py-0.5 rounded text-xs',
                  step.stepType === 'research'
                    ? 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200'
                    : 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200'
                )}
              >
                {step.stepType}
              </span>
              {step.needsSearch && (
                <span title="Needs search">
                  <SearchIcon className="h-3.5 w-3.5 text-muted-foreground" />
                </span>
              )}
            </div>
          </div>
          {step.description && (
            <p className="text-sm text-muted-foreground mt-1">{step.description}</p>
          )}

          {/* Source hints summary */}
          {step.sourceHints.length > 0 && (
            <div className="mt-2">
              <button
                type="button"
                onClick={() => setIsExpanded(!isExpanded)}
                className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
              >
                <ChevronIcon
                  className={cn(
                    'h-3 w-3 transition-transform',
                    isExpanded && 'rotate-90'
                  )}
                />
                <span>
                  {step.sourceHints.length} source{step.sourceHints.length !== 1 ? 's' : ''} configured
                </span>
              </button>

              {isExpanded && (
                <div className="mt-2 space-y-1 pl-4">
                  {step.sourceHints.map((hint) => (
                    <div
                      key={hint.sourceName}
                      className="flex items-center gap-2 text-xs"
                    >
                      <span
                        className={cn(
                          'w-4 h-4 rounded-full flex items-center justify-center text-[10px] font-medium',
                          hint.priority === 1
                            ? 'bg-green-100 text-green-700'
                            : hint.priority === 2
                            ? 'bg-blue-100 text-blue-700'
                            : 'bg-gray-100 text-gray-700'
                        )}
                      >
                        {hint.priority}
                      </span>
                      <span className="font-medium">{hint.sourceName}</span>
                      {hint.queryHint && (
                        <span className="text-muted-foreground italic truncate">
                          "{hint.queryHint}"
                        </span>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// Icons
function ClockIcon({ className }: { className?: string }) {
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
      <polyline points="12 6 12 12 16 14" />
    </svg>
  );
}

function LightbulbIcon({ className }: { className?: string }) {
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
      <path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5" />
      <path d="M9 18h6" />
      <path d="M10 22h4" />
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

function EditIcon({ className }: { className?: string }) {
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
      <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
      <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
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
      <path d="m9 18 6-6-6-6" />
    </svg>
  );
}

export default PlanReviewModal;
