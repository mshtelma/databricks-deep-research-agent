/**
 * PlanEditor - Editable list of plan steps with source configuration.
 *
 * Features (T044):
 * - Editable list of plan steps
 * - Per-step source selection (multi-select dropdown)
 * - Source priority adjustment (1/2/3 buttons)
 * - Query hint text input per source
 * - Add/remove/reorder step controls
 */

import * as React from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { cn } from '@/lib/utils';
import type {
  PlanStepWithSources,
  StepSourceHint,
  AvailableSource,
  DataSourceType,
} from '@/types/dataSources';

interface PlanEditorProps {
  steps: PlanStepWithSources[];
  availableSources: AvailableSource[];
  onStepsChange: (steps: PlanStepWithSources[]) => void;
  disabled?: boolean;
  className?: string;
}

export function PlanEditor({
  steps,
  availableSources,
  onStepsChange,
  disabled = false,
  className,
}: PlanEditorProps) {
  const [draggedIndex, setDraggedIndex] = React.useState<number | null>(null);

  const handleStepChange = (index: number, updates: Partial<PlanStepWithSources>) => {
    const newSteps = [...steps];
    const currentStep = newSteps[index];
    if (currentStep) {
      newSteps[index] = { ...currentStep, ...updates };
      onStepsChange(newSteps);
    }
  };

  const handleAddStep = () => {
    const newStep: PlanStepWithSources = {
      id: `step-${Date.now()}`,
      title: '',
      description: '',
      stepType: 'research',
      needsSearch: true,
      status: 'pending',
      sourceHints: [],
    };
    onStepsChange([...steps, newStep]);
  };

  const handleRemoveStep = (index: number) => {
    if (steps.length <= 1) return; // Keep at least one step
    const newSteps = steps.filter((_, i) => i !== index);
    onStepsChange(newSteps);
  };

  const handleMoveStep = (fromIndex: number, toIndex: number) => {
    if (toIndex < 0 || toIndex >= steps.length) return;
    const newSteps = [...steps];
    const [moved] = newSteps.splice(fromIndex, 1);
    if (moved) {
      newSteps.splice(toIndex, 0, moved);
      onStepsChange(newSteps);
    }
  };

  const handleDragStart = (index: number) => {
    setDraggedIndex(index);
  };

  const handleDragOver = (e: React.DragEvent, index: number) => {
    e.preventDefault();
    if (draggedIndex === null || draggedIndex === index) return;
    handleMoveStep(draggedIndex, index);
    setDraggedIndex(index);
  };

  const handleDragEnd = () => {
    setDraggedIndex(null);
  };

  return (
    <div className={cn('space-y-3', className)}>
      {steps.map((step, index) => (
        <div
          key={step.id}
          draggable={!disabled}
          onDragStart={() => handleDragStart(index)}
          onDragOver={(e) => handleDragOver(e, index)}
          onDragEnd={handleDragEnd}
          className={cn(
            'rounded-lg border bg-card p-4',
            draggedIndex === index && 'opacity-50 border-primary',
            disabled && 'opacity-60'
          )}
        >
          {/* Step Header */}
          <div className="flex items-center gap-2 mb-3">
            <div
              className={cn(
                'flex items-center justify-center w-6 h-6 rounded-full text-xs font-medium',
                'bg-primary/10 text-primary'
              )}
            >
              {index + 1}
            </div>
            <div className="flex items-center gap-1 cursor-grab" title="Drag to reorder">
              <GripIcon className="h-4 w-4 text-muted-foreground" />
            </div>
            <div className="flex-1">
              <Input
                value={step.title}
                onChange={(e) => handleStepChange(index, { title: e.target.value })}
                placeholder="Step title..."
                disabled={disabled}
                className="font-medium"
              />
            </div>
            <div className="flex items-center gap-1">
              <Button
                variant="ghost"
                size="icon"
                onClick={() => handleMoveStep(index, index - 1)}
                disabled={disabled || index === 0}
                title="Move up"
              >
                <ChevronUpIcon className="h-4 w-4" />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => handleMoveStep(index, index + 1)}
                disabled={disabled || index === steps.length - 1}
                title="Move down"
              >
                <ChevronDownIcon className="h-4 w-4" />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => handleRemoveStep(index)}
                disabled={disabled || steps.length <= 1}
                title="Remove step"
                className="text-muted-foreground hover:text-destructive"
              >
                <TrashIcon className="h-4 w-4" />
              </Button>
            </div>
          </div>

          {/* Step Description */}
          <div className="mb-3">
            <textarea
              value={step.description}
              onChange={(e) => handleStepChange(index, { description: e.target.value })}
              placeholder="Describe what this step should accomplish..."
              rows={2}
              disabled={disabled}
              className={cn(
                'w-full resize-none rounded-md border border-input bg-transparent px-3 py-2 text-sm',
                'placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring',
                'disabled:cursor-not-allowed disabled:opacity-50'
              )}
            />
          </div>

          {/* Step Type */}
          <div className="flex items-center gap-4 mb-3">
            <div className="flex items-center gap-2">
              <span className="text-sm text-muted-foreground">Type:</span>
              <select
                value={step.stepType}
                onChange={(e) =>
                  handleStepChange(index, {
                    stepType: e.target.value as 'research' | 'analysis',
                  })
                }
                disabled={disabled}
                className="rounded-md border border-input bg-background px-2 py-1 text-sm"
              >
                <option value="research">Research</option>
                <option value="analysis">Analysis</option>
              </select>
            </div>
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={step.needsSearch}
                onChange={(e) => handleStepChange(index, { needsSearch: e.target.checked })}
                disabled={disabled}
                className="rounded border-input"
              />
              Needs Search
            </label>
          </div>

          {/* Source Hints */}
          <div>
            <div className="text-sm font-medium mb-2 flex items-center gap-2">
              <span>Source Hints</span>
              <span className="text-xs text-muted-foreground">
                ({step.sourceHints.length} selected)
              </span>
            </div>
            <StepSourceHintsEditor
              sourceHints={step.sourceHints}
              availableSources={availableSources}
              onChange={(hints) => handleStepChange(index, { sourceHints: hints })}
              disabled={disabled}
            />
          </div>
        </div>
      ))}

      {/* Add Step Button */}
      <Button variant="outline" onClick={handleAddStep} disabled={disabled} className="w-full">
        <PlusIcon className="h-4 w-4 mr-2" />
        Add Step
      </Button>
    </div>
  );
}

interface StepSourceHintsEditorProps {
  sourceHints: StepSourceHint[];
  availableSources: AvailableSource[];
  onChange: (hints: StepSourceHint[]) => void;
  disabled?: boolean;
}

function StepSourceHintsEditor({
  sourceHints,
  availableSources,
  onChange,
  disabled,
}: StepSourceHintsEditorProps) {
  const [showPicker, setShowPicker] = React.useState(false);

  const selectedSourceNames = new Set(sourceHints.map((h) => h.sourceName));

  const handleAddSource = (source: AvailableSource) => {
    if (selectedSourceNames.has(source.name)) return;
    const newHint: StepSourceHint = {
      sourceName: source.name,
      sourceType: source.type,
      priority: 2,
    };
    onChange([...sourceHints, newHint]);
    setShowPicker(false);
  };

  const handleRemoveHint = (sourceName: string) => {
    onChange(sourceHints.filter((h) => h.sourceName !== sourceName));
  };

  const handlePriorityChange = (sourceName: string, priority: 1 | 2 | 3) => {
    onChange(
      sourceHints.map((h) => (h.sourceName === sourceName ? { ...h, priority } : h))
    );
  };

  const handleQueryHintChange = (sourceName: string, queryHint: string) => {
    onChange(
      sourceHints.map((h) =>
        h.sourceName === sourceName ? { ...h, queryHint: queryHint || undefined } : h
      )
    );
  };

  const unselectedSources = availableSources.filter(
    (s) => !selectedSourceNames.has(s.name)
  );

  return (
    <div className="space-y-2">
      {/* Selected source hints */}
      {sourceHints.map((hint) => (
        <div
          key={hint.sourceName}
          className="flex items-start gap-2 p-2 rounded-md border bg-muted/30"
        >
          <SourceTypeIcon type={hint.sourceType} className="h-4 w-4 mt-1" />
          <div className="flex-1 min-w-0 space-y-2">
            <div className="flex items-center justify-between gap-2">
              <span className="text-sm font-medium truncate">{hint.sourceName}</span>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => handleRemoveHint(hint.sourceName)}
                disabled={disabled}
                className="h-6 w-6 text-muted-foreground hover:text-destructive"
              >
                <XIcon className="h-3 w-3" />
              </Button>
            </div>

            {/* Priority selector */}
            <div className="flex items-center gap-2">
              <span className="text-xs text-muted-foreground">Priority:</span>
              <div className="flex gap-1">
                {([1, 2, 3] as const).map((p) => (
                  <button
                    key={p}
                    type="button"
                    onClick={() => handlePriorityChange(hint.sourceName, p)}
                    disabled={disabled}
                    className={cn(
                      'w-6 h-6 rounded text-xs font-medium transition-colors',
                      hint.priority === p
                        ? 'bg-primary text-primary-foreground'
                        : 'bg-muted hover:bg-muted/80 text-muted-foreground'
                    )}
                  >
                    {p}
                  </button>
                ))}
              </div>
              <span className="text-xs text-muted-foreground">
                {hint.priority === 1 ? '(High)' : hint.priority === 2 ? '(Medium)' : '(Low)'}
              </span>
            </div>

            {/* Query hint input */}
            <Input
              value={hint.queryHint || ''}
              onChange={(e) => handleQueryHintChange(hint.sourceName, e.target.value)}
              placeholder="Query hint for this source..."
              disabled={disabled}
              className="text-xs h-7"
            />
          </div>
        </div>
      ))}

      {/* Add source button / picker */}
      {unselectedSources.length > 0 && (
        <div className="relative">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowPicker(!showPicker)}
            disabled={disabled}
            className="w-full"
          >
            <PlusIcon className="h-3 w-3 mr-1" />
            Add Source
          </Button>

          {showPicker && (
            <div className="absolute z-10 mt-1 w-full rounded-md border bg-popover p-2 shadow-lg max-h-48 overflow-y-auto">
              {unselectedSources.map((source) => (
                <button
                  key={source.id}
                  type="button"
                  onClick={() => handleAddSource(source)}
                  className="w-full flex items-center gap-2 p-2 rounded hover:bg-muted text-left"
                >
                  <SourceTypeIcon type={source.type} className="h-4 w-4" />
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-medium truncate">{source.name}</div>
                    {source.description && (
                      <div className="text-xs text-muted-foreground truncate">
                        {source.description}
                      </div>
                    )}
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>
      )}

      {sourceHints.length === 0 && unselectedSources.length === 0 && (
        <p className="text-xs text-muted-foreground text-center py-2">
          No sources available
        </p>
      )}
    </div>
  );
}

function SourceTypeIcon({
  type,
  className,
}: {
  type: DataSourceType;
  className?: string;
}) {
  switch (type) {
    case 'vector_search':
      return <SearchIcon className={cn('text-blue-600', className)} />;
    case 'genie':
      return <DatabaseIcon className={cn('text-purple-600', className)} />;
    case 'knowledge_assistant':
      return <BrainIcon className={cn('text-emerald-600', className)} />;
    case 'web_search':
      return <GlobeIcon className={cn('text-orange-600', className)} />;
    default:
      return <CubeIcon className={cn('text-slate-600', className)} />;
  }
}

// Icons
function GripIcon({ className }: { className?: string }) {
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
      <circle cx="9" cy="12" r="1" />
      <circle cx="9" cy="5" r="1" />
      <circle cx="9" cy="19" r="1" />
      <circle cx="15" cy="12" r="1" />
      <circle cx="15" cy="5" r="1" />
      <circle cx="15" cy="19" r="1" />
    </svg>
  );
}

function ChevronUpIcon({ className }: { className?: string }) {
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
      <path d="m18 15-6-6-6 6" />
    </svg>
  );
}

function ChevronDownIcon({ className }: { className?: string }) {
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

function TrashIcon({ className }: { className?: string }) {
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
      <path d="M3 6h18" />
      <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6" />
      <path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2" />
    </svg>
  );
}

function PlusIcon({ className }: { className?: string }) {
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
      <path d="M12 5v14M5 12h14" />
    </svg>
  );
}

function XIcon({ className }: { className?: string }) {
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
      <path d="M18 6 6 18M6 6l12 12" />
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

function DatabaseIcon({ className }: { className?: string }) {
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
      <ellipse cx="12" cy="5" rx="9" ry="3" />
      <path d="M3 5v14a9 3 0 0 0 18 0V5" />
      <path d="M3 12a9 3 0 0 0 18 0" />
    </svg>
  );
}

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
      <path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 1.98-3A2.5 2.5 0 0 1 9.5 2Z" />
      <path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-1.98-3A2.5 2.5 0 0 0 14.5 2Z" />
    </svg>
  );
}

function GlobeIcon({ className }: { className?: string }) {
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
      <path d="M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20" />
      <path d="M2 12h20" />
    </svg>
  );
}

function CubeIcon({ className }: { className?: string }) {
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
      <path d="m21 16-9 5-9-5V8l9-5 9 5v8z" />
      <path d="m3.27 6.96 8.73 4.84 8.73-4.84" />
      <line x1="12" x2="12" y1="22" y2="11.8" />
    </svg>
  );
}

export default PlanEditor;
