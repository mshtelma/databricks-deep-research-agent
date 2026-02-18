/**
 * AgentPresetSteps - Manage preset steps for a custom agent.
 *
 * Features:
 * - Drag-and-drop step reordering
 * - Add new step button
 * - Per-step: title, description, source hints, required toggle
 * - Delete step with confirmation
 */

import * as React from 'react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import type { PresetStep, PresetStepSourceHint } from '@/types/customAgents';
import type { DataSourceType } from '@/types/discovery';

/** Minimal source shape compatible with SelectableSource from AgentBuilder. */
type SourceItem = { id: string; name: string; type: string; description: string | null };

interface AgentPresetStepsProps {
  /** List of preset steps */
  steps: PresetStep[];
  /** Callback when steps are modified */
  onChange: (steps: PresetStep[]) => void;
  /** Available sources for selection */
  availableSources: SourceItem[];
  /** Whether the editor is read-only */
  readOnly?: boolean;
  /** Additional CSS classes */
  className?: string;
}

export function AgentPresetSteps({
  steps,
  onChange,
  availableSources,
  readOnly = false,
  className,
}: AgentPresetStepsProps) {
  const [draggedIndex, setDraggedIndex] = React.useState<number | null>(null);
  const [dragOverIndex, setDragOverIndex] = React.useState<number | null>(null);
  const [deleteConfirmId, setDeleteConfirmId] = React.useState<string | null>(null);
  const [editingStepId, setEditingStepId] = React.useState<string | null>(null);

  const handleDragStart = (index: number) => {
    if (readOnly) return;
    setDraggedIndex(index);
  };

  const handleDragOver = (e: React.DragEvent, index: number) => {
    e.preventDefault();
    if (readOnly) return;
    setDragOverIndex(index);
  };

  const handleDragEnd = () => {
    if (draggedIndex === null || dragOverIndex === null || draggedIndex === dragOverIndex) {
      setDraggedIndex(null);
      setDragOverIndex(null);
      return;
    }

    const newSteps = [...steps];
    const [removed] = newSteps.splice(draggedIndex, 1);
    if (!removed) return;
    newSteps.splice(dragOverIndex, 0, removed);

    // Update order property
    const reorderedSteps = newSteps.map((step, idx) => ({
      ...step,
      order: idx,
    }));

    onChange(reorderedSteps);
    setDraggedIndex(null);
    setDragOverIndex(null);
  };

  const handleAddStep = () => {
    const newStep: PresetStep = {
      id: `temp-${Date.now()}`,
      agentId: steps[0]?.agentId || '',
      title: '',
      description: null,
      order: steps.length,
      isRequired: false,
      sourceScope: null,
      sourceHints: [],
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    };
    onChange([...steps, newStep]);
    setEditingStepId(newStep.id);
  };

  const handleRemoveStep = (stepId: string) => {
    const newSteps = steps
      .filter((s) => s.id !== stepId)
      .map((step, idx) => ({ ...step, order: idx }));
    onChange(newSteps);
    setDeleteConfirmId(null);
  };

  const handleUpdateStep = (stepId: string, updates: Partial<PresetStep>) => {
    onChange(
      steps.map((step) =>
        step.id === stepId ? { ...step, ...updates, updatedAt: new Date().toISOString() } : step
      )
    );
  };

  const handleMoveUp = (index: number) => {
    if (index === 0) return;
    const newSteps = [...steps];
    const current = newSteps[index];
    const previous = newSteps[index - 1];
    if (!current || !previous) return;
    newSteps[index] = previous;
    newSteps[index - 1] = current;
    const reorderedSteps = newSteps.map((step, idx) => ({ ...step, order: idx }));
    onChange(reorderedSteps);
  };

  const handleMoveDown = (index: number) => {
    if (index === steps.length - 1) return;
    const newSteps = [...steps];
    const current = newSteps[index];
    const next = newSteps[index + 1];
    if (!current || !next) return;
    newSteps[index] = next;
    newSteps[index + 1] = current;
    const reorderedSteps = newSteps.map((step, idx) => ({ ...step, order: idx }));
    onChange(reorderedSteps);
  };

  return (
    <div className={cn('space-y-3', className)}>
      {steps.length === 0 ? (
        <div className="text-sm text-muted-foreground p-6 border rounded-lg border-dashed text-center">
          No preset steps defined. Add steps to create a fixed research workflow.
        </div>
      ) : (
        <div className="space-y-2">
          {steps.map((step, index) => (
            <PresetStepCard
              key={step.id}
              step={step}
              index={index}
              isEditing={editingStepId === step.id}
              isDragging={draggedIndex === index}
              isDragOver={dragOverIndex === index && draggedIndex !== index}
              isDeleteConfirm={deleteConfirmId === step.id}
              readOnly={readOnly}
              availableSources={availableSources}
              totalSteps={steps.length}
              onDragStart={() => handleDragStart(index)}
              onDragOver={(e) => handleDragOver(e, index)}
              onDragEnd={handleDragEnd}
              onEdit={() => setEditingStepId(editingStepId === step.id ? null : step.id)}
              onUpdate={(updates) => handleUpdateStep(step.id, updates)}
              onMoveUp={() => handleMoveUp(index)}
              onMoveDown={() => handleMoveDown(index)}
              onDeleteClick={() => setDeleteConfirmId(step.id)}
              onDeleteConfirm={() => handleRemoveStep(step.id)}
              onDeleteCancel={() => setDeleteConfirmId(null)}
            />
          ))}
        </div>
      )}

      {/* Add step button */}
      {!readOnly && (
        <Button type="button" variant="outline" onClick={handleAddStep} className="w-full">
          <PlusIcon className="h-4 w-4 mr-2" />
          Add Step
        </Button>
      )}
    </div>
  );
}

// =============================================================================
// Preset Step Card
// =============================================================================

interface PresetStepCardProps {
  step: PresetStep;
  index: number;
  isEditing: boolean;
  isDragging: boolean;
  isDragOver: boolean;
  isDeleteConfirm: boolean;
  readOnly: boolean;
  availableSources: SourceItem[];
  totalSteps: number;
  onDragStart: () => void;
  onDragOver: (e: React.DragEvent) => void;
  onDragEnd: () => void;
  onEdit: () => void;
  onUpdate: (updates: Partial<PresetStep>) => void;
  onMoveUp: () => void;
  onMoveDown: () => void;
  onDeleteClick: () => void;
  onDeleteConfirm: () => void;
  onDeleteCancel: () => void;
}

function PresetStepCard({
  step,
  index,
  isEditing,
  isDragging,
  isDragOver,
  isDeleteConfirm,
  readOnly,
  availableSources,
  totalSteps,
  onDragStart,
  onDragOver,
  onDragEnd,
  onEdit,
  onUpdate,
  onMoveUp,
  onMoveDown,
  onDeleteClick,
  onDeleteConfirm,
  onDeleteCancel,
}: PresetStepCardProps) {
  const [showSourcePicker, setShowSourcePicker] = React.useState(false);

  const handleAddSourceHint = (source: SourceItem) => {
    const newHint: PresetStepSourceHint = {
      sourceId: source.id,
      sourceName: source.name,
      sourceType: source.type as DataSourceType,
      priority: 2,
      queryHint: null,
    };
    onUpdate({ sourceHints: [...step.sourceHints, newHint] });
    setShowSourcePicker(false);
  };

  const handleRemoveSourceHint = (sourceId: string) => {
    onUpdate({
      sourceHints: step.sourceHints.filter((h) => h.sourceId !== sourceId),
    });
  };

  const handleUpdateSourceHint = (sourceId: string, updates: Partial<PresetStepSourceHint>) => {
    onUpdate({
      sourceHints: step.sourceHints.map((h) =>
        h.sourceId === sourceId ? { ...h, ...updates } : h
      ),
    });
  };

  const selectedSourceIds = step.sourceHints.map((h) => h.sourceId);
  const unselectedSources = availableSources.filter((s) => !selectedSourceIds.includes(s.id));

  return (
    <div
      draggable={!readOnly && !isEditing}
      onDragStart={onDragStart}
      onDragOver={onDragOver}
      onDragEnd={onDragEnd}
      className={cn(
        'group border rounded-lg transition-all',
        !readOnly && !isEditing && 'cursor-grab active:cursor-grabbing',
        isDragging && 'opacity-50 ring-2 ring-primary',
        isDragOver && 'ring-2 ring-primary ring-dashed'
      )}
    >
      {/* Header */}
      <div className="flex items-start gap-3 p-3">
        {/* Step number */}
        <div
          className={cn(
            'flex items-center justify-center h-7 w-7 rounded-full shrink-0',
            'bg-primary/10 text-primary text-sm font-medium'
          )}
        >
          {index + 1}
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          {isEditing ? (
            <div className="space-y-3">
              {/* Title input */}
              <div>
                <label className="text-xs text-muted-foreground mb-1 block">Step Title *</label>
                <Input
                  value={step.title}
                  onChange={(e) => onUpdate({ title: e.target.value })}
                  placeholder="e.g., Research market trends"
                />
              </div>

              {/* Description */}
              <div>
                <label className="text-xs text-muted-foreground mb-1 block">Description</label>
                <textarea
                  value={step.description || ''}
                  onChange={(e) => onUpdate({ description: e.target.value || null })}
                  placeholder="Describe what this step should accomplish..."
                  rows={2}
                  className={cn(
                    'w-full resize-none rounded-md border border-input bg-transparent px-3 py-2 text-sm',
                    'placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring'
                  )}
                />
              </div>

              {/* Required toggle */}
              <label className="flex items-center gap-2 text-sm cursor-pointer">
                <input
                  type="checkbox"
                  checked={step.isRequired}
                  onChange={(e) => onUpdate({ isRequired: e.target.checked })}
                  className="rounded border-input"
                />
                Required step (cannot be skipped)
              </label>

              {/* Per-step source scope override (T042) */}
              <div>
                <label className="text-xs text-muted-foreground mb-1 block">Source Scope</label>
                <select
                  value={step.sourceScope || ''}
                  onChange={(e) => onUpdate({ sourceScope: e.target.value || null })}
                  className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                >
                  <option value="">Use Agent Default</option>
                  <option value="all">All Sources</option>
                  <option value="enterprise_only">Enterprise Only</option>
                  <option value="web_only">Web Only</option>
                </select>
                <p className="text-xs text-muted-foreground mt-1">
                  Override the agent's source scope for this step only
                </p>
              </div>

              {/* Source hints */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <label className="text-xs text-muted-foreground">Source Hints</label>
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={() => setShowSourcePicker(true)}
                    disabled={unselectedSources.length === 0}
                    className="h-7 text-xs"
                  >
                    <PlusIcon className="h-3 w-3 mr-1" />
                    Add Source
                  </Button>
                </div>

                {step.sourceHints.length === 0 ? (
                  <p className="text-xs text-muted-foreground">
                    No source hints. The agent will use default source selection.
                  </p>
                ) : (
                  <div className="space-y-2">
                    {step.sourceHints.map((hint) => (
                      <SourceHintCard
                        key={hint.sourceId}
                        hint={hint}
                        onUpdate={(updates) => handleUpdateSourceHint(hint.sourceId, updates)}
                        onRemove={() => handleRemoveSourceHint(hint.sourceId)}
                      />
                    ))}
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div>
              <div className="flex items-center gap-2">
                <h4 className={cn('font-medium text-sm', !step.title && 'text-muted-foreground italic')}>
                  {step.title || 'Untitled step'}
                </h4>
                {step.isRequired && (
                  <span className="px-1.5 py-0.5 rounded text-xs bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-200">
                    Required
                  </span>
                )}
                {step.sourceScope && (
                  <span className="px-1.5 py-0.5 rounded text-xs bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200">
                    {step.sourceScope === 'enterprise_only' ? 'Enterprise' : step.sourceScope === 'web_only' ? 'Web' : 'All'}
                  </span>
                )}
              </div>
              {step.description && (
                <p className="text-xs text-muted-foreground mt-0.5 line-clamp-2">
                  {step.description}
                </p>
              )}
              {step.sourceHints.length > 0 && (
                <div className="flex items-center gap-1 mt-1.5">
                  <SourceIcon className="h-3 w-3 text-muted-foreground" />
                  <span className="text-xs text-muted-foreground">
                    {step.sourceHints.length} source hint{step.sourceHints.length > 1 ? 's' : ''}
                  </span>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Actions */}
        {!readOnly && (
          <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity shrink-0">
            {/* Move buttons */}
            <Button
              type="button"
              variant="ghost"
              size="icon"
              onClick={onMoveUp}
              disabled={index === 0}
              className="h-7 w-7"
              title="Move up"
            >
              <ChevronUpIcon className="h-4 w-4" />
            </Button>
            <Button
              type="button"
              variant="ghost"
              size="icon"
              onClick={onMoveDown}
              disabled={index === totalSteps - 1}
              className="h-7 w-7"
              title="Move down"
            >
              <ChevronDownIcon className="h-4 w-4" />
            </Button>

            {/* Edit */}
            <Button
              type="button"
              variant="ghost"
              size="icon"
              onClick={onEdit}
              className={cn('h-7 w-7', isEditing && 'bg-muted')}
              title={isEditing ? 'Close editor' : 'Edit step'}
            >
              {isEditing ? <CheckIcon className="h-3.5 w-3.5" /> : <EditIcon className="h-3.5 w-3.5" />}
            </Button>

            {/* Delete */}
            {isDeleteConfirm ? (
              <div className="flex items-center gap-1">
                <Button
                  type="button"
                  variant="destructive"
                  size="sm"
                  onClick={onDeleteConfirm}
                  className="h-7 text-xs"
                >
                  Confirm
                </Button>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={onDeleteCancel}
                  className="h-7 text-xs"
                >
                  Cancel
                </Button>
              </div>
            ) : (
              <Button
                type="button"
                variant="ghost"
                size="icon"
                onClick={onDeleteClick}
                className="h-7 w-7 text-muted-foreground hover:text-destructive"
                title="Remove step"
              >
                <TrashIcon className="h-3.5 w-3.5" />
              </Button>
            )}
          </div>
        )}

        {/* Drag handle */}
        {!readOnly && !isEditing && (
          <div className="opacity-30 group-hover:opacity-100 transition-opacity shrink-0">
            <GripIcon className="h-5 w-5 text-muted-foreground" />
          </div>
        )}
      </div>

      {/* Source picker modal */}
      {showSourcePicker && (
        <SourcePickerModal
          sources={unselectedSources}
          onSelect={handleAddSourceHint}
          onClose={() => setShowSourcePicker(false)}
        />
      )}
    </div>
  );
}

// =============================================================================
// Source Hint Card
// =============================================================================

interface SourceHintCardProps {
  hint: PresetStepSourceHint;
  onUpdate: (updates: Partial<PresetStepSourceHint>) => void;
  onRemove: () => void;
}

function SourceHintCard({ hint, onUpdate, onRemove }: SourceHintCardProps) {
  return (
    <div className="flex items-start gap-2 p-2 bg-muted/50 rounded-md">
      <SourceTypeIcon type={hint.sourceType} className="h-4 w-4 text-muted-foreground mt-0.5" />
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium truncate">{hint.sourceName}</p>
        <Input
          value={hint.queryHint || ''}
          onChange={(e) => onUpdate({ queryHint: e.target.value || null })}
          placeholder="Query hint (optional)..."
          className="mt-1 text-xs h-7"
        />
      </div>
      <div className="flex items-center gap-1">
        <select
          value={hint.priority}
          onChange={(e) => onUpdate({ priority: parseInt(e.target.value, 10) as 1 | 2 | 3 })}
          className="rounded border border-input bg-background px-2 py-1 text-xs"
          title="Priority"
        >
          <option value={1}>Required</option>
          <option value={2}>Recommended</option>
          <option value={3}>Optional</option>
        </select>
        <Button
          type="button"
          variant="ghost"
          size="icon"
          onClick={onRemove}
          className="h-6 w-6 text-muted-foreground hover:text-destructive"
        >
          <XIcon className="h-3 w-3" />
        </Button>
      </div>
    </div>
  );
}

// =============================================================================
// Source Picker Modal
// =============================================================================

interface SourcePickerModalProps {
  sources: SourceItem[];
  onSelect: (source: SourceItem) => void;
  onClose: () => void;
}

function SourcePickerModal({ sources, onSelect, onClose }: SourcePickerModalProps) {
  const [search, setSearch] = React.useState('');

  const filteredSources = sources.filter(
    (s) =>
      s.name.toLowerCase().includes(search.toLowerCase()) ||
      (s.description && s.description.toLowerCase().includes(search.toLowerCase()))
  );

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="fixed inset-0 bg-black/50" onClick={onClose} />
      <div className="relative z-50 w-full max-w-md bg-background rounded-lg shadow-lg p-4">
        <h4 className="font-medium mb-3">Select Source</h4>
        <Input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search sources..."
          className="mb-3"
        />
        <div className="max-h-64 overflow-y-auto space-y-2">
          {filteredSources.length === 0 ? (
            <div className="text-sm text-muted-foreground text-center py-4">
              No sources available
            </div>
          ) : (
            filteredSources.map((source) => (
              <button
                key={source.id}
                type="button"
                onClick={() => onSelect(source)}
                className="w-full flex items-center gap-3 p-2 rounded-md hover:bg-muted text-left"
              >
                <SourceTypeIcon type={source.type} className="h-4 w-4 text-muted-foreground" />
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium truncate">{source.name}</p>
                  {source.description && (
                    <p className="text-xs text-muted-foreground truncate">{source.description}</p>
                  )}
                </div>
              </button>
            ))
          )}
        </div>
        <div className="flex justify-end mt-4">
          <Button variant="outline" onClick={onClose}>
            Cancel
          </Button>
        </div>
      </div>
    </div>
  );
}

// =============================================================================
// Icons
// =============================================================================

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
      <path d="M17 3a2.85 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z" />
      <path d="m15 5 4 4" />
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
      <circle cx="9" cy="5" r="1" />
      <circle cx="9" cy="12" r="1" />
      <circle cx="9" cy="19" r="1" />
      <circle cx="15" cy="5" r="1" />
      <circle cx="15" cy="12" r="1" />
      <circle cx="15" cy="19" r="1" />
    </svg>
  );
}

function SourceIcon({ className }: { className?: string }) {
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
      <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" />
      <path d="M3 12c0 1.66 4 3 9 3s9-1.34 9-3" />
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

function SourceTypeIcon({ type, className }: { type: string; className?: string }) {
  switch (type) {
    case 'web_search':
      return (
        <svg
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          className={className}
        >
          <circle cx="12" cy="12" r="10" />
          <path d="M2 12h20" />
          <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
        </svg>
      );
    case 'vector_search':
      return (
        <svg
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          className={className}
        >
          <ellipse cx="12" cy="5" rx="9" ry="3" />
          <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" />
          <path d="M3 12c0 1.66 4 3 9 3s9-1.34 9-3" />
        </svg>
      );
    case 'genie':
      return (
        <svg
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          className={className}
        >
          <path d="M3 3v18h18" />
          <path d="m19 9-5 5-4-4-3 3" />
        </svg>
      );
    case 'knowledge_assistant':
      return (
        <svg
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          className={className}
        >
          <path d="M12 8V4H8" />
          <rect width="16" height="12" x="4" y="8" rx="2" />
          <path d="M2 14h2M20 14h2M15 13v2M9 13v2" />
        </svg>
      );
    default:
      return (
        <svg
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          className={className}
        >
          <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z" />
          <polyline points="14 2 14 8 20 8" />
        </svg>
      );
  }
}

export default AgentPresetSteps;
