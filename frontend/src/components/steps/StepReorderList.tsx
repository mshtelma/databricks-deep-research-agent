/**
 * StepReorderList - Drag-and-drop reorderable list of research steps.
 *
 * Features:
 * - Drag-and-drop reorderable list of steps
 * - Add new step button
 * - Edit step button (opens ManualStepEditor)
 * - Remove step button with confirmation
 * - Step number indicators
 */

import * as React from 'react'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import type { ManualStep } from './ManualStepEditor'

interface StepReorderListProps {
  /** List of steps to display */
  steps: ManualStep[]
  /** Callback when steps are reordered or modified */
  onChange: (steps: ManualStep[]) => void
  /** Callback when a step is selected for editing */
  onEdit: (step: ManualStep) => void
  /** Whether the list is in read-only mode */
  readOnly?: boolean
  /** Additional CSS classes */
  className?: string
}

export function StepReorderList({
  steps,
  onChange,
  onEdit,
  readOnly = false,
  className,
}: StepReorderListProps) {
  const [draggedIndex, setDraggedIndex] = React.useState<number | null>(null)
  const [dragOverIndex, setDragOverIndex] = React.useState<number | null>(null)
  const [deleteConfirmId, setDeleteConfirmId] = React.useState<string | null>(null)

  const handleDragStart = (index: number) => {
    if (readOnly) return
    setDraggedIndex(index)
  }

  const handleDragOver = (e: React.DragEvent, index: number) => {
    e.preventDefault()
    if (readOnly) return
    setDragOverIndex(index)
  }

  const handleDragEnd = () => {
    if (draggedIndex === null || dragOverIndex === null || draggedIndex === dragOverIndex) {
      setDraggedIndex(null)
      setDragOverIndex(null)
      return
    }

    // Reorder steps
    const newSteps = [...steps]
    const [removed] = newSteps.splice(draggedIndex, 1)
    if (!removed) return
    newSteps.splice(dragOverIndex, 0, removed)

    // Update order property
    const reorderedSteps = newSteps.map((step, idx) => ({
      ...step,
      order: idx,
    }))

    onChange(reorderedSteps)
    setDraggedIndex(null)
    setDragOverIndex(null)
  }

  const handleAddStep = () => {
    const newStep: ManualStep = {
      id: `step-${Date.now()}`,
      title: '',
      objective: '',
      sources: [],
      order: steps.length,
    }
    onChange([...steps, newStep])
    onEdit(newStep)
  }

  const handleRemoveStep = (stepId: string) => {
    const newSteps = steps
      .filter((s) => s.id !== stepId)
      .map((step, idx) => ({ ...step, order: idx }))
    onChange(newSteps)
    setDeleteConfirmId(null)
  }

  const handleMoveUp = (index: number) => {
    if (index === 0) return
    const newSteps = [...steps]
    const current = newSteps[index]
    const previous = newSteps[index - 1]
    if (!current || !previous) return
    newSteps[index] = previous
    newSteps[index - 1] = current
    const reorderedSteps = newSteps.map((step, idx) => ({
      ...step,
      order: idx,
    }))
    onChange(reorderedSteps)
  }

  const handleMoveDown = (index: number) => {
    if (index === steps.length - 1) return
    const newSteps = [...steps]
    const current = newSteps[index]
    const next = newSteps[index + 1]
    if (!current || !next) return
    newSteps[index] = next
    newSteps[index + 1] = current
    const reorderedSteps = newSteps.map((step, idx) => ({
      ...step,
      order: idx,
    }))
    onChange(reorderedSteps)
  }

  return (
    <div className={cn('space-y-2', className)}>
      {steps.length === 0 ? (
        <div className="text-sm text-muted-foreground p-6 border rounded-lg border-dashed text-center">
          No steps defined. Add your first research step below.
        </div>
      ) : (
        <div className="space-y-2">
          {steps.map((step, index) => (
            <div
              key={step.id}
              draggable={!readOnly}
              onDragStart={() => handleDragStart(index)}
              onDragOver={(e) => handleDragOver(e, index)}
              onDragEnd={handleDragEnd}
              className={cn(
                'group flex items-start gap-3 p-3 border rounded-lg transition-all',
                !readOnly && 'cursor-grab active:cursor-grabbing',
                draggedIndex === index && 'opacity-50 ring-2 ring-primary',
                dragOverIndex === index &&
                  draggedIndex !== index &&
                  'ring-2 ring-primary ring-dashed'
              )}
            >
              {/* Step number */}
              <div
                className={cn(
                  'flex items-center justify-center h-7 w-7 rounded-full shrink-0',
                  'bg-primary/10 text-primary text-sm font-medium'
                )}
              >
                {index + 1}
              </div>

              {/* Step content */}
              <div className="flex-1 min-w-0">
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <h4 className={cn('font-medium text-sm', !step.title && 'text-muted-foreground italic')}>
                      {step.title || 'Untitled step'}
                    </h4>
                    {step.objective && (
                      <p className="text-xs text-muted-foreground mt-0.5 line-clamp-2">
                        {step.objective}
                      </p>
                    )}
                    {step.sources.length > 0 && (
                      <div className="flex items-center gap-1 mt-1.5">
                        <SourceIcon className="h-3 w-3 text-muted-foreground" />
                        <span className="text-xs text-muted-foreground">
                          {step.sources.length} source{step.sources.length > 1 ? 's' : ''}
                        </span>
                      </div>
                    )}
                  </div>

                  {/* Actions */}
                  {!readOnly && (
                    <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                      {/* Move buttons */}
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon"
                        onClick={() => handleMoveUp(index)}
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
                        onClick={() => handleMoveDown(index)}
                        disabled={index === steps.length - 1}
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
                        onClick={() => onEdit(step)}
                        className="h-7 w-7"
                        title="Edit step"
                      >
                        <EditIcon className="h-3.5 w-3.5" />
                      </Button>

                      {/* Delete */}
                      {deleteConfirmId === step.id ? (
                        <div className="flex items-center gap-1">
                          <Button
                            type="button"
                            variant="destructive"
                            size="sm"
                            onClick={() => handleRemoveStep(step.id)}
                            className="h-7 text-xs"
                          >
                            Confirm
                          </Button>
                          <Button
                            type="button"
                            variant="ghost"
                            size="sm"
                            onClick={() => setDeleteConfirmId(null)}
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
                          onClick={() => setDeleteConfirmId(step.id)}
                          className="h-7 w-7 text-muted-foreground hover:text-destructive"
                          title="Remove step"
                        >
                          <TrashIcon className="h-3.5 w-3.5" />
                        </Button>
                      )}
                    </div>
                  )}
                </div>
              </div>

              {/* Drag handle indicator */}
              {!readOnly && (
                <div className="opacity-30 group-hover:opacity-100 transition-opacity shrink-0">
                  <GripIcon className="h-5 w-5 text-muted-foreground" />
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Add step button */}
      {!readOnly && (
        <Button
          type="button"
          variant="outline"
          onClick={handleAddStep}
          className="w-full"
        >
          <PlusIcon className="h-4 w-4 mr-2" />
          Add Step
        </Button>
      )}
    </div>
  )
}

// Icons
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
  )
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
  )
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
  )
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
  )
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
  )
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
  )
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
  )
}

export default StepReorderList
