/**
 * ManualStepEditor - Edit a single manual research step.
 *
 * Features:
 * - Step title input
 * - Objective/description textarea
 * - Source selection area (integrate with SourceBrowser or multi-select)
 * - Per-source custom prompt inputs
 * - Optional filter configuration for Vector Search sources (key-value pairs)
 */

import * as React from 'react'
import { cn } from '@/lib/utils'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import type { DataSource } from '@/types/dataSources'

// Re-export StepSourcePicker for convenience
export { StepSourcePicker } from './StepSourcePicker'

/**
 * Attachment of a source to a step with optional custom prompt.
 */
export interface StepSourceAttachment {
  sourceId: string
  sourceName: string
  sourceType: DataSource['type']
  customPrompt?: string
  priority: 1 | 2 | 3
  filters?: Record<string, string>
}

/**
 * Manual step definition for user-defined research workflows.
 */
export interface ManualStep {
  id: string
  title: string
  objective: string
  sources: StepSourceAttachment[]
  order: number
}

interface ManualStepEditorProps {
  /** The step being edited */
  step: ManualStep
  /** Callback when step is updated */
  onChange: (step: ManualStep) => void
  /** Available sources for selection */
  availableSources: DataSource[]
  /** Whether the editor is in read-only mode */
  readOnly?: boolean
  /** Additional CSS classes */
  className?: string
}

export function ManualStepEditor({
  step,
  onChange,
  availableSources,
  readOnly = false,
  className,
}: ManualStepEditorProps) {
  const [showSourcePicker, setShowSourcePicker] = React.useState(false)

  const handleTitleChange = (title: string) => {
    onChange({ ...step, title })
  }

  const handleObjectiveChange = (objective: string) => {
    onChange({ ...step, objective })
  }

  const handleAddSource = (source: DataSource) => {
    const newAttachment: StepSourceAttachment = {
      sourceId: source.id,
      sourceName: source.name,
      sourceType: source.type,
      priority: 2,
    }
    onChange({
      ...step,
      sources: [...step.sources, newAttachment],
    })
    setShowSourcePicker(false)
  }

  const handleRemoveSource = (sourceId: string) => {
    onChange({
      ...step,
      sources: step.sources.filter((s) => s.sourceId !== sourceId),
    })
  }

  const handleUpdateSourcePrompt = (sourceId: string, customPrompt: string) => {
    onChange({
      ...step,
      sources: step.sources.map((s) =>
        s.sourceId === sourceId ? { ...s, customPrompt } : s
      ),
    })
  }

  const handleUpdateSourcePriority = (sourceId: string, priority: 1 | 2 | 3) => {
    onChange({
      ...step,
      sources: step.sources.map((s) =>
        s.sourceId === sourceId ? { ...s, priority } : s
      ),
    })
  }

  const handleUpdateSourceFilters = (sourceId: string, filters: Record<string, string>) => {
    onChange({
      ...step,
      sources: step.sources.map((s) =>
        s.sourceId === sourceId ? { ...s, filters } : s
      ),
    })
  }

  const selectedSourceIds = step.sources.map((s) => s.sourceId)
  const unselectedSources = availableSources.filter(
    (s) => !selectedSourceIds.includes(s.id)
  )

  return (
    <div className={cn('space-y-4', className)}>
      {/* Step Title */}
      <div>
        <label className="text-sm font-medium mb-1.5 block">Step Title *</label>
        <Input
          value={step.title}
          onChange={(e) => handleTitleChange(e.target.value)}
          placeholder="e.g., Research market trends"
          disabled={readOnly}
        />
      </div>

      {/* Step Objective */}
      <div>
        <label className="text-sm font-medium mb-1.5 block">Objective</label>
        <textarea
          value={step.objective}
          onChange={(e) => handleObjectiveChange(e.target.value)}
          placeholder="Describe what this step should accomplish..."
          rows={3}
          disabled={readOnly}
          className={cn(
            'w-full resize-none rounded-md border border-input bg-transparent px-3 py-2 text-sm',
            'placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring',
            'disabled:cursor-not-allowed disabled:opacity-50'
          )}
        />
      </div>

      {/* Sources Section */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <label className="text-sm font-medium">Data Sources</label>
          {!readOnly && (
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => setShowSourcePicker(true)}
              disabled={unselectedSources.length === 0}
            >
              <PlusIcon className="h-3.5 w-3.5 mr-1" />
              Add Source
            </Button>
          )}
        </div>

        {step.sources.length === 0 ? (
          <div className="text-sm text-muted-foreground p-4 border rounded-md border-dashed text-center">
            No sources selected. Add sources to define where to search.
          </div>
        ) : (
          <div className="space-y-3">
            {step.sources.map((attachment) => (
              <SourceAttachmentCard
                key={attachment.sourceId}
                attachment={attachment}
                onRemove={() => handleRemoveSource(attachment.sourceId)}
                onUpdatePrompt={(prompt) =>
                  handleUpdateSourcePrompt(attachment.sourceId, prompt)
                }
                onUpdatePriority={(priority) =>
                  handleUpdateSourcePriority(attachment.sourceId, priority)
                }
                onUpdateFilters={(filters) =>
                  handleUpdateSourceFilters(attachment.sourceId, filters)
                }
                readOnly={readOnly}
              />
            ))}
          </div>
        )}
      </div>

      {/* Source Picker Modal */}
      {showSourcePicker && (
        <SourcePickerModal
          sources={unselectedSources}
          onSelect={handleAddSource}
          onClose={() => setShowSourcePicker(false)}
        />
      )}
    </div>
  )
}

/**
 * Card displaying a source attachment with custom prompt input.
 */
function SourceAttachmentCard({
  attachment,
  onRemove,
  onUpdatePrompt,
  onUpdatePriority,
  onUpdateFilters,
  readOnly,
}: {
  attachment: StepSourceAttachment
  onRemove: () => void
  onUpdatePrompt: (prompt: string) => void
  onUpdatePriority: (priority: 1 | 2 | 3) => void
  onUpdateFilters: (filters: Record<string, string>) => void
  readOnly: boolean
}) {
  const [showFilters, setShowFilters] = React.useState(false)
  const [newFilterKey, setNewFilterKey] = React.useState('')
  const [newFilterValue, setNewFilterValue] = React.useState('')

  const handleAddFilter = () => {
    if (newFilterKey.trim() && newFilterValue.trim()) {
      const filters = { ...attachment.filters, [newFilterKey.trim()]: newFilterValue.trim() }
      onUpdateFilters(filters)
      setNewFilterKey('')
      setNewFilterValue('')
    }
  }

  const handleRemoveFilter = (key: string) => {
    const filters = { ...attachment.filters }
    delete filters[key]
    onUpdateFilters(filters)
  }

  return (
    <div className="border rounded-lg p-3 space-y-3">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <SourceTypeIcon type={attachment.sourceType} className="h-4 w-4 text-muted-foreground" />
          <span className="font-medium text-sm">{attachment.sourceName}</span>
        </div>
        <div className="flex items-center gap-2">
          {/* Priority selector */}
          <select
            value={attachment.priority}
            onChange={(e) => onUpdatePriority(parseInt(e.target.value, 10) as 1 | 2 | 3)}
            disabled={readOnly}
            className="rounded border border-input bg-background px-2 py-1 text-xs"
            title="Source priority (1=Required, 2=Recommended, 3=Optional)"
          >
            <option value={1}>Required</option>
            <option value={2}>Recommended</option>
            <option value={3}>Optional</option>
          </select>
          {!readOnly && (
            <Button
              type="button"
              variant="ghost"
              size="icon"
              onClick={onRemove}
              className="h-7 w-7 text-muted-foreground hover:text-destructive"
            >
              <XIcon className="h-3.5 w-3.5" />
            </Button>
          )}
        </div>
      </div>

      {/* Custom prompt */}
      <div>
        <label className="text-xs text-muted-foreground mb-1 block">
          Custom search prompt (optional)
        </label>
        <Input
          value={attachment.customPrompt || ''}
          onChange={(e) => onUpdatePrompt(e.target.value)}
          placeholder="Specific instructions for this source..."
          disabled={readOnly}
          className="text-sm"
        />
      </div>

      {/* Filters for Vector Search */}
      {attachment.sourceType === 'vector_search' && (
        <div>
          <button
            type="button"
            onClick={() => setShowFilters(!showFilters)}
            className="text-xs text-muted-foreground hover:text-foreground flex items-center gap-1"
          >
            <FilterIcon className="h-3 w-3" />
            Metadata Filters
            {attachment.filters && Object.keys(attachment.filters).length > 0 && (
              <span className="bg-muted px-1.5 py-0.5 rounded-full">
                {Object.keys(attachment.filters).length}
              </span>
            )}
            <ChevronIcon
              className={cn('h-3 w-3 transition-transform', showFilters && 'rotate-180')}
            />
          </button>

          {showFilters && (
            <div className="mt-2 space-y-2">
              {/* Existing filters */}
              {attachment.filters && Object.entries(attachment.filters).map(([key, value]) => (
                <div key={key} className="flex items-center gap-2 text-xs">
                  <code className="bg-muted px-1.5 py-0.5 rounded">{key}</code>
                  <span>=</span>
                  <code className="bg-muted px-1.5 py-0.5 rounded">{value}</code>
                  {!readOnly && (
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon"
                      onClick={() => handleRemoveFilter(key)}
                      className="h-5 w-5"
                    >
                      <XIcon className="h-3 w-3" />
                    </Button>
                  )}
                </div>
              ))}

              {/* Add new filter */}
              {!readOnly && (
                <div className="flex items-center gap-2">
                  <Input
                    value={newFilterKey}
                    onChange={(e) => setNewFilterKey(e.target.value)}
                    placeholder="Key"
                    className="text-xs h-7 flex-1"
                  />
                  <span className="text-muted-foreground">=</span>
                  <Input
                    value={newFilterValue}
                    onChange={(e) => setNewFilterValue(e.target.value)}
                    placeholder="Value"
                    className="text-xs h-7 flex-1"
                  />
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={handleAddFilter}
                    disabled={!newFilterKey.trim() || !newFilterValue.trim()}
                    className="h-7 text-xs"
                  >
                    Add
                  </Button>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

/**
 * Simple modal for selecting a source.
 */
function SourcePickerModal({
  sources,
  onSelect,
  onClose,
}: {
  sources: DataSource[]
  onSelect: (source: DataSource) => void
  onClose: () => void
}) {
  const [search, setSearch] = React.useState('')

  const filteredSources = sources.filter(
    (s) =>
      s.name.toLowerCase().includes(search.toLowerCase()) ||
      (s.description && s.description.toLowerCase().includes(search.toLowerCase()))
  )

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
  )
}

function FilterIcon({ className }: { className?: string }) {
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
      <polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3" />
    </svg>
  )
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
  )
}

function SourceTypeIcon({ type, className }: { type: DataSource['type']; className?: string }) {
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
      )
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
      )
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
      )
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
      )
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
      )
  }
}

export default ManualStepEditor
