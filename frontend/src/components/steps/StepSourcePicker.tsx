/**
 * StepSourcePicker - Multi-select dropdown/list for picking sources for a step.
 *
 * Features:
 * - Multi-select dropdown or list for picking sources
 * - Custom prompt input field for each selected source
 * - Priority selector (1/2/3) per source
 * - Visual indicator of source type (icon)
 */

import * as React from 'react'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import type { DataSource } from '@/types/dataSources'
import { getSourceTypeLabel } from '@/types/dataSources'
import type { StepSourceAttachment } from './ManualStepEditor'

interface StepSourcePickerProps {
  /** Available sources to select from */
  availableSources: DataSource[]
  /** Currently selected source attachments */
  selectedSources: StepSourceAttachment[]
  /** Callback when selection changes */
  onChange: (sources: StepSourceAttachment[]) => void
  /** Whether the picker is disabled */
  disabled?: boolean
  /** Placeholder text */
  placeholder?: string
  /** Additional CSS classes */
  className?: string
}

export function StepSourcePicker({
  availableSources,
  selectedSources,
  onChange,
  disabled = false,
  placeholder = 'Select sources...',
  className,
}: StepSourcePickerProps) {
  const [isOpen, setIsOpen] = React.useState(false)
  const [search, setSearch] = React.useState('')
  const containerRef = React.useRef<HTMLDivElement>(null)

  // Close on click outside
  React.useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setIsOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  const selectedIds = new Set(selectedSources.map((s) => s.sourceId))

  const filteredSources = availableSources.filter(
    (source) =>
      source.name.toLowerCase().includes(search.toLowerCase()) ||
      (source.description && source.description.toLowerCase().includes(search.toLowerCase()))
  )

  const handleToggleSource = (source: DataSource) => {
    if (selectedIds.has(source.id)) {
      // Remove
      onChange(selectedSources.filter((s) => s.sourceId !== source.id))
    } else {
      // Add
      const newAttachment: StepSourceAttachment = {
        sourceId: source.id,
        sourceName: source.name,
        sourceType: source.type,
        priority: 2,
      }
      onChange([...selectedSources, newAttachment])
    }
  }

  const handleUpdatePriority = (sourceId: string, priority: 1 | 2 | 3) => {
    onChange(
      selectedSources.map((s) =>
        s.sourceId === sourceId ? { ...s, priority } : s
      )
    )
  }

  const handleUpdatePrompt = (sourceId: string, customPrompt: string) => {
    onChange(
      selectedSources.map((s) =>
        s.sourceId === sourceId ? { ...s, customPrompt } : s
      )
    )
  }

  const handleRemove = (sourceId: string) => {
    onChange(selectedSources.filter((s) => s.sourceId !== sourceId))
  }

  return (
    <div ref={containerRef} className={cn('relative', className)}>
      {/* Trigger button */}
      <button
        type="button"
        onClick={() => !disabled && setIsOpen(!isOpen)}
        disabled={disabled}
        className={cn(
          'w-full flex items-center justify-between px-3 py-2 rounded-md border border-input',
          'bg-background text-sm transition-colors',
          'hover:bg-accent hover:text-accent-foreground',
          disabled && 'cursor-not-allowed opacity-50'
        )}
      >
        <span className={cn(!selectedSources.length && 'text-muted-foreground')}>
          {selectedSources.length > 0
            ? `${selectedSources.length} source${selectedSources.length > 1 ? 's' : ''} selected`
            : placeholder}
        </span>
        <ChevronIcon className={cn('h-4 w-4 transition-transform', isOpen && 'rotate-180')} />
      </button>

      {/* Dropdown */}
      {isOpen && (
        <div className="absolute z-50 top-full left-0 right-0 mt-1 bg-popover border rounded-md shadow-lg">
          {/* Search */}
          <div className="p-2 border-b">
            <Input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search sources..."
              className="h-8"
            />
          </div>

          {/* Source list */}
          <div className="max-h-48 overflow-y-auto">
            {filteredSources.length === 0 ? (
              <div className="p-4 text-center text-sm text-muted-foreground">
                No sources found
              </div>
            ) : (
              filteredSources.map((source) => {
                const isSelected = selectedIds.has(source.id)
                return (
                  <button
                    key={source.id}
                    type="button"
                    onClick={() => handleToggleSource(source)}
                    className={cn(
                      'w-full flex items-center gap-3 px-3 py-2 text-left',
                      'hover:bg-muted transition-colors',
                      isSelected && 'bg-primary/5'
                    )}
                  >
                    <div
                      className={cn(
                        'h-4 w-4 rounded border flex items-center justify-center shrink-0',
                        isSelected
                          ? 'bg-primary border-primary text-primary-foreground'
                          : 'border-input'
                      )}
                    >
                      {isSelected && <CheckIcon className="h-3 w-3" />}
                    </div>
                    <SourceTypeIcon type={source.type} className="h-4 w-4 text-muted-foreground shrink-0" />
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium truncate">{source.name}</p>
                      <p className="text-xs text-muted-foreground truncate">
                        {source.description || getSourceTypeLabel(source.type)}
                      </p>
                    </div>
                  </button>
                )
              })
            )}
          </div>
        </div>
      )}

      {/* Selected sources with configuration */}
      {selectedSources.length > 0 && (
        <div className="mt-3 space-y-2">
          {selectedSources.map((attachment) => (
            <SelectedSourceCard
              key={attachment.sourceId}
              attachment={attachment}
              onRemove={() => handleRemove(attachment.sourceId)}
              onUpdatePriority={(priority) =>
                handleUpdatePriority(attachment.sourceId, priority)
              }
              onUpdatePrompt={(prompt) =>
                handleUpdatePrompt(attachment.sourceId, prompt)
              }
              disabled={disabled}
            />
          ))}
        </div>
      )}
    </div>
  )
}

/**
 * Card for a selected source with configuration options.
 */
function SelectedSourceCard({
  attachment,
  onRemove,
  onUpdatePriority,
  onUpdatePrompt,
  disabled,
}: {
  attachment: StepSourceAttachment
  onRemove: () => void
  onUpdatePriority: (priority: 1 | 2 | 3) => void
  onUpdatePrompt: (prompt: string) => void
  disabled: boolean
}) {
  return (
    <div className="border rounded-md p-2.5 space-y-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <SourceTypeIcon
            type={attachment.sourceType}
            className="h-4 w-4 text-muted-foreground"
          />
          <span className="text-sm font-medium">{attachment.sourceName}</span>
        </div>
        <div className="flex items-center gap-2">
          <PriorityBadge
            priority={attachment.priority}
            onChange={onUpdatePriority}
            disabled={disabled}
          />
          {!disabled && (
            <Button
              type="button"
              variant="ghost"
              size="icon"
              onClick={onRemove}
              className="h-6 w-6 text-muted-foreground hover:text-destructive"
            >
              <XIcon className="h-3.5 w-3.5" />
            </Button>
          )}
        </div>
      </div>
      <Input
        value={attachment.customPrompt || ''}
        onChange={(e) => onUpdatePrompt(e.target.value)}
        placeholder="Custom search prompt for this source..."
        className="h-8 text-xs"
        disabled={disabled}
      />
    </div>
  )
}

/**
 * Priority badge with selector.
 */
function PriorityBadge({
  priority,
  onChange,
  disabled,
}: {
  priority: 1 | 2 | 3
  onChange: (priority: 1 | 2 | 3) => void
  disabled: boolean
}) {
  const colors: Record<number, string> = {
    1: 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400',
    2: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400',
    3: 'bg-gray-100 text-gray-800 dark:bg-gray-900/30 dark:text-gray-400',
  }
  const labels: Record<number, string> = {
    1: 'Required',
    2: 'Recommended',
    3: 'Optional',
  }

  return (
    <select
      value={priority}
      onChange={(e) => onChange(parseInt(e.target.value, 10) as 1 | 2 | 3)}
      disabled={disabled}
      className={cn(
        'text-[10px] font-medium px-1.5 py-0.5 rounded border-0 cursor-pointer',
        colors[priority],
        disabled && 'cursor-not-allowed opacity-50'
      )}
      title={`Priority: ${labels[priority]}`}
    >
      <option value={1}>P1</option>
      <option value={2}>P2</option>
      <option value={3}>P3</option>
    </select>
  )
}

// Icons
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

function SourceTypeIcon({ type, className }: { type: DataSource['type']; className?: string }) {
  switch (type) {
    case 'web_search':
      return (
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className={className}>
          <circle cx="12" cy="12" r="10" />
          <path d="M2 12h20" />
          <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
        </svg>
      )
    case 'vector_search':
      return (
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className={className}>
          <ellipse cx="12" cy="5" rx="9" ry="3" />
          <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" />
          <path d="M3 12c0 1.66 4 3 9 3s9-1.34 9-3" />
        </svg>
      )
    case 'genie':
      return (
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className={className}>
          <path d="M3 3v18h18" />
          <path d="m19 9-5 5-4-4-3 3" />
        </svg>
      )
    case 'knowledge_assistant':
      return (
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className={className}>
          <path d="M12 8V4H8" />
          <rect width="16" height="12" x="4" y="8" rx="2" />
          <path d="M2 14h2M20 14h2M15 13v2M9 13v2" />
        </svg>
      )
    default:
      return (
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className={className}>
          <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z" />
          <polyline points="14 2 14 8 20 8" />
        </svg>
      )
  }
}

export default StepSourcePicker
