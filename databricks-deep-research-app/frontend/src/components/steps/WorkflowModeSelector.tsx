/**
 * WorkflowModeSelector - Toggle between research workflow modes.
 *
 * Modes:
 * - Planner (AI): AI generates and executes research steps
 * - Manual: User defines all research steps
 * - Hybrid: User defines initial steps, AI adds more based on findings
 */

import * as React from 'react'
import { cn } from '@/lib/utils'

export type WorkflowMode = 'planner' | 'manual' | 'hybrid'

interface WorkflowModeSelectorProps {
  /** Currently selected mode */
  value: WorkflowMode
  /** Callback when mode changes */
  onChange: (mode: WorkflowMode) => void
  /** Whether the selector is disabled */
  disabled?: boolean
  /** Additional CSS classes */
  className?: string
}

const WORKFLOW_MODES: {
  value: WorkflowMode
  label: string
  shortLabel: string
  description: string
}[] = [
  {
    value: 'planner',
    label: 'Planner (AI)',
    shortLabel: 'Planner',
    description: 'AI generates and executes research steps automatically',
  },
  {
    value: 'manual',
    label: 'Manual',
    shortLabel: 'Manual',
    description: 'You define all research steps and sources',
  },
  {
    value: 'hybrid',
    label: 'Hybrid',
    shortLabel: 'Hybrid',
    description: 'Define initial steps, AI adds more based on findings',
  },
]

export function WorkflowModeSelector({
  value,
  onChange,
  disabled = false,
  className,
}: WorkflowModeSelectorProps) {
  const [showTooltip, setShowTooltip] = React.useState<WorkflowMode | null>(null)

  return (
    <div className={cn('flex items-center gap-1', className)}>
      <span className="text-xs text-muted-foreground mr-1">Workflow:</span>
      <div className="relative flex gap-1 rounded-md border border-input p-0.5 bg-muted/50">
        {WORKFLOW_MODES.map((mode) => (
          <div key={mode.value} className="relative">
            <button
              type="button"
              onClick={() => onChange(mode.value)}
              disabled={disabled}
              onMouseEnter={() => setShowTooltip(mode.value)}
              onMouseLeave={() => setShowTooltip(null)}
              className={cn(
                'px-2 py-1 text-xs rounded transition-colors relative',
                'focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring',
                value === mode.value
                  ? 'bg-background text-foreground shadow-sm'
                  : 'text-muted-foreground hover:text-foreground hover:bg-background/50',
                disabled && 'cursor-not-allowed opacity-50'
              )}
              aria-label={mode.label}
              aria-pressed={value === mode.value}
            >
              {mode.shortLabel}
            </button>

            {/* Tooltip */}
            {showTooltip === mode.value && (
              <div
                className={cn(
                  'absolute z-50 top-full left-1/2 -translate-x-1/2 mt-2',
                  'px-3 py-2 bg-popover text-popover-foreground rounded-md shadow-md border',
                  'text-xs whitespace-nowrap',
                  'animate-in fade-in-0 zoom-in-95 duration-100'
                )}
              >
                <div className="font-medium mb-0.5">{mode.label}</div>
                <div className="text-muted-foreground max-w-[200px] whitespace-normal">
                  {mode.description}
                </div>
                {/* Arrow */}
                <div className="absolute -top-1 left-1/2 -translate-x-1/2 w-2 h-2 bg-popover border-l border-t rotate-45" />
              </div>
            )}
          </div>
        ))}
      </div>
      <InfoButton />
    </div>
  )
}

/**
 * Info button with tooltip explaining all modes.
 */
function InfoButton() {
  const [showHelp, setShowHelp] = React.useState(false)

  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => setShowHelp(!showHelp)}
        className="p-1 text-muted-foreground hover:text-foreground transition-colors rounded"
        aria-label="Learn about workflow modes"
      >
        <InfoIcon className="h-3.5 w-3.5" />
      </button>

      {showHelp && (
        <>
          {/* Backdrop */}
          <div
            className="fixed inset-0 z-40"
            onClick={() => setShowHelp(false)}
          />

          {/* Help popup */}
          <div
            className={cn(
              'absolute z-50 right-0 top-full mt-2',
              'w-72 p-3 bg-popover text-popover-foreground rounded-lg shadow-lg border',
              'animate-in fade-in-0 slide-in-from-top-1 duration-150'
            )}
          >
            <h4 className="font-medium text-sm mb-2">Workflow Modes</h4>
            <div className="space-y-3">
              {WORKFLOW_MODES.map((mode) => (
                <div key={mode.value} className="text-xs">
                  <div className="font-medium text-foreground">{mode.label}</div>
                  <div className="text-muted-foreground mt-0.5">{mode.description}</div>
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  )
}

/**
 * Compact badge showing current workflow mode.
 */
export function WorkflowModeBadge({ mode }: { mode: WorkflowMode }) {
  const config = WORKFLOW_MODES.find((m) => m.value === mode)
  if (!config) return null

  return (
    <span
      className={cn(
        'inline-flex items-center px-2 py-0.5 rounded text-xs font-medium',
        mode === 'planner' && 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400',
        mode === 'manual' && 'bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-400',
        mode === 'hybrid' && 'bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-400'
      )}
      title={config.description}
    >
      {config.shortLabel}
    </span>
  )
}

// Icons
function InfoIcon({ className }: { className?: string }) {
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
      <path d="M12 16v-4" />
      <path d="M12 8h.01" />
    </svg>
  )
}

export default WorkflowModeSelector
