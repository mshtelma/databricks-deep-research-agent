/**
 * Active Jobs Indicator Component
 *
 * Shows a badge with the count of running research jobs.
 * When clicked, displays a dropdown with job details and cancel options.
 */

import * as React from 'react'
import { useActiveJobs, useCancelJob } from '@/hooks/useResearchJobs'
import { cn } from '@/lib/utils'
import type { Job } from '@/api/client'

interface ActiveJobsIndicatorProps {
  className?: string
  onNavigateToChat?: (chatId: string) => void
}

export function ActiveJobsIndicator({ className, onNavigateToChat }: ActiveJobsIndicatorProps) {
  const { data, isLoading } = useActiveJobs()
  const cancelMutation = useCancelJob()
  const [isOpen, setIsOpen] = React.useState(false)
  const dropdownRef = React.useRef<HTMLDivElement>(null)
  const buttonRef = React.useRef<HTMLButtonElement>(null)

  const jobs = data?.jobs || []
  const activeCount = data?.activeCount || 0
  const limit = data?.limit || 2

  // Close dropdown on click outside
  React.useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(e.target as Node) &&
        buttonRef.current &&
        !buttonRef.current.contains(e.target as Node)
      ) {
        setIsOpen(false)
      }
    }

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside)
      return () => document.removeEventListener('mousedown', handleClickOutside)
    }
  }, [isOpen])

  // Close on escape
  React.useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        setIsOpen(false)
      }
    }
    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [isOpen])

  // Don't render if loading or no active jobs
  if (isLoading || activeCount === 0) {
    return null
  }

  const handleCancel = (e: React.MouseEvent, sessionId: string) => {
    e.stopPropagation()
    cancelMutation.mutate(sessionId)
  }

  const handleJobClick = (job: Job) => {
    if (onNavigateToChat) {
      onNavigateToChat(job.chatId)
    }
    setIsOpen(false)
  }

  return (
    <div className={cn('relative', className)}>
      {/* Indicator button */}
      <button
        ref={buttonRef}
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className={cn(
          'flex items-center gap-2 px-3 py-2 rounded-md w-full',
          'text-sm font-medium transition-colors',
          'bg-blue-50 text-blue-700 hover:bg-blue-100',
          'dark:bg-blue-950 dark:text-blue-300 dark:hover:bg-blue-900'
        )}
        aria-expanded={isOpen}
        aria-haspopup="true"
      >
        <LoaderIcon className="h-4 w-4 animate-spin" />
        <span>{activeCount}/{limit} running</span>
        <ChevronIcon className={cn('h-4 w-4 ml-auto transition-transform', isOpen && 'rotate-180')} />
      </button>

      {/* Dropdown */}
      {isOpen && (
        <div
          ref={dropdownRef}
          className={cn(
            'absolute bottom-full left-0 right-0 mb-1 z-50',
            'rounded-md border bg-popover shadow-lg',
            'animate-in fade-in-0 slide-in-from-bottom-2'
          )}
        >
          {/* Header */}
          <div className="px-3 py-2 border-b">
            <p className="text-sm font-medium text-foreground">Active Research Jobs</p>
            <p className="text-xs text-muted-foreground mt-0.5">
              {activeCount === limit
                ? 'Job limit reached - wait for completion'
                : `${limit - activeCount} more jobs allowed`}
            </p>
          </div>

          {/* Job list */}
          <div className="max-h-64 overflow-y-auto p-1">
            {jobs.map((job) => (
              <JobListItem
                key={job.sessionId}
                job={job}
                onClick={() => handleJobClick(job)}
                onCancel={(e) => handleCancel(e, job.sessionId)}
                isCancelling={cancelMutation.isPending && cancelMutation.variables === job.sessionId}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

interface JobListItemProps {
  job: Job
  onClick: () => void
  onCancel: (e: React.MouseEvent) => void
  isCancelling?: boolean
}

function JobListItem({ job, onClick, onCancel, isCancelling }: JobListItemProps) {
  // Format progress
  const progress =
    job.currentStep !== null && job.totalSteps !== null
      ? `Step ${job.currentStep + 1}/${job.totalSteps}`
      : 'Starting...'

  // Truncate query for display
  const truncatedQuery = job.query.length > 60 ? `${job.query.slice(0, 60)}...` : job.query

  return (
    <div
      onClick={onClick}
      className={cn(
        'flex items-start gap-2 p-2 rounded cursor-pointer',
        'hover:bg-accent hover:text-accent-foreground',
        'transition-colors'
      )}
    >
      {/* Spinner */}
      <div className="flex-shrink-0 mt-0.5">
        <LoaderIcon className="h-4 w-4 animate-spin text-blue-500" />
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium truncate">{truncatedQuery}</p>
        <div className="flex items-center gap-2 mt-0.5">
          <span className="text-xs text-muted-foreground">{progress}</span>
          <span className="text-xs px-1.5 py-0.5 rounded bg-muted text-muted-foreground">
            {job.queryMode === 'deep_research' ? 'Deep' : job.queryMode === 'web_search' ? 'Web' : 'Simple'}
          </span>
        </div>
      </div>

      {/* Cancel button */}
      <button
        type="button"
        onClick={onCancel}
        disabled={isCancelling}
        className={cn(
          'flex-shrink-0 p-1 rounded',
          'text-muted-foreground hover:text-destructive hover:bg-destructive/10',
          'transition-colors',
          isCancelling && 'opacity-50 cursor-not-allowed'
        )}
        title="Cancel job"
        aria-label="Cancel job"
      >
        {isCancelling ? (
          <LoaderIcon className="h-4 w-4 animate-spin" />
        ) : (
          <XIcon className="h-4 w-4" />
        )}
      </button>
    </div>
  )
}

// Icons

function LoaderIcon({ className }: { className?: string }) {
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
      <path d="M21 12a9 9 0 1 1-6.219-8.56" />
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
      <path d="M18 6 6 18" />
      <path d="m6 6 12 12" />
    </svg>
  )
}
