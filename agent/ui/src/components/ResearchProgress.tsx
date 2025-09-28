import { useEffect, useRef, useState } from 'react'
import {
  CheckCircle,
  Circle,
  Loader2,
  XCircle,
  MinusCircle,
  ChevronDown,
  ChevronUp,
  PlayCircle
} from 'lucide-react'
import { cn } from '../utils/cn'
import { formatTime } from '../utils/formatters'
import { ProgressItem, StructuredProgress } from '../types/progress'

interface ResearchProgressProps {
  structuredProgress: StructuredProgress
  isStreaming?: boolean
  className?: string
  showTimestamps?: boolean
  autoScroll?: boolean
  collapsible?: boolean
}

export function ResearchProgress({
  structuredProgress,
  isStreaming = false,
  className = '',
  showTimestamps = true,
  autoScroll = true,
  collapsible = false
}: ResearchProgressProps) {
  const [collapsed, setCollapsed] = useState(false)
  const activeItemRef = useRef<HTMLDivElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  // Extract structured data
  const workflowPhases = structuredProgress.workflowPhases
  const planSteps = structuredProgress.planSteps
  const allItems = [...workflowPhases, ...planSteps]

  // Auto-scroll to active item
  useEffect(() => {
    if (autoScroll && activeItemRef.current && containerRef.current) {
      activeItemRef.current.scrollIntoView({
        behavior: 'smooth',
        block: 'nearest'
      })
    }
  }, [allItems, autoScroll])

  const getStatusIcon = (status: ProgressItem['status'], isActive: boolean = false) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-green-500" />
      case 'active':
        return <Loader2 className="w-4 h-4 text-databricks-blue animate-spin" />
      case 'failed':
        return <XCircle className="w-4 h-4 text-red-500" />
      case 'skipped':
        return <MinusCircle className="w-4 h-4 text-yellow-500" />
      default:
        return isActive ?
          <PlayCircle className="w-4 h-4 text-databricks-blue animate-pulse" /> :
          <Circle className="w-4 h-4 text-gray-400" />
    }
  }

  const getStatusStyles = (status: ProgressItem['status']) => {
    switch (status) {
      case 'completed':
        return 'bg-green-50 border-green-200 text-green-800'
      case 'active':
        return 'bg-blue-50 border-databricks-blue text-databricks-blue shadow-sm'
      case 'failed':
        return 'bg-red-50 border-red-200 text-red-800'
      case 'skipped':
        return 'bg-yellow-50 border-yellow-200 text-yellow-800'
      default:
        return 'bg-gray-50 border-gray-200 text-gray-600'
    }
  }

  const getStatusBadge = (status: ProgressItem['status']) => {
    switch (status) {
      case 'completed':
        return <span className="text-xs bg-green-100 text-green-700 px-2 py-1 rounded-full">Completed</span>
      case 'active':
        return (
          <span className="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded-full flex items-center gap-1">
            <div className="w-1 h-1 bg-blue-500 rounded-full animate-pulse" />
            Running
          </span>
        )
      case 'failed':
        return <span className="text-xs bg-red-100 text-red-700 px-2 py-1 rounded-full">Failed</span>
      case 'skipped':
        return <span className="text-xs bg-yellow-100 text-yellow-700 px-2 py-1 rounded-full">Skipped</span>
      default:
        return <span className="text-xs bg-gray-100 text-gray-600 px-2 py-1 rounded-full">Pending</span>
    }
  }

  const completedCount = allItems.filter(item => item.status === 'completed').length
  const totalCount = allItems.length
  const progressPercentage = structuredProgress.overallProgress || 0

  if (!allItems.length) {
    return null
  }

  return (
    <div className={cn("bg-white border border-gray-200 rounded-lg", className)} ref={containerRef}>
      {/* Header */}
      <div className="p-3 border-b border-gray-100">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <h3 className="text-sm font-semibold text-gray-800">Research Progress</h3>
            {isStreaming && (
              <div className="flex items-center gap-1">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                <span className="text-xs text-green-600 font-medium">Live</span>
              </div>
            )}
          </div>

          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-500">
              {completedCount}/{totalCount}
            </span>
            <span className="text-xs bg-databricks-blue text-white px-2 py-1 rounded-full">
              {Math.round(progressPercentage)}%
            </span>
            {collapsible && (
              <button
                onClick={() => setCollapsed(!collapsed)}
                className="p-1 hover:bg-gray-100 rounded"
              >
                {collapsed ? (
                  <ChevronDown className="w-4 h-4 text-gray-500" />
                ) : (
                  <ChevronUp className="w-4 h-4 text-gray-500" />
                )}
              </button>
            )}
          </div>
        </div>

        {/* Progress Bar */}
        <div className="mt-2 w-full bg-gray-200 rounded-full h-2">
          <div
            className="bg-gradient-to-r from-databricks-orange to-databricks-blue h-2 rounded-full transition-all duration-500"
            style={{ width: `${progressPercentage}%` }}
          />
        </div>
      </div>

      {/* Progress Items */}
      {!collapsed && (
        <div className="p-3 space-y-3 max-h-80 overflow-y-auto">
          {/* Agent Progress Section */}
          {workflowPhases.length > 0 && (
            <div className="space-y-2">
              <h4 className="text-xs font-semibold text-gray-600 uppercase tracking-wide">Agent Progress</h4>
              <div className="space-y-2">
                {workflowPhases.map((item) => (
                  <div
                    key={item.id}
                    ref={item.status === 'active' ? activeItemRef : null}
                    className={cn(
                      "flex items-center gap-3 p-2 rounded-lg border transition-all duration-200",
                      getStatusStyles(item.status),
                      item.status === 'active' && "animate-pulse"
                    )}
                  >
                    <div className="flex-shrink-0">
                      {getStatusIcon(item.status)}
                    </div>
                    <div className="flex-grow min-w-0">
                      <p className={cn(
                        "text-sm font-medium",
                        item.status === 'completed' && "line-through opacity-70"
                      )}>
                        {item.label}
                      </p>
                    </div>
                    <div className="flex-shrink-0">
                      {getStatusBadge(item.status)}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Research Steps Section */}
          {planSteps.length > 0 && (
            <div className="space-y-2">
              <h4 className="text-xs font-semibold text-gray-600 uppercase tracking-wide">Research Steps</h4>
              <div className="space-y-2">
                {planSteps.map((item) => (
                  <div
                    key={item.id}
                    ref={item.status === 'active' ? activeItemRef : null}
                    className={cn(
                      "flex items-start gap-3 p-3 rounded-lg border transition-all duration-200",
                      getStatusStyles(item.status),
                      item.status === 'active' && "animate-pulse"
                    )}
                  >
                    <div className="flex-shrink-0 mt-0.5">
                      {getStatusIcon(item.status)}
                    </div>
                    <div className="flex-grow min-w-0">
                      <div className="flex items-start justify-between gap-2">
                        <div className="flex-grow">
                          <div className="flex items-center gap-2 mb-1">
                            {item.stepNumber && (
                              <span className="text-xs font-medium opacity-60">
                                Step {item.stepNumber}
                              </span>
                            )}
                            {item.status === 'active' && (
                              <div className="flex space-x-1">
                                <div className="w-1 h-1 bg-current rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                                <div className="w-1 h-1 bg-current rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                                <div className="w-1 h-1 bg-current rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                              </div>
                            )}
                          </div>
                          <p className={cn(
                            "text-sm font-medium",
                            item.status === 'completed' && "line-through opacity-70"
                          )}>
                            {item.label}
                          </p>
                          {item.result && (
                            <p className="text-xs mt-2 p-2 bg-white bg-opacity-50 rounded border">
                              <span className="font-medium">Result: </span>
                              {item.result}
                            </p>
                          )}
                          {showTimestamps && item.timestamp && (
                            <p className="text-xs opacity-60 mt-1">
                              {item.status === 'completed' ? 'Completed' : 'Started'} at {formatTime(item.timestamp)}
                              {item.duration && ` (${item.duration}s)`}
                            </p>
                          )}
                        </div>
                        <div className="flex-shrink-0">
                          {getStatusBadge(item.status)}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Overall Status */}
          {structuredProgress.elapsedTime && (
            <div className="pt-2 border-t border-gray-100 mt-3">
              <div className="flex items-center justify-between text-xs text-gray-500">
                <span>
                  Phase: {structuredProgress.currentPhase || 'Unknown'}
                </span>
                <span>
                  Elapsed: {Math.round(structuredProgress.elapsedTime)}s
                </span>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
