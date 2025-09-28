// PlanTracker component
import { CheckCircle, Clock, Play, SkipForward } from 'lucide-react'
import { PlanMetadata, PlanStep } from '../types/agents'
import { cn } from '../utils/cn'

interface PlanTrackerProps {
  planDetails: PlanMetadata
  className?: string
}

export function PlanTracker({ planDetails, className = '' }: PlanTrackerProps) {
  const getStatusIcon = (status: PlanStep['status']) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-green-500" />
      case 'in_progress':
        return <Play className="w-4 h-4 text-blue-500 animate-pulse" />
      case 'skipped':
        return <SkipForward className="w-4 h-4 text-gray-400" />
      default:
        return <Clock className="w-4 h-4 text-gray-400" />
    }
  }

  const getStatusColor = (status: PlanStep['status']) => {
    switch (status) {
      case 'completed':
        return 'bg-green-50 border-green-200 text-green-800'
      case 'in_progress':
        return 'bg-blue-50 border-blue-200 text-blue-800'
      case 'skipped':
        return 'bg-gray-50 border-gray-200 text-gray-600'
      default:
        return 'bg-gray-50 border-gray-200 text-gray-600'
    }
  }

  const completedSteps = planDetails.steps.filter(step => step.status === 'completed').length
  const totalSteps = planDetails.steps.length
  const progressPercentage = totalSteps > 0 ? (completedSteps / totalSteps) * 100 : 0

  return (
    <div className={cn("space-y-3", className)}>
      {/* Plan Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-800">Research Plan</h3>
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-500">
            {completedSteps}/{totalSteps} completed
          </span>
          {planDetails.quality && (
            <span className="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded-full">
              Quality: {Math.round(planDetails.quality * 100)}%
            </span>
          )}
        </div>
      </div>

      {/* Progress Bar */}
      <div className="w-full bg-gray-200 rounded-full h-2">
        <div
          className="bg-gradient-to-r from-databricks-orange to-databricks-blue h-2 rounded-full transition-all duration-500"
          style={{ width: `${progressPercentage}%` }}
        />
      </div>

      {/* Steps List */}
      <div className="space-y-2">
        {planDetails.steps.map((step, index) => (
          <div
            key={step.id}
            className={cn(
              "flex items-start gap-3 p-3 rounded-lg border transition-all duration-200",
              getStatusColor(step.status),
              step.status === 'in_progress' && "shadow-sm"
            )}
          >
            <div className="flex-shrink-0 mt-0.5">
              {getStatusIcon(step.status)}
            </div>

            <div className="flex-grow min-w-0">
              <div className="flex items-center gap-2 mb-1">
                <span className="text-xs font-medium opacity-60">
                  Step {index + 1}
                </span>
                {step.status === 'in_progress' && (
                  <div className="flex space-x-1">
                    <div className="w-1 h-1 bg-current rounded-full thinking-dot" />
                    <div className="w-1 h-1 bg-current rounded-full thinking-dot" />
                    <div className="w-1 h-1 bg-current rounded-full thinking-dot" />
                  </div>
                )}
              </div>

              <p className={cn(
                "text-sm",
                step.status === 'completed' && "line-through opacity-70"
              )}>
                {step.description}
              </p>

              {step.result && (
                <p className="text-xs mt-2 p-2 bg-white bg-opacity-50 rounded border">
                  <span className="font-medium">Result: </span>
                  {step.result}
                </p>
              )}

              {step.completedAt && (
                <p className="text-xs opacity-60 mt-1">
                  Completed at {new Date(step.completedAt).toLocaleTimeString()}
                </p>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Plan Status */}
      <div className="flex items-center justify-between text-xs text-gray-500 pt-2 border-t">
        <span>Status: {planDetails.status}</span>
        <span>Iteration: {planDetails.iterations}</span>
      </div>
    </div>
  )
}