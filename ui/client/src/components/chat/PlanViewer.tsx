import { useState } from 'react'
import { ChevronDown, ChevronRight, Clock, CheckCircle, AlertCircle, RotateCcw } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { PlanMetadata } from '@/types/chat'
import { cn } from '@/lib/utils'

interface PlanViewerProps {
  planData: PlanMetadata
  className?: string
  isStreaming?: boolean
}

export function PlanViewer({ planData, className = '', isStreaming = false }: PlanViewerProps) {
  const [isExpanded, setIsExpanded] = useState(true)

  if (!planData || !planData.steps || planData.steps.length === 0) {
    return null
  }

  const completedSteps = planData.steps.filter(step => step.status === 'completed').length
  const inProgressSteps = planData.steps.filter(step => step.status === 'in_progress').length
  const pendingSteps = planData.steps.filter(step => step.status === 'pending').length
  const skippedSteps = planData.steps.filter(step => step.status === 'skipped').length
  
  const totalSteps = planData.steps.length
  const progressPercentage = (completedSteps / totalSteps) * 100

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-green-600" />
      case 'in_progress':
        return <RotateCcw className="w-4 h-4 text-blue-600 animate-spin" />
      case 'skipped':
        return <AlertCircle className="w-4 h-4 text-yellow-600" />
      default:
        return <Clock className="w-4 h-4 text-gray-400" />
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'bg-green-50 border-green-200'
      case 'in_progress':
        return 'bg-blue-50 border-blue-200'
      case 'skipped':
        return 'bg-yellow-50 border-yellow-200'
      default:
        return 'bg-gray-50 border-gray-200'
    }
  }

  const getStatusBadgeVariant = (status: string) => {
    switch (status) {
      case 'draft':
        return 'secondary'
      case 'approved':
        return 'default'
      case 'executing':
        return 'default'
      case 'completed':
        return 'default'
      default:
        return 'secondary'
    }
  }

  return (
    <div className={`bg-white dark:bg-gray-800 border rounded-lg ${className}`}>
      <Collapsible open={isExpanded} onOpenChange={setIsExpanded}>
        <CollapsibleTrigger className="w-full p-4 flex items-center justify-between hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-purple-100 rounded-full flex items-center justify-center">
              <span className="text-purple-600 text-sm">📋</span>
            </div>
            <div className="text-left">
              <h3 className="text-sm font-medium">
                Research Plan
                {isStreaming && (
                  <span className="ml-2 inline-flex items-center gap-1">
                    <div className="w-2 h-2 bg-purple-500 rounded-full animate-pulse" />
                    <span className="text-xs text-purple-600 dark:text-purple-400">Executing</span>
                  </span>
                )}
              </h3>
              <p className="text-xs text-gray-500">
                {completedSteps}/{totalSteps} steps completed
                {isStreaming && <span className="ml-1">(updating)</span>}
                {planData.iterations > 1 && (
                  <span className="ml-2">• Iteration {planData.iterations}</span>
                )}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant={getStatusBadgeVariant(planData.status)}>
              {planData.status}
            </Badge>
            {planData.quality && (
              <Badge variant="secondary" className="text-xs">
                {Math.round(planData.quality * 100)}% quality
              </Badge>
            )}
            {isExpanded ? (
              <ChevronDown className="w-4 h-4 text-gray-400" />
            ) : (
              <ChevronRight className="w-4 h-4 text-gray-400" />
            )}
          </div>
        </CollapsibleTrigger>

        <CollapsibleContent className="px-4 pb-4">
          {/* Progress Overview */}
          <div className="mb-4">
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm text-gray-600">Overall Progress</span>
              <span className="text-sm font-medium">{Math.round(progressPercentage)}%</span>
            </div>
            <Progress value={progressPercentage} className="h-2" />
            
            {/* Status Summary */}
            <div className="flex gap-4 mt-2 text-xs text-gray-500">
              <span>{completedSteps} completed</span>
              {inProgressSteps > 0 && <span>{inProgressSteps} in progress</span>}
              <span>{pendingSteps} pending</span>
              {skippedSteps > 0 && <span>{skippedSteps} skipped</span>}
            </div>
          </div>

          {/* Plan Iterations Info */}
          {planData.iterations > 1 && (
            <div className="mb-4 p-3 bg-blue-50 dark:bg-blue-950 rounded-lg">
              <div className="flex items-center gap-2 text-blue-700 dark:text-blue-300">
                <RotateCcw className="w-4 h-4" />
                <span className="text-sm font-medium">
                  Plan refined {planData.iterations - 1} time{planData.iterations > 2 ? 's' : ''}
                </span>
              </div>
              <p className="text-xs text-blue-600 dark:text-blue-400 mt-1">
                This plan has been iteratively improved for better coverage and quality.
              </p>
            </div>
          )}

          {/* Step List */}
          <div className="space-y-2">
            {planData.steps.map((step, index) => (
              <div
                key={step.id}
                className={cn(
                  "p-3 border rounded-lg transition-colors",
                  getStatusColor(step.status)
                )}
              >
                <div className="flex items-start gap-3">
                  <div className="flex-shrink-0 mt-0.5">
                    {getStatusIcon(step.status)}
                  </div>
                  <div className="flex-grow min-w-0">
                    <div className="flex items-center justify-between mb-1">
                      <h4 className="text-sm font-medium">
                        Step {index + 1}
                      </h4>
                      <div className="flex items-center gap-2">
                        <Badge 
                          variant="outline" 
                          className="text-xs capitalize"
                        >
                          {step.status.replace('_', ' ')}
                        </Badge>
                        {step.completedAt && (
                          <span className="text-xs text-gray-500">
                            {new Date(step.completedAt).toLocaleTimeString()}
                          </span>
                        )}
                      </div>
                    </div>
                    
                    <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
                      {step.description}
                    </p>

                    {step.result && (
                      <div className="mt-2 p-2 bg-green-50 dark:bg-green-950 rounded text-xs text-green-700 dark:text-green-300">
                        <strong>Result:</strong> {step.result}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Additional Context */}
          {planData.hasEnoughContext && (
            <div className="mt-4 p-3 bg-green-50 dark:bg-green-950 rounded-lg">
              <div className="flex items-center gap-2 text-green-700 dark:text-green-300">
                <CheckCircle className="w-4 h-4" />
                <span className="text-sm font-medium">Plan has sufficient context</span>
              </div>
              <p className="text-xs text-green-600 dark:text-green-400 mt-1">
                The plan contains enough information to proceed directly to reporting.
              </p>
            </div>
          )}
        </CollapsibleContent>
      </Collapsible>
    </div>
  )
}