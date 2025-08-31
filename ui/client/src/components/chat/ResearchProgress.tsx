import React from 'react'
import { Search, Database, Brain, FileText, Loader2 } from 'lucide-react'
import { useChatStore } from '@/stores/chatStore'
import { Progress } from '@/components/ui/progress'
import { Badge } from '@/components/ui/badge'
import { cn } from '@/lib/utils'

export function ResearchProgress() {
  const { researchProgress, isLoading } = useChatStore()
  
  if (!isLoading || researchProgress.currentPhase === 'complete') {
    return null
  }
  
  const phaseConfig = {
    querying: {
      icon: Search,
      label: 'Generating search queries',
      description: 'Analyzing your question and planning research strategy',
      progress: 20,
      color: 'text-blue-600'
    },
    searching: {
      icon: Database,
      label: 'Searching sources',
      description: 'Gathering information from multiple sources',
      progress: 45,
      color: 'text-orange-600'
    },
    analyzing: {
      icon: Brain,
      label: 'Analyzing results',
      description: 'Processing and evaluating search results',
      progress: 70,
      color: 'text-purple-600'
    },
    synthesizing: {
      icon: FileText,
      label: 'Synthesizing response',
      description: 'Compiling comprehensive research-backed answer',
      progress: 90,
      color: 'text-green-600'
    },
    complete: {
      icon: FileText,
      label: 'Complete',
      description: 'Research complete',
      progress: 100,
      color: 'text-green-600'
    }
  }
  
  const config = phaseConfig[researchProgress.currentPhase]
  const Icon = config.icon
  
  return (
    <div className="w-full max-w-4xl mx-auto mb-4">
      <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-4 shadow-sm">
        <div className="flex items-center gap-3 mb-3">
          <div className={cn("flex-shrink-0", config.color)}>
            <Loader2 className="h-5 w-5 animate-spin" />
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center justify-between">
              <h3 className="font-medium text-gray-900 dark:text-gray-100">
                {config.label}
              </h3>
              <div className="flex gap-2">
                {researchProgress.queriesGenerated > 0 && (
                  <Badge variant="secondary" className="text-xs">
                    {researchProgress.queriesGenerated} queries
                  </Badge>
                )}
                {researchProgress.sourcesFound > 0 && (
                  <Badge variant="secondary" className="text-xs">
                    {researchProgress.sourcesFound} sources
                  </Badge>
                )}
                {researchProgress.iterationsComplete > 0 && (
                  <Badge variant="secondary" className="text-xs">
                    {researchProgress.iterationsComplete} iterations
                  </Badge>
                )}
              </div>
            </div>
            <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
              {config.description}
            </p>
          </div>
        </div>
        
        <div className="space-y-2">
          <Progress value={config.progress} className="h-2" />
          <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400">
            <span>Research in progress...</span>
            <span>{config.progress}%</span>
          </div>
        </div>
        
        {/* Phase indicators */}
        <div className="flex justify-between mt-4 text-xs">
          {Object.entries(phaseConfig).slice(0, -1).map(([phase, phaseInfo]) => {
            const PhaseIcon = phaseInfo.icon
            const isActive = phase === researchProgress.currentPhase
            const isCompleted = config.progress > phaseInfo.progress
            
            return (
              <div
                key={phase}
                className={cn(
                  "flex flex-col items-center gap-1 transition-colors",
                  isActive ? phaseInfo.color : isCompleted ? "text-green-500" : "text-gray-300 dark:text-gray-600"
                )}
              >
                <PhaseIcon className="h-4 w-4" />
                <span className="capitalize">{phase}</span>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}