import React, { useMemo, useEffect, useState } from 'react'
import { Clock, TrendingUp } from 'lucide-react'
import { useChatStore } from '@/stores/chatStore'
import { Progress } from '@/components/ui/progress'
import { Badge } from '@/components/ui/badge'
import { cn } from '@/lib/utils'

// Animated counter hook for smooth number transitions
function useAnimatedCounter(target: number, duration: number = 1000) {
  const [count, setCount] = useState(target)
  
  useEffect(() => {
    if (target === count) return
    
    const startTime = Date.now()
    const startCount = count
    const difference = target - startCount
    
    const timer = setInterval(() => {
      const elapsedTime = Date.now() - startTime
      const progress = Math.min(elapsedTime / duration, 1)
      
      // Easing function for smooth animation
      const easedProgress = 1 - Math.pow(1 - progress, 3)
      
      const currentCount = Math.round(startCount + (difference * easedProgress))
      setCount(currentCount)
      
      if (progress >= 1) {
        clearInterval(timer)
        setCount(target)
      }
    }, 16) // ~60fps
    
    return () => clearInterval(timer)
  }, [target, count, duration])
  
  return count
}

export function ResearchProgress() {
  const { researchProgress, isLoading } = useChatStore()
  
  // Enhanced phase configuration with better icons and styling - now includes all 7 agent phases
  const phaseConfig = useMemo(() => ({
    querying: {
      icon: '🔍',
      label: 'Analyzing Query',
      description: 'Understanding your question and generating search strategies',
      color: 'text-blue-600',
      bgColor: 'bg-blue-50 dark:bg-blue-950',
      borderColor: 'border-blue-200 dark:border-blue-800'
    },
    preparing: {
      icon: '📋',
      label: 'Preparing Search',
      description: 'Organizing search execution with rate limiting and batching',
      color: 'text-indigo-600',
      bgColor: 'bg-indigo-50 dark:bg-indigo-950',
      borderColor: 'border-indigo-200 dark:border-indigo-800'
    },
    searching: {
      icon: '🌐', 
      label: 'Searching Web',
      description: 'Gathering information from external web sources',
      color: 'text-orange-600',
      bgColor: 'bg-orange-50 dark:bg-orange-950',
      borderColor: 'border-orange-200 dark:border-orange-800'
    },
    searching_internal: {
      icon: '🗄️',
      label: 'Internal Search',
      description: 'Searching internal knowledge base and vector database',
      color: 'text-cyan-600',
      bgColor: 'bg-cyan-50 dark:bg-cyan-950',
      borderColor: 'border-cyan-200 dark:border-cyan-800'
    },
    aggregating: {
      icon: '📊',
      label: 'Aggregating Results',
      description: 'Combining and deduplicating search results',
      color: 'text-yellow-600',
      bgColor: 'bg-yellow-50 dark:bg-yellow-950',
      borderColor: 'border-yellow-200 dark:border-yellow-800'
    },
    analyzing: {
      icon: '🤔',
      label: 'Analyzing Results',
      description: 'Processing and evaluating search results for insights',
      color: 'text-purple-600',
      bgColor: 'bg-purple-50 dark:bg-purple-950',
      borderColor: 'border-purple-200 dark:border-purple-800'
    },
    synthesizing: {
      icon: '✍️',
      label: 'Synthesizing Response', 
      description: 'Compiling comprehensive research-backed answer',
      color: 'text-green-600',
      bgColor: 'bg-green-50 dark:bg-green-950',
      borderColor: 'border-green-200 dark:border-green-800'
    },
    complete: {
      icon: '✅',
      label: 'Research Complete',
      description: 'Analysis finished successfully',
      color: 'text-green-600',
      bgColor: 'bg-green-50 dark:bg-green-950',
      borderColor: 'border-green-200 dark:border-green-800'
    }
  }), [])
  
  // Get current config with fallback for unknown phases
  const currentConfig = phaseConfig[researchProgress.currentPhase] || {
    icon: '⚙️',
    label: 'Processing...',
    description: 'Working on your request',
    color: 'text-gray-600',
    bgColor: 'bg-gray-50 dark:bg-gray-950',
    borderColor: 'border-gray-200 dark:border-gray-800'
  }
  const progress = researchProgress.progressPercentage || 0
  
  // Animated counters for smooth transitions - ALWAYS call hooks before any conditionals
  const animatedProgress = useAnimatedCounter(Math.round(progress), 800)
  const animatedQueries = useAnimatedCounter(researchProgress.queriesGenerated, 500)
  const animatedSources = useAnimatedCounter(researchProgress.sourcesFound, 500)
  const animatedVector = useAnimatedCounter(researchProgress.vectorResultsCount || 0, 500)
  const animatedIterations = useAnimatedCounter(researchProgress.iterationsComplete, 500)
  
  // Early return AFTER all hooks have been called
  if (!isLoading || researchProgress.currentPhase === 'complete') {
    return null
  }
  
  // Format elapsed time
  const formatTime = (seconds: number) => {
    if (seconds < 60) return `${Math.round(seconds)}s`
    const minutes = Math.floor(seconds / 60)
    const remainingSeconds = Math.round(seconds % 60)
    return `${minutes}m ${remainingSeconds}s`
  }
  
  // Format ETA
  const formatETA = (seconds: number) => {
    if (seconds < 60) return `~${Math.round(seconds)}s`
    const minutes = Math.floor(seconds / 60)
    return `~${minutes}m`
  }
  
  return (
    <div className="w-full max-w-4xl mx-auto mb-4">
      <div className={cn(
        "border rounded-lg p-6 shadow-sm transition-all duration-300",
        "bg-white dark:bg-gray-800",
        currentConfig?.borderColor || "border-gray-200 dark:border-gray-700"
      )}>
        
        {/* Main Progress Section */}
        <div className="flex items-start gap-4 mb-4">
          {/* Phase Icon */}
          <div className={cn(
            "flex-shrink-0 w-12 h-12 rounded-full flex items-center justify-center text-xl transition-all duration-500 shadow-lg",
            currentConfig?.bgColor
          )}>
            <span className="animate-pulse duration-2000">{currentConfig?.icon}</span>
          </div>
          
          {/* Progress Info */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center justify-between mb-2">
              <h3 className="font-semibold text-lg text-gray-900 dark:text-gray-100">
                {currentConfig?.label}
              </h3>
              <div className="flex items-center gap-2">
                <span className={cn("text-2xl font-bold transition-colors duration-300", currentConfig?.color)}>
                  {animatedProgress}%
                </span>
              </div>
            </div>
            
            {/* Current Operation */}
            {researchProgress.currentOperation && (
              <p className="text-sm text-gray-600 dark:text-gray-300 mb-2">
                {researchProgress.currentOperation}
              </p>
            )}
            
            <p className="text-sm text-gray-500 dark:text-gray-400 mb-3">
              {currentConfig?.description}
            </p>
            
            {/* Progress Bar */}
            <div className="space-y-2">
              <Progress value={progress} className="h-3 transition-all duration-700 ease-out" />
              
              {/* Time and Stats Row */}
              <div className="flex justify-between items-center text-xs">
                <div className="flex items-center gap-4 text-gray-500 dark:text-gray-400">
                  {researchProgress.elapsedTime && researchProgress.elapsedTime > 0 && (
                    <div className="flex items-center gap-1">
                      <Clock className="h-3 w-3" />
                      {formatTime(researchProgress.elapsedTime)}
                    </div>
                  )}
                  {researchProgress.estimatedTimeRemaining && researchProgress.estimatedTimeRemaining > 0 && (
                    <div className="flex items-center gap-1">
                      <TrendingUp className="h-3 w-3" />
                      ETA {formatETA(researchProgress.estimatedTimeRemaining)}
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
        
        {/* Metadata Badges */}
        <div className="flex flex-wrap gap-2 mb-4">
          {animatedQueries > 0 && (
            <Badge variant="secondary" className="text-xs transition-all duration-300 hover:scale-105">
              🔍 {animatedQueries} queries
            </Badge>
          )}
          {animatedSources > 0 && (
            <Badge variant="secondary" className="text-xs transition-all duration-300 hover:scale-105">
              🌐 {animatedSources} sources
            </Badge>
          )}
          {animatedVector > 0 && (
            <Badge variant="secondary" className="text-xs transition-all duration-300 hover:scale-105">
              🗄️ {animatedVector} internal results
            </Badge>
          )}
          {animatedIterations > 0 && (
            <Badge variant="secondary" className="text-xs transition-all duration-300 hover:scale-105">
              🔄 {animatedIterations} iterations
            </Badge>
          )}
          {researchProgress.currentNode && (
            <Badge variant="outline" className="text-xs transition-all duration-300 hover:scale-105">
              📍 {researchProgress.currentNode}
            </Badge>
          )}
        </div>
        
        {/* Phase Progress Timeline */}
        <div className="relative flex justify-between items-center">
          {Object.entries(phaseConfig).filter(([phase]) => phase !== 'complete').map(([phase, config], index) => {
            const isActive = phase === researchProgress.currentPhase
            const phaseOrder = ['querying', 'preparing', 'searching', 'searching_internal', 'aggregating', 'analyzing', 'synthesizing']
            const currentPhaseIndex = phaseOrder.indexOf(researchProgress.currentPhase)
            const isCompleted = currentPhaseIndex > index || researchProgress.currentPhase === 'complete'
            
            return (
              <React.Fragment key={phase}>
                <div className="flex flex-col items-center relative z-10">
                  {/* Phase Icon */}
                  <div className={cn(
                    "w-8 h-8 rounded-full flex items-center justify-center text-sm transition-all duration-500 mb-1 shadow-sm",
                    isActive && "shadow-md scale-110",
                    isActive ? config.bgColor : isCompleted ? "bg-green-100 dark:bg-green-900" : "bg-gray-100 dark:bg-gray-700"
                  )}>
                    <span className={cn(
                      "transition-all duration-300",
                      isActive && "animate-pulse",
                      !isActive && !isCompleted && "opacity-60"
                    )}>
                      {isCompleted ? "✅" : config.icon}
                    </span>
                  </div>
                  
                  {/* Phase Label */}
                  <span className={cn(
                    "text-xs font-medium transition-all duration-300",
                    isActive && "font-semibold",
                    isActive ? config.color : isCompleted ? "text-green-600" : "text-gray-400"
                  )}>
                    {config.label}
                  </span>
                </div>
                
                {/* Connection Line between phases */}
                {index < phaseOrder.length - 1 && (
                  <div className={cn(
                    "flex-1 h-0.5 transition-all duration-700 mx-2 rounded-full",
                    isCompleted ? "bg-green-500 shadow-sm" : "bg-gray-200 dark:bg-gray-600"
                  )} />
                )}
              </React.Fragment>
            )
          })}
        </div>
      </div>
    </div>
  )
}