import React from 'react'
import { useChatStore } from '@/stores/chatStore'
import { cn } from '@/lib/utils'

interface WorkflowVisualizerProps {
  className?: string
  compact?: boolean
}

export function WorkflowVisualizer({ className = '', compact = false }: WorkflowVisualizerProps) {
  const { researchProgress, intermediateEvents } = useChatStore()

  const agents = [
    {
      id: 'coordinator',
      name: 'Coordinator',
      icon: '🎯',
      color: 'bg-blue-500',
      lightColor: 'bg-blue-100',
      description: 'Query analysis & routing'
    },
    {
      id: 'background_investigation',
      name: 'Background',
      icon: '🔍', 
      color: 'bg-indigo-500',
      lightColor: 'bg-indigo-100',
      description: 'Initial context gathering'
    },
    {
      id: 'planning',
      name: 'Planner',
      icon: '📋',
      color: 'bg-purple-500',
      lightColor: 'bg-purple-100',
      description: 'Research plan creation'
    },
    {
      id: 'research',
      name: 'Researcher',
      icon: '🔬',
      color: 'bg-orange-500',
      lightColor: 'bg-orange-100',
      description: 'Information gathering'
    },
    {
      id: 'fact_checking',
      name: 'Fact Checker',
      icon: '🔎',
      color: 'bg-red-500',
      lightColor: 'bg-red-100',
      description: 'Accuracy verification'
    },
    {
      id: 'reporting',
      name: 'Reporter',
      icon: '📄',
      color: 'bg-green-500',
      lightColor: 'bg-green-100',
      description: 'Report synthesis'
    }
  ]

  const currentPhase = researchProgress.currentPhase
  const currentAgentIndex = agents.findIndex(agent => agent.id === currentPhase)
  
  // Get recent handoffs from events
  const recentHandoffs = intermediateEvents
    .filter(event => event.event_type === 'agent_handoff' as any)
    .slice(-3)

  // Get progress percentage for animation
  const progressPercentage = researchProgress.progressPercentage || 0

  if (compact) {
    return (
      <div className={`flex items-center gap-2 ${className}`}>
        {agents.map((agent, index) => {
          const isActive = agent.id === currentPhase
          const isCompleted = currentAgentIndex > index || currentPhase === 'complete'
          
          return (
            <div
              key={agent.id}
              className={cn(
                "flex items-center justify-center w-8 h-8 rounded-full text-sm font-medium transition-all duration-500",
                isActive && "scale-110 shadow-lg",
                isActive ? agent.color + " text-white" : 
                isCompleted ? agent.lightColor + " text-gray-700" : 
                "bg-gray-100 text-gray-400"
              )}
              title={`${agent.name}: ${agent.description}`}
            >
              <span className={cn("transition-all duration-300", isActive && "animate-pulse")}>
                {agent.icon}
              </span>
            </div>
          )
        })}
      </div>
    )
  }

  return (
    <div className={`bg-white dark:bg-gray-800 rounded-lg border p-6 ${className}`}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold">Multi-Agent Workflow</h3>
        <div className="text-sm text-gray-500">
          {Math.round(progressPercentage)}% Complete
        </div>
      </div>

      {/* Main Workflow Visualization */}
      <div className="relative">
        {/* Progress Line */}
        <div className="absolute top-12 left-8 right-8 h-0.5 bg-gray-200 dark:bg-gray-700">
          <div 
            className="h-full bg-blue-500 transition-all duration-1000 ease-out"
            style={{ width: `${(currentAgentIndex / (agents.length - 1)) * 100}%` }}
          />
        </div>

        {/* Agent Nodes */}
        <div className="relative flex justify-between">
          {agents.map((agent, index) => {
            const isActive = agent.id === currentPhase
            const isCompleted = currentAgentIndex > index || currentPhase === 'complete'
            const isPending = currentAgentIndex < index && currentPhase !== 'complete'
            
            return (
              <div key={agent.id} className="flex flex-col items-center relative">
                {/* Agent Node */}
                <div
                  className={cn(
                    "w-16 h-16 rounded-full flex items-center justify-center text-xl font-medium transition-all duration-500 shadow-md relative z-10",
                    isActive && "scale-110 shadow-xl animate-pulse",
                    isActive ? agent.color + " text-white" : 
                    isCompleted ? agent.lightColor + " text-gray-800 border-2 border-green-300" : 
                    isPending ? "bg-gray-100 text-gray-400 border-2 border-dashed border-gray-300" :
                    "bg-gray-50 text-gray-300"
                  )}
                >
                  <span>{agent.icon}</span>
                  
                  {/* Active Agent Pulse Ring */}
                  {isActive && (
                    <div className={`absolute inset-0 rounded-full ${agent.color} opacity-20 animate-ping`} />
                  )}
                  
                  {/* Completion Checkmark */}
                  {isCompleted && !isActive && (
                    <div className="absolute -top-1 -right-1 w-6 h-6 bg-green-500 text-white rounded-full flex items-center justify-center text-xs">
                      ✓
                    </div>
                  )}
                </div>

                {/* Agent Label */}
                <div className="mt-3 text-center">
                  <div className={cn(
                    "text-sm font-medium transition-colors duration-300",
                    isActive ? "text-blue-600" : 
                    isCompleted ? "text-green-600" : 
                    "text-gray-400"
                  )}>
                    {agent.name}
                  </div>
                  <div className="text-xs text-gray-500 mt-1 max-w-20">
                    {agent.description}
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* Recent Activity */}
      {recentHandoffs.length > 0 && (
        <div className="mt-6 pt-4 border-t">
          <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Recent Handoffs
          </h4>
          <div className="space-y-1">
            {recentHandoffs.map((handoff, index) => (
              <div key={handoff.id} className="text-xs text-gray-600 dark:text-gray-400">
                <span className="text-blue-600">{handoff.data.from_agent}</span>
                <span className="mx-1">→</span>
                <span className="text-green-600">{handoff.data.to_agent}</span>
                <span className="ml-2 text-gray-500">
                  {new Date(handoff.timestamp * 1000).toLocaleTimeString()}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Quality Metrics */}
      {(researchProgress.factualityScore || researchProgress.researchQualityScore || researchProgress.coverageScore) && (
        <div className="mt-4 pt-4 border-t">
          <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Quality Metrics
          </h4>
          <div className="grid grid-cols-3 gap-4 text-center">
            {researchProgress.factualityScore && (
              <div>
                <div className="text-lg font-semibold text-blue-600">
                  {Math.round(researchProgress.factualityScore * 100)}%
                </div>
                <div className="text-xs text-gray-500">Factuality</div>
              </div>
            )}
            {researchProgress.researchQualityScore && (
              <div>
                <div className="text-lg font-semibold text-green-600">
                  {Math.round(researchProgress.researchQualityScore * 100)}%
                </div>
                <div className="text-xs text-gray-500">Quality</div>
              </div>
            )}
            {researchProgress.coverageScore && (
              <div>
                <div className="text-lg font-semibold text-orange-600">
                  {Math.round(researchProgress.coverageScore * 100)}%
                </div>
                <div className="text-xs text-gray-500">Coverage</div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}