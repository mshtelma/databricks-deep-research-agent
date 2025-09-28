import { useState, useEffect } from 'react'
import { ChevronDown, ChevronUp, Brain, Search, CheckCircle, Clock, AlertTriangle, Zap } from 'lucide-react'
import { IntermediateEvent, IntermediateEventType, AgentActivity } from '../types/agents'
import { cn } from '../utils/cn'
import { getAgentIcon, getAgentColor } from '../utils/formatters'

interface StreamingProgressProps {
  events: IntermediateEvent[]
  isActive?: boolean
  className?: string
}

export function StreamingProgress({ events, isActive = true, className = '' }: StreamingProgressProps) {
  const [expandedActivities, setExpandedActivities] = useState<Set<string>>(new Set())
  const [activities, setActivities] = useState<AgentActivity[]>([])

  // Process events into agent activities
  useEffect(() => {
    const processedActivities: { [key: string]: AgentActivity } = {}

    events.forEach(event => {
      const agent = event.data.agent || event.data.current_agent || event.data.from_agent || 'System'

      if (!processedActivities[agent]) {
        processedActivities[agent] = {
          agent,
          currentAction: '',
          status: 'thinking',
          events: [],
          queries: [],
          findings: []
        }
      }

      const activity = processedActivities[agent]
      activity.events.push(event)

      // Update activity based on event type
      switch (event.event_type) {
        case IntermediateEventType.AGENT_START:
          activity.currentAction = event.data.action || `${agent} starting...`
          activity.status = 'thinking'
          break

        case IntermediateEventType.LLM_THINKING:
          activity.currentAction = event.data.content_preview || 'Processing...'
          activity.status = 'thinking'
          break

        case IntermediateEventType.QUERY_GENERATED:
        case IntermediateEventType.QUERY_EXECUTING:
          const query = event.data.query || event.data.parameters?.query
          if (query && !activity.queries.includes(query)) {
            activity.queries.push(query)
          }
          activity.currentAction = `Searching: ${query || 'Processing query'}`
          activity.status = 'working'
          break

        case IntermediateEventType.TOOL_CALL_START:
          const toolQuery = event.data.parameters?.query
          if (toolQuery && !activity.queries.includes(toolQuery)) {
            activity.queries.push(toolQuery)
          }
          activity.currentAction = `🔍 ${event.data.action || 'Searching'}`
          activity.status = 'working'
          break

        case IntermediateEventType.SEARCH_RESULTS_FOUND:
        case IntermediateEventType.TOOL_CALL_COMPLETE:
          const resultCount = event.data.results_count || event.data.result_count || 0
          activity.currentAction = `✓ Found ${resultCount} results`
          if (event.data.result_summary && !activity.findings.includes(event.data.result_summary)) {
            activity.findings.push(event.data.result_summary)
          }
          break

        case IntermediateEventType.SYNTHESIS_PROGRESS:
          const preview = event.data.content_preview
          if (preview && !activity.findings.includes(preview)) {
            activity.findings.push(preview)
          }
          activity.currentAction = 'Analyzing findings...'
          activity.status = 'working'
          break

        case IntermediateEventType.AGENT_COMPLETE:
        case IntermediateEventType.ACTION_COMPLETE:
          activity.currentAction = `✓ ${agent} completed`
          activity.status = 'completed'
          break

        case IntermediateEventType.TOOL_CALL_ERROR:
          activity.currentAction = `⚠️ Error: ${event.data.error_message}`
          activity.status = 'error'
          break
      }
    })

    setActivities(Object.values(processedActivities))
  }, [events])

  const toggleActivity = (agent: string) => {
    const newExpanded = new Set(expandedActivities)
    if (newExpanded.has(agent)) {
      newExpanded.delete(agent)
    } else {
      newExpanded.add(agent)
    }
    setExpandedActivities(newExpanded)
  }

  const getStatusIcon = (status: AgentActivity['status']) => {
    switch (status) {
      case 'thinking':
        return <Brain className="w-4 h-4 animate-pulse" />
      case 'working':
        return <Zap className="w-4 h-4 animate-bounce" />
      case 'completed':
        return <CheckCircle className="w-4 h-4" />
      case 'error':
        return <AlertTriangle className="w-4 h-4" />
      default:
        return <Clock className="w-4 h-4" />
    }
  }

  if (!isActive || activities.length === 0) {
    return null
  }

  return (
    <div className={cn("space-y-2 mb-4", className)}>
      {activities.map((activity) => {
        const agentColorClass = getAgentColor(activity.agent)
        const agentIcon = getAgentIcon(activity.agent)

        return (
          <div
            key={activity.agent}
            className={cn(
              "border rounded-lg transition-all duration-200 hover:shadow-sm",
              `agent-badge ${agentColorClass}`
            )}
          >
            {/* Activity Header */}
            <div
              className="flex items-center gap-3 p-3 cursor-pointer"
              onClick={() => toggleActivity(activity.agent)}
            >
              <div className="flex items-center gap-2 flex-shrink-0">
                <span className="text-lg">{agentIcon}</span>
                {getStatusIcon(activity.status)}
              </div>

              <div className="flex-grow min-w-0">
                <div className="flex items-center gap-2">
                  <span className="font-medium text-sm">{activity.agent}</span>
                  {activity.status === 'thinking' && (
                    <div className="flex space-x-1">
                      <div className="w-1 h-1 bg-current rounded-full thinking-dot" />
                      <div className="w-1 h-1 bg-current rounded-full thinking-dot" />
                      <div className="w-1 h-1 bg-current rounded-full thinking-dot" />
                    </div>
                  )}
                </div>
                <p className="text-xs opacity-75 truncate">{activity.currentAction}</p>
              </div>

              <div className="flex-shrink-0">
                {expandedActivities.has(activity.agent) ? (
                  <ChevronUp className="w-4 h-4" />
                ) : (
                  <ChevronDown className="w-4 h-4" />
                )}
              </div>
            </div>

            {/* Expanded Details */}
            {expandedActivities.has(activity.agent) && (
              <div className="px-3 pb-3 space-y-3">
                {/* Queries */}
                {activity.queries.length > 0 && (
                  <div>
                    <h4 className="text-xs font-medium opacity-75 mb-1">Queries:</h4>
                    <div className="space-y-1">
                      {activity.queries.map((query, idx) => (
                        <div key={idx} className="flex items-start gap-2">
                          <Search className="w-3 h-3 mt-0.5 flex-shrink-0 opacity-60" />
                          <p className="text-xs opacity-80 break-words">{query}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Recent Findings */}
                {activity.findings.length > 0 && (
                  <div>
                    <h4 className="text-xs font-medium opacity-75 mb-1">Recent Findings:</h4>
                    <div className="space-y-1">
                      {activity.findings.slice(-3).map((finding, idx) => (
                        <div key={idx} className="flex items-start gap-2">
                          <CheckCircle className="w-3 h-3 mt-0.5 flex-shrink-0 opacity-60" />
                          <p className="text-xs opacity-80 break-words line-clamp-2">{finding}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Event Count */}
                <div className="text-xs opacity-50">
                  {activity.events.length} events • Last: {new Date(activity.events[activity.events.length - 1]?.timestamp * 1000).toLocaleTimeString()}
                </div>
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}
