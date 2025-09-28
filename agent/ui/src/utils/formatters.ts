import { format } from 'date-fns'

export function formatTime(timestamp: number | Date): string {
  const date = typeof timestamp === 'number' ? new Date(timestamp * 1000) : timestamp
  return format(date, 'HH:mm:ss')
}

export function formatRelativeTime(timestamp: number): string {
  const now = Date.now()
  const diff = now - (timestamp * 1000)

  if (diff < 1000) return 'Just now'
  if (diff < 60000) return `${Math.floor(diff / 1000)}s ago`
  if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`
  return formatTime(timestamp)
}

export function getAgentIcon(agent: string): string {
  const agentLower = agent.toLowerCase()
  if (agentLower.includes('coordinator')) return '🎯'
  if (agentLower.includes('planner')) return '📋'
  if (agentLower.includes('researcher') || agentLower.includes('research')) return '🔬'
  if (agentLower.includes('fact') || agentLower.includes('checker')) return '🔎'
  if (agentLower.includes('reporter')) return '📄'
  if (agentLower.includes('background')) return '🔍'
  return '🤖'
}

export function getAgentColor(agent: string): string {
  const agentLower = agent.toLowerCase()
  if (agentLower.includes('coordinator')) return 'coordinator'
  if (agentLower.includes('planner')) return 'planner'
  if (agentLower.includes('researcher') || agentLower.includes('research')) return 'researcher'
  if (agentLower.includes('fact') || agentLower.includes('checker')) return 'checker'
  if (agentLower.includes('reporter')) return 'reporter'
  return 'coordinator' // Default
}

export function formatConfidenceScore(score?: number): string {
  if (!score) return 'N/A'
  return `${Math.round(score * 100)}%`
}

export function formatSourceCount(count: number): string {
  if (count === 0) return 'No sources'
  if (count === 1) return '1 source'
  return `${count} sources`
}