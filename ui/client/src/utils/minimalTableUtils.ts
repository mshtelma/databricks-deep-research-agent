/**
 * Minimal table utilities - do NOT modify table content
 * Only provide detection and minimal cleanup without destroying structure
 */

/**
 * Minimal preprocessing - only handle critical issues without modifying tables
 */
export function preprocessMarkdownMinimal(input: string): string {
  if (!input) return input
  
  // Only normalize line endings, nothing else
  return input.replace(/\r\n/g, '\n')
}

/**
 * Check if content appears to contain a markdown table
 */
export function containsTable(content: string): boolean {
  const lines = content.split('\n')
  let pipeLines = 0
  let separatorFound = false
  
  for (const line of lines) {
    const trimmed = line.trim()
    
    // Count lines with table structure
    if (trimmed.startsWith('|') && trimmed.endsWith('|') && trimmed.split('|').length >= 3) {
      pipeLines++
    }
    
    // Check for separator
    if (/^[\|\s\-:]+$/.test(trimmed) && trimmed.includes('-') && trimmed.includes('|')) {
      separatorFound = true
    }
  }
  
  // Need at least 2 pipe lines and a separator for a valid table
  return pipeLines >= 2 && separatorFound
}