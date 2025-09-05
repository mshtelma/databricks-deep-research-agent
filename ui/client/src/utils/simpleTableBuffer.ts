/**
 * Simple table buffering for streaming
 * Preserves table content while showing placeholders during streaming
 */

export interface SimpleTableBuffer {
  displayContent: string  // What to show during streaming (with placeholders)
  rawContent: string      // The actual raw content including tables
  hasTable: boolean
}

/**
 * Process streaming content with placeholders while preserving raw content
 * Returns both display content (with placeholders) and raw content (preserved)
 */
export function processStreamingWithPlaceholders(content: string): { display: string; raw: string } {
  // Always preserve the raw content
  const raw = content
  
  const lines = content.split('\n')
  const output: string[] = []
  let inTable = false
  let tableStartLine = -1
  
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    const trimmed = line.trim()
    
    // Detect table start
    if (!inTable && trimmed.includes('|') && (trimmed.startsWith('|') || trimmed.includes(' | '))) {
      inTable = true
      tableStartLine = i
      // Add placeholder once at the start of a table
      if (tableStartLine === i) {
        output.push('```')
        output.push('📊 Table is being received... Please wait for complete rendering.')
        output.push('```')
      }
    } else if (inTable) {
      // Check if still in table
      if (trimmed.includes('|') || trimmed === '') {
        // Still in table, don't output these lines (they're replaced by placeholder)
        continue
      } else {
        // Table ended
        inTable = false
        output.push(line) // Add the non-table line
      }
    } else {
      // Regular content
      output.push(line)
    }
  }
  
  return {
    display: output.join('\n'),
    raw: raw  // Preserve the original content with tables
  }
}

/**
 * Remove placeholders from content
 * This is no longer needed since we preserve raw content
 */
export function removePlaceholders(content: string): string {
  const lines = content.split('\n')
  const output: string[] = []
  let skipPlaceholder = false
  
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    
    if (line === '```' && i + 1 < lines.length) {
      const nextLine = lines[i + 1]
      if (nextLine.includes('📊') || nextLine.includes('Table is being received')) {
        // Skip placeholder block
        skipPlaceholder = true
        // Find closing ```
        for (let j = i + 2; j < lines.length; j++) {
          if (lines[j] === '```') {
            i = j
            skipPlaceholder = false
            break
          }
        }
        continue
      }
    }
    
    if (!skipPlaceholder) {
      output.push(line)
    }
  }
  
  return output.join('\n')
}