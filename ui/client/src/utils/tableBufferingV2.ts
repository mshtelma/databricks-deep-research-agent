/**
 * Enhanced table buffering utility for streaming content
 * Properly preserves table structure during streaming
 */

export interface TableBufferV2 {
  inTable: boolean
  tableLines: string[]
  bufferedContent: string[]
  lastWasTable: boolean
}

/**
 * Create initial table buffer state
 */
export function createTableBufferV2(): TableBufferV2 {
  return {
    inTable: false,
    tableLines: [],
    bufferedContent: [],
    lastWasTable: false
  }
}

/**
 * Check if a line is a markdown table row
 */
function isTableLine(line: string): boolean {
  const trimmed = line.trim()
  
  // Empty lines are not table lines
  if (!trimmed) return false
  
  // Must contain at least one pipe
  if (!trimmed.includes('|')) return false
  
  // Check for table row pattern (starts and ends with |, or is a separator)
  const isRow = trimmed.startsWith('|') && trimmed.endsWith('|')
  const isSeparator = /^[\|\s\-:]+$/.test(trimmed) && trimmed.includes('-')
  
  return isRow || isSeparator
}

/**
 * Check if a line is a table separator
 */
function isTableSeparator(line: string): boolean {
  const trimmed = line.trim()
  // More lenient separator detection
  return /^[\|\s\-:]+$/.test(trimmed) && trimmed.includes('-') && trimmed.includes('|')
}

/**
 * Process streaming content with improved table buffering
 */
export function processStreamingContentV2(
  content: string,
  buffer: TableBufferV2
): { processedContent: string; updatedBuffer: TableBufferV2 } {
  const lines = content.split('\n')
  const newBuffer: TableBufferV2 = {
    ...buffer,
    bufferedContent: []
  }
  
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    const trimmed = line.trim()
    
    // Check if this line is part of a table
    const isTable = isTableLine(line)
    
    if (!newBuffer.inTable && isTable) {
      // Starting a new table
      newBuffer.inTable = true
      newBuffer.tableLines = [line]
      newBuffer.lastWasTable = true
    } else if (newBuffer.inTable && isTable) {
      // Continue building the table
      newBuffer.tableLines.push(line)
    } else if (newBuffer.inTable && !isTable) {
      // Table might be ending, but check if it's just an empty line within the table
      if (trimmed === '' && i + 1 < lines.length) {
        // Look ahead to see if the table continues
        const nextLine = lines[i + 1]
        if (isTableLine(nextLine)) {
          // Table continues, include the empty line
          newBuffer.tableLines.push(line)
          continue
        }
      }
      
      // Table has ended, add the complete table
      if (newBuffer.tableLines.length > 0) {
        // Add table placeholder while streaming
        newBuffer.bufferedContent.push('```')
        newBuffer.bufferedContent.push('📊 Table loading...')
        newBuffer.bufferedContent.push('```')
        newBuffer.bufferedContent.push('')
      }
      
      // Add the non-table line
      newBuffer.bufferedContent.push(line)
      newBuffer.inTable = false
      newBuffer.lastWasTable = false
    } else {
      // Regular content, not in a table
      newBuffer.bufferedContent.push(line)
      newBuffer.lastWasTable = false
    }
  }
  
  // If we're still in a table at the end, keep buffering
  if (newBuffer.inTable && newBuffer.tableLines.length > 0) {
    // Show placeholder for incomplete table
    newBuffer.bufferedContent.push('```')
    newBuffer.bufferedContent.push('📊 Receiving table data...')
    newBuffer.bufferedContent.push('```')
  }
  
  return {
    processedContent: newBuffer.bufferedContent.join('\n'),
    updatedBuffer: newBuffer
  }
}

/**
 * Finalize content when streaming completes
 */
export function finalizeBufferedContentV2(
  content: string,
  buffer: TableBufferV2
): string {
  // If there's an incomplete table at the end, render it
  if (buffer.inTable && buffer.tableLines.length > 0) {
    const lines = content.split('\n')
    const result: string[] = []
    let skipTable = false
    
    for (const line of lines) {
      // Skip placeholder blocks
      if (line === '```') {
        skipTable = !skipTable
        continue
      }
      if (skipTable || line.includes('📊')) {
        continue
      }
      result.push(line)
    }
    
    return result.join('\n')
  }
  
  // Remove any table placeholders from final content
  const lines = content.split('\n')
  const result: string[] = []
  let _skipPlaceholder = false
  
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    
    if (line === '```' && i + 1 < lines.length && lines[i + 1].includes('📊')) {
      _skipPlaceholder = true
      i += 2 // Skip the placeholder and closing ```
      continue
    }
    
    result.push(line)
  }
  
  return result.join('\n')
}

/**
 * Clean markdown content to remove any processing artifacts
 */
export function cleanMarkdownContent(content: string): string {
  // Remove duplicate separators and clean up table formatting
  const lines = content.split('\n')
  const cleaned: string[] = []
  let lastWasSeparator = false
  
  for (const line of lines) {
    const isSep = isTableSeparator(line)
    
    // Skip duplicate separators
    if (isSep && lastWasSeparator) {
      continue
    }
    
    // Skip malformed separator patterns
    if (line.trim() === '| --- |' || line.trim() === '|---|') {
      continue
    }
    
    cleaned.push(line)
    lastWasSeparator = isSep
  }
  
  return cleaned.join('\n')
}