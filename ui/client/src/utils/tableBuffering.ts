/**
 * Table buffering utility for streaming content
 * Detects table boundaries and buffers complete tables before rendering
 */

export interface TableBuffer {
  inTable: boolean
  tableStartIndex: number
  tableLines: string[]
  currentBuffer: string
  lastProcessedIndex: number
}

/**
 * Detects if a line is likely the start of a markdown table
 */
function isTableStart(line: string, nextLine?: string): boolean {
  const trimmed = line.trim()
  
  // Check for pipe-delimited row
  if (trimmed.startsWith('|') && trimmed.endsWith('|')) {
    // Need at least 2 pipes (start and end) plus one separator
    const pipeCount = (trimmed.match(/\|/g) || []).length
    if (pipeCount >= 3) {
      // Check if next line is a separator
      if (nextLine) {
        const nextTrimmed = nextLine.trim()
        // Separator pattern: contains dashes and pipes
        if (nextTrimmed.match(/^\|?\s*[:?\-]+\s*(\|\s*[:?\-]+\s*)*\|?\s*$/)) {
          return true
        }
      }
      // Even without separator, could be table if it has proper structure
      return true
    }
  }
  
  return false
}

/**
 * Detects if a line is a table separator row
 */
function isTableSeparator(line: string): boolean {
  const trimmed = line.trim()
  // Match various separator formats
  return !!(
    trimmed.match(/^\|?\s*[:?\-]+\s*(\|\s*[:?\-]+\s*)*\|?\s*$/) ||
    trimmed.match(/^\|(\s*:?-+:?\s*\|)+\s*$/) ||
    trimmed.match(/^[\|\s\-:]+$/) && trimmed.includes('-') && trimmed.includes('|')
  )
}

/**
 * Detects if a line is part of a table row
 */
function isTableRow(line: string): boolean {
  const trimmed = line.trim()
  // Must have pipes and be formatted like a table row
  return trimmed.includes('|') && (
    (trimmed.startsWith('|') && trimmed.endsWith('|')) ||
    isTableSeparator(line)
  )
}

/**
 * Process streaming content and buffer tables
 */
export function processStreamingContent(
  content: string,
  buffer: TableBuffer
): { processedContent: string; updatedBuffer: TableBuffer } {
  const lines = content.split('\n')
  const outputLines: string[] = []
  const newBuffer = { ...buffer }
  
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    const nextLine = lines[i + 1]
    
    if (!newBuffer.inTable) {
      // Not in a table, check if we're starting one
      if (isTableStart(line, nextLine)) {
        newBuffer.inTable = true
        newBuffer.tableStartIndex = outputLines.length
        newBuffer.tableLines = [line]
        // Add placeholder for table while buffering
        outputLines.push('```')
        outputLines.push('📊 Receiving table data...')
        outputLines.push('```')
      } else {
        // Regular content, pass through
        outputLines.push(line)
      }
    } else {
      // We're in a table
      if (isTableRow(line) || isTableSeparator(line)) {
        // Still in table, buffer the line
        newBuffer.tableLines.push(line)
      } else if (line.trim() === '') {
        // Empty line might be table end or just spacing
        // Look ahead to see if table continues
        let tableContines = false
        for (let j = i + 1; j < Math.min(i + 3, lines.length); j++) {
          if (isTableRow(lines[j])) {
            tableContines = true
            break
          }
        }
        
        if (tableContines) {
          // Table continues, buffer the empty line
          newBuffer.tableLines.push(line)
        } else {
          // Table ended, flush the buffer
          const completeTable = newBuffer.tableLines.join('\n')
          
          // Replace placeholder with actual table
          outputLines.splice(newBuffer.tableStartIndex, 3, completeTable)
          
          // Reset buffer
          newBuffer.inTable = false
          newBuffer.tableLines = []
          newBuffer.tableStartIndex = -1
          
          // Add the current line (empty line after table)
          outputLines.push(line)
        }
      } else {
        // Non-table content, table has ended
        const completeTable = newBuffer.tableLines.join('\n')
        
        // Replace placeholder with actual table
        outputLines.splice(newBuffer.tableStartIndex, 3, completeTable)
        
        // Reset buffer
        newBuffer.inTable = false
        newBuffer.tableLines = []
        newBuffer.tableStartIndex = -1
        
        // Add the current line
        outputLines.push(line)
      }
    }
  }
  
  // Update current buffer with remaining content
  newBuffer.currentBuffer = outputLines.join('\n')
  
  return {
    processedContent: outputLines.join('\n'),
    updatedBuffer: newBuffer
  }
}

/**
 * Create initial table buffer state
 */
export function createTableBuffer(): TableBuffer {
  return {
    inTable: false,
    tableStartIndex: -1,
    tableLines: [],
    currentBuffer: '',
    lastProcessedIndex: 0
  }
}

/**
 * Finalize buffered content when streaming ends
 */
export function finalizeBufferedContent(buffer: TableBuffer): string {
  if (buffer.inTable && buffer.tableLines.length > 0) {
    // We have an incomplete table at the end, render it as-is
    const lines = buffer.currentBuffer.split('\n')
    const completeTable = buffer.tableLines.join('\n')
    
    // Replace placeholder with actual table
    lines.splice(buffer.tableStartIndex, 3, completeTable)
    
    return lines.join('\n')
  }
  
  return buffer.currentBuffer
}

/**
 * Check if content has an incomplete table at the end
 */
export function hasIncompleteTable(content: string): boolean {
  const lines = content.split('\n')
  let inTable = false
  let tableHasData = false
  
  for (let i = lines.length - 1; i >= Math.max(0, lines.length - 10); i--) {
    const line = lines[i]
    
    if (isTableRow(line)) {
      inTable = true
      if (!isTableSeparator(line)) {
        tableHasData = true
      }
    } else if (line.trim() !== '' && inTable) {
      // Found non-table content before table, so table is complete
      return false
    }
  }
  
  // If we found a table at the end with data but no clear ending, it might be incomplete
  return inTable && tableHasData
}