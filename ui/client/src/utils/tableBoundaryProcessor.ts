/**
 * Table Boundary Processor
 * 
 * Handles tables wrapped in boundary markers for safe streaming and processing.
 * Boundary format:
 * <!-- TABLE_START -->
 * | table | content |
 * <!-- TABLE_END -->
 */

import { simplifiedTableProcessor } from './simplifiedTableProcessor'

export interface BoundaryProcessorResult {
  processed: string
  tablesFound: number
  boundariesRemoved: number
  issues: string[]
}

/**
 * Process content with table boundaries
 */
export function processTableBoundaries(content: string): BoundaryProcessorResult {
  if (!content) {
    return {
      processed: content,
      tablesFound: 0,
      boundariesRemoved: 0,
      issues: []
    }
  }

  const issues: string[] = []
  let tablesFound = 0
  let boundariesRemoved = 0

  // Pattern to match table boundaries
  const tablePattern = /<!-- TABLE_START(?:\s+cols:\d+\s+rows:\d+)?\s*-->([\s\S]*?)<!-- TABLE_END -->/g
  
  // Process each bounded table
  let processed = content.replace(tablePattern, (match, tableContent) => {
    tablesFound++
    boundariesRemoved += 2 // Start and end markers
    
    // Validate the table content
    const validation = validateBoundedTable(tableContent)
    if (validation.issues.length > 0) {
      issues.push(...validation.issues)
    }
    
    // Process the table content with simplified processor
    const result = simplifiedTableProcessor(tableContent.trim())
    
    // Return processed table without boundaries
    return result.processed
  })
  
  // Check for unmatched boundaries
  const unmatchedStarts = (processed.match(/<!-- TABLE_START/g) || []).length
  const unmatchedEnds = (processed.match(/<!-- TABLE_END -->/g) || []).length
  
  if (unmatchedStarts > 0) {
    issues.push(`Found ${unmatchedStarts} unmatched TABLE_START markers`)
    // Remove unmatched starts
    processed = processed.replace(/<!-- TABLE_START(?:\s+cols:\d+\s+rows:\d+)?\s*-->/g, '')
    boundariesRemoved += unmatchedStarts
  }
  
  if (unmatchedEnds > 0) {
    issues.push(`Found ${unmatchedEnds} unmatched TABLE_END markers`)
    // Remove unmatched ends
    processed = processed.replace(/<!-- TABLE_END -->/g, '')
    boundariesRemoved += unmatchedEnds
  }
  
  return {
    processed,
    tablesFound,
    boundariesRemoved,
    issues
  }
}

/**
 * Validate a bounded table
 */
function validateBoundedTable(content: string): { isValid: boolean; issues: string[] } {
  const issues: string[] = []
  const lines = content.trim().split('\n').filter(line => line.trim())
  
  if (lines.length === 0) {
    issues.push('Empty table within boundaries')
    return { isValid: false, issues }
  }
  
  // Check for proper table structure
  let hasHeader = false
  let hasSeparator = false
  let hasData = false
  
  for (const line of lines) {
    const trimmed = line.trim()
    
    // Check for header (first non-empty line should be header)
    if (!hasHeader && trimmed.startsWith('|') && trimmed.endsWith('|')) {
      hasHeader = true
      continue
    }
    
    // Check for separator
    if (hasHeader && !hasSeparator && /^\|\s*(-+\s*\|)+\s*$/.test(trimmed)) {
      hasSeparator = true
      continue
    }
    
    // Check for data
    if (hasHeader && hasSeparator && trimmed.startsWith('|') && trimmed.endsWith('|')) {
      hasData = true
    }
    
    // Check for malformed patterns
    if (trimmed.includes('|---|')) {
      issues.push('Condensed separator pattern detected within bounded table')
    }
    
    if (trimmed.includes('| --- |') && trimmed.split('|').some(cell => {
      const c = cell.trim()
      return c && c !== '---' && c !== ''
    })) {
      issues.push('Mixed content and separators on same line within bounded table')
    }
  }
  
  // Validate structure
  if (!hasHeader) {
    issues.push('No valid header found in bounded table')
  }
  if (!hasSeparator && lines.length > 1) {
    issues.push('No separator row found in bounded table')
  }
  if (!hasData && lines.length > 2) {
    issues.push('No data rows found in bounded table')
  }
  
  return {
    isValid: issues.length === 0,
    issues
  }
}

/**
 * Add boundaries to tables in content
 */
export function addTableBoundaries(content: string): string {
  if (!content) return content
  
  // Pattern to match tables (simple heuristic)
  const lines = content.split('\n')
  const result: string[] = []
  let inTable = false
  let tableBuffer: string[] = []
  
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    const trimmed = line.trim()
    const nextLine = i < lines.length - 1 ? lines[i + 1].trim() : ''
    
    // Check if this looks like a table start
    if (!inTable && trimmed.startsWith('|') && trimmed.endsWith('|') && 
        nextLine.match(/^\|\s*(-+\s*\|)+\s*$/)) {
      // Start of table
      inTable = true
      result.push('<!-- TABLE_START -->')
      tableBuffer = [line]
    } else if (inTable && trimmed.startsWith('|') && trimmed.endsWith('|')) {
      // Continue table
      tableBuffer.push(line)
    } else if (inTable && trimmed === '') {
      // Might be end of table or just empty line in table
      tableBuffer.push(line)
    } else if (inTable) {
      // End of table
      result.push(...tableBuffer)
      result.push('<!-- TABLE_END -->')
      result.push(line)
      inTable = false
      tableBuffer = []
    } else {
      // Regular content
      result.push(line)
    }
  }
  
  // Handle table at end of content
  if (inTable && tableBuffer.length > 0) {
    result.push(...tableBuffer)
    result.push('<!-- TABLE_END -->')
  }
  
  return result.join('\n')
}

/**
 * Check if content has table boundaries
 */
export function hasTableBoundaries(content: string): boolean {
  return content.includes('<!-- TABLE_START') && content.includes('<!-- TABLE_END -->')
}

/**
 * Extract metadata from table boundary markers
 */
export function extractTableMetadata(boundary: string): { cols?: number; rows?: number } | null {
  const match = boundary.match(/<!-- TABLE_START\s+cols:(\d+)\s+rows:(\d+)\s*-->/)
  if (match) {
    return {
      cols: parseInt(match[1], 10),
      rows: parseInt(match[2], 10)
    }
  }
  return null
}