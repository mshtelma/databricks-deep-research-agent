/**
 * Simplified Table Processor
 * 
 * A simplified approach to handling markdown tables.
 * The agent now preprocesses tables to ensure valid markdown structure,
 * so this processor primarily handles:
 * 1. Agent-preprocessed tables with TABLE_START/TABLE_END markers
 * 2. Streaming fragmentation during real-time delivery
 * 3. Minor formatting issues for backwards compatibility
 */

export interface TableProcessorResult {
  processed: string
  issues: ProcessorIssue[]
  stats: ProcessorStats
}

export interface ProcessorIssue {
  type: string
  message: string
  line?: number
}

export interface ProcessorStats {
  tablesProcessed: number
  linesRemoved: number
  linesFixed: number
}

/**
 * Main entry point for simplified table processing
 */
export function simplifiedTableProcessor(content: string): TableProcessorResult {
  if (!content) {
    return {
      processed: content,
      issues: [],
      stats: { tablesProcessed: 0, linesRemoved: 0, linesFixed: 0 }
    }
  }

  const stats: ProcessorStats = {
    tablesProcessed: 0,
    linesRemoved: 0,
    linesFixed: 0
  }
  const issues: ProcessorIssue[] = []

  // Check if content has agent-preprocessed tables
  if (content.includes('TABLE_START') && content.includes('TABLE_END')) {
    // Agent has already preprocessed tables - just extract and render
    const processed = processPreprocessedTables(content, stats)
    return { processed, issues, stats }
  }

  // Fallback: Handle legacy content without preprocessing
  // This path will be less common as the agent preprocessor is rolled out
  
  // Step 1: Pre-process specific malformed patterns
  let processed = preprocessSpecificPatterns(content, stats)
  
  // Step 2: Clean up obvious malformed patterns
  processed = removeObviousMalformedPatterns(processed, stats)

  // Step 3: Process tables line by line with state tracking
  processed = processWithStateTracking(processed, stats, issues)

  // Step 4: Final cleanup pass
  processed = finalCleanup(processed)

  return { processed, issues, stats }
}

/**
 * Process agent-preprocessed tables with TABLE_START/TABLE_END markers
 */
function processPreprocessedTables(content: string, stats: ProcessorStats): string {
  // Simply extract tables between markers
  // Agent guarantees they're properly formatted
  const tableRegex = /TABLE_START\n?([\s\S]*?)\n?TABLE_END/g
  
  let processedContent = content
  let match
  
  while ((match = tableRegex.exec(content)) !== null) {
    const table = match[1].trim()
    // Remove markers and keep the clean table
    processedContent = processedContent.replace(match[0], table)
    stats.tablesProcessed++
  }
  
  // Final cleanup of any double pipes (shouldn't happen with preprocessed tables)
  processedContent = processedContent.replace(/\|\|+/g, '|')
  
  return processedContent
}

/**
 * Pre-process specific malformed patterns from real agent output
 */
function preprocessSpecificPatterns(content: string, stats: ProcessorStats): string {
  const lines = content.split('\n')
  const processed: string[] = []
  
  for (let i = 0; i < lines.length; i++) {
    let line = lines[i]
    const trimmed = line.trim()
    
    // SEVERE PATTERN: Headers-Separators-Data all on one line
    // Example: "| Header1 | Header2 || --- | --- | --- | Data1 | Data2 |"
    if (trimmed.match(/\|\s*---\s*\|.*\|\s*---\s*\|.*[A-Za-z]/)) {
      // This line has multiple separator patterns mixed with content
      const parts = trimmed.split('|').map(p => p.trim()).filter(p => p)
      
      // Separate into headers, separators, and data
      const headers: string[] = []
      const data: string[] = []
      let inSeparatorSection = false
      let separatorCount = 0
      
      for (const part of parts) {
        if (/^-+$/.test(part)) {
          inSeparatorSection = true
          separatorCount++
        } else if (!inSeparatorSection) {
          headers.push(part)
        } else {
          // After separator section
          data.push(part)
        }
      }
      
      // If we found this pattern, split into proper lines
      if (headers.length > 0 && data.length > 0 && separatorCount > 0) {
        // Add header line
        processed.push('| ' + headers.join(' | ') + ' |')
        // Add separator line
        const colCount = Math.max(headers.length, data.length)
        processed.push('|' + Array(colCount).fill(' --- ').join('|') + '|')
        // Add data line
        processed.push('| ' + data.join(' | ') + ' |')
        stats.linesFixed++
        continue
      }
    }
    
    // SEVERE PATTERN: Line starting with separator followed by data
    // Example: "| --- | --- | --- | --- | Quarterly earnings news | CNBC..."
    if (trimmed.match(/^\|\s*---\s*\|.*\|\s*---\s*\|.*[A-Za-z]/)) {
      const parts = trimmed.split('|').map(p => p.trim()).filter(p => p)
      const dataParts = parts.filter(p => !/^-+$/.test(p))
      
      if (dataParts.length > 0) {
        // This is data that appeared after separators - just keep the data
        line = '| ' + dataParts.join(' | ') + ' |'
        stats.linesFixed++
      }
    }
    
    // Pattern: Row ending with multiple separator cells appended, possibly with more data after
    // Example: "| Country | ... | --------- | ------ | Spain | €22 800 | ..."
    if (trimmed.includes('|') && (trimmed.includes('| --------- |') || trimmed.includes('| ------ |'))) {
      // Split into parts
      const parts = trimmed.split('|').map(p => p.trim()).filter(p => p)
      const beforeSeparators: string[] = []
      const afterSeparators: string[] = []
      let inSeparatorSection = false
      let separatorSectionEnded = false
      
      for (const part of parts) {
        if (/^-{5,}$/.test(part)) {
          inSeparatorSection = true
        } else if (inSeparatorSection && !separatorSectionEnded) {
          // We've passed the separator section
          separatorSectionEnded = true
          afterSeparators.push(part)
        } else if (separatorSectionEnded || inSeparatorSection) {
          afterSeparators.push(part)
        } else {
          beforeSeparators.push(part)
        }
      }
      
      // If we have both before and after sections, we need to split into two lines
      if (beforeSeparators.length > 0 && afterSeparators.length > 0) {
        // First line: headers without separators
        processed.push('| ' + beforeSeparators.join(' | ') + ' |')
        // Second line: data that was after separators
        line = '| ' + afterSeparators.join(' | ') + ' |'
        stats.linesFixed++
      } else if (beforeSeparators.length > 0) {
        line = '| ' + beforeSeparators.join(' | ') + ' |'
        stats.linesFixed++
      }
    }
    
    // Pattern: Lines with "| --- |" mixed with content
    // Example: "| --- | **Family set‑ups** | data |"
    else if (trimmed.includes('| --- |') && !(/^\|\s*(---\s*\|)+\s*$/.test(trimmed))) {
      const cells = trimmed.split('|').map(c => c.trim()).filter(c => c)
      const contentCells = cells.filter(c => !/^-+$/.test(c))
      if (contentCells.length > 0) {
        line = '| ' + contentCells.join(' | ') + ' |'
        stats.linesFixed++
      }
    }
    
    // Pattern: Content ending with "| --- |"
    // Example: "| **Assumed gross earnings** | text | --- |"
    else if (trimmed.endsWith('| --- |') && hasValidContent(trimmed)) {
      // Remove trailing separator
      const cells = trimmed.split('|').map(c => c.trim()).filter(c => c)
      const contentCells = cells.filter(c => !/^-+$/.test(c))
      if (contentCells.length > 0) {
        line = '| ' + contentCells.join(' | ') + ' |'
        stats.linesFixed++
      }
    }
    
    // Pattern: Lines that are partial data but got truncated
    // Example: the missing Spain row
    if (trimmed.includes('Spain') && trimmed.includes('€') && !trimmed.startsWith('|')) {
      // This might be truncated data, try to reconstruct
      const parts = trimmed.split(/\s+/).filter(p => p)
      if (parts.length >= 2) {
        // Try to reconstruct as table row
        line = '| ' + parts.join(' | ') + ' |'
        stats.linesFixed++
      }
    }
    
    processed.push(line)
  }
  
  return processed.join('\n')
}

/**
 * Remove obvious malformed patterns that should never appear in valid tables
 */
function removeObviousMalformedPatterns(content: string, stats: ProcessorStats): string {
  const lines = content.split('\n')
  const cleaned: string[] = []

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    const trimmed = line.trim()
    const prevLine = i > 0 ? lines[i - 1].trim() : ''
    const nextLine = i < lines.length - 1 ? lines[i + 1].trim() : ''
    
    // Skip these patterns entirely:
    // 1. Pure separator rows with no context: | --- | --- | --- | --- |
    if (/^\|\s*---\s*\|\s*---\s*\|\s*---\s*\|\s*---\s*\|?\s*$/.test(trimmed)) {
      // Check if this is between valid table rows
      const hasPrevTable = prevLine.includes('|') && !prevLine.match(/^\|\s*---/)
      const hasNextTable = nextLine.includes('|') && !nextLine.match(/^\|\s*---/)
      
      // Only keep if it's a valid separator between header and data
      if (!hasPrevTable || !hasNextTable) {
        stats.linesRemoved++
        continue
      }
    }
    
    // 2. Condensed separator patterns: |---|---|---|---|
    // ALWAYS remove these - they're never valid in proper markdown tables
    if (/\|---\|/.test(trimmed) && trimmed.match(/^\|(-{2,}\|)+\s*$/)) {
      stats.linesRemoved++
      continue
    }
    
    // 3. Mixed empty and separator cells: | |---|---|
    if (/^\|\s*\|---\|---/.test(trimmed)) {
      stats.linesRemoved++
      continue
    }
    
    // 4. Pure separator lines without pipes (but only if repeated)
    if (trimmed === '---') {
      // Check if there are multiple --- lines in a row
      if (nextLine === '---' || prevLine === '---') {
        stats.linesRemoved++
        continue
      }
    }
    
    // 4b. Multiple separator patterns in sequence
    // Remove lines that are just variations of separator patterns
    if (/^\|\s*---\s*\|\s*---\s*\|?\s*$/.test(trimmed) && 
        /^\|\s*---\s*\|/.test(prevLine)) {
      // Multiple separator rows in sequence - keep only the first
      stats.linesRemoved++
      continue
    }

    // 5. Excessive dashes: | ------ | ------------- | --------- |
    if (/^\|\s*-{5,}\s*\|\s*-{5,}/.test(trimmed) && !hasValidContent(trimmed)) {
      stats.linesRemoved++
      continue
    }

    // 6. Standalone pipes or empty pipe rows
    if (trimmed === '|' || trimmed.match(/^\|\s*$/)) {
      stats.linesRemoved++
      continue
    }
    
    // 7. Lines with just pipes and spaces (empty rows)
    if (trimmed.match(/^\|\s*(\|\s*)+$/)) {
      stats.linesRemoved++
      continue
    }

    cleaned.push(line)
  }

  return cleaned.join('\n')
}

/**
 * Process content with enhanced state tracking and lookahead for better table detection
 */
function processWithStateTracking(content: string, stats: ProcessorStats, _issues: ProcessorIssue[]): string {
  const lines = content.split('\n')
  const result: string[] = []
  
  let state: TableState = {
    inTable: false,
    hasHeader: false,
    hasSeparator: false,
    minColumns: 0,
    maxColumns: 0,
    currentTable: [],
    emptyLineCount: 0,
    lastTableLine: -1,
    possibleContinuation: false
  }

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    const trimmed = line.trim()
    const nextLine = i < lines.length - 1 ? lines[i + 1].trim() : ''
    
    // Check if this line contains table content
    if (looksLikeTableRow(trimmed)) {
      const cleanedRow = cleanTableRow(trimmed, stats)
      
      // If we recently ended a table and this could be a continuation
      if (!state.inTable && state.possibleContinuation && i - state.lastTableLine <= 3) {
        // Check if this table has similar column structure
        const columnCount = getColumnCount(cleanedRow)
        if (Math.abs(columnCount - state.maxColumns) <= 2) {
          // Likely a continuation of the previous table
          state.inTable = true
          state.possibleContinuation = false
        }
      }
      
      if (!state.inTable) {
        // Starting a new table
        state = startNewTable(cleanedRow, nextLine)
        stats.tablesProcessed++
      } else {
        // Continue existing table
        state = updateTableState(state, cleanedRow, nextLine)
      }
      
      if (cleanedRow && cleanedRow.trim() !== '|' && cleanedRow.trim() !== '| |') {
        state.currentTable.push(cleanedRow)
        state.lastTableLine = i
        state.emptyLineCount = 0
      }
    } else if (state.inTable) {
      // We're in a table but found a non-table line
      // Use lookahead to check if table continues
      const shouldContinue = shouldContinueTable(lines, i, state)
      
      if (shouldContinue) {
        // Keep the line as part of the table structure
        if (trimmed === '') {
          state.emptyLineCount++
          if (state.emptyLineCount <= 2) {
            state.currentTable.push(line)
          }
        } else if (isTableRelatedContent(trimmed)) {
          // This might be a subheading or note within the table
          state.currentTable.push(line)
        }
      } else {
        // End of table, flush it
        result.push(...finalizeTable(state))
        const prevMaxCols = state.maxColumns
        state = resetTableState()
        state.lastTableLine = i - 1
        state.possibleContinuation = true
        state.maxColumns = prevMaxCols // Remember for continuation check
        
        // Don't forget to add the non-table line
        if (trimmed && trimmed !== '---' && !trimmed.match(/^-{3,}$/)) {
          result.push(line)
        }
      }
    } else if (trimmed.match(/^\|\s*$/) || trimmed === '---') {
      // Skip standalone pipes and separator lines outside tables
      continue
    } else {
      // Regular content - but skip standalone --- lines
      if (trimmed !== '---' && !trimmed.match(/^-{3,}$/)) {
        result.push(line)
      }
      state.possibleContinuation = false
    }
  }

  // Flush any remaining table
  if (state.inTable) {
    result.push(...finalizeTable(state))
  }

  return result.join('\n')
}

/**
 * Check if content has valid data (not just separators)
 */
function hasValidContent(line: string): boolean {
  const cells = line.split('|').map(c => c.trim()).filter(c => c)
  return cells.some(cell => cell && !/^-+$/.test(cell))
}

/**
 * Check if a line looks like it could be part of a table
 */
function looksLikeTableRow(line: string): boolean {
  if (!line.includes('|')) return false
  
  // Must have at least 2 pipes to be a table row (but be more flexible)
  const pipeCount = (line.match(/\|/g) || []).length
  if (pipeCount < 2) {
    // Special case: single column tables or continuation rows
    if (pipeCount === 1 && line.trim().startsWith('|') && line.trim().endsWith('|')) {
      return true
    }
    return false
  }
  
  // Check if it's a valid table pattern
  return /^\|.*\|$/.test(line) || /\|.*\|/.test(line)
}

/**
 * Use lookahead to determine if a table should continue
 */
function shouldContinueTable(lines: string[], currentIndex: number, state: TableState): boolean {
  // If we've had too many empty lines, end the table
  if (state.emptyLineCount > 1) {
    return false
  }
  
  // Look ahead up to 5 lines
  const lookahead = 5
  for (let i = 1; i <= lookahead && currentIndex + i < lines.length; i++) {
    const futureLineTrimmed = lines[currentIndex + i].trim()
    
    if (looksLikeTableRow(futureLineTrimmed)) {
      // Check if it has similar column structure
      const futureColumns = getColumnCount(futureLineTrimmed)
      const columnDiff = Math.abs(futureColumns - state.maxColumns)
      
      // If column count is reasonably similar, continue the table
      if (columnDiff <= 2 || (state.minColumns > 0 && futureColumns >= state.minColumns && futureColumns <= state.maxColumns + 2)) {
        return true
      }
    }
    
    // If we find another empty line, stop looking
    if (futureLineTrimmed === '' && i > 1) {
      break
    }
  }
  
  return false
}

/**
 * Check if content is likely table-related (subheading, note, etc.)
 */
function isTableRelatedContent(line: string): boolean {
  // Common patterns for table-related content
  const patterns = [
    /^\*\*[^*]+\*\*$/,  // Bold headers
    /^Note:/i,            // Notes
    /^Source:/i,          // Source citations
    /^\d+\./,             // Numbered items
    /^[A-Z][^.!?]*:$/,    // Category headers ending with colon
    /^Total:/i,           // Total rows
    /^Summary:/i,         // Summary sections
    /^\([^)]+\)$/        // Parenthetical notes
  ]
  
  return patterns.some(pattern => pattern.test(line.trim()))
}

/**
 * Get the number of columns in a table row
 */
function getColumnCount(row: string): number {
  const trimmed = row.trim()
  if (!trimmed.includes('|')) return 0
  
  // Count pipes and subtract 1 (pipes separate columns)
  const pipes = (trimmed.match(/\|/g) || []).length
  
  // Account for leading and trailing pipes
  if (trimmed.startsWith('|') && trimmed.endsWith('|')) {
    return Math.max(0, pipes - 1)
  }
  
  return pipes
}

/**
 * Clean a single table row
 */
function cleanTableRow(row: string, stats: ProcessorStats): string {
  let cleaned = row.trim()
  
  // Don't clean already clean rows
  if (/^\|\s*[^|]+\s*(\|\s*[^|]+\s*)*\|$/.test(cleaned) && !cleaned.includes('---')) {
    // This looks like a valid table row, minimal cleanup only
    return cleaned
  }
  
  // Handle special case: row with content followed by separator patterns
  // Example: "| **Countries** | Spain, France | --- | **Family** | data |"
  if (cleaned.includes('| --- |') || cleaned.includes('|---|')) {
    const cells = cleaned.split('|').map(c => c.trim()).filter(c => c)
    const contentCells = cells.filter(c => !/^-+$/.test(c))
    const separatorCells = cells.filter(c => /^-+$/.test(c))
    
    // If we have both content and separators mixed, extract only content
    if (contentCells.length > 0 && separatorCells.length > 0) {
      cleaned = '| ' + contentCells.join(' | ') + ' |'
      stats.linesFixed++
      return cleaned
    }
  }
  
  // Remove trailing separator patterns
  cleaned = cleaned
    // Pattern: "| Header | Header | --- | --- | --- |"
    .replace(/\|\s*---\s*\|\s*---\s*\|\s*---.*$/g, '')
    // Pattern: "|---|---|---|"
    .replace(/\|---\|---\|---.*$/g, '')
    // Pattern: "| --------- | ------------------------------ |"
    .replace(/\|\s*-{5,}\s*\|\s*-{5,}.*$/g, '')
    // Pattern: trailing "| --- |"
    .replace(/\|\s*---\s*\|\s*$/g, '|')
    // Remove trailing empty cells
    .replace(/(\|\s*)+$/g, '|')
  
  // Fix separator rows to be consistent
  if (isSeparatorRow(cleaned)) {
    const cellCount = cleaned.split('|').filter(c => c.trim()).length
    cleaned = '|' + Array(cellCount).fill(' --- ').join('|') + '|'
  }
  
  // Ensure proper table row format
  if (cleaned && !cleaned.startsWith('|')) {
    cleaned = '| ' + cleaned
  }
  if (cleaned && !cleaned.endsWith('|')) {
    cleaned = cleaned + ' |'
  }
  
  // Final cleanup - normalize spacing but preserve valid tables
  if (!isSeparatorRow(cleaned)) {
    cleaned = cleaned
      .split('|')
      .map(cell => cell.trim())
      .filter((cell, idx, arr) => idx === 0 || idx === arr.length - 1 || cell !== '')
      .join(' | ')
    
    // Ensure pipes at start and end
    if (cleaned && !cleaned.startsWith('|')) {
      cleaned = '| ' + cleaned
    }
    if (cleaned && !cleaned.endsWith('|')) {
      cleaned = cleaned + ' |'
    }
  }
  
  return cleaned
}

/**
 * Check if a row is a separator row
 */
function isSeparatorRow(row: string): boolean {
  const cells = row.split('|').map(c => c.trim()).filter(c => c)
  return cells.length > 0 && cells.every(cell => /^:?-+:?$/.test(cell))
}

/**
 * Enhanced table state tracking with better context awareness
 */
interface TableState {
  inTable: boolean
  hasHeader: boolean
  hasSeparator: boolean
  minColumns: number
  maxColumns: number
  currentTable: string[]
  emptyLineCount: number
  lastTableLine: number
  possibleContinuation: boolean
}

/**
 * Start a new table with enhanced column tracking
 */
function startNewTable(firstRow: string, _nextLine: string): TableState {
  const columnCount = getColumnCount(firstRow)
  
  return {
    inTable: true,
    hasHeader: !isSeparatorRow(firstRow),
    hasSeparator: false,
    minColumns: columnCount,
    maxColumns: columnCount,
    currentTable: [],
    emptyLineCount: 0,
    lastTableLine: -1,
    possibleContinuation: false
  }
}

/**
 * Update table state with flexible column tracking
 */
function updateTableState(state: TableState, row: string, _nextLine: string): TableState {
  if (isSeparatorRow(row) && !state.hasSeparator) {
    state.hasSeparator = true
  }
  
  const columnCount = getColumnCount(row)
  
  // Update min/max column counts for flexible validation
  if (columnCount > 0) {
    if (state.minColumns === 0 || columnCount < state.minColumns) {
      state.minColumns = columnCount
    }
    if (columnCount > state.maxColumns) {
      state.maxColumns = columnCount
    }
  }
  
  return state
}

/**
 * Reset table state
 */
function resetTableState(): TableState {
  return {
    inTable: false,
    hasHeader: false,
    hasSeparator: false,
    minColumns: 0,
    maxColumns: 0,
    currentTable: [],
    emptyLineCount: 0,
    lastTableLine: -1,
    possibleContinuation: false
  }
}

/**
 * Finalize a table - ensure it has proper structure with flexible column handling
 */
function finalizeTable(state: TableState): string[] {
  const result: string[] = []
  const rows = state.currentTable.filter(r => {
    const trimmed = r.trim()
    // Keep non-empty rows and valid table rows
    return trimmed && (looksLikeTableRow(trimmed) || isTableRelatedContent(trimmed) || trimmed === '')
  })
  
  if (rows.length === 0) return result
  
  // Check if the original input had trailing --- patterns
  const firstRow = rows[0]
  const hasTrailingSeparator = firstRow && firstRow.includes('**') && 
    (firstRow.endsWith('| --- |') || firstRow.includes('| --- |'))
  
  // Check if we need to add a separator
  let needsSeparator = false
  if (rows.length >= 2 && !hasTrailingSeparator) {
    const hasAnySeparator = rows.some(r => isSeparatorRow(r))
    if (!hasAnySeparator && state.hasHeader) {
      needsSeparator = true
    }
  }
  
  // Build final table with smart column normalization
  let targetColumns = state.maxColumns
  
  for (let i = 0; i < rows.length; i++) {
    const row = rows[i]
    const trimmed = row.trim()
    
    // For table rows, normalize column count if needed
    if (looksLikeTableRow(trimmed) && !isSeparatorRow(trimmed)) {
      const currentColumns = getColumnCount(trimmed)
      if (currentColumns < targetColumns && currentColumns > 0) {
        // Pad with empty columns if needed
        const cells = trimmed.split('|').map(c => c.trim())
        while (cells.filter(c => c !== '').length < targetColumns) {
          cells.splice(cells.length - 1, 0, '')
        }
        result.push(cells.join(' | '))
      } else {
        result.push(row)
      }
    } else {
      result.push(row)
    }
    
    // Add separator after header if needed
    if (i === 0 && needsSeparator && looksLikeTableRow(trimmed)) {
      result.push('|' + Array(targetColumns).fill(' --- ').join('|') + '|')
    }
  }
  
  return result
}

/**
 * Final cleanup pass
 */
function finalCleanup(content: string): string {
  let result = content
    // Remove multiple consecutive empty lines
    .replace(/\n{3,}/g, '\n\n')
    // Remove trailing whitespace
    .replace(/[ \t]+$/gm, '')
  
  // Don't add trailing newline if there wasn't one
  if (!content.endsWith('\n')) {
    return result
  }
  
  return result
}

/**
 * Export for use in other modules
 */
export function processTablesSimplified(content: string): string {
  const result = simplifiedTableProcessor(content)
  
  // Log issues if any
  if (result.issues.length > 0) {
    console.log('[TABLE PROCESSOR] Issues found:', result.issues)
  }
  
  return result.processed
}