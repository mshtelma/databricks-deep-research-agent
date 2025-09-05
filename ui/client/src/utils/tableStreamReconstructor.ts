/**
 * Table Stream Reconstructor
 * 
 * Intelligently reconstructs markdown tables from streaming chunks,
 * handling cases where tables are split across multiple delta events.
 */

export interface TableBoundary {
  startIndex: number
  endIndex: number
  isComplete: boolean
  content: string
}

export interface ReconstructionResult {
  display: string      // Content to display (with placeholders for incomplete tables)
  raw: string         // Raw accumulated content
  hasIncompleteTable: boolean
  tables: TableBoundary[]
}

export class TableStreamReconstructor {
  private buffer: string = ''
  private completedContent: string = ''
  private tableBuffer: string[] = []
  private inTable: boolean = false
  private currentTableStartLine: number = -1
  private minColumns: number = 0
  private maxColumns: number = 0
  private emptyLineCount: number = 0
  private lastTableContent: string = ''
  private recentTableColumns: number = 0
  
  /**
   * Add a streaming chunk and reconstruct tables
   */
  addChunk(chunk: string): ReconstructionResult {
    this.buffer += chunk
    
    // Process complete lines
    const lines = this.buffer.split('\n')
    const incompleteLine = lines.pop() || '' // Keep last incomplete line in buffer
    this.buffer = incompleteLine
    
    for (const line of lines) {
      this.processLine(line)
    }
    
    return this.getResult()
  }
  
  /**
   * Finalize processing and get remaining content
   */
  finalize(): ReconstructionResult {
    // Process any remaining buffer content
    if (this.buffer) {
      this.processLine(this.buffer)
      this.buffer = ''
    }
    
    // Flush any incomplete table
    if (this.inTable && this.tableBuffer.length > 0) {
      this.flushTable(false)
    }
    
    return this.getResult()
  }
  
  /**
   * Process a single line with enhanced table detection
   */
  private processLine(line: string): void {
    const isTableLine = this.isTableRow(line)
    const trimmed = line.trim()
    
    if (isTableLine) {
      const columnCount = this.getColumnCount(line)
      
      // Check if this might be a continuation of a recent table
      if (!this.inTable && this.recentTableColumns > 0 && 
          Math.abs(columnCount - this.recentTableColumns) <= 2) {
        // Likely a continuation
        this.inTable = true
        this.tableBuffer = [line]
        this.minColumns = columnCount
        this.maxColumns = columnCount
      } else if (!this.inTable) {
        // Start of a new table
        this.inTable = true
        this.tableBuffer = [line]
        this.minColumns = columnCount
        this.maxColumns = columnCount
      } else {
        // Continue building table
        this.tableBuffer.push(line)
        // Update column range
        if (columnCount < this.minColumns) this.minColumns = columnCount
        if (columnCount > this.maxColumns) this.maxColumns = columnCount
      }
      
      this.emptyLineCount = 0
    } else if (this.inTable) {
      // We're in a table but found a non-table line
      if (trimmed === '') {
        this.emptyLineCount++
        // Allow up to 2 empty lines within a table
        if (this.emptyLineCount <= 2) {
          this.tableBuffer.push(line)
        } else {
          // Too many empty lines, end the table
          this.flushTable(true)
          this.completedContent += line + '\n'
        }
      } else if (this.isTableRelatedContent(trimmed)) {
        // This might be a subheading or note within the table
        this.tableBuffer.push(line)
        this.emptyLineCount = 0
      } else {
        // Table has ended
        this.flushTable(true)
        this.completedContent += line + '\n'
      }
    } else {
      // Regular non-table line
      this.completedContent += line + '\n'
      // Reset continuation tracking if we've moved far from the last table
      if (this.emptyLineCount > 3) {
        this.recentTableColumns = 0
      }
      this.emptyLineCount++
    }
  }
  
  /**
   * Flush the current table buffer with enhanced handling
   */
  private flushTable(isComplete: boolean): void {
    if (this.tableBuffer.length === 0) return
    
    // Filter out excessive empty lines but keep structure
    const filteredBuffer = this.tableBuffer.filter((line, idx) => {
      const trimmed = line.trim()
      // Keep non-empty lines and limited empty lines for spacing
      if (trimmed) return true
      // Keep first empty line after content for spacing
      if (idx > 0 && this.tableBuffer[idx - 1].trim()) return true
      return false
    })
    
    const tableContent = filteredBuffer.join('\n')
    
    if (isComplete && this.isValidTable(tableContent)) {
      // Complete and valid table
      this.completedContent += tableContent + '\n'
      this.lastTableContent = tableContent
      this.recentTableColumns = this.maxColumns
    } else if (isComplete) {
      // Complete but potentially malformed - try to fix
      const fixed = this.fixTableIssues(tableContent)
      this.completedContent += fixed + '\n'
      this.lastTableContent = fixed
      this.recentTableColumns = this.maxColumns
    } else {
      // Incomplete table - add with placeholder
      this.completedContent += '```\n📊 Table is being received... Please wait for complete rendering.\n```\n'
      // Keep raw table in a hidden section for later reconstruction
      this.completedContent += `<!-- INCOMPLETE_TABLE_START\n${tableContent}\nINCOMPLETE_TABLE_END -->\n`
    }
    
    this.tableBuffer = []
    this.inTable = false
    this.minColumns = 0
    this.maxColumns = 0
    this.emptyLineCount = 0
  }
  
  /**
   * Get current reconstruction result
   */
  private getResult(): ReconstructionResult {
    const tables = this.extractTables(this.completedContent)
    const hasIncompleteTable = this.inTable && this.tableBuffer.length > 0
    
    let display = this.completedContent
    
    // If there's an incomplete table being built, add placeholder
    if (hasIncompleteTable) {
      display += '\n```\n📊 Table is being received... Please wait for complete rendering.\n```\n'
    }
    
    return {
      display,
      raw: this.completedContent + (this.tableBuffer.length > 0 ? '\n' + this.tableBuffer.join('\n') : ''),
      hasIncompleteTable,
      tables
    }
  }
  
  /**
   * Check if a line is a table row
   */
  private isTableRow(line: string): boolean {
    const trimmed = line.trim()
    if (!trimmed || !trimmed.includes('|')) return false
    
    // Check for basic table structure
    const pipeCount = (trimmed.match(/\|/g) || []).length
    if (pipeCount < 2) return false
    
    // Should ideally start and end with pipes (but be flexible for streaming)
    if (trimmed.startsWith('|') || trimmed.includes(' | ')) {
      return true
    }
    
    // Check if it's a separator row
    if (/^[\s\|\-:]+$/.test(trimmed) && trimmed.includes('-')) {
      return true
    }
    
    return false
  }
  
  /**
   * Check if a table is valid with flexible column validation
   */
  private isValidTable(content: string): boolean {
    const lines = content.split('\n').filter(l => {
      const trimmed = l.trim()
      return trimmed && (this.isTableRow(trimmed) || this.isTableRelatedContent(trimmed))
    })
    
    if (lines.length < 2) return false
    
    // Find header and separator (may not be consecutive)
    let headerIdx = -1
    let separatorIdx = -1
    
    for (let i = 0; i < Math.min(lines.length, 5); i++) {
      if (headerIdx === -1 && this.isTableRow(lines[i]) && !this.isSeparatorRow(lines[i])) {
        headerIdx = i
      }
      if (separatorIdx === -1 && this.isSeparatorRow(lines[i])) {
        separatorIdx = i
      }
    }
    
    if (headerIdx === -1 || separatorIdx === -1) return false
    
    const header = lines[headerIdx]
    const separator = lines[separatorIdx]
    
    // Check column consistency with more flexibility
    const headerCols = this.getColumnCount(header)
    const separatorCols = this.getColumnCount(separator)
    
    // Allow up to 2 column difference for flexibility
    if (Math.abs(headerCols - separatorCols) > 2) return false
    
    return true
  }
  
  /**
   * Check if a line is a separator row
   */
  private isSeparatorRow(line: string): boolean {
    const trimmed = line.trim()
    if (!trimmed.includes('|')) return false
    
    // Remove pipes and check if content is mostly dashes
    const cells = trimmed.split('|').map(c => c.trim()).filter(c => c)
    if (cells.length === 0) return false
    
    // All cells should be separator patterns
    return cells.every(cell => /^:?-+:?$/.test(cell))
  }
  
  /**
   * Check if content is likely table-related
   */
  private isTableRelatedContent(line: string): boolean {
    const patterns = [
      /^\*\*[^*]+\*\*$/,     // Bold headers
      /^Note:/i,               // Notes
      /^Source:/i,             // Source citations  
      /^\d+\./,                // Numbered items
      /^[A-Z][^.!?]*:$/,       // Category headers
      /^Total:/i,              // Total rows
      /^Summary:/i,            // Summary sections
      /^\([^)]+\)$/,          // Parenthetical notes
      /^[A-Z]{2,}/             // Country codes, abbreviations
    ]
    
    return patterns.some(pattern => pattern.test(line))
  }
  
  /**
   * Get the number of columns in a table row
   */
  private getColumnCount(row: string): number {
    const trimmed = row.trim()
    if (!trimmed.includes('|')) return 0
    
    // Count pipes and account for structure
    const pipes = (trimmed.match(/\|/g) || []).length
    
    // Account for leading and trailing pipes
    if (trimmed.startsWith('|') && trimmed.endsWith('|')) {
      return Math.max(1, pipes - 1)
    }
    
    return Math.max(1, pipes)
  }
  
  /**
   * Fix common table issues with enhanced patterns
   */
  private fixTableIssues(content: string): string {
    let fixed = content
    
    // Fix double pipes
    fixed = fixed.replace(/\|\|+/g, '|')
    
    // Fix inline separators (separators mixed with content)
    fixed = fixed.replace(/(\|[^|\n]+)\|(---+\|)/g, '$1\n$2')
    fixed = fixed.replace(/\| --- \| --- \| ([A-Za-z])/g, '\n| $1')
    
    // Fix condensed separators
    fixed = fixed.replace(/\|---|---|---|/g, '| --- | --- | --- |')
    
    // Ensure proper spacing around pipes
    fixed = fixed.replace(/\|([^\s|])/g, '| $1')
    fixed = fixed.replace(/([^\s|])\|/g, '$1 |')
    
    // Remove trailing separators from data rows
    fixed = fixed.replace(/(\|[^|\n]+)\| --- \|$/gm, '$1 |')
    
    return fixed
  }
  
  /**
   * Extract table boundaries from content
   */
  private extractTables(content: string): TableBoundary[] {
    const tables: TableBoundary[] = []
    const lines = content.split('\n')
    
    let inTable = false
    let tableStart = -1
    let tableLines: string[] = []
    
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i]
      
      if (this.isTableRow(line)) {
        if (!inTable) {
          inTable = true
          tableStart = i
          tableLines = [line]
        } else {
          tableLines.push(line)
        }
      } else if (inTable) {
        // Table ended
        const tableContent = tableLines.join('\n')
        tables.push({
          startIndex: tableStart,
          endIndex: i - 1,
          isComplete: this.isValidTable(tableContent),
          content: tableContent
        })
        
        inTable = false
        tableStart = -1
        tableLines = []
      }
    }
    
    // Handle table at end of content
    if (inTable && tableLines.length > 0) {
      const tableContent = tableLines.join('\n')
      tables.push({
        startIndex: tableStart,
        endIndex: lines.length - 1,
        isComplete: this.isValidTable(tableContent),
        content: tableContent
      })
    }
    
    return tables
  }
  
  /**
   * Reset the reconstructor
   */
  reset(): void {
    this.buffer = ''
    this.completedContent = ''
    this.tableBuffer = []
    this.inTable = false
    this.currentTableStartLine = -1
    this.minColumns = 0
    this.maxColumns = 0
    this.emptyLineCount = 0
    this.lastTableContent = ''
    this.recentTableColumns = 0
  }
}

/**
 * Enhanced streaming processor that uses the reconstructor
 */
export function processStreamingWithTableReconstruction(content: string): ReconstructionResult {
  const reconstructor = new TableStreamReconstructor()
  
  // Process the entire content as if it was streamed
  const lines = content.split('\n')
  for (const line of lines) {
    reconstructor.addChunk(line + '\n')
  }
  
  return reconstructor.finalize()
}

/**
 * Validate if content has properly formed tables
 */
export function validateTableIntegrity(content: string): {
  valid: boolean
  issues: string[]
} {
  const issues: string[] = []
  
  // Check for double pipes
  if (content.includes('||')) {
    issues.push('Contains double pipes (||)')
  }
  
  // Check for broken separators
  if (/\|---\|---\|---\|---\|---\|/.test(content)) {
    issues.push('Contains broken separator pattern')
  }
  
  // Check for inline separators
  if (/\w+.*\|---/.test(content)) {
    issues.push('Contains inline separator with content')
  }
  
  // Extract and validate each table
  const reconstructor = new TableStreamReconstructor()
  const result = processStreamingWithTableReconstruction(content)
  
  for (const table of result.tables) {
    if (!table.isComplete) {
      issues.push(`Incomplete table at line ${table.startIndex}`)
    }
    
    // Check column consistency
    const lines = table.content.split('\n')
    if (lines.length >= 2) {
      const headerCols = (lines[0].match(/\|/g) || []).length - 1
      for (let i = 1; i < lines.length; i++) {
        const lineCols = (lines[i].match(/\|/g) || []).length - 1
        if (Math.abs(lineCols - headerCols) > 1) {
          issues.push(`Column mismatch in table at line ${table.startIndex + i}`)
        }
      }
    }
  }
  
  return {
    valid: issues.length === 0,
    issues
  }
}