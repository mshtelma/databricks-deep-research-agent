/**
 * Table Validator - Detects and reports malformed table patterns
 */

export interface TableValidationResult {
  isValid: boolean
  issues: TableIssue[]
  stats: TableStats
  suggestions: string[]
}

export interface TableIssue {
  type: TableIssueType
  severity: 'error' | 'warning' | 'info'
  line?: number
  column?: number
  message: string
  pattern?: string
}

export interface TableStats {
  tableCount: number
  totalLines: number
  malformedTables: number
  inlineTables: number
  properTables: number
}

export enum TableIssueType {
  DUPLICATE_SEPARATOR = 'duplicate_separator',
  INLINE_TABLE = 'inline_table',
  MERGED_ROWS = 'merged_rows',
  INCONSISTENT_COLUMNS = 'inconsistent_columns',
  ORPHANED_SEPARATOR = 'orphaned_separator',
  MALFORMED_SEPARATOR = 'malformed_separator',
  EMPTY_CELLS = 'empty_cells',
  MISSING_PIPES = 'missing_pipes',
}

/**
 * Patterns that indicate malformed tables
 */
const MALFORMED_PATTERNS = {
  // Multiple separator lines in a row
  duplicateSeparators: /(\| ?-+ ?\|.*\n){2,}/g,

  // Inline separators (all on one line)
  inlineSeparators: /\|.*\| ?-+ ?\|.*\| ?-+ ?\|/g,

  // Orphaned separator lines
  orphanedSeparators: /^\s*\| ?-+ ?\|\s*$/gm,

  // Mixed content and separators
  mergedHeaderSeparator: /\|[^|]*[A-Za-z][^|]*\| ?-+ ?\|/g,

  // Empty pipe cells
  emptyPipeCells: /\|\s*\|---\|/g,

  // Broken separator patterns
  brokenSeparators: /\|---\|---\|---\|---\|---\|---\|/g,
};

/**
 * Validate table content and return detailed results
 */
export function validateTableContent(content: string): TableValidationResult {
  // Handle null/undefined input
  if (!content) {
    return {
      isValid: true,
      issues: [],
      stats: { tableCount: 0, totalLines: 0, malformedTables: 0, inlineTables: 0, properTables: 0 },
      suggestions: []
    }
  }

  const lines = content.split('\n')
  const issues: TableIssue[] = []
  const stats: TableStats = {
    tableCount: 0,
    totalLines: lines.length,
    malformedTables: 0,
    inlineTables: 0,
    properTables: 0,
  }

  // Check for various malformed patterns
  checkDuplicateSeparators(content, issues)
  checkInlineTables(content, issues, stats)
  checkOrphanedSeparators(content, issues)
  checkMergedContent(content, issues)
  checkColumnConsistency(lines, issues)

  // Count tables
  const tables = detectTables(lines)
  stats.tableCount = tables.length

  for (const table of tables) {
    if (table.isProper) {
      stats.properTables++
    } else if (table.isInline) {
      stats.inlineTables++
    } else {
      stats.malformedTables++
    }
  }

  // Generate suggestions
  const suggestions = generateSuggestions(issues, stats)

  return {
    isValid: issues.filter(i => i.severity === 'error').length === 0,
    issues,
    stats,
    suggestions,
  }
}

/**
 * Check for duplicate separator patterns
 */
function checkDuplicateSeparators(content: string, issues: TableIssue[]) {
  const matches = content.match(MALFORMED_PATTERNS.duplicateSeparators)
  if (matches) {
    matches.forEach(match => {
      issues.push({
        type: TableIssueType.DUPLICATE_SEPARATOR,
        severity: 'error',
        message: 'Found duplicate separator lines in table',
        pattern: match.substring(0, 50) + '...',
      })
    })
  }
}

/**
 * Check for inline tables (headers, separators, and data on same line)
 */
function checkInlineTables(content: string, issues: TableIssue[], stats: TableStats) {
  const lines = content.split('\n')

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]

    // Check if line contains both text and separator patterns
    if (line.includes('| --- |') || line.includes('|---|')) {
      const segments = line.split('|').map(s => s.trim())
      const hasSeparators = segments.some(s => /^-+$/.test(s))
      const hasContent = segments.some(s => s && !/^-+$/.test(s) && s !== '')

      if (hasSeparators && hasContent) {
        issues.push({
          type: TableIssueType.INLINE_TABLE,
          severity: 'error',
          line: i + 1,
          message: 'Table headers/data and separators are on the same line',
          pattern: line.substring(0, 100) + '...',
        })
        stats.inlineTables++
      }
    }

    // Check for the specific broken pattern
    if (MALFORMED_PATTERNS.brokenSeparators.test(line)) {
      issues.push({
        type: TableIssueType.MALFORMED_SEPARATOR,
        severity: 'error',
        line: i + 1,
        message: 'Malformed separator pattern detected',
        pattern: line.substring(0, 100) + '...',
      })
    }
  }
}

/**
 * Check for orphaned separator lines
 */
function checkOrphanedSeparators(content: string, issues: TableIssue[]) {
  const lines = content.split('\n')

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim()

    // Check if this is a separator line
    if (/^\| ?-+ ?\|/.test(line) || line === '---') {
      // Check if there's a header before and data after
      const hasPrevLine = i > 0 && lines[i - 1].includes('|')
      const hasNextLine = i < lines.length - 1 && lines[i + 1].includes('|')

      if (!hasPrevLine && !hasNextLine) {
        issues.push({
          type: TableIssueType.ORPHANED_SEPARATOR,
          severity: 'warning',
          line: i + 1,
          message: 'Orphaned separator line with no associated table data',
        })
      }
    }
  }
}

/**
 * Check for merged content (multiple data items on same line)
 */
function checkMergedContent(content: string, issues: TableIssue[]) {
  const matches = content.match(MALFORMED_PATTERNS.mergedHeaderSeparator)
  if (matches) {
    matches.forEach(match => {
      issues.push({
        type: TableIssueType.MERGED_ROWS,
        severity: 'error',
        message: 'Table content and separators are merged on the same line',
        pattern: match.substring(0, 50) + '...',
      })
    })
  }
}

/**
 * Check for column consistency across table rows
 */
function checkColumnConsistency(lines: string[], issues: TableIssue[]) {
  const tables = detectTables(lines)

  for (const table of tables) {
    if (table.rows.length < 2) continue

    const columnCounts = table.rows.map(row => {
      const cells = row.split('|').map(c => c.trim()).filter(c => c !== '')
      return cells.length
    })

    const uniqueCounts = [...new Set(columnCounts)]
    if (uniqueCounts.length > 1) {
      issues.push({
        type: TableIssueType.INCONSISTENT_COLUMNS,
        severity: 'warning',
        line: table.startLine,
        message: `Table has inconsistent column counts: ${uniqueCounts.join(', ')} columns`,
      })
    }
  }
}

/**
 * Detect tables in content
 */
interface DetectedTable {
  startLine: number
  endLine: number
  rows: string[]
  isProper: boolean
  isInline: boolean
}

function detectTables(lines: string[]): DetectedTable[] {
  const tables: DetectedTable[] = []
  let currentTable: DetectedTable | null = null

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]

    if (line.includes('|')) {
      if (!currentTable) {
        currentTable = {
          startLine: i + 1,
          endLine: i + 1,
          rows: [line],
          isProper: false,
          isInline: false,
        }
      } else {
        currentTable.rows.push(line)
        currentTable.endLine = i + 1
      }
    } else if (currentTable) {
      // End of table
      analyzeTable(currentTable)
      tables.push(currentTable)
      currentTable = null
    }
  }

  if (currentTable) {
    analyzeTable(currentTable)
    tables.push(currentTable)
  }

  return tables
}

/**
 * Analyze a detected table to determine its type
 */
function analyzeTable(table: DetectedTable) {
  if (table.rows.length >= 3) {
    // Check if second row is a separator
    const secondRow = table.rows[1]
    if (/^\|[\s\-:]+\|/.test(secondRow.trim())) {
      table.isProper = true
    }
  }

  // Check if it's an inline table
  for (const row of table.rows) {
    if (row.includes('| --- |') || row.includes('|---|')) {
      const hasContent = row.split('|').some(cell => {
        const trimmed = cell.trim()
        return trimmed && !/^-+$/.test(trimmed)
      })
      if (hasContent) {
        table.isInline = true
        table.isProper = false
        break
      }
    }
  }
}

/**
 * Generate suggestions based on issues found
 */
function generateSuggestions(issues: TableIssue[], stats: TableStats): string[] {
  const suggestions: string[] = []

  if (stats.inlineTables > 0) {
    suggestions.push('Split inline tables so headers, separators, and data are on separate lines')
  }

  if (issues.some(i => i.type === TableIssueType.DUPLICATE_SEPARATOR)) {
    suggestions.push('Remove duplicate separator lines - each table should have only one separator row')
  }

  if (issues.some(i => i.type === TableIssueType.INCONSISTENT_COLUMNS)) {
    suggestions.push('Ensure all rows in a table have the same number of columns')
  }

  if (issues.some(i => i.type === TableIssueType.MERGED_ROWS)) {
    suggestions.push('Separate merged table rows - each row should be on its own line')
  }

  if (stats.malformedTables > 0) {
    suggestions.push('Review and fix malformed table structures')
  }

  return suggestions
}

/**
 * Fix common table issues automatically
 */
export function autoFixTableIssues(content: string): string {
  // Handle null/undefined input
  if (!content) {
    return content
  }

  let fixed = content

  // Fix duplicate separators
  fixed = fixed.replace(/(\| ?-+ ?\|.*\n){2,}/g, (match) => {
    const lines = match.split('\n')
    return lines[0] + '\n' // Keep only the first separator
  })

  // Fix inline tables and clean up noise lines
  const lines = fixed.split('\n')
  const processedLines: string[] = []
  let inTableRegion = false

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    const trimmed = line.trim()

    // Detect table region start/end
    if (trimmed.includes('|')) {
      inTableRegion = true
    } else if (inTableRegion && trimmed === '') {
      // Continue in table region through empty lines
    } else if (inTableRegion && !trimmed.includes('|') && !isNoiseTableLine(trimmed)) {
      inTableRegion = false
    }

    // Skip noise lines within table regions
    if (inTableRegion && isNoiseTableLine(trimmed)) {
      continue // Skip this line entirely
    }

    // Skip bare pipe-only lines
    if (trimmed === '|') {
      continue
    }

    if (line.includes('| --- |') || line.includes('|---|')) {
      const reconstructed = reconstructInlineTable(line)
      if (reconstructed) {
        processedLines.push(...reconstructed.split('\n'))
      } else {
        processedLines.push(line)
      }
    } else {
      processedLines.push(line)
    }
  }

  return processedLines.join('\n')
}

/**
 * Check if a line is table noise that should be removed
 */
function isNoiseTableLine(line: string): boolean {
  // Lines with only dashes and spaces (like "---    ---    ---")
  if (/^[-\s]+$/.test(line) && line.includes('---')) {
    return true
  }

  // Lines with repeated dash patterns but no pipes
  if (/^---(\s+---)+\s*$/.test(line)) {
    return true
  }

  // Lines with excessive separator patterns like "| --------- | ------------------------------ |"
  if (/^\|\s*-{5,}\s*\|\s*-{5,}/.test(line)) {
    return true
  }

  // Lines with just | --- | repeated excessively
  if (/^\|\s*---\s*\|\s*$/.test(line) || /^(\|\s*---\s*){3,}\|?\s*$/.test(line)) {
    return true
  }

  return false
}

/**
 * Reconstruct an inline table
 */
function reconstructInlineTable(line: string): string | null {
  const segments = line.split('|').map(s => s.trim()).filter(s => s)

  // Find separator segments
  const separatorIndices: number[] = []
  segments.forEach((seg, idx) => {
    if (/^-+$/.test(seg)) {
      separatorIndices.push(idx)
    }
  })

  if (separatorIndices.length === 0) return null

  // Extract parts
  const firstSepIndex = separatorIndices[0]
  const lastSepIndex = separatorIndices[separatorIndices.length - 1]

  const headers = segments.slice(0, firstSepIndex)
  const data = segments.slice(lastSepIndex + 1)

  if (headers.length === 0) return null

  // Build proper table
  const result: string[] = []

  // Add header row
  result.push('| ' + headers.join(' | ') + ' |')

  // Add separator row
  result.push('|' + headers.map(() => ' --- ').join('|') + '|')

  // Add data rows
  const columnCount = headers.length
  for (let i = 0; i < data.length; i += columnCount) {
    const row = data.slice(i, i + columnCount)
    while (row.length < columnCount) {
      row.push('') // Pad with empty cells
    }
    result.push('| ' + row.join(' | ') + ' |')
  }

  return result.join('\n')
}