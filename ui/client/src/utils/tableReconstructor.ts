/**
 * Table reconstructor - completely rebuilds malformed tables
 * This takes a different approach: extract data, then rebuild clean tables
 */

interface TableData {
  headers: string[]
  rows: string[][]
}

/**
 * Extract table data from malformed markdown
 */
function extractTableData(lines: string[]): TableData | null {
  const headers: string[] = []
  const dataRows: string[][] = []
  let foundHeader = false
  let _skipNextSeparator = false
  
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    const trimmed = line.trim()
    
    // Skip empty lines
    if (!trimmed) continue
    
    // Skip broken separator patterns
    if (trimmed === '| --- | --- | --- | --- | --- | --- |') continue
    if (trimmed === '|---|---|---|---|---|---|') continue
    if (/^\|\s*-+\s*\|\s*-+\s*\|\s*-+\s*\|\s*-+\s*\|\s*-+\s*\|\s*-+\s*\|/.test(trimmed)) continue
    
    // Check if this is a separator row (should come after header)
    const isSeparator = /^[\|\s\-:]+$/.test(trimmed) && trimmed.includes('-')
    
    if (isSeparator) {
      if (!foundHeader) {
        _skipNextSeparator = true
      }
      continue // Skip all separator rows
    }
    
    // Process table rows
    if (trimmed.includes('|')) {
      let cells: string[] = []
      
      // Handle rows that start with |
      if (trimmed.startsWith('|') && trimmed.endsWith('|')) {
        cells = trimmed
          .slice(1, -1) // Remove leading/trailing pipes
          .split('|')
          .map(cell => cell.trim())
      } else if (trimmed.includes(' | ')) {
        // Handle rows without leading/trailing pipes
        cells = trimmed.split('|').map(cell => cell.trim())
      } else {
        continue // Skip malformed rows
      }
      
      // Filter out separator fragments and empty cells at the end
      cells = cells.filter((cell, idx) => {
        // Keep non-empty cells that aren't just '---'
        if (cell === '---') return false
        // Keep empty cells in the middle but not trailing ones
        if (cell === '' && idx === cells.length - 1) return false
        return true
      })
      
      // Skip if no valid cells
      if (cells.length === 0) continue
      
      // Skip rows that are just fragments like "| --- | --- | --- | --- | --- | --- | --- |"
      if (cells.every(cell => cell === '---' || cell === '')) continue
      
      // First valid row with enough cells becomes headers
      if (!foundHeader && cells.length >= 3) {
        headers.push(...cells)
        foundHeader = true
      } else if (foundHeader && cells.length > 0) {
        // Only add rows that have actual data
        if (cells.some(cell => cell !== '' && cell !== '---')) {
          dataRows.push(cells)
        }
      }
    }
  }
  
  // Return null if no valid table found
  if (headers.length === 0) return null
  
  return { headers, rows: dataRows }
}

/**
 * Rebuild a clean markdown table from extracted data
 */
function rebuildTable(data: TableData): string[] {
  const { headers, rows } = data
  const result: string[] = []
  
  // Ensure all rows have same number of columns
  const columnCount = headers.length
  
  // Build header row
  result.push('| ' + headers.join(' | ') + ' |')
  
  // Build separator row
  result.push('| ' + headers.map(() => '---').join(' | ') + ' |')
  
  // Build data rows
  for (const row of rows) {
    // Pad row to match column count
    const paddedRow = [...row]
    while (paddedRow.length < columnCount) {
      paddedRow.push('')
    }
    // Truncate if too many columns
    paddedRow.length = columnCount
    
    result.push('| ' + paddedRow.join(' | ') + ' |')
  }
  
  return result
}

/**
 * Process content and reconstruct all tables
 */
export function reconstructTables(content: string): string {
  const lines = content.split('\n')
  const output: string[] = []
  let i = 0
  
  while (i < lines.length) {
    const line = lines[i]
    const trimmed = line.trim()
    
    // Check if this might be the start of a table
    if (trimmed.includes('|') && (trimmed.startsWith('|') || trimmed.includes(' | '))) {
      // Collect all potential table lines
      const tableLines: string[] = []
      let j = i
      
      while (j < lines.length) {
        const currentLine = lines[j].trim()
        
        // Continue collecting if line contains pipes
        if (currentLine.includes('|')) {
          tableLines.push(lines[j])
          j++
        } else if (currentLine === '' && j + 1 < lines.length) {
          // Check if table continues after empty line
          const nextLine = lines[j + 1].trim()
          if (nextLine.includes('|')) {
            j++ // Skip empty line
            continue
          } else {
            break // Table ended
          }
        } else {
          break // No more table lines
        }
      }
      
      // Try to extract and rebuild table
      const tableData = extractTableData(tableLines)
      if (tableData && tableData.headers.length > 0) {
        // Add rebuilt table
        const rebuiltTable = rebuildTable(tableData)
        output.push(...rebuiltTable)
        output.push('') // Add empty line after table
        i = j
      } else {
        // Not a valid table, keep original line
        output.push(line)
        i++
      }
    } else {
      // Regular content
      output.push(line)
      i++
    }
  }
  
  return output.join('\n')
}

/**
 * Clean up content before reconstruction
 */
export function cleanupBeforeReconstruction(content: string): string {
  // Remove code block placeholders for tables
  const lines = content.split('\n')
  const cleaned: string[] = []
  let inPlaceholder = false
  
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    
    // Skip placeholder blocks
    if (line === '```' && i + 1 < lines.length) {
      const nextLine = lines[i + 1]
      if (nextLine.includes('📊') || nextLine.includes('Table loading') || nextLine.includes('Receiving table')) {
        // Skip entire placeholder block
        inPlaceholder = true
        // Find closing ```
        for (let j = i + 2; j < lines.length; j++) {
          if (lines[j] === '```') {
            i = j
            inPlaceholder = false
            break
          }
        }
        continue
      }
    }
    
    if (!inPlaceholder) {
      cleaned.push(line)
    }
  }
  
  return cleaned.join('\n')
}

/**
 * Main processing function for final table content
 */
export function processTablesFinal(content: string): string {
  // Step 1: Clean up any placeholders
  const cleaned = cleanupBeforeReconstruction(content)
  
  // Step 2: Reconstruct all tables
  const reconstructed = reconstructTables(cleaned)
  
  return reconstructed
}

/**
 * Simple table detection for buffering during streaming
 */
export function detectTableBoundaries(content: string): { start: number; end: number }[] {
  const lines = content.split('\n')
  const tables: { start: number; end: number }[] = []
  let tableStart = -1
  
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim()
    
    if (line.includes('|') && tableStart === -1) {
      // Potential table start
      tableStart = i
    } else if (tableStart !== -1 && !line.includes('|') && line !== '') {
      // Table ended
      tables.push({ start: tableStart, end: i - 1 })
      tableStart = -1
    }
  }
  
  // Handle table at end
  if (tableStart !== -1) {
    tables.push({ start: tableStart, end: lines.length - 1 })
  }
  
  return tables
}