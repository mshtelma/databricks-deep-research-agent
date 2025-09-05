/**
 * Smart table processor that fixes malformed tables while preserving structure
 */

interface TableCell {
  content: string
  isHeader: boolean
}

interface _TableRow {
  cells: TableCell[]
  isSeparator: boolean
}

/**
 * Parse a markdown table line into cells
 */
function parseTableRow(line: string): string[] {
  const trimmed = line.trim()
  if (!trimmed.startsWith('|') || !trimmed.endsWith('|')) {
    return []
  }
  
  // Remove leading and trailing pipes, then split
  const inner = trimmed.slice(1, -1)
  return inner.split('|').map(cell => cell.trim())
}

/**
 * Check if a line is a separator row
 */
function isSeparatorRow(line: string): boolean {
  const trimmed = line.trim()
  return /^[\|\s\-:]+$/.test(trimmed) && trimmed.includes('-')
}

/**
 * Smart table processor that fixes malformed streaming tables
 */
export function processTableContent(content: string): string {
  const lines = content.split('\n')
  const output: string[] = []
  let i = 0
  
  while (i < lines.length) {
    const line = lines[i]
    const trimmed = line.trim()
    
    // Check if this looks like a table row
    if (trimmed.startsWith('|') || isSeparatorRow(line)) {
      // Collect all consecutive table lines
      const tableLines: string[] = []
      let j = i
      
      while (j < lines.length) {
        const currentLine = lines[j]
        const currentTrimmed = currentLine.trim()
        
        // Include table rows and separators
        if (currentTrimmed.startsWith('|') || isSeparatorRow(currentLine)) {
          tableLines.push(currentLine)
          j++
        } else if (currentTrimmed === '' && j + 1 < lines.length) {
          // Check if table continues after empty line
          const nextLine = lines[j + 1]?.trim() || ''
          if (nextLine.startsWith('|') || isSeparatorRow(nextLine)) {
            tableLines.push(currentLine)
            j++
          } else {
            break
          }
        } else {
          break
        }
      }
      
      // Process the collected table
      if (tableLines.length > 0) {
        const processedTable = cleanupTable(tableLines)
        output.push(...processedTable)
        i = j
      } else {
        output.push(line)
        i++
      }
    } else {
      output.push(line)
      i++
    }
  }
  
  return output.join('\n')
}

/**
 * Clean up a table by removing duplicate separators and fixing structure
 */
function cleanupTable(tableLines: string[]): string[] {
  const cleaned: string[] = []
  let lastWasSeparator = false
  let columnCount = 0
  let hasHeader = false
  let hasSeparator = false
  
  // First pass: determine table structure
  for (const line of tableLines) {
    const cells = parseTableRow(line)
    if (cells.length > 0 && !isSeparatorRow(line)) {
      columnCount = Math.max(columnCount, cells.length)
    }
  }
  
  // Second pass: clean up the table
  for (let i = 0; i < tableLines.length; i++) {
    const line = tableLines[i]
    const trimmed = line.trim()
    
    // Skip empty lines within table
    if (!trimmed) {
      continue
    }
    
    // Handle separator rows
    if (isSeparatorRow(line)) {
      // Skip duplicate separators
      if (lastWasSeparator) {
        continue
      }
      
      // Skip malformed separators like "| --- |" without proper structure
      if (trimmed === '| --- |' || trimmed === '|---|') {
        continue
      }
      
      // Only add separator after header and if we haven't added one yet
      if (!hasSeparator && hasHeader) {
        // Create a proper separator with correct column count
        const separator = '|' + ' --- |'.repeat(columnCount)
        cleaned.push(separator)
        hasSeparator = true
        lastWasSeparator = true
      }
    } else if (trimmed.startsWith('|')) {
      // Regular table row
      const cells = parseTableRow(line)
      
      // Skip rows that are just broken fragments
      if (cells.length === 1 && cells[0] === '---') {
        continue
      }
      
      // Skip completely empty rows
      if (cells.every(cell => cell === '' || cell === '---')) {
        continue
      }
      
      // Add the row
      cleaned.push(line)
      
      if (!hasHeader) {
        hasHeader = true
      }
      
      lastWasSeparator = false
    }
  }
  
  return cleaned
}

/**
 * Process content for final display (after streaming is complete)
 */
export function finalizeTableContent(content: string): string {
  // Remove any leftover placeholders
  const lines = content.split('\n')
  const final: string[] = []
  let _skipPlaceholder = false
  
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    
    // Skip placeholder blocks
    if (line === '```' && i + 1 < lines.length) {
      const nextLine = lines[i + 1]
      if (nextLine.includes('📊') || nextLine.includes('Table loading') || nextLine.includes('Receiving table')) {
        // Skip the code block with placeholder
        i += 2 // Skip ``` and placeholder and closing ```
        while (i < lines.length && lines[i] !== '```') {
          i++
        }
        continue
      }
    }
    
    final.push(line)
  }
  
  // Process tables to fix any remaining issues
  return processTableContent(final.join('\n'))
}