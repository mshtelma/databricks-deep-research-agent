/**
 * Simplified table utilities that don't destroy valid markdown tables
 * Only fixes truly broken patterns without aggressive removal
 */

/**
 * Minimal preprocessing for markdown - only fix critical issues
 */
export function preprocessMarkdownSimple(input: string): string {
  if (!input) return input
  
  let text = input
  
  // Normalize Windows newlines
  text = text.replace(/\r\n/g, '\n')
  
  // Handle <br> tags in table cells - convert to line breaks within cells
  text = text.replace(/(<br\s*\/?>)/gi, '  \n')
  
  // Only fix obviously broken patterns
  const lines = text.split('\n')
  const cleaned: string[] = []
  
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    const trimmed = line.trim()
    
    // Skip only clearly malformed fragments like "| |---|---|" (empty first cell with separator)
    if (trimmed.match(/^\|\s*\|---\|/)) {
      continue
    }
    
    // Keep all other lines including valid table separators
    cleaned.push(line)
  }
  
  return cleaned.join('\n')
}

/**
 * Check if content appears to have a table
 */
export function hasTable(content: string): boolean {
  const lines = content.split('\n')
  let pipeCount = 0
  let separatorFound = false
  
  for (const line of lines) {
    const trimmed = line.trim()
    
    // Check for table rows
    if (trimmed.startsWith('|') && trimmed.endsWith('|')) {
      pipeCount++
    }
    
    // Check for separator
    if (trimmed.match(/^\|?\s*[:?\-]+\s*(\|\s*[:?\-]+\s*)*\|?\s*$/)) {
      separatorFound = true
    }
    
    // If we have both pipes and a separator, it's likely a table
    if (pipeCount >= 2 && separatorFound) {
      return true
    }
  }
  
  return false
}

/**
 * Extract tables from content and return them separately
 * This allows special handling of tables if needed
 */
export function extractTables(content: string): { tables: string[]; contentWithPlaceholders: string } {
  const lines = content.split('\n')
  const tables: string[] = []
  const output: string[] = []
  let currentTable: string[] = []
  let inTable = false
  
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    const trimmed = line.trim()
    const nextLine = lines[i + 1]?.trim() || ''
    
    // Check if we're starting a table
    if (!inTable && trimmed.startsWith('|') && trimmed.endsWith('|')) {
      // Look for separator in next line
      if (nextLine.match(/^\|?\s*[:?\-]+\s*(\|\s*[:?\-]+\s*)*\|?\s*$/)) {
        inTable = true
        currentTable = [line]
      } else {
        output.push(line)
      }
    } else if (inTable) {
      // Check if still in table
      if (trimmed.startsWith('|') || trimmed.match(/^\|?\s*[:?\-]+\s*(\|\s*[:?\-]+\s*)*\|?\s*$/)) {
        currentTable.push(line)
      } else if (trimmed === '') {
        // Empty line might be end of table
        // Look ahead to see if table continues
        let tableContinues = false
        for (let j = i + 1; j < Math.min(i + 3, lines.length); j++) {
          const checkLine = lines[j].trim()
          if (checkLine.startsWith('|')) {
            tableContinues = true
            break
          }
        }
        
        if (tableContinues) {
          currentTable.push(line)
        } else {
          // Table ended
          tables.push(currentTable.join('\n'))
          output.push(`[TABLE_${tables.length - 1}]`)
          output.push(line)
          inTable = false
          currentTable = []
        }
      } else {
        // Non-table content, table has ended
        tables.push(currentTable.join('\n'))
        output.push(`[TABLE_${tables.length - 1}]`)
        output.push(line)
        inTable = false
        currentTable = []
      }
    } else {
      output.push(line)
    }
  }
  
  // Handle table at end
  if (inTable && currentTable.length > 0) {
    tables.push(currentTable.join('\n'))
    output.push(`[TABLE_${tables.length - 1}]`)
  }
  
  return {
    tables,
    contentWithPlaceholders: output.join('\n')
  }
}

/**
 * Restore tables from placeholders
 */
export function restoreTables(content: string, tables: string[]): string {
  let result = content
  
  for (let i = 0; i < tables.length; i++) {
    result = result.replace(`[TABLE_${i}]`, tables[i])
  }
  
  return result
}