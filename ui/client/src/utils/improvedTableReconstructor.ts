/**
 * Improved table reconstruction that handles inline and malformed tables
 * Handles complex cases like:
 * - Tables with headers, separators, and data merged on single lines
 * - Multiple separator patterns (---, | --- |, |---|)
 * - Tables with inconsistent column counts
 * - Tables with orphaned separators
 */

import { validateTableContent, autoFixTableIssues } from './tableValidator'

/**
 * Process tables in the final content - handling all malformed patterns
 */
export function processTablesImproved(content: string): string {
  if (!content) return content

  // First, validate the content to understand what issues we're dealing with
  const validation = validateTableContent(content)

  // Debug logging
  if (validation.issues.length > 0) {
    console.log('[TABLE RECONSTRUCT] Found table issues:', {
      errorCount: validation.issues.filter(i => i.severity === 'error').length,
      warningCount: validation.issues.filter(i => i.severity === 'warning').length,
      inlineTables: validation.stats.inlineTables,
      malformedTables: validation.stats.malformedTables,
    })
  }

  // Apply auto-fixes for common issues
  let processed = autoFixTableIssues(content)

  // Then handle more complex reconstruction
  const lines = processed.split('\n')
  const processedLines: string[] = []
  let tableBuffer: string[] = []
  let inTable = false

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    const trimmed = line.trim()

    // Skip completely empty lines outside tables
    if (!trimmed && !inTable) {
      processedLines.push(line)
      continue
    }

    // Detect table start or continuation
    if (trimmed.includes('|')) {
      // Check for malformed patterns that need special handling
      if (isMalformedTableLine(trimmed)) {
        console.log('[TABLE RECONSTRUCT] Malformed line detected:', trimmed.substring(0, 100))

        // Buffer this line for reconstruction
        tableBuffer.push(line)
        inTable = true

        // Check if next line is also part of table
        const nextLine = i < lines.length - 1 ? lines[i + 1] : ''
        if (!nextLine.includes('|')) {
          // End of malformed table, reconstruct it
          const reconstructed = reconstructBufferedTable(tableBuffer)
          processedLines.push(...reconstructed)
          tableBuffer = []
          inTable = false
        }
      } else if (inTable) {
        // Continue buffering table lines
        tableBuffer.push(line)
      } else {
        // Normal table line
        processedLines.push(line)
      }
    } else if (inTable) {
      // End of table, reconstruct buffered content
      const reconstructed = reconstructBufferedTable(tableBuffer)
      processedLines.push(...reconstructed)
      tableBuffer = []
      inTable = false

      // Add the non-table line
      processedLines.push(line)
    } else {
      // Regular non-table content
      processedLines.push(line)
    }
  }

  // Handle any remaining buffered table
  if (tableBuffer.length > 0) {
    const reconstructed = reconstructBufferedTable(tableBuffer)
    processedLines.push(...reconstructed)
  }

  return processedLines.join('\n')
}

/**
 * Check if a line contains malformed table patterns
 */
function isMalformedTableLine(line: string): boolean {
  // Check for inline separators with content
  if ((line.includes('| --- |') || line.includes('|---|')) &&
    line.split('|').some(cell => {
      const trimmed = cell.trim()
      return trimmed && !/^-+$/.test(trimmed) && trimmed !== ''
    })) {
    return true
  }

  // Check for multiple separator patterns
  if ((line.match(/\| --- \|/g) || []).length > 3) {
    return true
  }

  // Check for the specific broken pattern from the user's example
  // 1) Condensed separators without spaces like "|---|---|"
  // 2) Repeated trailing groups mixed into content
  if (/\|-{3,}\|-{3,}\|/.test(line) || /\|\s*---\s*\|\s*---\s*\|.*\|\s*---\s*\|\s*---\s*\|/.test(line)) {
    return true
  }

  // NEW: Check for trailing separators (from actual agent output)
  // Pattern: headers followed by | --- | --- | --- |
  if (/\|[^|]+\|[^|]+\|.*\| --- \| --- \| --- \|$/.test(line)) {
    return true
  }

  // NEW: Check for step/description separator pattern
  if (/\| ------ \| ------------- \| --------- \|/.test(line)) {
    return true
  }

  // NEW: Check for mixed content and separators in same line
  if (line.includes('| --- |') && !(/^\s*\|(\s*---\s*\|)+\s*$/.test(line))) {
    // Line has separator pattern but also other content
    return true
  }

  // NEW: Check for excessive separator patterns like "| --------- | ------------------------------ |"
  if (/\|\s*-{5,}\s*\|\s*-{5,}/.test(line)) {
    return true
  }

  // NEW: Check for content mixed with multiple empty columns or separators 
  if (line.includes('|') && (line.match(/\|\s*\|/g) || []).length > 2) {
    // Multiple consecutive empty columns often indicate malformed content
    return true
  }

  return false
}

/**
 * Reconstruct a buffered table with multiple issues
 */
function reconstructBufferedTable(buffer: string[]): string[] {
  if (buffer.length === 0) return []

  console.log('[TABLE RECONSTRUCT] Reconstructing buffered table with', buffer.length, 'lines')

  // Analyze the buffer to understand its structure
  const allContent = buffer.join(' ')

  // Try to identify if this is a single inline table
  if (buffer.length === 1 || (buffer.length <= 3 && allContent.includes('---'))) {
    const reconstructed = reconstructInlineTable(allContent)
    if (reconstructed) {
      return reconstructed.split('\n')
    }
  }

  // For multi-line malformed tables, try to extract structure
  const result: string[] = []
  const cells: string[] = []
  let foundSeparator = false

  for (const line of buffer) {
    const segments = line.split('|').map(s => s.trim()).filter(s => s)

    for (const segment of segments) {
      if (/^-{3,}$/.test(segment)) {
        foundSeparator = true
      } else if (segment && !isTableNoise(segment)) {
        // Only add valid content cells, skip noise patterns
        cells.push(segment)
      }
    }
  }

  /**
   * Check if a segment is table noise that should be skipped
   */
  function isTableNoise(segment: string): boolean {
    // Skip segments that are just separators or noise
    if (/^-{2,}$/.test(segment)) return true
    if (segment === '---') return true
    if (/^[\s\-]+$/.test(segment)) return true
    // Skip empty or whitespace-only segments
    if (!segment || segment.trim() === '') return true
    return false
  }

  if (foundSeparator && cells.length > 0) {
    // Try to reconstruct based on detected cells
    // For financial tables like the user's example, common column counts are 4-7
    const possibleColumnCounts = [4, 5, 6, 7, 3]

    for (const colCount of possibleColumnCounts) {
      if (cells.length >= colCount) {
        // Check if this column count makes sense
        const remainder = cells.length % colCount
        if (remainder <= 2 || cells.length <= colCount * 2) {
          // This might be our column count
          const headers = cells.slice(0, colCount)
          const data = cells.slice(colCount)

          // Build table
          result.push('| ' + headers.join(' | ') + ' |')
          result.push('|' + headers.map(() => ' --- ').join('|') + '|')

          for (let i = 0; i < data.length; i += colCount) {
            const row = data.slice(i, Math.min(i + colCount, data.length))
            while (row.length < colCount) {
              row.push('') // Pad incomplete rows
            }
            result.push('| ' + row.join(' | ') + ' |')
          }

          return result
        }
      }
    }
  }

  // Fallback: return cleaned buffer with proper table structure
  const cleanedLines: string[] = []
  let allTableLines: string[] = []

  // First pass: collect all valid table lines
  for (const line of buffer) {
    const cleaned = cleanLine(line)
    if (cleaned && cleaned.includes('|')) {
      allTableLines.push(cleaned)
    }
  }

  if (allTableLines.length === 0) {
    return []
  }

  // If we have multiple lines, create a proper table structure
  if (allTableLines.length >= 2) {
    // First line is headers
    cleanedLines.push(allTableLines[0])

    // Add separator row
    const headerParts = allTableLines[0].split('|').filter(s => s.trim())
    const separatorRow = '|' + headerParts.map(() => ' --- ').join('|') + '|'
    cleanedLines.push(separatorRow)

    // Add remaining lines as data
    for (let i = 1; i < allTableLines.length; i++) {
      cleanedLines.push(allTableLines[i])
    }
  } else {
    // Single line - just return it
    cleanedLines.push(allTableLines[0])
  }

  return cleanedLines.filter(line => line.trim())
}

/**
 * Clean up a single line with malformed table patterns
 */
function cleanLine(line: string): string | null {
  let cleaned = line.trim()

  if (!cleaned || !cleaned.includes('|')) {
    return null
  }

  // Handle trailing separators pattern: "| Header | Header | --- | --- | --- |"
  if (/\|[^|]+\|[^|]+\|.*\| --- \| --- \| --- \|$/.test(cleaned)) {
    // Extract the actual headers
    const parts = cleaned.split('|').map(s => s.trim()).filter(s => s)
    const headers = parts.filter(p => !/^-+$/.test(p))

    if (headers.length > 0) {
      // Return just the headers
      return '| ' + headers.join(' | ') + ' |'
    }
  }

  // Handle step/description separator pattern
  if (/\| ------ \| ------------- \| --------- \|/.test(cleaned)) {
    // Skip this line, we'll generate proper separators
    return null
  }

  // Clean up other obvious issues
  cleaned = cleaned
    // Remove all variations of condensed separator patterns completely
    .replace(/\|---\|---\|---\|---\|---\|.*$/g, '')
    .replace(/\|---\|---\|---\|---\|.*$/g, '')
    .replace(/\|---\|---\|---\|.*$/g, '')
    .replace(/\|---\|---\|$/g, '')
    .replace(/\|---\|$/g, '')
    // Also handle patterns that might be embedded in content
    .replace(/\s*\|---\|---\|---\|---\|.*$/g, '')
    .replace(/\s*\|---\|---\|---\|.*$/g, '')
    .replace(/\s*\|---\|---\|.*$/g, '')
    .replace(/\s*\|---\|.*$/g, '')
    // Remove any trailing condensed patterns with any amount of separators
    .replace(/\s*\|\s*---\s*\|\s*---.*$/g, '')
    // Remove excessive separator patterns like "| --------- | ------------------------------ |"
    .replace(/\|\s*-{5,}\s*\|\s*-{5,}.*$/g, '')
    // Remove excessive | --- | patterns that are standalone or trailing
    .replace(/^(\|\s*---\s*){3,}\|?\s*$/g, '')
    .replace(/\s*(\|\s*---\s*){3,}\|?\s*$/g, '')
    // Handle empty trailing separators and multiple consecutive ones
    .replace(/\s*\|\s*\|\s*$/g, '') // Remove trailing | |
    .replace(/\s*\|\s*$/, '') // Remove single trailing |
    // Normalize remaining separators like "|---|---|" -> "| --- | --- |" if they're standalone
    .replace(/^\s*\|-{3}\|\s*$/g, '| --- |')
    .replace(/^\s*\|-{3}(?:\|-{3})+\|\s*$/g, (match) => {
      const count = (match.match(/\|-{3}/g) || []).length
      return '|' + Array(count).fill(' --- ').join('|') + '|'
    })

  return cleaned.trim() || null
}

/**
 * Reconstruct an inline table that's been merged into a single line
 */
function reconstructInlineTable(line: string): string | null {
  // Check if this looks like a table with headers and separators inline
  // Example: "Title| Header1 | Header2 | --- | --- | Data1 | Data2 |"

  // Split by pipes but preserve the structure
  const segments = line.split('|').map(s => s.trim())

  // Find separator segments (containing only dashes, spaces, colons)
  const separatorIndices: number[] = []
  segments.forEach((seg, idx) => {
    if (/^[\s\-:]+$/.test(seg) && seg.includes('-')) {
      separatorIndices.push(idx)
    }
  })

  if (separatorIndices.length === 0) {
    return null // No separator found, not a table
  }

  // Find where the separator section starts
  const firstSepIndex = separatorIndices[0]
  const lastSepIndex = separatorIndices[separatorIndices.length - 1]

  // Extract parts
  const beforeSeparator = segments.slice(0, firstSepIndex)
  const _separatorParts = segments.slice(firstSepIndex, lastSepIndex + 1)
  const afterSeparator = segments.slice(lastSepIndex + 1)

  // Determine if there's leading text before the table
  let leadingText = ''
  let headerParts = beforeSeparator

  // If the first segment doesn't look like a header, it might be leading text
  if (beforeSeparator.length > 0 && !beforeSeparator[0].includes('|')) {
    // Check if it's descriptive text rather than a header
    const firstPart = beforeSeparator[0]
    if (firstPart.length > 30 || firstPart.includes('–') || firstPart.includes(':')) {
      leadingText = firstPart
      headerParts = beforeSeparator.slice(1)
    }
  }

  // Build the reconstructed table
  const result: string[] = []

  // Add leading text if present
  if (leadingText) {
    result.push(leadingText)
    result.push('') // Add blank line before table
  }

  // Reconstruct the header row
  if (headerParts.length > 0) {
    const headerRow = '| ' + headerParts.filter(h => h).join(' | ') + ' |'
    result.push(headerRow)
  }

  // Reconstruct the separator row
  const numColumns = headerParts.filter(h => h).length
  if (numColumns > 0) {
    const separatorRow = '|' + Array(numColumns).fill('---').map(s => ` ${s} `).join('|') + '|'
    result.push(separatorRow)
  }

  // Process data rows
  // Group remaining segments into rows based on column count
  const dataSegments = afterSeparator.filter(s => s)
  if (dataSegments.length > 0 && numColumns > 0) {
    for (let i = 0; i < dataSegments.length; i += numColumns) {
      const rowData = dataSegments.slice(i, i + numColumns)
      if (rowData.length > 0) {
        // Pad row if needed
        while (rowData.length < numColumns) {
          rowData.push('')
        }
        const dataRow = '| ' + rowData.join(' | ') + ' |'
        result.push(dataRow)
      }
    }
  }

  return result.join('\n')
}