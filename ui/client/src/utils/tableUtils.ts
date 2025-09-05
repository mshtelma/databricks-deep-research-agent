/**
 * Utility functions for table processing and manipulation
 */

/**
 * Represents a detected table structure with confidence scoring
 */
interface TableStructure {
  startLine: number
  endLine: number
  columnCount: number
  confidence: number
  hasHeader: boolean
  hasSeparator: boolean
  hasData: boolean
  lines: string[]
}

/**
 * Split a markdown table row into cells, preserving escaped pipes (\|) inside cells
 */
function splitRowPreserveEscapes(row: string): string[] {
  const trimmed = row.trim()
  if (!(trimmed.startsWith('|') && trimmed.endsWith('|'))) return []
  const inner = trimmed.slice(1, -1)
  const cells: string[] = []
  let buffer = ''
  for (let i = 0; i < inner.length; i++) {
    const ch = inner[i]
    if (ch === '|' && inner[i - 1] !== '\\') {
      cells.push(buffer.trim())
      buffer = ''
      continue
    }
    buffer += ch
  }
  cells.push(buffer.trim())
  return cells
}

/**
 * Analyzes text to detect table structures with confidence scoring
 */
function detectTableStructures(text: string): TableStructure[] {
  const lines = text.split('\n')
  const structures: TableStructure[] = []
  let currentStructure: TableStructure | null = null

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    const trimmedLine = line.trim()
    const nextLine = lines[i + 1]?.trim() || ''
    const _prevLine = i > 0 ? lines[i - 1].trim() : ''

    // Pattern 1: Standard markdown table row
    const isStandardRow = /^\|.*\|$/.test(trimmedLine) && trimmedLine.split('|').length > 2

    // Pattern 2: Separator row
    const isSeparatorRow = /^\|?\s*[-:]+\s*(\|\s*[-:]+\s*)*\|?$/.test(trimmedLine)

    // Pattern 3: Potential header without pipes
    const isPotentialHeader = !trimmedLine.includes('|') &&
      trimmedLine.length > 0 &&
      (nextLine === '---' || nextLine.match(/^-{3,}$/))

    // Pattern 4: Content that looks like table data
    const hasTableIndicators = trimmedLine.includes('|') && trimmedLine.split('|').length >= 2

    if (isStandardRow) {
      if (!currentStructure) {
        currentStructure = {
          startLine: i,
          endLine: i,
          columnCount: trimmedLine.split('|').length - 1,
          confidence: 0.8,
          hasHeader: true,
          hasSeparator: false,
          hasData: false,
          lines: [line]
        }
      } else {
        currentStructure.endLine = i
        currentStructure.lines.push(line)
        if (isSeparatorRow && !currentStructure.hasSeparator) {
          currentStructure.hasSeparator = true
          currentStructure.confidence += 0.1
        } else if (!isSeparatorRow && currentStructure.hasSeparator) {
          currentStructure.hasData = true
          currentStructure.confidence += 0.1
        }
      }
    } else if (hasTableIndicators && currentStructure) {
      // Continue table even if not perfectly formatted
      currentStructure.endLine = i
      currentStructure.lines.push(line)
    } else if (isPotentialHeader && !currentStructure) {
      // Start potential table from header without pipes
      currentStructure = {
        startLine: i,
        endLine: i,
        columnCount: 0, // Will be determined later
        confidence: 0.3,
        hasHeader: true,
        hasSeparator: false,
        hasData: false,
        lines: [line]
      }
    } else if (currentStructure && trimmedLine === '') {
      // Empty line might continue table
      currentStructure.lines.push(line)
    } else if (currentStructure && (trimmedLine === '---' || trimmedLine.match(/^-{3,}$/))) {
      // Standalone --- lines after empty lines are content dividers, not table separators
      // Only treat as table separator if it's IMMEDIATELY after a table row (no empty lines)
      const prevIndex = i - 1
      if (prevIndex >= 0) {
        const _prevLine = lines[prevIndex].trim()

        // If previous line is empty, this is not a table separator
        if (_prevLine === '') {
          // End the table structure here
          if (currentStructure.hasHeader) {
            structures.push(currentStructure)
          }
          currentStructure = null
        } else if (_prevLine.includes('|')) {
          // Previous line is a table row, this could be a separator
          currentStructure.lines.push(line)
          if (!currentStructure.hasSeparator) {
            currentStructure.hasSeparator = true
          }
        } else {
          // Previous line is not table-related, end structure
          if (currentStructure.hasHeader) {
            structures.push(currentStructure)
          }
          currentStructure = null
        }
      }
    } else if (currentStructure) {
      // End current structure
      if (currentStructure.hasHeader && (currentStructure.hasSeparator || currentStructure.hasData)) {
        structures.push(currentStructure)
      }
      currentStructure = null
    }
  }

  // Add final structure if exists
  if (currentStructure && currentStructure.hasHeader) {
    structures.push(currentStructure)
  }

  return structures
}

/**
 * Determines if text contains an incomplete table that shouldn't be rendered
 */
export function isIncompleteTable(text: string): boolean {
  const _structures = detectTableStructures(text)

  for (const structure of _structures) {
    // Don't treat single-row tables as incomplete (they might be data from cleaned tables)
    const lineCount = structure.lines.filter(l => {
      const trimmed = l.trim()
      return trimmed && !trimmed.match(/^\|?\s*[-:]+\s*(\|\s*[-:]+\s*)*\|?$/)
    }).length

    // Only mark as incomplete if it's clearly a broken table structure
    // A single data row without header/separator is OK (might be from cleaned table),
    // but if it's a header-only row, treat as incomplete
    if (lineCount === 1 && !structure.hasSeparator && !structure.hasHeader) {
      continue // Don't mark single non-header rows as incomplete
    }

    // Incomplete if has header but no separator or data
    if (structure.hasHeader && !structure.hasSeparator && !structure.hasData) {
      return true
    }
    // Incomplete if has header and separator but no data
    if (structure.hasHeader && structure.hasSeparator && !structure.hasData) {
      return true
    }
    // Check for malformed separators that would break rendering
    for (const line of structure.lines) {
      if (line.trim().match(/^\|\s*---\s*\|\s*---\s*\|\s*---\s*\|\s*---\s*\|$/)) {
        return true // This is the broken pattern we're seeing
      }
    }
  }

  return false
}

/**
 * Intelligently fixes broken table separators using context-aware detection
 */
export function fixBrokenTableSeparators(text: string): string {
  const lines = text.split('\n')
  const processedLines: string[] = []
  let i = 0

  while (i < lines.length) {
    const line = lines[i]
    const trimmedLine = line.trim()

    // Skip empty lines
    if (trimmedLine === '') {
      processedLines.push(line)
      i++
      continue
    }

    // Detect potential table header patterns
    const hasTablePipes = trimmedLine.includes('|')
    const pipeCount = (trimmedLine.match(/\|/g) || []).length
    const isProperTableRow = /^\|.*\|$/.test(trimmedLine) && pipeCount >= 2
    const isPartialTableRow = hasTablePipes && pipeCount >= 1
    const isSeparatorRow = /^\|?\s*[-:]+\s*(\|\s*[-:]+\s*)*\|?$/.test(trimmedLine)

    // Don't process if it's already a separator row
    if (isSeparatorRow) {
      processedLines.push(line)
      i++
      continue
    }

    // Check if this could be a table header
    if (isProperTableRow || isPartialTableRow) {
      // Look ahead for potential broken separators
      let separatorCandidates: number[] = []
      let tableDataStart = -1
      let j = i + 1

      // Scan ahead up to 10 lines for patterns
      while (j < lines.length && j <= i + 10) {
        const scanLine = lines[j].trim()

        // Check for separator patterns
        if (scanLine === '---' ||
          scanLine.match(/^-{3,}$/) ||
          scanLine.match(/^[-\s]+$/) && scanLine.includes('-')) {
          separatorCandidates.push(j)
        } else if (scanLine.includes('|') && !scanLine.match(/^\|?\s*[-:]+\s*(\|\s*[-:]+\s*)*\|?$/)) {
          // Found actual table data
          tableDataStart = j
          break
        } else if (scanLine !== '' && !scanLine.match(/^\s*$/)) {
          // Non-empty, non-table line
          if (separatorCandidates.length === 0) {
            break // No separator found before content
          }
        }
        j++
      }

      // Determine if we should fix this table
      const shouldFix = separatorCandidates.length > 0 &&
        (tableDataStart > -1 || separatorCandidates.length >= 2)

      if (shouldFix) {
        // Calculate column count from various sources
        let columnCount = 2 // minimum

        // From header
        if (isProperTableRow) {
          columnCount = Math.max(2, pipeCount - 1)
        } else if (isPartialTableRow) {
          // Estimate from partial row
          const parts = trimmedLine.split('|').filter(p => p.trim())
          columnCount = Math.max(2, parts.length)
        }

        // From data rows if available
        if (tableDataStart > -1) {
          const dataLine = lines[tableDataStart].trim()
          const dataPipes = (dataLine.match(/\|/g) || []).length
          if (dataLine.startsWith('|') && dataLine.endsWith('|')) {
            columnCount = Math.max(columnCount, dataPipes - 1)
          } else {
            const dataParts = dataLine.split('|').filter(p => p.trim())
            columnCount = Math.max(columnCount, dataParts.length)
          }
        }

        // Format header properly
        let headerLine = trimmedLine
        if (!headerLine.startsWith('|')) {
          headerLine = '| ' + headerLine
        }
        if (!headerLine.endsWith('|')) {
          headerLine = headerLine + ' |'
        }

        // Ensure header has correct column count
        const headerParts = headerLine.split('|').filter(p => p !== '')
        if (headerParts.length < columnCount) {
          // Pad with empty columns
          const padding = columnCount - headerParts.length
          headerLine = headerLine.replace(/\|$/, '') +
            Array(padding).fill(' | ').join('') + ' |'
        }

        processedLines.push(headerLine)

        // Add proper separator
        const separator = '|' + Array(columnCount).fill(' --- ').join('|') + '|'
        processedLines.push(separator)

        // Skip all the separator candidate lines
        const skipTo = tableDataStart > -1 ? tableDataStart :
          Math.max(...separatorCandidates) + 1

        // Preserve non-separator, non-empty interleaved lines (e.g., amounts like €7,200)
        if (tableDataStart === -1) {
          for (let k = i + 1; k < skipTo; k++) {
            const interLine = lines[k]
            const interTrim = interLine.trim()
            const isDashSeparator = interTrim === '---' || interTrim.match(/^-{3,}$/) || (interTrim.match(/^[-\s]+$/) && interTrim.includes('-'))
            if (interTrim !== '' && !isDashSeparator) {
              processedLines.push(interLine)
            }
          }
        }
        i = skipTo
        continue
      }
    }

    // Advanced pattern: Multi-line header reconstruction
    if (!hasTablePipes && trimmedLine.length > 0 && i + 2 < lines.length) {
      // Check if this could be part of a broken table header
      let potentialHeaders: string[] = []
      let separatorCount = 0
      let scanIndex = i
      let foundTableData = false
      let dataLineIndex = -1

      // Collect potential header components
      while (scanIndex < lines.length && scanIndex <= i + 10) {
        const scanLine = lines[scanIndex].trim()

        if (scanLine === '---' || scanLine.match(/^-{3,}$/)) {
          separatorCount++
        } else if (scanLine === '') {
          // Skip empty lines
        } else if (scanLine.includes('|')) {
          // Found table data
          foundTableData = true
          dataLineIndex = scanIndex
          break
        } else if (scanLine.match(/^[\w\s\-‑€$£¥,.()]+$/)) {
          // Potential header text (including currency and special chars)
          potentialHeaders.push(scanLine)
        } else if (separatorCount > 0) {
          // Stop if we hit non-header content after separators
          break
        }
        scanIndex++
      }

      // Reconstruct if we have headers and table data
      if (potentialHeaders.length >= 2 && foundTableData && dataLineIndex > -1) {
        const dataLine = lines[dataLineIndex].trim()
        const dataParts = dataLine.split('|').filter(p => p.trim())

        // Only reconstruct if column counts make sense
        if (dataParts.length >= potentialHeaders.length) {
          // Build the header row
          const headerRow = '| ' + potentialHeaders.join(' | ') + ' |'
          processedLines.push(headerRow)

          // Add separator
          const separator = '|' + Array(potentialHeaders.length).fill(' --- ').join('|') + '|'
          processedLines.push(separator)

          // Skip to the data line
          i = dataLineIndex
          continue
        }
      }
    }

    // Default: keep the line as-is
    processedLines.push(lines[i])
    i++
  }

  return processedLines.join('\n')
}

export function preprocessMarkdown(input: string): string {
  if (!input) return input
  let text = input

  // Normalize Windows newlines
  text = text.replace(/\r\n/g, '\n')

  // Handle <br> tags in table cells - convert to line breaks within cells
  text = text.replace(/(<br\s*\/?>)/gi, '  \n')

  // First pass: Remove malformed separator patterns that break rendering
  const lines = text.split('\n')
  const preCleaned: string[] = []

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    const trimmedLine = line.trim()

    // Pattern 1: Remove broken separator rows like: | --- | --- | --- | --- |
    // These are malformed because --- should not have spaces and pipes around each one
    if (trimmedLine.match(/^\|\s*---\s*\|\s*---\s*\|\s*---\s*(\|\s*---\s*)*\|?\s*$/)) {
      // Skip this malformed separator entirely
      continue
    }

    // Pattern 2: Remove broken rows like: |---|---|---|---|
    // when they appear without proper context
    if (trimmedLine.match(/^\|---\|---(\|---)*\|?$/)) {
      const _prevLine = i > 0 ? lines[i - 1].trim() : ''
      const nextLine = i < lines.length - 1 ? lines[i + 1].trim() : ''

      // Only keep if it's a valid separator between header and data
      if (_prevLine.match(/^\|.*\|$/) && nextLine.match(/^\|.*\|$/)) {
        const prevCols = _prevLine.split('|').length - 2
        const nextCols = nextLine.split('|').length - 2
        const sepCols = trimmedLine.split('|').length - 2

        // Column counts should roughly match
        if (Math.abs(prevCols - sepCols) <= 1 && Math.abs(nextCols - sepCols) <= 1) {
          preCleaned.push(line)
        }
        // Otherwise skip this malformed separator
      }
      // Skip if not in proper table context
    }
    // Pattern 3: Remove incomplete table headers like: | |---|---|
    // These appear to be malformed partial headers
    else if (trimmedLine.match(/^\|\s*\|---\|---(\|---)*\|?$/)) {
      // Skip this malformed partial header
      continue
    } else {
      preCleaned.push(line)
    }
  }

  text = preCleaned.join('\n')

  // Sanitize data rows: remove trailing separator-only cells appended to data rows
  {
    const lines2 = text.split('\n')
    const sanitized: string[] = []
    for (const rawLine of lines2) {
      const trimmed = rawLine.trim()
      // Drop stray pipe-only lines
      if (trimmed === '|') {
        continue
      }
      if (/^\|.*\|$/.test(trimmed)) {
        // Process table row cells
        let cells = splitRowPreserveEscapes(trimmed)
        // Remove trailing separator-like cells (---, :---, ---:, :---:)
        while (cells.length > 0 && /^(?::)?-{3,}(?::)?$/.test(cells[cells.length - 1])) {
          cells.pop()
        }
        // If the original raw line had contiguous pipes (e.g., "||"),
        // collapse by removing empty cells entirely for this row
        if (rawLine.includes('||')) {
          cells = cells.filter(c => c !== '')
        }
        const rebuilt = cells.length > 0 ? `| ${cells.join(' | ')} |` : ''
        if (rebuilt) {
          sanitized.push(rebuilt)
        }
      } else {
        sanitized.push(rawLine)
      }
    }
    text = sanitized.join('\n')
  }

  // Fix broken table separators
  text = fixBrokenTableSeparators(text)

  // Collapse multiple consecutive pipes to single (except escaped \| context)
  // We only collapse when the extra pipe is a delimiter, not an escaped literal within a cell
  text = text.replace(/\|{2,}/g, (m) => m.length > 1 ? '|' : m)

  // Clean up pipe spacing
  text = text.replace(/\|[ \t]{2,}/g, '| ') // Multiple spaces after pipe (spaces/tabs only)
  text = text.replace(/[ \t]{2,}\|/g, ' |') // Multiple spaces before pipe (spaces/tabs only)

  // Final validation: decide whether we need to consider wrapping incomplete tables
  const _structures = detectTableStructures(text)
  let hasIncompleteTable = false
  if (isIncompleteTable(text)) {
    hasIncompleteTable = true
  } else {
    // Detect any partial table rows (start with '|' but no closing '|')
    const partialRowRegex = /^\|.*[^|]\s*$/
    hasIncompleteTable = text.split('\n').some(l => {
      const t = l.trim()
      return partialRowRegex.test(t) && !t.match(/^\|?\s*[-:]+\s*(\|\s*[-:]+\s*)*\|?$/)
    })
  }

  // Only apply code block wrapping if we have any incomplete tables or partial rows
  if (hasIncompleteTable) {
    const finalLines = text.split('\n')
    const wrappedLines: string[] = []
    let currentTable: string[] = []
    let inTable = false
    let sawPartialRow = false

    for (let i = 0; i < finalLines.length; i++) {
      const line = finalLines[i]
      const trimmedLine = line.trim()
      const isSeparator = /^\|?\s*[-:]+\s*(\|\s*[-:]+\s*)*\|?$/.test(trimmedLine)
      const isTableRow = /^\|.*\|$/.test(trimmedLine) && trimmedLine.split('|').length >= 3
      const isPartialRow = trimmedLine.startsWith('|') && !trimmedLine.endsWith('|') && !isSeparator

      if ((isTableRow || isSeparator || isPartialRow) && !inTable) {
        inTable = true
        currentTable = [line]
        sawPartialRow = isPartialRow
      } else if ((isTableRow || isSeparator || isPartialRow) && inTable) {
        currentTable.push(line)
        if (isPartialRow) sawPartialRow = true
      } else if (inTable && trimmedLine === '') {
        currentTable.push(line)
      } else if (inTable) {
        // End of table - check row count
        const tableRows = currentTable.filter(l => {
          const trimmed = l.trim()
          return trimmed && trimmed.includes('|') &&
            !trimmed.match(/^\|?\s*[-:]+\s*(\|\s*[-:]+\s*)*\|?$/)
        }).length

        // Wrap if the table is incomplete OR we saw partial rows
        if ((tableRows > 1 || sawPartialRow) && isIncompleteTable(currentTable.join('\n'))) {
          wrappedLines.push('```')
          wrappedLines.push(...currentTable)
          wrappedLines.push('```')
        } else {
          wrappedLines.push(...currentTable)
        }
        wrappedLines.push(line)
        inTable = false
        currentTable = []
        sawPartialRow = false
      } else {
        wrappedLines.push(line)
      }
    }

    // Handle table at end
    if (inTable && currentTable.length > 0) {
      const tableRows = currentTable.filter(l => {
        const trimmed = l.trim()
        return trimmed && trimmed.includes('|') &&
          !trimmed.match(/^\|?\s*[-:]+\s*(\|\s*[-:]+\s*)*\|?$/)
      }).length

      if ((tableRows > 1 || sawPartialRow) && isIncompleteTable(currentTable.join('\n'))) {
        wrappedLines.push('```')
        wrappedLines.push(...currentTable)
        wrappedLines.push('```')
      } else {
        wrappedLines.push(...currentTable)
      }
    }

    text = wrappedLines.join('\n')
  }

  // Final cleanup: Ensure tables are properly formatted
  const finalLines = text.split('\n')
  const output: string[] = []
  let i = 0
  let withinTable = false
  let expectedCols = 0
  let lastDataRowIndex = -1

  while (i < finalLines.length) {
    const line = finalLines[i]
    const trimmedLine = line.trim()

    // Check for table rows
    if (trimmedLine.includes('|')) {
      const isProperRow = /^\|.*\|$/.test(trimmedLine)
      const isSeparator = /^\|?\s*[-:]+\s*(\|\s*[-:]+\s*)*\|?$/.test(trimmedLine)
      const cellCount = splitRowPreserveEscapes(trimmedLine).length
      const isSingleCellRow = isProperRow && cellCount === 1

      if (isProperRow || isSeparator) {
        // Track table boundaries and expected column counts
        if (isProperRow && !isSeparator) {
          // Determine expected columns from header the first time we encounter it
          const cols = Math.max(1, splitRowPreserveEscapes(trimmedLine).length)
          if (!withinTable) {
            withinTable = true
            expectedCols = cols
            lastDataRowIndex = -1
          }
        }

        // If we encounter a single-cell row while inside a table and previous data row has missing cell,
        // merge this single cell into the previous data row instead of pushing a broken row
        if (
          withinTable &&
          isSingleCellRow &&
          lastDataRowIndex >= 0 &&
          expectedCols > 1
        ) {
          const prev = output[lastDataRowIndex]
          const prevTrim = prev.trim()
          // Count previous row cells
          const prevCells = splitRowPreserveEscapes(prevTrim)
          if (prevCells.length > 0 && prevCells.length === expectedCols - 1) {
            // Extract the single cell content
            const singleCell = trimmedLine.slice(1, -1).trim()
            const mergedCells = [...prevCells, singleCell]
            const mergedRow = `| ${mergedCells.join(' | ')} |`
            output[lastDataRowIndex] = mergedRow
            // Do not push the current single-cell row
            i++
            continue
          }
        }

        output.push(line)

        // Check if we need to add a separator after header
        // Only add separator if next line is also a table row (not for single rows)
        // AND we don't already have a separator
        if (!isSeparator && i + 1 < finalLines.length) {
          const nextLine = finalLines[i + 1].trim()
          const isNextSeparator = /^\|?\s*[-:]+\s*(\|\s*[-:]+\s*)*\|?$/.test(nextLine)
          const isNextData = /^\|.*\|$/.test(nextLine) && !isNextSeparator && nextLine.split('|').length >= 3

          // Only add separator between actual table rows, not for standalone rows
          if (isNextData) {
            // Look ahead to see if this is actually a multi-row table
            let hasMoreRows = false
            for (let j = i + 2; j < Math.min(i + 5, finalLines.length); j++) {
              const checkLine = finalLines[j].trim()
              if (checkLine.match(/^\|.*\|$/) && checkLine.split('|').length >= 3) {
                hasMoreRows = true
                break
              }
              if (checkLine && !checkLine.match(/^\|?\s*[-:]+\s*(\|\s*[-:]+\s*)*\|?$/) && checkLine !== '') {
                break // Non-table content
              }
            }

            // Only add separator if this is part of a multi-row table
            if (hasMoreRows) {
              const colCount = Math.max(2, splitRowPreserveEscapes(trimmedLine).length)
              output.push('|' + Array(colCount).fill(' --- ').join('|') + '|')
            }
          }
        }

        // Update last data row index after we may have added an auto-separator
        if (isProperRow && !isSeparator) {
          lastDataRowIndex = output.length - 1
        }
      } else if (trimmedLine.startsWith('|') || trimmedLine.endsWith('|')) {
        // Partial table row - try to fix
        let fixed = trimmedLine
        if (!fixed.startsWith('|')) fixed = '| ' + fixed
        if (!fixed.endsWith('|')) fixed = fixed + ' |'
        output.push(fixed)
        withinTable = true
        expectedCols = Math.max(expectedCols, splitRowPreserveEscapes(fixed).length)
        lastDataRowIndex = output.length - 1
      } else {
        // Line contains pipes but isn't a table row
        // Check if it's mixed content
        const pipeIndex = line.indexOf('|')
        if (pipeIndex > 0 && pipeIndex < line.length - 1) {
          // Could be text followed by table
          const beforePipe = line.substring(0, pipeIndex).trim()
          const fromPipe = line.substring(pipeIndex)

          if (beforePipe && /^\|.*\|$/.test(fromPipe.trim())) {
            // Split them
            output.push(beforePipe)
            output.push(fromPipe)
          } else {
            output.push(line)
          }
        } else {
          output.push(line)
        }
      }
    } else {
      // Leaving a table context when encountering non-table content
      withinTable = false
      expectedCols = 0
      lastDataRowIndex = -1
      output.push(line)
    }
    i++
  }

  return output.join('\n')
}