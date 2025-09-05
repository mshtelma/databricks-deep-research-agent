/**
 * UNIFIED TABLE PROCESSOR
 * 
 * This is the ONE AND ONLY table processor that handles:
 * - Streaming and final content processing
 * - All malformed table patterns
 * - Noise cleanup
 * - Structure normalization
 * - Performance optimization
 * 
 * Replaces all other table processors for consistency and simplicity.
 */

export interface TableProcessingOptions {
    streaming?: boolean
    preserveCodeBlocks?: boolean
    maxTableSize?: number
}

interface TableCell {
    content: string
    original: string
}

interface TableRow {
    cells: TableCell[]
    isHeader: boolean
    isSeparator: boolean
    originalLine: string
}

interface ParsedTable {
    rows: TableRow[]
    startLine: number
    endLine: number
    isComplete: boolean
}

/**
 * The main unified table processor
 */
export function processTablesUnified(
    content: string,
    options: TableProcessingOptions = {}
): string {
    // Handle null/undefined/empty inputs
    if (!content) return content

    const {
        streaming = false,
        preserveCodeBlocks = true,
        maxTableSize = 50000 // Increased limit for large tables
    } = options

    // Step 1: Protect code blocks from processing
    const { content: protectedContent, codeBlocks } = preserveCodeBlocks
        ? protectCodeBlocks(content)
        : { content, codeBlocks: [] }

    // Step 2: Parse all tables in the content
    const { tables, nonTableLines } = parseAllTables(protectedContent)

    // Step 3: Process each table based on mode
    const processedTables = tables.map(table => {
        if (streaming && !table.isComplete) {
            return createTablePlaceholder()
        }
        // Force inline tables to be complete for normalization
        if (!table.isComplete && table.rows.length > 1) {
            table.isComplete = true
        }
        return normalizeTable(table, maxTableSize)
    })

    // Step 4: Reconstruct the final content
    const result = reconstructContent(nonTableLines, processedTables, tables)

    // Step 5: Restore code blocks
    return preserveCodeBlocks ? restoreCodeBlocks(result, codeBlocks) : result
}

/**
 * Protect code blocks from table processing
 */
function protectCodeBlocks(content: string): { content: string; codeBlocks: string[] } {
    const codeBlocks: string[] = []
    let protectedContent = content

    // Match ```...``` blocks
    const codeBlockRegex = /```[\s\S]*?```/g
    let match
    let index = 0

    while ((match = codeBlockRegex.exec(content)) !== null) {
        const placeholder = `__CODE_BLOCK_${index}__`
        codeBlocks.push(match[0])
        protectedContent = protectedContent.replace(match[0], placeholder)
        index++
    }

    return { content: protectedContent, codeBlocks }
}

/**
 * Restore code blocks after processing
 */
function restoreCodeBlocks(content: string, codeBlocks: string[]): string {
    let restored = content
    codeBlocks.forEach((block, index) => {
        const placeholder = `__CODE_BLOCK_${index}__`
        restored = restored.replace(placeholder, block)
    })
    return restored
}

/**
 * Parse all tables from content
 */
function parseAllTables(content: string): { tables: ParsedTable[]; nonTableLines: (string | null)[] } {
    const lines = content.split('\n')
    const tables: ParsedTable[] = []
    const nonTableLines: (string | null)[] = new Array(lines.length).fill(null)

    let i = 0
    while (i < lines.length) {
        const line = lines[i]

        if (isTableLine(line)) {
            // Found start of a table, collect all table lines
            const tableLines: string[] = []
            const startLine = i

            while (i < lines.length && (isTableLine(lines[i]) || isTableNoise(lines[i]) || isEmptyLineBetweenTable(lines, i))) {
                if (!isTableNoise(lines[i])) {
                    tableLines.push(lines[i])
                }
                i++
            }

            if (tableLines.length > 0) {
                const parsedTable = parseTable(tableLines, startLine, i - 1)
                tables.push(parsedTable)

                // Mark these lines as processed
                for (let j = startLine; j < i; j++) {
                    nonTableLines[j] = null
                }
            }
        } else {
            nonTableLines[i] = line
            i++
        }
    }

    return { tables, nonTableLines }
}

/**
 * Check if a line is part of a table
 */
function isTableLine(line: string): boolean {
    const trimmed = line.trim()

    // Must contain pipes
    if (!trimmed.includes('|')) return false

    // Basic table patterns
    if (trimmed.startsWith('|') && trimmed.endsWith('|')) return true
    if (trimmed.includes(' | ')) return true

    // Handle inline separators
    if (trimmed.includes('|') && (trimmed.includes('---') || trimmed.includes('-'))) return true

    return false
}

/**
 * Check if a line is table noise that should be removed
 */
function isTableNoise(line: string): boolean {
    const trimmed = line.trim()

    // Lines with only dashes and spaces
    if (/^[-\s]+$/.test(trimmed) && trimmed.includes('---')) return true

    // Lines like "---    ---    ---"
    if (/^---(\s+---)+\s*$/.test(trimmed)) return true

    // Empty pipe-only lines
    if (trimmed === '|' || /^\|+$/.test(trimmed)) return true

    return false
}

/**
 * Check if an empty line is between table parts
 */
function isEmptyLineBetweenTable(lines: string[], index: number): boolean {
    if (lines[index].trim() !== '') return false

    // Check if next line is a table line
    if (index + 1 < lines.length && isTableLine(lines[index + 1])) {
        return true
    }

    return false
}

/**
 * Parse a single table from collected lines
 */
function parseTable(lines: string[], startLine: number, endLine: number): ParsedTable {
    // Check if this is an inline table first
    if (lines.length === 1 && lines[0].includes('---')) {
        const inlineRows = parseInlineTable(lines[0])
        if (inlineRows.length > 0) {
            return {
                rows: inlineRows,
                startLine,
                endLine,
                isComplete: false // Inline tables are treated as incomplete for special handling
            }
        }
    }

    const rows: TableRow[] = []

    for (const line of lines) {
        const row = parseTableRow(line)
        if (row) {
            rows.push(row)
        }
    }

    // Determine if table is complete
    const isComplete = hasValidStructure(rows)

    return {
        rows,
        startLine,
        endLine,
        isComplete
    }
}

/**
 * Parse an inline table like: | Header1 | Header2 | --- | --- | Data1 | Data2 |
 */
function parseInlineTable(line: string): TableRow[] {
    const parts = smartSplitByPipe(line)

    // Remove empty first/last parts
    let cleanParts = parts
    if (cleanParts[0].trim() === '') cleanParts = cleanParts.slice(1)
    if (cleanParts.length > 0 && cleanParts[cleanParts.length - 1].trim() === '') {
        cleanParts = cleanParts.slice(0, -1)
    }

    // Find separator positions
    const separatorIndices: number[] = []
    cleanParts.forEach((part, index) => {
        const trimmed = part.trim()
        if (/^-+$/.test(trimmed) || trimmed === '---') {
            separatorIndices.push(index)
        }
    })

    if (separatorIndices.length === 0) return []

    // Split into before/after separator sections
    const firstSeparator = separatorIndices[0]
    const lastSeparator = separatorIndices[separatorIndices.length - 1]

    const headerParts = cleanParts.slice(0, firstSeparator)
    const dataParts = cleanParts.slice(lastSeparator + 1)

    const rows: TableRow[] = []

    // Add header row if we have headers
    if (headerParts.length > 0) {
        const headerCells = headerParts.map(part => ({
            content: cleanCellContent(part),
            original: part
        }))

        rows.push({
            cells: headerCells,
            isHeader: true,
            isSeparator: false,
            originalLine: line
        })
    }

    // Add data row if we have data
    if (dataParts.length > 0) {
        const dataCells = dataParts.map(part => ({
            content: cleanCellContent(part),
            original: part
        }))

        rows.push({
            cells: dataCells,
            isHeader: false,
            isSeparator: false,
            originalLine: line
        })
    }

    return rows
}

/**
 * Parse a single table row
 */
function parseTableRow(line: string): TableRow | null {
    const trimmed = line.trim()
    if (!trimmed) return null

    // Check if it's a separator row
    const isSeparator = /^[\|\s\-:]+$/.test(trimmed) && trimmed.includes('-')

    // Split by pipes, but be smart about pipes inside code/quotes
    let parts = smartSplitByPipe(line)

    // Remove empty first/last parts if they exist (from leading/trailing |)
    if (parts[0].trim() === '') parts = parts.slice(1)
    if (parts.length > 0 && parts[parts.length - 1].trim() === '') parts = parts.slice(0, -1)

    // Handle inline separators (content mixed with separators)
    if (!isSeparator && trimmed.includes('---')) {
        parts = extractContentFromInlineSeparators(parts)
    }

    const cells: TableCell[] = parts.map(part => ({
        content: cleanCellContent(part),
        original: part
    }))

    return {
        cells,
        isHeader: false, // Will be determined later
        isSeparator,
        originalLine: line
    }
}

/**
 * Smart split by pipe that respects code blocks and quotes
 */
function smartSplitByPipe(line: string): string[] {
    const parts: string[] = []
    let current = ''
    let inCode = false
    let inQuotes = false
    let quoteChar = ''

    for (let i = 0; i < line.length; i++) {
        const char = line[i]
        const prevChar = i > 0 ? line[i - 1] : ''

        if (char === '`' && prevChar !== '\\') {
            inCode = !inCode
            current += char
        } else if (!inCode && (char === '"' || char === "'") && prevChar !== '\\') {
            if (!inQuotes) {
                inQuotes = true
                quoteChar = char
            } else if (char === quoteChar) {
                inQuotes = false
                quoteChar = ''
            }
            current += char
        } else if (char === '|' && !inCode && !inQuotes) {
            parts.push(current)
            current = ''
        } else {
            current += char
        }
    }

    parts.push(current)
    return parts
}

/**
 * Extract content from inline separators
 */
function extractContentFromInlineSeparators(parts: string[]): string[] {
    const content: string[] = []

    for (const part of parts) {
        const trimmed = part.trim()
        // Skip separator parts
        if (/^-+$/.test(trimmed) || trimmed === '---') {
            continue
        }
        // Keep content parts
        if (trimmed && trimmed !== '') {
            content.push(part)
        }
    }

    return content
}

/**
 * Clean cell content
 */
function cleanCellContent(content: string): string {
    const trimmed = content.trim()

    // Don't clean if it looks like it contains meaningful content
    if (trimmed.includes('€') || trimmed.includes('$') || trimmed.includes('`') || trimmed.includes('"')) {
        return trimmed
    }

    // Only remove separator-only content
    if (/^-+$/.test(trimmed)) {
        return ''
    }

    return trimmed
}

/**
 * Check if table has valid structure
 */
function hasValidStructure(rows: TableRow[]): boolean {
    if (rows.length === 0) return false

    // Should have at least one non-separator row
    const contentRows = rows.filter(row => !row.isSeparator)
    if (contentRows.length === 0) return false

    // Check if table seems complete (has enough structure)
    // For inline separators, if we have content mixed with separators, it's incomplete
    const hasInlineSeparators = rows.some(row =>
        !row.isSeparator && row.originalLine.includes('---')
    )

    if (hasInlineSeparators) return false

    // Check for incomplete patterns that suggest streaming
    const separatorRows = rows.filter(row => row.isSeparator)

    // If we have separators but they look incomplete (like "| --- |" without matching columns)
    if (separatorRows.length > 0) {
        const firstContentRow = contentRows[0]
        const firstSeparatorRow = separatorRows[0]

        if (firstContentRow && firstSeparatorRow) {
            const contentColumns = firstContentRow.cells.length
            const separatorColumns = firstSeparatorRow.cells.length

            // If separator has fewer columns than content, it's likely incomplete
            if (separatorColumns < contentColumns) {
                return false
            }
        }
    }

    // If we have separator rows and content rows, it's complete
    const hasSeparators = rows.some(row => row.isSeparator)
    if (hasSeparators && contentRows.length > 1) return true

    // If we have multiple content rows, assume complete
    if (contentRows.length >= 2) return true

    // Single row might be incomplete (streaming)
    return false
}

/**
 * Normalize a parsed table
 */
function normalizeTable(table: ParsedTable, maxSize: number): string[] {
    if (table.rows.length === 0) return []

    // Performance check
    const totalCells = table.rows.reduce((sum, row) => sum + row.cells.length, 0)
    if (totalCells > maxSize) {
        return ['⚠️ Table too large to display safely']
    }

    // Identify structure
    const structure = analyzeTableStructure(table.rows)

    // Rebuild the table
    return rebuildNormalizedTable(table.rows, structure)
}

/**
 * Analyze table structure
 */
function analyzeTableStructure(rows: TableRow[]) {
    const nonSeparatorRows = rows.filter(row => !row.isSeparator)

    if (nonSeparatorRows.length === 0) {
        return { columnCount: 0, hasHeaders: false, separatorIndex: -1 }
    }

    // Determine column count from the most common row length
    const columnCounts = nonSeparatorRows.map(row => row.cells.length)
    const columnCount = mode(columnCounts) || Math.max(...columnCounts)

    // Find separator row
    const separatorIndex = rows.findIndex(row => row.isSeparator)

    // Assume first non-separator row is header
    const hasHeaders = nonSeparatorRows.length > 1

    return { columnCount, hasHeaders, separatorIndex }
}

/**
 * Get the most common value in an array
 */
function mode(arr: number[]): number | null {
    if (arr.length === 0) return null

    const counts: { [key: number]: number } = {}
    let maxCount = 0
    let modeValue = arr[0]

    for (const value of arr) {
        counts[value] = (counts[value] || 0) + 1
        if (counts[value] > maxCount) {
            maxCount = counts[value]
            modeValue = value
        }
    }

    return modeValue
}

/**
 * Rebuild a normalized table
 */
function rebuildNormalizedTable(rows: TableRow[], structure: { columnCount: number; hasHeaders: boolean }): string[] {
    const result: string[] = []
    const { columnCount, hasHeaders } = structure

    if (columnCount === 0) return result

    let headerAdded = false
    let separatorAdded = false

    for (const row of rows) {
        if (row.isSeparator) continue // Skip original separators

        // Normalize row to correct column count
        const normalizedCells = normalizeRowCells(row.cells, columnCount)
        const tableRow = '| ' + normalizedCells.join(' | ') + ' |'

        result.push(tableRow)

        // Add separator after first row if this is a header table
        if (hasHeaders && !headerAdded && !separatorAdded) {
            const separator = '| ' + Array(columnCount).fill('---').join(' | ') + ' |'
            result.push(separator)
            separatorAdded = true
        }

        headerAdded = true
    }

    return result
}

/**
 * Normalize row cells to match column count
 */
function normalizeRowCells(cells: TableCell[], targetCount: number): string[] {
    const normalized: string[] = []

    for (let i = 0; i < targetCount; i++) {
        if (i < cells.length) {
            normalized.push(cells[i].content || '')
        } else {
            normalized.push('') // Pad with empty cells
        }
    }

    return normalized
}

/**
 * Create a placeholder for incomplete tables during streaming
 */
function createTablePlaceholder(): string[] {
    return [
        '```',
        '📊 Table loading...',
        '```'
    ]
}

/**
 * Reconstruct the final content
 */
function reconstructContent(
    nonTableLines: (string | null)[],
    processedTables: string[][],
    originalTables: ParsedTable[]
): string {
    const result: string[] = []
    let tableIndex = 0

    for (let i = 0; i < nonTableLines.length; i++) {
        if (nonTableLines[i] !== null) {
            result.push(nonTableLines[i] as string)
        } else {
            // This was a table line, check if we're at the start of a table
            if (tableIndex < originalTables.length && i === originalTables[tableIndex].startLine) {
                // Add the processed table
                result.push(...processedTables[tableIndex])
                tableIndex++
            }
        }
    }

    return result.join('\n')
}

/**
 * Legacy compatibility - streaming version
 */
export function processStreamingTables(content: string): { display: string; raw: string } {
    const processed = processTablesUnified(content, { streaming: true })
    return {
        display: processed,
        raw: content
    }
}

/**
 * Legacy compatibility - final version
 */
export function processFinalTables(content: string): string {
    return processTablesUnified(content, { streaming: false })
}
