import { test, expect } from '@playwright/test'
import { validateTableContent, autoFixTableIssues, TableIssueType } from '../src/utils/tableValidator'

test.describe('Table Validator Error Handling', () => {
    test.describe('Input Validation', () => {
        test('should handle null and undefined input gracefully', async () => {
            expect(() => validateTableContent(null as any)).not.toThrow()
            expect(() => validateTableContent(undefined as any)).not.toThrow()
            expect(() => autoFixTableIssues(null as any)).not.toThrow()
            expect(() => autoFixTableIssues(undefined as any)).not.toThrow()
        })

        test('should handle empty string input', async () => {
            const result = validateTableContent('')
            expect(result.isValid).toBe(true)
            expect(result.issues).toHaveLength(0)
            expect(result.stats.tableCount).toBe(0)
        })

        test('should handle very large input without crashing', async () => {
            const largeInput = '| Col1 | Col2 |\n'.repeat(10000) + '| --- | --- |\n'.repeat(1000)
            const start = Date.now()

            expect(() => validateTableContent(largeInput)).not.toThrow()
            expect(() => autoFixTableIssues(largeInput)).not.toThrow()

            const duration = Date.now() - start
            expect(duration).toBeLessThan(5000) // Should complete within 5 seconds
        })
    })

    test.describe('Malformed Pattern Detection', () => {
        test('should detect duplicate separators', async () => {
            const input = `| Header1 | Header2 |
| --- | --- |
| --- | --- |
| Data1 | Data2 |`

            const result = validateTableContent(input)
            expect(result.isValid).toBe(false)
            expect(result.issues.some(i => i.type === TableIssueType.DUPLICATE_SEPARATOR)).toBe(true)
        })

        test('should detect inline tables', async () => {
            const input = `| Header1 | Header2 | --- | --- | Data1 | Data2 |`

            const result = validateTableContent(input)
            expect(result.isValid).toBe(false)
            expect(result.issues.some(i => i.type === TableIssueType.INLINE_TABLE)).toBe(true)
            expect(result.stats.inlineTables).toBeGreaterThan(0)
        })

        test('should detect malformed separators', async () => {
            const input = `| Header1 | Header2 |
|---|---|---|---|---|---|
| Data1 | Data2 |`

            const result = validateTableContent(input)
            expect(result.isValid).toBe(false)
            expect(result.issues.some(i => i.type === TableIssueType.MALFORMED_SEPARATOR)).toBe(true)
        })

        test('should detect orphaned separators', async () => {
            const input = `Some text
| --- | --- |
More text`

            const result = validateTableContent(input)
            expect(result.issues.some(i => i.type === TableIssueType.ORPHANED_SEPARATOR)).toBe(true)
        })

        test('should detect inconsistent columns', async () => {
            const input = `| Header1 | Header2 |
| --- | --- |
| Data1 | Data2 |
| Data3 | Data4 | Data5 | Data6 |`

            const result = validateTableContent(input)
            expect(result.issues.some(i => i.type === TableIssueType.INCONSISTENT_COLUMNS)).toBe(true)
        })
    })

    test.describe('Auto-Fix Functionality', () => {
        test('should fix duplicate separators', async () => {
            const input = `| Header1 | Header2 |
| --- | --- |
| --- | --- |
| Data1 | Data2 |`

            const fixed = autoFixTableIssues(input)
            const lines = fixed.split('\n')
            const separatorLines = lines.filter(line => /^\| --- \| --- \|$/.test(line.trim()))
            expect(separatorLines.length).toBe(1)
        })

        test('should remove noise lines', async () => {
            const input = `| Header1 | Header2 |
---    ---    ---
| --- | --- |
| Data1 | Data2 |
---    ---    ---`

            const fixed = autoFixTableIssues(input)
            expect(fixed).not.toContain('---    ---    ---')
            expect(fixed).toContain('| Header1 | Header2 |')
            expect(fixed).toContain('| Data1 | Data2 |')
        })

        test('should handle mixed valid and invalid content', async () => {
            const input = `# Title
| Valid | Table |
| --- | --- |
| Data | Here |

---    ---    ---

| Another | Table | --- | --- |
| With | Issues |`

            const fixed = autoFixTableIssues(input)
            expect(fixed).toContain('# Title')
            expect(fixed).toContain('| Valid | Table |')
            expect(fixed).not.toContain('---    ---    ---')
        })
    })

    test.describe('Edge Cases for Validator', () => {
        test('should handle tables with only separators', async () => {
            const input = `| --- | --- | --- |
| --- | --- | --- |
| --- | --- | --- |`

            const result = validateTableContent(input)
            expect(result.stats.tableCount).toBeGreaterThanOrEqual(0)

            const fixed = autoFixTableIssues(input)
            expect(fixed.length).toBeLessThan(input.length) // Should be cleaned up
        })

        test('should handle unicode in tables', async () => {
            const input = `| 中文 | العربية | Ελληνικά |
| --- | --- | --- |
| データ | معلومات | δεδομένα |`

            const result = validateTableContent(input)
            expect(result.isValid).toBe(true)

            const fixed = autoFixTableIssues(input)
            expect(fixed).toContain('中文')
            expect(fixed).toContain('العربية')
            expect(fixed).toContain('Ελληνικά')
        })

        test('should handle tables with markdown formatting', async () => {
            const input = `| **Bold** | *Italic* | \`Code\` |
| --- | --- | --- |
| [Link](url) | ![Image](url) | ~~Strike~~ |`

            const result = validateTableContent(input)
            expect(result.isValid).toBe(true)

            const fixed = autoFixTableIssues(input)
            expect(fixed).toContain('**Bold**')
            expect(fixed).toContain('[Link](url)')
        })

        test('should handle extremely malformed input gracefully', async () => {
            const input = `||||||||||||
|---|---|---|---|---|---||||
||||||data||||||||
|||||||---|---|---|
||||||||||||||||||`

            expect(() => validateTableContent(input)).not.toThrow()
            expect(() => autoFixTableIssues(input)).not.toThrow()

            const result = validateTableContent(input)
            expect(result.issues.length).toBeGreaterThan(0)
        })

        test('should handle input with only pipe characters', async () => {
            const input = `|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||`

            const result = validateTableContent(input)
            const fixed = autoFixTableIssues(input)

            expect(result).toBeTruthy()
            expect(fixed.length).toBeLessThanOrEqual(input.length) // Should be cleaned up or unchanged
        })

        test('should handle tables mixed with code blocks', async () => {
            const input = `\`\`\`
| This | Is | Not | A | Table |
| --- | --- | --- | --- | --- |
\`\`\`

| But | This | Is |
| --- | --- | --- |
| Real | Table | Data |`

            const result = validateTableContent(input)
            const fixed = autoFixTableIssues(input)

            expect(fixed).toContain('```')
            expect(fixed).toContain('| But | This | Is |')
        })
    })

    test.describe('Performance and Stress Tests', () => {
        test('should handle deeply nested malformed patterns', async () => {
            const input = `| Header |${' | --- |'.repeat(50)}
${'| Data |'.repeat(25)}${'| --- |'.repeat(25)}
| More | Data |${'| --- | --- |'.repeat(20)}`

            const start = Date.now()
            const result = validateTableContent(input)
            const fixed = autoFixTableIssues(input)
            const duration = Date.now() - start

            expect(duration).toBeLessThan(2000) // Should complete within 2 seconds
            expect(result).toBeTruthy()
            expect(fixed).toBeTruthy()
        })

        test('should handle repetitive malformed patterns', async () => {
            const patterns = [
                '|---|---|---|',
                '| --- | --- | --- |',
                '---    ---    ---',
                '| | | |',
                '||||||||'
            ]

            const input = patterns.join('\n').repeat(100)

            const start = Date.now()
            expect(() => validateTableContent(input)).not.toThrow()
            expect(() => autoFixTableIssues(input)).not.toThrow()
            const duration = Date.now() - start

            expect(duration).toBeLessThan(3000) // Should complete within 3 seconds
        })

        test('should handle memory efficiently with large tables', async () => {
            // Create a table with many columns and rows
            const columnCount = 50
            const rowCount = 200

            const headers = Array.from({ length: columnCount }, (_, i) => `Header${i}`).join(' | ')
            const separator = Array.from({ length: columnCount }, () => '---').join(' | ')
            const rows = Array.from({ length: rowCount }, (_, i) =>
                Array.from({ length: columnCount }, (_, j) => `Data${i}-${j}`).join(' | ')
            )

            const input = `| ${headers} |
| ${separator} |
${rows.map(row => `| ${row} |`).join('\n')}`

            const initialMemory = process.memoryUsage().heapUsed

            const result = validateTableContent(input)
            const fixed = autoFixTableIssues(input)

            const finalMemory = process.memoryUsage().heapUsed
            const memoryIncrease = finalMemory - initialMemory

            // Memory increase should be reasonable (< 50MB)
            expect(memoryIncrease).toBeLessThan(50 * 1024 * 1024)
            expect(result.stats.tableCount).toBeGreaterThan(0)
            expect(fixed.length).toBeGreaterThan(0)
        })
    })
})
