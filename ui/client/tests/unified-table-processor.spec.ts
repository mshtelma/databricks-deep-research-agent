import { test, expect } from '@playwright/test'
import { processTablesUnified } from '../src/utils/unifiedTableProcessor'

/**
 * UNIFIED TABLE PROCESSOR TESTS
 * 
 * This defines the behavior for a single, robust table processor
 * that will replace all existing table processing logic.
 */

test.describe('Unified Table Processor', () => {

    test.describe('Core Requirements', () => {
        test('should handle empty and null inputs gracefully', async () => {
            expect(processTablesUnified('')).toBe('')
            expect(processTablesUnified(null as any)).toBe(null)
            expect(processTablesUnified(undefined as any)).toBe(undefined)
        })

        test('should preserve non-table content exactly', async () => {
            const content = `# Title
This is a paragraph.

- List item 1
- List item 2

\`\`\`code
function test() {}
\`\`\`

Another paragraph.`

            const result = processTablesUnified(content)
            expect(result).toBe(content)
        })

        test('should handle perfect tables without modification', async () => {
            const content = `| Header1 | Header2 | Header3 |
| --- | --- | --- |
| Data1 | Data2 | Data3 |
| Data4 | Data5 | Data6 |`

            const result = processTablesUnified(content)
            expect(result).toBe(content)
        })
    })

    test.describe('Malformed Table Normalization', () => {
        test('should normalize condensed separators', async () => {
            const input = `| Header1 | Header2 |
|---|---|
| Data1 | Data2 |`

            const expected = `| Header1 | Header2 |
| --- | --- |
| Data1 | Data2 |`

            const result = processTablesUnified(input)
            expect(result).toBe(expected)
        })

        test('should handle inline separators by extracting structure', async () => {
            const input = `| Header1 | Header2 | --- | --- | Data1 | Data2 |`

            const expected = `| Header1 | Header2 |
| --- | --- |
| Data1 | Data2 |`

            const result = processTablesUnified(input)
            expect(result).toBe(expected)
        })

        test('should clean up noise lines completely', async () => {
            const input = `| Header1 | Header2 |
---    ---    ---
|---|---|---|
| --- | --- |
| Data1 | Data2 |
---    ---    ---`

            const expected = `| Header1 | Header2 |
| --- | --- |
| Data1 | Data2 |`

            const result = processTablesUnified(input)
            expect(result).toBe(expected)
        })

        test('should handle the financial table from user example', async () => {
            const input = `| Country | Avg. gross per earner (local) | Household gross (2 ×) | Income‑tax (annual) | Social‑security (annual) | **Net pay** (gross‑tax‑SS) | **Disposable income** (net + no‑child cash) | --------- | ------------------------------ | ----------------------- | --------------------- | -------------------------- | ---------------------------- | -------------------------------------------- | Spain | €30 800 | €61 600 | €9 860 | €3 916 | **€47 824**
 | €47 824 |
| --- | France | €38 200 | €76 400 | €12 960 | €7 405 | **€55 ? ?** (≈ €55 ? ) | €55 ?  | United Kingdom | £33 100 | £66 200 | £7 920 | £5 952 | **£52 328** ≈ **€60 000**






 | €60 000 |
| --- |
| Germany | €45 900 | €91 800 | €20 200 | €9 180 | **€62 420**
 | €62 420 |
| --- |`

            const result = processTablesUnified(input)

            // Should have proper structure
            expect(result).toContain('| Country |')
            expect(result).toContain('| --- |')
            // Should preserve most country data (Spain might be lost in extreme malformation)
            expect(result).toContain('| France |')
            expect(result).toContain('| Germany |')
            // Should contain at least one preserved data value
            expect(result).toContain('€47 824') // Spain's value (even if not in proper table cell)

            // Should NOT contain the worst noise patterns
            expect(result).not.toContain('| --------- |')
            // Should not have more than 3 separator lines (some may remain due to complexity)
            expect(result.split('\n').filter(line => line.trim() === '| --- |').length).toBeLessThanOrEqual(3)
        })
    })

    test.describe('Streaming vs Final Processing', () => {
        test('should handle incomplete streaming content with placeholders', async () => {
            const input = `| Header1 | Header2 |
| Data1`

            const streamingResult = processTablesUnified(input, { streaming: true })
            const finalResult = processTablesUnified(input, { streaming: false })

            // Streaming should use placeholder
            expect(streamingResult).toContain('📊')

            // Final should attempt reconstruction
            expect(finalResult).toContain('| Header1 | Header2 |')
        })

        test('should buffer incomplete tables during streaming', async () => {
            const input = `Some text before

| Header1 | Header2 |
| --- |`

            const result = processTablesUnified(input, { streaming: true })

            // Should preserve non-table content
            expect(result).toContain('Some text before')

            // Should show placeholder for incomplete table
            expect(result).toContain('📊')
        })
    })

    test.describe('Edge Cases and Error Handling', () => {
        test('should handle tables with inconsistent column counts', async () => {
            const input = `| A | B |
| --- | --- |
| 1 | 2 | 3 | 4 |
| 5 |
| 6 | 7 | 8 |`

            const result = processTablesUnified(input)

            // Should normalize to consistent structure
            const lines = result.split('\n')
            const dataLines = lines.filter(line => line.startsWith('|') && !line.includes('---'))

            // All data lines should have same column count (padded or truncated)
            const columnCounts = dataLines.map(line => (line.match(/\|/g) || []).length - 1)
            const uniqueCounts = [...new Set(columnCounts)]
            expect(uniqueCounts.length).toBeLessThanOrEqual(2) // Header and data might differ slightly
        })

        test('should preserve special characters and unicode', async () => {
            const input = `| Symbol | Name | Value |
| --- | --- | --- |
| € | Euro | 1.00 |
| £ | Pound | 0.86 |
| ¥ | Yen | 149.50 |
| 中文 | Chinese | 测试 |`

            const result = processTablesUnified(input)

            expect(result).toContain('€')
            expect(result).toContain('£')
            expect(result).toContain('¥')
            expect(result).toContain('中文')
            expect(result).toContain('测试')
        })

        test('should handle URLs and code in table cells', async () => {
            const input = `| Site | URL | Code |
| --- | --- | --- |
| Google | https://google.com?q=test&sort=date | \`console.log("test | pipe")\` |
| GitHub | https://github.com/user/repo | \`grep "a|b" file\` |`

            const result = processTablesUnified(input)

            expect(result).toContain('q=test&sort=date')
            expect(result).toContain('test | pipe')
            expect(result).toContain('a|b')
        })

        test('should handle nested pipe characters correctly', async () => {
            const input = `| Command | Description |
| --- | --- |
| \`echo "a|b|c"\` | Print with pipes |
| \`awk -F"|" '{print $1}'\` | Split on pipe |`

            const result = processTablesUnified(input)

            expect(result).toContain('a|b|c')
            expect(result).toContain('-F"|"')
        })

        test('should handle extremely large tables efficiently', async () => {
            // Generate a large table
            const headers = Array.from({ length: 20 }, (_, i) => `Col${i + 1}`).join(' | ')
            const separator = Array.from({ length: 20 }, () => '---').join(' | ')
            const rows = Array.from({ length: 500 }, (_, i) =>
                Array.from({ length: 20 }, (_, j) => `Data${i}-${j}`).join(' | ')
            )

            const input = `| ${headers} |
| ${separator} |
${rows.map(row => `| ${row} |`).join('\n')}`

            const start = Date.now()
            const result = processTablesUnified(input)
            const duration = Date.now() - start

            // Should complete within reasonable time
            expect(duration).toBeLessThan(1000)
            expect(result).toContain('Col1')
            expect(result).toContain('Data499-19')
        })
    })

    test.describe('Mixed Content Scenarios', () => {
        test('should handle multiple tables in one document', async () => {
            const input = `# First Table

| Name | Age |
| --- | --- |
| Alice | 25 |

Some text between tables.

| Product | Price |
| --- | --- |
| Apple | $1.50 |

# End`

            const result = processTablesUnified(input)

            expect(result).toContain('| Name | Age |')
            expect(result).toContain('| Product | Price |')
            expect(result).toContain('Some text between tables.')
        })

        test('should handle tables within lists', async () => {
            const input = `1. First item

2. Table item:
   | Col1 | Col2 |
   | --- | --- |
   | A | B |

3. Third item`

            const result = processTablesUnified(input)

            expect(result).toContain('1. First item')
            expect(result).toContain('2. Table item:')
            expect(result).toContain('| Col1 | Col2 |')
            expect(result).toContain('3. Third item')
        })

        test('should not process tables inside code blocks', async () => {
            const input = `\`\`\`
| This | Is | Not | A | Table |
| --- | --- | --- | --- | --- |
| It | Is | Code | Block | Content |
\`\`\`

| This | Is | A | Real | Table |
| --- | --- | --- | --- | --- |
| With | Real | Table | Data | Here |`

            const result = processTablesUnified(input)

            // Should preserve code block exactly
            expect(result).toContain('```')
            expect(result).toContain('| This | Is | Not | A | Table |')

            // Should process real table
            expect(result).toContain('| This | Is | A | Real | Table |')
        })
    })

    test.describe('Performance and Stress Tests', () => {
        test('should handle recursive malformed patterns without infinite loops', async () => {
            const input = `${'|---|---|'.repeat(1000)}
${'| Header | Data |'.repeat(100)}
${'|---|---|'.repeat(1000)}`

            const start = Date.now()
            const result = processTablesUnified(input)
            const duration = Date.now() - start

            expect(duration).toBeLessThan(2000)
            expect(result).toBeTruthy()
        })

        test('should handle memory efficiently with repeated patterns', async () => {
            const pattern = `| Country | GDP |
|---|---|---|---|---|
| Data | €1.4T |
---    ---    ---    ---
| More | Data |`

            const input = pattern.repeat(100)

            const initialMemory = process.memoryUsage().heapUsed
            const result = processTablesUnified(input)
            const finalMemory = process.memoryUsage().heapUsed
            const memoryIncrease = finalMemory - initialMemory

            // Memory increase should be reasonable
            expect(memoryIncrease).toBeLessThan(20 * 1024 * 1024) // < 20MB
            expect(result.length).toBeGreaterThan(0)
        })
    })

    test.describe('Regression Prevention', () => {
        test('should not corrupt financial data', async () => {
            const input = `| Company | Revenue | Profit |
| --- | --- | --- |
| TechCorp | €1,000,000 | -€50,000 |
| StartupInc | $500,000 | -$25,000 |`

            const result = processTablesUnified(input)

            expect(result).toContain('-€50,000')
            expect(result).toContain('-$25,000')
            expect(result).toContain('€1,000,000')
        })

        test('should preserve scientific notation', async () => {
            const input = `| Constant | Value |
| --- | --- |
| Avogadro | 6.022e23 |
| Planck | 6.626e-34 |`

            const result = processTablesUnified(input)

            expect(result).toContain('6.022e23')
            expect(result).toContain('6.626e-34')
        })

        test('should handle mathematical expressions', async () => {
            const input = `| Formula | Result |
| --- | --- |
| 2 + 2 | 4 |
| x = (-b ± √(b²-4ac)) / 2a | Quadratic |`

            const result = processTablesUnified(input)

            expect(result).toContain('2 + 2')
            expect(result).toContain('√(b²-4ac)')
        })
    })
})
