import { test, expect } from '@playwright/test'
import { processTablesImproved } from '../src/utils/improvedTableReconstructor'
import { validateTableContent } from '../src/utils/tableValidator'

test.describe('Table Regression Tests', () => {
    test('should not break on GitHub issue #123 pattern', async () => {
        // Simulated issue: Tables with special characters break reconstruction
        const input = `| Symbol | Value |
| --- | --- |
| € | Euro |
| £ | Pound |
| ¥ | Yen |`

        const result = processTablesImproved(input)
        expect(result).toContain('€')
        expect(result).toContain('£')
        expect(result).toContain('¥')
    })

    test('should preserve URLs with query parameters', async () => {
        // Regression: URLs were being broken by pipe character in query params
        const input = `| API | Endpoint |
| --- | --- |
| Users | /api/users?filter=active&sort=name |
| Orders | /api/orders?date=2024-01-01&status=pending |`

        const result = processTablesImproved(input)
        expect(result).toContain('filter=active&sort=name')
        expect(result).toContain('date=2024-01-01&status=pending')
    })

    test('should handle financial data with negative values', async () => {
        // Regression: Negative numbers with dashes confused separator detection
        const input = `| Company | Profit |
| --- | --- |
| TechCorp | -€50,000 |
| StartupInc | -$25,000 |`

        const result = processTablesImproved(input)
        expect(result).toContain('-€50,000')
        expect(result).toContain('-$25,000')
    })

    test('should not corrupt code snippets in tables', async () => {
        // Regression: Code with pipes was breaking table structure
        const input = `| Language | Pipe Operator |
| --- | --- |
| Bash | \`cat file.txt | grep "pattern"\` |
| PowerShell | \`Get-Process | Where-Object {$_.CPU -gt 100}\` |`

        const result = processTablesImproved(input)
        expect(result).toContain('cat file.txt | grep')
        expect(result).toContain('Get-Process | Where-Object')
    })

    test('should handle scientific notation', async () => {
        // Regression: Scientific notation with 'e' was being processed incorrectly
        const input = `| Constant | Value |
| --- | --- |
| Avogadro | 6.022e23 |
| Planck | 6.626e-34 |`

        const result = processTablesImproved(input)
        expect(result).toContain('6.022e23')
        expect(result).toContain('6.626e-34')
    })
})
