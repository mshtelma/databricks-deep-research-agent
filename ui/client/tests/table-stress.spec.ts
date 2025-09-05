import { test, expect } from '@playwright/test'
import { processTablesImproved } from '../src/utils/improvedTableReconstructor'

test.describe('Table Stress Tests', () => {
    test('should handle recursive malformed patterns', async () => {
        const input = `| Header |${' |---|'.repeat(100)}
${'|---|'.repeat(200)}
| Data | Data |${' |---|'.repeat(50)}`

        const start = Date.now()
        const result = processTablesImproved(input)
        const duration = Date.now() - start

        expect(duration).toBeLessThan(3000)
        expect(result).toBeTruthy()
    })

    test('should handle extreme separator combinations', async () => {
        const patterns = [
            '|---|---|---|---|---|---|---|---|---|---|',
            '| --------- | ------------- | --------- |',
            '| --- | --- | --- | --- | --- | --- |',
            '|||||||||||||||||||||||',
            '---    ---    ---    ---    ---'
        ].join('\n').repeat(50)

        expect(() => processTablesImproved(patterns)).not.toThrow()
    })

    test('should handle corrupted streaming patterns', async () => {
        const input = `| Country | GDP |
| Fra| Fra| Fra| Fra| Fra| Fra
|---|---|---|Fra|---|---|Fra
| Spain | €1.4T |Fra|Fra|Fra
|---|---|---Fra---Fra---Fra
| Germany |Fra€4.2TFra|Fra|Fra`

        const result = processTablesImproved(input)
        expect(result).toContain('Country')
        expect(result).toContain('Spain')
    })

    test('should handle massive inline separators', async () => {
        const input = `| Header1 | Header2 | ${'| --- |'.repeat(200)} Data1 | Data2 |`

        const start = Date.now()
        const result = processTablesImproved(input)
        const duration = Date.now() - start

        expect(duration).toBeLessThan(2000)
        expect(result).toContain('Header1')
    })
})
