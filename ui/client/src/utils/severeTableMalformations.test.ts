import { describe, it, expect } from 'vitest'
import { simplifiedTableProcessor } from './simplifiedTableProcessor'

describe('Severe Table Malformation Tests - New Agent Issues', () => {
  
  describe('Issue 1: Headers-Separators-Data on Single Line', () => {
    it('should split headers, separators and data that are merged on single line', () => {
      const input = `**Microsoft (MSFT) – Sentiment Overview**| Sentiment Driver | Indicator | Tone | Key Points |
| --- | --- | --- | --- | Quarterly earnings news | CNBC – "stock dipped as investors focused on disappointing Azure revenue" | Mixed/Negative | The share price fell after Azure growth missed some expectations, creating short‑term downside pressure. |`

      const result = simplifiedTableProcessor(input)
      
      // Should have proper structure
      expect(result.processed).toContain('**Microsoft (MSFT) – Sentiment Overview**')
      expect(result.processed).toContain('| Sentiment Driver | Indicator | Tone | Key Points |')
      expect(result.processed).toContain('| --- | --- | --- | --- |')
      expect(result.processed).toContain('| Quarterly earnings news |')
      
      // Should not have merged lines
      expect(result.processed).not.toContain('| --- | --- | --- | --- | Quarterly earnings news')
    })

    it('should handle multiple instances of merged headers-separators-data', () => {
      const input = `| Sentiment Driver | Indicator | Tone | Key Points |
| --- | --- | --- | --- | Azure growth performance | CNBC – "revenue from Azure grew 39%" | Positive | Azure's growth outpaced analyst estimates. |
| --- | --- | --- | --- | Management outlook | CNBC – "management called for accelerating" | Positive | Forward‑looking guidance signals confidence. |`

      const result = simplifiedTableProcessor(input)
      
      expect(result.processed).toContain('| Azure growth performance |')
      expect(result.processed).toContain('| Management outlook |')
      expect(result.processed).not.toContain('| --- | --- | --- | --- | Azure')
      expect(result.processed).not.toContain('| --- | --- | --- | --- | Management')
    })
  })

  describe('Issue 2: Extreme Separator Duplication', () => {
    it('should remove multiple consecutive empty separator blocks', () => {
      const input = `| Country | Data |
| --- | --- |
|---|---|

|

| --- | --- |
|---|---|

|

| --- | --- |
|---|---|

| Spain | Value |`

      const result = simplifiedTableProcessor(input)
      
      expect(result.processed).toContain('| Country | Data |')
      expect(result.processed).toContain('| Spain | Value |')
      
      // Should have at most one separator row
      const separatorCount = (result.processed.match(/\| --- \| --- \|/g) || []).length
      expect(separatorCount).toBeLessThanOrEqual(1)
      
      // Should not have condensed separators
      expect(result.processed).not.toContain('|---|---|')
      
      // Should not have multiple empty lines
      expect(result.processed).not.toMatch(/\n{3,}/)
    })

    it('should handle extreme nested separator patterns', () => {
      const input = `| **Goal** – Compare the **after‑tax disposable income** | Data |
| --- | --- |
|---|---|

| --- | --- |
|---|---|

|

| --- | --- |
|---|---|

|

| --- | --- |
|---|---|

| 1️⃣ Married couple **without** children | 2️⃣ Married couple **with one** 3‑year‑old child |`

      const result = simplifiedTableProcessor(input)
      
      expect(result.processed).toContain('| **Goal**')
      expect(result.processed).toContain('| 1️⃣ Married couple')
      expect(result.processed).not.toContain('|---|---|')
      
      // Count total lines to ensure we're not creating excessive output
      const lineCount = result.processed.split('\n').length
      expect(lineCount).toBeLessThan(10)
    })
  })

  describe('Issue 3: Mixed Content-Separator Lines', () => {
    it('should fix lines with content followed by separator patterns', () => {
      const input = `| Item | Assumption |
| --- | --- | **Gross salary per partner** | **€45,000** (≈ US $48,500) – a typical middle‑income wage in Western Europe. |
| --- | **Household gross income** | €90,000 (both partners). |
| --- | **Currency conversion** | All figures are shown in **Euro (€)**. | --- |`

      const result = simplifiedTableProcessor(input)
      
      expect(result.processed).toContain('| Item | Assumption |')
      expect(result.processed).toContain('| **Gross salary per partner** | **€45,000**')
      expect(result.processed).toContain('| **Household gross income** | €90,000')
      
      // Should not have mixed content and separators
      expect(result.processed).not.toContain('| --- | --- | **Gross salary')
      expect(result.processed).not.toContain('| --- | **Household')
      expect(result.processed).not.toContain('Euro (€)**. | --- |')
    })

    it('should handle trailing separators on data rows', () => {
      const input = `| **Tax filing status** | Joint filing (married‑couple) where the law permits | --- |
| **Social‑security** | Employee‑share only | --- |
| **Child‑related cash benefits** | Only the **universal, non‑means‑tested** benefits | --- |`

      const result = simplifiedTableProcessor(input)
      
      // Should remove trailing | --- |
      expect(result.processed).toContain('| **Tax filing status** | Joint filing')
      expect(result.processed).toContain('| **Social‑security** | Employee‑share only |')
      // We allow a separator row between header and data, just not trailing on data
      expect(result.processed).not.toContain('benefits | --- |')
    })
  })

  describe('Issue 4: Complex Multi-Pattern Tables', () => {
    it('should handle tables with all malformation patterns combined', () => {
      const input = `| Country | Gross HH Income (€) | Income‑Tax (€) | Social‑Security (€) | Net Disposable (€) | --- | --- | --- | --- | --- |

| **Spain** | 90 000 | 13 200 | 6 750 | 70 050 |
|---|---|---|---|

| --- | --- | --- | --- |
|---|---|---|---|

|

| --- | --- | --- | --- |
|---|---|---|---|

| --- | --- | --- | --- |
|---|---|---|---|

| **France** | 90 000 | 16 500 | 7 200 | 66 300 |
|---|---|---|---|

| --- | --- | --- | --- |
|---|---|---|---|`

      const result = simplifiedTableProcessor(input)
      
      expect(result.processed).toContain('| Country | Gross HH Income')
      expect(result.processed).toContain('| **Spain** | 90 000')
      expect(result.processed).toContain('| **France** | 90 000')
      
      // Should not have any malformed patterns
      expect(result.processed).not.toContain('|---|---|---|---|')
      // A single properly formatted separator row is OK
      const separatorMatches = (result.processed.match(/\| --- \| --- \| --- \| --- \|/g) || [])
      expect(separatorMatches.length).toBeLessThanOrEqual(1)
      
      // Should have clean structure
      const lines = result.processed.split('\n').filter(l => l.trim())
      expect(lines.length).toBeLessThan(15) // Allow for proper table structure
    })
  })

  describe('Issue 5: Analyst Rating Tables', () => {
    it('should handle analyst rating snapshot tables', () => {
      const input = `**Analyst Rating Snapshot (as of the latest available data)**| Company | Rating Trend* | Recent Moves (if any) | Source |
| --- | --- | --- | --- | Microsoft (MSFT) | Slightly Positive | Upgrades noted on Yahoo Finance's "analyst estimates" page | [Source: Yahoo Finance] | Apple (AAPL) | Neutral/Positive (historical) | No specific upgrade/downgrade cited in provided sources | [Source: Yahoo Finance] |`

      const result = simplifiedTableProcessor(input)
      
      expect(result.processed).toContain('**Analyst Rating Snapshot')
      expect(result.processed).toContain('| Company | Rating Trend')
      expect(result.processed).toContain('| Microsoft (MSFT) |')
      expect(result.processed).toContain('| Apple (AAPL) |')
      
      // Should not have merged header and data
      expect(result.processed).not.toContain('| --- | --- | --- | --- | Microsoft')
    })
  })

  describe('Issue 6: Empty Rows and Pipes', () => {
    it('should handle tables with empty pipe rows', () => {
      const input = `| Header 1 | Header 2 |
|

| --- | --- |
|

| Data 1 | Data 2 |
|`

      const result = simplifiedTableProcessor(input)
      
      expect(result.processed).toContain('| Header 1 | Header 2 |')
      expect(result.processed).toContain('| Data 1 | Data 2 |')
      // Check that we don't have standalone pipes
      const lines = result.processed.split('\n')
      const hasSoloPipe = lines.some(l => l.trim() === '|')
      expect(hasSoloPipe).toBe(false)
    })
  })

  describe('Issue 7: Tax Calculation Tables', () => {
    it('should handle complex tax calculation tables', () => {
      const input = `| Country | Income‑Tax Rate(s) Used | Social‑Security Rate (employee) | Child‑Benefit (none) | Key Sources | --- | --- | --- | --- | --- | Spain – 2024 | 19 % up to €12 450, 24 % €12 451‑€20 200 | 7.5 % (common payroll) | — | Source: Spanish Tax Agency |---|---|---|---|

| --- | --- | --- | --- |
|---|---|---|---|`

      const result = simplifiedTableProcessor(input)
      
      expect(result.processed).toContain('| Country | Income‑Tax Rate(s) Used')
      expect(result.processed).toContain('| Spain – 2024 |')
      expect(result.processed).not.toContain('| --- | --- | --- | --- | --- | Spain')
      expect(result.processed).not.toContain('|---|---|---|---|')
    })
  })
})

describe('Performance Tests for Severe Malformations', () => {
  it('should handle extremely malformed tables efficiently', () => {
    // Generate a table with 100 separator blocks
    const separatorBlock = `| --- | --- |
|---|---|

|

`
    const input = `| Header 1 | Header 2 |
${separatorBlock.repeat(100)}
| Data 1 | Data 2 |`

    const start = performance.now()
    const result = simplifiedTableProcessor(input)
    const end = performance.now()

    expect(end - start).toBeLessThan(100) // Should process quickly
    expect(result.processed).toContain('| Header 1 | Header 2 |')
    expect(result.processed).toContain('| Data 1 | Data 2 |')
    expect(result.processed).not.toContain('|---|---|')
  })

  it('should handle massive single-line merged tables', () => {
    const rows = Array(50).fill(0).map((_, i) => 
      `| --- | --- | --- | --- | Row ${i} Col 1 | Row ${i} Col 2 | Row ${i} Col 3 | Row ${i} Col 4 |`
    )
    const input = rows.join('\n')

    const start = performance.now()
    const result = simplifiedTableProcessor(input)
    const end = performance.now()

    expect(end - start).toBeLessThan(200)
    expect(result.processed).toContain('| Row 0 Col 1')
    expect(result.processed).toContain('| Row 49 Col 1')
    expect(result.processed).not.toContain('| --- | --- | --- | --- | Row')
  })
})