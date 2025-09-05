import { describe, it, expect } from 'vitest'
import { simplifiedTableProcessor } from './simplifiedTableProcessor'

describe('Malformed Table Tests - Real Agent Output Issues', () => {
  
  describe('Issue 1: Mixed Header-Separator Pattern', () => {
    it('should handle tables with headers and separators mixed on same line', () => {
      const input = `## Scope & Methodology  

| Item | Description |
| ------ | ------------- |

| **Countries** | Spain, France, United Kingdom, Switzerland (canton Zug), Germany, Poland, Bulgaria |
| --- | **Family set‑ups** | 1️⃣ Married couple **without** children 2️⃣ Married couple **with one** child (age 3) |
| --- |`

      const result = simplifiedTableProcessor(input)
      
      expect(result.processed).toContain('| Item | Description |')
      expect(result.processed).toContain('| --- | --- |')
      expect(result.processed).toContain('| **Countries** | Spain, France, United Kingdom')
      expect(result.processed).not.toContain('| --- | **Family')
      expect(result.issues.length).toBe(0)
    })
  })

  describe('Issue 2: Complex Mixed Content Tables', () => {
    it('should handle tables with mixed content and separator patterns', () => {
      const input = `| **Assumed gross earnings** | Each spouse earns the **median annual gross wage** of the respective country (‑‑ see *Source 1*). The two‑earner household therefore has a **combined gross income** equal to twice the national median. | --- |

| **Currencies** | All results are expressed in **Euro (€)** for an apples‑to‑apples comparison. Local‑currency amounts are converted using 2024 average exchange rates (ECB) – e.g., 1 CHF = 0.92 €, 1 GBP = 1.17 €, 1 PLN = 0.20 €, 1 BGN = 0.51 € (*Source 2*). | --- |`

      const result = simplifiedTableProcessor(input)
      
      expect(result.processed).toContain('| **Assumed gross earnings** |')
      // The trailing | --- | should be removed from content cells
      expect(result.processed).not.toContain('| --- |$')
      expect(result.issues.length).toBe(0)
    })
  })

  describe('Issue 3: Multiple Broken Separator Rows', () => {
    it('should handle tables with multiple broken separator patterns', () => {
      const input = `| Country | Median gross wage (per spouse) | Combined gross income | Total tax + social contributions | Net disposable income* | --------- | ------------------------------- | ----------------------- | -------------------------------- | ------------------------ | Spain | €22 800 | €45 600 | €13 800 (30.3 %) | **€31 800**

 |

| France | €23 500 | €47 000 | €15 200 (32.3 %) | **€31 800**
|---|---|---|---| --- | --- | --- | --- |
|---|---|---|---|
| --- | --- | --- | --- |
| --- | --- | --- | --- |`

      const result = simplifiedTableProcessor(input)
      
      expect(result.processed).toContain('| Country | Median gross wage')
      expect(result.processed).toContain('| Spain | €22 800')
      expect(result.processed).toContain('| France | €23 500')
      expect(result.processed).not.toContain('|---|---|---|---|')
      // The separator row is valid, so we don't check for its absence
      // We just ensure the malformed patterns are gone
      const separatorCount = (result.processed.match(/\| --- \| --- \| --- \| --- \|/g) || []).length
      expect(separatorCount).toBeLessThanOrEqual(1) // At most one valid separator row
    })
  })

  describe('Issue 4: Nested Table Structures', () => {
    it('should handle nested table-like structures', () => {
      const input = `| United Kingdom | £28 000 ≈ €32 800 | €65 600 | £19 200 ≈ €22 500 (34.3 %) | **€43 100**|---|---|---|---|

| --- | --- | --- | --- |
|---|---|---|---|
| --- | --- | --- | --- |
| --- | --- | --- | --- |`

      const result = simplifiedTableProcessor(input)
      
      expect(result.processed).toContain('| United Kingdom | £28 000')
      expect(result.processed).not.toContain('|---|---|---|---|')
      expect(result.processed.split('| --- | --- | --- | --- |').length).toBe(1)
    })
  })

  describe('Issue 5: Orphaned Content Between Tables', () => {
    it('should handle orphaned content between table rows', () => {
      const input = `| Switzerland (Zug) | CHF 70 000 ≈ €64 400 | €128 800 | CHF 41 000 ≈ €37 700 (29.3 %) | **€91 100**|---|---|---|---|

| --- | --- | --- | --- |
|---|---|---|---|
| --- | --- | --- | --- |
| --- | --- | --- | --- |




 | Germany | €30 000 | €60 000 | €18 300 (30.5 %) | **€41 700**
|---|---|---|---| --- | --- | --- | --- |
|---|---|---|---|`

      const result = simplifiedTableProcessor(input)
      
      expect(result.processed).toContain('| Switzerland (Zug)')
      expect(result.processed).toContain('| Germany | €30 000')
      expect(result.processed).not.toContain('|---|---|---|')
    })
  })

  describe('Issue 6: Tables with Excessive Separators', () => {
    it('should handle tables with excessive separator patterns', () => {
      const input = `| Country | Child‑related cash benefit (annual) | Tax‑credit/allowance impact* | Net disposable income (incl. child benefit) | --------- | ------------------------------------ | ------------------------------ | ---------------------------------------------- | Spain | €300 (family allowance) | €250 reduction in taxable income → €75 tax saving | **€32 175**`

      const result = simplifiedTableProcessor(input)
      
      expect(result.processed).toContain('| Country | Child‑related cash benefit')
      expect(result.processed).toContain('| Spain | €300')
      expect(result.processed).not.toContain('| --------- |')
    })
  })

  describe('Issue 7: Comparative Summary Table', () => {
    it('should handle comparative summary tables correctly', () => {
      const input = `| Country | Net disposable (no child) | Net disposable (1 child) | Increment from child benefit | --------- | --------------------------- | --------------------------- | ------------------------------ |

| **Switzerland (Zug)** | €91 100 | €94 600 | **+€3 500** (≈ 3.9 %) |
| --- | --- | --- | United Kingdom | €43 100 | €46 730 | **+€3 630** (≈ 8.4 %) | Germany | €41 700 | €42 350 | **+€650** (≈ 1.6 %) |`

      const result = simplifiedTableProcessor(input)
      
      expect(result.processed).toContain('| Country | Net disposable (no child)')
      expect(result.processed).toContain('| **Switzerland (Zug)** | €91 100')
      expect(result.processed).not.toContain('| --- | --- | --- | United Kingdom')
    })
  })

  describe('Issue 8: Sources Table with Mixed Format', () => {
    it('should handle sources table with mixed formatting', () => {
      const input = `| # | Source | What it provided |
| --- | -------- | ------------------ | 1 | **OECD – "Median wages" 2023** (OECD.stat) | Median annual gross earnings for each country (used as per‑spouse income). | 2 | **European Central Bank – "Average exchange rates 2024"**
 | Conversion rates: 1 CHF = 0.92 €, 1 GBP = 1.17 €, 1 PLN = 0.20 €, 1 BGN = 0.51 €. |
| --- | 3 | **Swiss Federal Tax Administration (SFTA) – "Tax Calculator for Zug, 2024"** | Income‑tax + social‑security rates for a married two‑earner household; built‑in child‑allowance. | --- |`

      const result = simplifiedTableProcessor(input)
      
      expect(result.processed).toContain('| # | Source | What it provided |')
      expect(result.processed).toContain('| 1 | **OECD')
      expect(result.processed).not.toContain('| --- | 3 |')
    })
  })

  describe('Issue 9: Empty Columns and Cells', () => {
    it('should handle tables with empty columns and cells', () => {
      const input = `| |---|---|
| United Kingdom | €100,000 |

---
---
€7,200
---`

      const result = simplifiedTableProcessor(input)
      
      expect(result.processed).toContain('United Kingdom')
      expect(result.processed).toContain('€100,000')
      expect(result.processed).toContain('€7,200')
      expect(result.processed).not.toContain('| |---|---|')
    })
  })

  describe('Issue 10: Malformed Multi-line Content', () => {
    it('should handle tables with malformed multi-line content', () => {
      const input = `| 4 | **OECD Tax‑Benefit Model 2024** (country‑specific runs) | Full calculation of income tax, payroll contributions, child‑allowances/credits, and net disposable income. | 5 | **PwC – "Worldwide Tax Summaries 2024"** (Spain, France, UK, Germany, Poland, Bulgaria) | Confirmation of tax brackets, payroll‑tax rates, and child‑benefit amounts. | 6 | **HM Revenue & Customs (UK) – "Child Benefit & Tax Credits 2024"**
 | Child Benefit rate (£2 920) and "Child Tax Credit" for one child. |
| --- | 7 | **Service-Public.fr – "Allocation familiale"**
 | French family allowance (€1 200) and child‑tax‑credit (€1 000). |`

      const result = simplifiedTableProcessor(input)
      
      expect(result.processed).toContain('| 4 | **OECD Tax‑Benefit Model')
      expect(result.processed).toContain('| 5 | **PwC')
      expect(result.processed).toContain('| 6 | **HM Revenue')
      expect(result.processed).not.toContain('| --- | 7 |')
    })
  })
})

describe('Edge Cases and Complex Scenarios', () => {
  
  it('should handle completely broken tables gracefully', () => {
    const input = `| --- | --- | --- | --- |
|--- |---|---|---|
| --- | --- | --- | --- |
| --- | --- | --- | --- |

| |---|---|
| Data | Value |`

    const result = simplifiedTableProcessor(input)
    
    expect(result.processed).toContain('| Data | Value |')
    expect(result.processed).not.toContain('| --- | --- | --- | --- |')
  })

  it('should preserve valid markdown tables', () => {
    const input = `| Header 1 | Header 2 | Header 3 |
| --- | --- | --- |
| Cell 1 | Cell 2 | Cell 3 |
| Cell 4 | Cell 5 | Cell 6 |`

    const result = simplifiedTableProcessor(input)
    
    expect(result.processed).toBe(input)
    expect(result.issues.length).toBe(0)
  })

  it('should handle tables with unicode and special characters', () => {
    const input = `| Currency | Symbol | Amount |
| --- | --- | --- |
| Euro | € | 100.50 |
| Pound | £ | 85.25 |
| Yen | ¥ | 11,000 |
| Bitcoin | ₿ | 0.0025 |`

    const result = simplifiedTableProcessor(input)
    
    expect(result.processed).toContain('€')
    expect(result.processed).toContain('£')
    expect(result.processed).toContain('¥')
    expect(result.processed).toContain('₿')
  })

  it('should handle tables with markdown formatting', () => {
    const input = `| Feature | Status | Notes |
| --- | --- | --- |
| **Bold** | ✅ | Works with **bold text** |
| *Italic* | ✅ | Works with *italic text* |
| [Links](url) | ⚠️ | [Click here](https://example.com) |
| \`code\` | ✅ | Inline \`code blocks\` work |`

    const result = simplifiedTableProcessor(input)
    
    expect(result.processed).toContain('**Bold**')
    expect(result.processed).toContain('*Italic*')
    expect(result.processed).toContain('[Links](url)')
    expect(result.processed).toContain('`code`')
  })
})

describe('Performance Tests', () => {
  it('should handle large tables efficiently', () => {
    const rows = Array(1000).fill(0).map((_, i) => 
      `| Row ${i} | Data ${i} | Value ${i} |`
    )
    const input = `| Index | Data | Value |
| --- | --- | --- |
${rows.join('\n')}`

    const start = performance.now()
    const result = simplifiedTableProcessor(input)
    const end = performance.now()

    expect(end - start).toBeLessThan(100) // Should process in under 100ms
    expect(result.processed).toContain('| Row 0 | Data 0 | Value 0 |')
    expect(result.processed).toContain('| Row 999 | Data 999 | Value 999 |')
  })

  it('should handle deeply nested broken patterns efficiently', () => {
    const brokenPattern = '| --- | --- | --- | --- |\n'
    const input = brokenPattern.repeat(100) + '| Data | Value |'

    const start = performance.now()
    const result = simplifiedTableProcessor(input)
    const end = performance.now()

    expect(end - start).toBeLessThan(50)
    expect(result.processed).toContain('| Data | Value |')
    expect(result.processed.split(brokenPattern).length).toBe(1)
  })
})