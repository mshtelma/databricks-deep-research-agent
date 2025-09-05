import { describe, it, expect } from 'vitest'
import { 
  isIncompleteTable, 
  fixBrokenTableSeparators, 
  preprocessMarkdown 
} from './tableUtils'

describe('isIncompleteTable', () => {
  it('should return true for incomplete streaming tables', () => {
    const incompleteTable = `| Header 1 | Header 2 |
|----------|`
    expect(isIncompleteTable(incompleteTable)).toBe(true)
  })

  it('should return false for complete tables', () => {
    const completeTable = `| Header 1 | Header 2 |
|----------|----------|
| Data 1   | Data 2   |`
    expect(isIncompleteTable(completeTable)).toBe(false)
  })

  it('should return true for tables with header but no separator', () => {
    const headerOnly = `| Column A | Column B |`
    expect(isIncompleteTable(headerOnly)).toBe(true)
  })

  it('should return true for tables with header and separator but no data', () => {
    const headerAndSeparator = `| Column A | Column B |
|----------|----------|`
    expect(isIncompleteTable(headerAndSeparator)).toBe(true)
  })

  it('should return false for non-table content', () => {
    const nonTable = `This is just regular text
with some lines
but no tables`
    expect(isIncompleteTable(nonTable)).toBe(false)
  })

  it('should detect malformed separator patterns', () => {
    const malformed = `| Header |
| --- | --- | --- | --- |`
    expect(isIncompleteTable(malformed)).toBe(true)
  })

  it('should handle empty strings', () => {
    expect(isIncompleteTable('')).toBe(false)
  })

  it('should handle tables with empty lines', () => {
    const tableWithEmptyLines = `| Header 1 | Header 2 |
|----------|----------|

| Data 1   | Data 2   |`
    expect(isIncompleteTable(tableWithEmptyLines)).toBe(false)
  })
})

describe('fixBrokenTableSeparators', () => {
  it('should fix tables with multiple dash lines as separators', () => {
    const brokenTable = `| Country | Population |
---
---
| USA | 331M |
| Germany | 83M |`

    const result = fixBrokenTableSeparators(brokenTable)
    
    expect(result).toContain('| Country | Population |')
    expect(result).toContain('| --- | --- |')
    expect(result).toContain('| USA | 331M |')
    expect(result).not.toContain('---\n---')
  })

  it('should fix the specific tax table format', () => {
    const taxTable = `After‑tax income | Country | Gross earnings |
---
---
Main employee‑social‑security charge
---
Income‑tax system
---
| Austria | €25,000 | €5,432 |`

    const result = fixBrokenTableSeparators(taxTable)
    
    expect(result).toContain('| After‑tax income | Country | Gross earnings |')
    expect(result).toContain('| --- | --- | --- |')
    expect(result).toContain('| Austria | €25,000 | €5,432 |')
  })

  it('should handle multi-line header reconstruction', () => {
    const multiLineHeader = `Product Name
---
Category
---
Price Range
---
| iPhone | Electronics | $999 |
| Shirt | Clothing | $29 |`

    const result = fixBrokenTableSeparators(multiLineHeader)
    
    expect(result).toContain('| Product Name | Category | Price Range |')
    expect(result).toContain('| --- | --- | --- |')
    expect(result).toContain('| iPhone | Electronics | $999 |')
  })

  it('should preserve well-formed tables', () => {
    const wellFormed = `| Country | Population | GDP |
|---------|------------|-----|
| USA | 331M | $21T |
| Germany | 83M | $4T |`

    const result = fixBrokenTableSeparators(wellFormed)
    expect(result).toBe(wellFormed)
  })

  it('should handle tables with currency symbols and special characters', () => {
    const currencyTable = `Country | Currency | Amount
---
---
| UK | £100 | €115 |
| Switzerland | CHF 200 | €210 |`

    const result = fixBrokenTableSeparators(currencyTable)
    
    expect(result).toContain('| Country | Currency | Amount |')
    expect(result).toContain('| --- | --- | --- |')
    expect(result).toContain('| UK | £100 | €115 |')
  })

  it('should not create false tables from content separators', () => {
    const contentWithSeparators = `This is a paragraph.

---

This is another paragraph.

---

Not a table at all.`

    const result = fixBrokenTableSeparators(contentWithSeparators)
    expect(result).toBe(contentWithSeparators)
  })

  it('should handle empty input', () => {
    expect(fixBrokenTableSeparators('')).toBe('')
  })
})

describe('preprocessMarkdown - Comprehensive Table Cases', () => {
  
  describe('Standard Markdown Tables', () => {
    it('should handle well-formed GFM tables', () => {
      const gfmTable = `| Header 1 | Header 2 | Header 3 |
| --- | --- | --- |
| Cell 1 | Cell 2 | Cell 3 |
| Cell 4 | Cell 5 | Cell 6 |`

      const result = preprocessMarkdown(gfmTable)
      expect(result).toContain('| Header 1 | Header 2 | Header 3 |')
      expect(result).toContain('| Cell 1 | Cell 2 | Cell 3 |')
    })

    it('should handle tables with alignment', () => {
      const alignedTable = `| Left | Center | Right |
| :--- | :---: | ---: |
| L1 | C1 | R1 |
| L2 | C2 | R2 |`

      const result = preprocessMarkdown(alignedTable)
      expect(result).toContain('| Left | Center | Right |')
      expect(result).toContain('| L1 | C1 | R1 |')
    })
  })

  describe('Broken Tax Table Format', () => {
    it('should fix the exact broken tax table from user', () => {
      const brokenTaxTable = `| --- | --- | --- | --- |
|--- |---|---|---|
| --- | --- | --- | --- |
| --- | --- | --- | --- |

| |---|---|
| United Kingdom | €100,000 |

---
---
€7,200
---
£2,400 ≈ €2,800 (child‑tax‑credit & universal‑credit)`

      const result = preprocessMarkdown(brokenTaxTable)
      
      // Should remove malformed separators
      expect(result).not.toContain('| --- | --- | --- | --- |')
      expect(result).not.toContain('|--- |---|---|---|')
      
      // Should preserve actual table content
      expect(result).toContain('United Kingdom')
      expect(result).toContain('€100,000')
      expect(result).toContain('€7,200')
    })

    it('should handle Switzerland tax table variant', () => {
      const swissTable = `| --- | --- | --- | --- |
|--- |---|---|---|
| --- | --- | --- | --- |
| --- | --- | --- | --- |

| |---|---|
| Switzerland – Zug | €100,000 |

---
---
€6,600
---
CHF 2,200 ≈ €2,300 (child‑allowance)`

      const result = preprocessMarkdown(swissTable)
      
      expect(result).not.toContain('| --- | --- | --- | --- |')
      expect(result).toContain('Switzerland – Zug')
      expect(result).toContain('€100,000')
      expect(result).toContain('€6,600')
    })
  })

  describe('Complex Financial Tables', () => {
    it('should handle financial data with multiple currencies', () => {
      const financialTable = `Asset Class | Q1 Return | Q2 Return | YTD
---
---
| Equities | $1.2M (↑5%) | $1.3M (↑8%) | $2.5M |
| Bonds | €500K (↓2%) | €480K (↓4%) | €980K |
| Crypto | ₿100 (↑15%) | ₿115 (↑15%) | ₿215 |`

      const result = preprocessMarkdown(financialTable)
      
      expect(result).toContain('| Asset Class | Q1 Return | Q2 Return | YTD |')
      expect(result).toContain('| Equities | $1.2M (↑5%) | $1.3M (↑8%) | $2.5M |')
      expect(result).toContain('₿100')
    })

    it('should handle accounting tables with parentheses', () => {
      const accountingTable = `Account | Debit | Credit | Balance
---
| Cash | 10,000 | (5,000) | 5,000 |
| A/R | 25,000 | (10,000) | 15,000 |
| Revenue | - | (50,000) | (50,000) |`

      const result = preprocessMarkdown(accountingTable)
      
      expect(result).toContain('| Account | Debit | Credit | Balance |')
      expect(result).toContain('| Cash | 10,000 | (5,000) | 5,000 |')
    })
  })

  describe('Scientific and Technical Tables', () => {
    it('should handle scientific notation and units', () => {
      const sciTable = `Element | Mass (g/mol) | Density | Melting Point
---
---
| H₂O | 18.015 | 1.0 g/cm³ | 0°C |
| NaCl | 58.44 | 2.16 g/cm³ | 801°C |
| Fe | 55.845 | 7.87 g/cm³ | 1538°C |`

      const result = preprocessMarkdown(sciTable)
      
      expect(result).toContain('| Element | Mass (g/mol) | Density | Melting Point |')
      expect(result).toContain('| H₂O | 18.015 | 1.0 g/cm³ | 0°C |')
    })

    it('should handle mathematical expressions', () => {
      const mathTable = `Function | Domain | Range | Derivative
---
| f(x) = x² | ℝ | [0, ∞) | 2x |
| g(x) = sin(x) | ℝ | [-1, 1] | cos(x) |
| h(x) = eˣ | ℝ | (0, ∞) | eˣ |`

      const result = preprocessMarkdown(mathTable)
      
      expect(result).toContain('| Function | Domain | Range | Derivative |')
      expect(result).toContain('| f(x) = x² | ℝ | [0, ∞) | 2x |')
    })
  })

  describe('Tables with HTML and Markdown Content', () => {
    it('should handle tables with inline markdown', () => {
      const mdTable = `| Feature | Status | Notes |
| --- | --- | --- |
| **Bold** | ✅ | Works with **bold text** |
| *Italic* | ✅ | Works with *italic text* |
| [Links](url) | ⚠️ | [Click here](https://example.com) |
| \`code\` | ✅ | Inline \`code blocks\` work |`

      const result = preprocessMarkdown(mdTable)
      
      expect(result).toContain('| Feature | Status | Notes |')
      expect(result).toContain('| **Bold** | ✅ | Works with **bold text** |')
      expect(result).toContain('[Click here](https://example.com)')
    })

    it('should handle tables with HTML breaks', () => {
      const htmlTable = `| Name | Address |
| --- | --- |
| John | 123 Main St<br>New York<br>10001 |
| Jane | 456 Oak Ave<br/>Boston<br />02134 |`

      const result = preprocessMarkdown(htmlTable)
      
      expect(result).toContain('| Name | Address |')
      expect(result).toContain('123 Main St')
    })
  })

  describe('Edge Cases and Malformed Tables', () => {
    it('should handle tables with missing cells', () => {
      const sparseTable = `| A | B | C | D |
| --- | --- | --- | --- |
| 1 | 2 | | 4 |
| | | 3 | |
| 5 | | | |`

      const result = preprocessMarkdown(sparseTable)
      
      expect(result).toContain('| A | B | C | D |')
      expect(result).toContain('| 1 | 2 | | 4 |')
    })

    it('should handle tables with pipes in content', () => {
      const pipesInContent = `| Command | Description |
| --- | --- |
| grep "a\\|b" | Search for a or b |
| awk '{print $1\\|$2}' | Print fields |`

      const result = preprocessMarkdown(pipesInContent)
      
      expect(result).toContain('| Command | Description |')
      expect(result).toContain('grep "a\\|b"')
    })

    it('should handle very wide tables', () => {
      const wideTable = `| A | B | C | D | E | F | G | H | I | J | K | L | M | N | O | P | Q | R | S | T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |`

      const result = preprocessMarkdown(wideTable)
      
      expect(result).toContain('| A | B | C | D | E |')
      expect(result).toContain('| 1 | 2 | 3 | 4 | 5 |')
    })

    it('should handle mixed content with tables', () => {
      const mixedContent = `Here is some text before the table.

| Header 1 | Header 2 |
| --- | --- |
| Data 1 | Data 2 |

Some text between tables.

---

Another section with a table:

Column A | Column B
---
---
| Value A | Value B |

Final paragraph.`

      const result = preprocessMarkdown(mixedContent)
      
      expect(result).toContain('Here is some text before the table.')
      expect(result).toContain('| Header 1 | Header 2 |')
      expect(result).toContain('Some text between tables.')
      expect(result).toContain('| Column A | Column B |')
      expect(result).toContain('Final paragraph.')
    })

    it('should handle streaming incomplete tables', () => {
      const streaming = `| Starting to build |
| --- |
| This table is still`

      const result = preprocessMarkdown(streaming)
      
      // Should wrap incomplete table in code block
      expect(result).toContain('```')
    })
  })

  describe('International and Unicode Tables', () => {
    it('should handle RTL and international characters', () => {
      const intlTable = `| العربية | 中文 | 日本語 | 한국어 |
| --- | --- | --- | --- |
| مرحبا | 你好 | こんにちは | 안녕하세요 |
| شكرا | 谢谢 | ありがとう | 감사합니다 |`

      const result = preprocessMarkdown(intlTable)
      
      expect(result).toContain('| العربية | 中文 | 日本語 | 한국어 |')
      expect(result).toContain('| مرحبا | 你好 | こんにちは | 안녕하세요 |')
    })

    it('should handle emoji in tables', () => {
      const emojiTable = `| Status | Emoji | Description |
| --- | --- | --- |
| Success | ✅ 🎉 | All tests passed! |
| Warning | ⚠️ 🔔 | Needs attention |
| Error | ❌ 🚨 | Critical failure |`

      const result = preprocessMarkdown(emojiTable)
      
      expect(result).toContain('| Status | Emoji | Description |')
      expect(result).toContain('| Success | ✅ 🎉 | All tests passed! |')
    })
  })

  describe('Performance and Stress Tests', () => {
    it('should handle very long tables efficiently', () => {
      // Create a 100-row table
      const rows = Array(100).fill(0).map((_, i) => 
        `| Row ${i} | Data ${i} | Value ${i} |`
      )
      const longTable = `| Index | Data | Value |
| --- | --- | --- |
${rows.join('\n')}`

      const start = performance.now()
      const result = preprocessMarkdown(longTable)
      const end = performance.now()

      expect(end - start).toBeLessThan(100) // Should process in under 100ms
      expect(result).toContain('| Row 0 | Data 0 | Value 0 |')
      expect(result).toContain('| Row 99 | Data 99 | Value 99 |')
    })

    it('should handle deeply nested broken separators', () => {
      const deeplyBroken = `Table Title
---
Column 1
---
---
Column 2
---
---
---
Column 3
---
---
---
---
| Data 1 | Data 2 | Data 3 |`

      const result = preprocessMarkdown(deeplyBroken)
      
      expect(result).not.toContain('---\n---\n---\n---')
      expect(result).toContain('Data 1')
    })
  })

  describe('Regression Tests', () => {
    it('should normalize Windows newlines', () => {
      const windowsText = 'Line 1\r\nLine 2\r\nLine 3'
      const result = preprocessMarkdown(windowsText)
      expect(result).toBe('Line 1\nLine 2\nLine 3')
    })

    it('should collapse multiple pipes', () => {
      const doublePipes = `| Column A || Column B |
|----------|----------|
| Data A || Data B |`

      const result = preprocessMarkdown(doublePipes)
      expect(result).not.toContain('||')
      expect(result).toContain('| Column A | Column B |')
    })

    it('should handle null/undefined gracefully', () => {
      expect(preprocessMarkdown(null as any)).toBe(null)
      expect(preprocessMarkdown(undefined as any)).toBe(undefined)
    })
  })
})

describe('Real-world Integration Tests', () => {
  it('should handle complex research report tables', () => {
    const researchTable = `## Tax Comparison Report

After analyzing the data, here are the findings:

After‑tax (disposable) income | Country | Gross earnings
---
---
Main social security
---
Tax system type
---

| United Kingdom | €100,000 | €72,000 |
| Germany | €100,000 | €58,000 |
| Switzerland | €100,000 | €76,000 |

Additional notes and analysis follow...`

    const result = preprocessMarkdown(researchTable)
    
    expect(result).toContain('## Tax Comparison Report')
    expect(result).toContain('| After‑tax (disposable) income | Country | Gross earnings |')
    expect(result).toContain('| United Kingdom | €100,000 | €72,000 |')
    expect(result).toContain('Additional notes and analysis follow...')
  })

  it('should handle tables embedded in markdown documents', () => {
    const document = `# Product Catalog

## Electronics

Here are our current offerings:

Product | Price | Stock
---
---
| MacBook Pro | $2,999 | In Stock |
| iPhone 15 | $999 | Limited |

---

## Accessories

Item | Cost
---
| Case | $49 |
| Charger | $29 |

Contact us for more information.`

    const result = preprocessMarkdown(document)
    
    expect(result).toContain('# Product Catalog')
    expect(result).toContain('| Product | Price | Stock |')
    expect(result).toContain('| MacBook Pro | $2,999 | In Stock |')
    expect(result).toContain('| Item | Cost |')
    expect(result).toContain('Contact us for more information.')
  })
})