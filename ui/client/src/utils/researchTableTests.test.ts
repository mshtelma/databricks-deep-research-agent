import { describe, it, expect } from 'vitest'
import { simplifiedTableProcessor } from './simplifiedTableProcessor'
import { TableStreamReconstructor } from './tableStreamReconstructor'

describe('Research Table Processing Tests', () => {
  
  describe('Multi-Section Research Tables', () => {
    it('should keep multi-section comparison tables together', () => {
      const input = `## GDP Comparison by Country

| Country | 2019 GDP | 2020 GDP | 2021 GDP | 2022 GDP | 2023 GDP |
| --- | --- | --- | --- | --- | --- |
| USA | $21.4T | $20.9T | $23.0T | $25.5T | $26.9T |
| China | $14.3T | $14.7T | $17.7T | $17.9T | $17.7T |

**European Countries**

| Country | 2019 GDP | 2020 GDP | 2021 GDP | 2022 GDP | 2023 GDP |
| --- | --- | --- | --- | --- | --- |
| Germany | $3.9T | $3.8T | $4.2T | $4.1T | $4.3T |
| France | $2.7T | $2.6T | $2.9T | $2.8T | $2.9T |
| UK | $2.8T | $2.7T | $3.1T | $3.1T | $3.2T |

Total EU: $15.2T (2023)`

      const result = simplifiedTableProcessor(input)
      
      // Should preserve both tables and the subheading
      expect(result.processed).toContain('| USA |')
      expect(result.processed).toContain('| China |')
      expect(result.processed).toContain('**European Countries**')
      expect(result.processed).toContain('| Germany |')
      expect(result.processed).toContain('| France |')
      expect(result.processed).toContain('Total EU:')
      
      // Tables should be properly structured
      const tables = result.processed.match(/\| --- \|/g) || []
      expect(tables.length).toBeGreaterThanOrEqual(2)
    })

    it('should handle financial comparison tables with notes', () => {
      const input = `| Investment Type | 5Y Return | 10Y Return | Risk Level | Min Investment |
| --- | --- | --- | --- | --- |
| S&P 500 Index | 11.2% | 12.5% | Medium | $1 |
| Corporate Bonds | 4.5% | 5.2% | Low | $1,000 |

Note: Returns are annualized and before taxes

| Real Estate Fund | 8.7% | 9.3% | Medium-High | $5,000 |
| Gold ETF | 6.1% | 3.8% | Medium | $50 |

Source: Market data as of Dec 2023`

      const result = simplifiedTableProcessor(input)
      
      // Should keep the table together with notes
      expect(result.processed).toContain('| S&P 500 Index |')
      expect(result.processed).toContain('Note: Returns are annualized')
      expect(result.processed).toContain('| Real Estate Fund |')
      expect(result.processed).toContain('Source: Market data')
      
      // Should not split the table at the note
      const lines = result.processed.split('\n')
      const tableLines = lines.filter(l => l.includes('|'))
      expect(tableLines.length).toBeGreaterThanOrEqual(5)
    })
  })

  describe('Tables with Variable Column Structures', () => {
    it('should handle tables with merged cells and spanning headers', () => {
      const input = `| Region | Q1 2023 | | Q2 2023 | | Q3 2023 | |
| --- | Revenue | Profit | Revenue | Profit | Revenue | Profit |
| --- | --- | --- | --- | --- | --- | --- |
| North America | $45M | $12M | $48M | $14M | $52M | $15M |
| Europe | $38M | $9M | $41M | $11M | $43M | $12M |
| Asia Pacific | $62M | $18M | $67M | $20M | $71M | $22M |`

      const result = simplifiedTableProcessor(input)
      
      expect(result.processed).toContain('| Region |')
      expect(result.processed).toContain('| North America |')
      expect(result.processed).toContain('$45M')
      expect(result.processed).toContain('| Asia Pacific |')
      
      // Should maintain the complex structure
      const hasProperStructure = result.processed.includes('Revenue') && 
                                 result.processed.includes('Profit')
      expect(hasProperStructure).toBe(true)
    })

    it('should handle tables with category rows and subtotals', () => {
      const input = `| Department | Headcount | Budget | YoY Growth |
| --- | --- | --- | --- |
**Technology Division**
| Engineering | 450 | $67.5M | +12% |
| Product | 125 | $18.8M | +8% |
| Design | 75 | $11.3M | +15% |
Technology Total | 650 | $97.6M | +11.3% |

**Operations Division**
| Sales | 320 | $48M | +18% |
| Marketing | 180 | $27M | +22% |
| Support | 210 | $31.5M | +5% |
Operations Total | 710 | $106.5M | +14.7% |

**Company Total** | **1,360** | **$204.1M** | **+13.1%** |`

      const result = simplifiedTableProcessor(input)
      
      // Should preserve all sections and totals
      expect(result.processed).toContain('**Technology Division**')
      expect(result.processed).toContain('| Engineering |')
      expect(result.processed).toContain('Technology Total')
      expect(result.processed).toContain('**Operations Division**')
      expect(result.processed).toContain('| Sales |')
      expect(result.processed).toContain('Operations Total')
      expect(result.processed).toContain('**Company Total**')
    })
  })

  describe('Scientific and Data Analysis Tables', () => {
    it('should handle climate data tables with projections', () => {
      const input = `| Continent | Avg Temp 2020 | Avg Temp 2023 | Projected 2030 | Projected 2050 |
| --- | --- | --- | --- | --- |
| Africa | 25.2°C | 25.8°C | 26.5°C | 28.1°C |
| Asia | 15.3°C | 15.7°C | 16.3°C | 17.8°C |

*Temperature anomalies relative to 1950-1980 baseline*

| Europe | 9.8°C | 10.4°C | 11.2°C | 12.9°C |
| North America | 11.2°C | 11.7°C | 12.4°C | 13.8°C |

Confidence interval: ±0.3°C for historical, ±0.8°C for projections`

      const result = simplifiedTableProcessor(input)
      
      expect(result.processed).toContain('| Africa |')
      expect(result.processed).toContain('*Temperature anomalies')
      expect(result.processed).toContain('| Europe |')
      expect(result.processed).toContain('Confidence interval:')
      
      // Should keep the table together despite notes
      const tables = result.processed.split('| --- |')
      expect(tables.length).toBeLessThanOrEqual(3) // Should be 1-2 separator rows, not many
    })

    it('should handle LLM model comparison matrices', () => {
      const input = `## Large Language Model Feature Comparison

| Model | Parameters | Context | Training Data | Release | License |
| --- | --- | --- | --- | --- | --- |
| GPT-4 | ~1.76T | 128k | Sept 2021 | 2023 | Proprietary |
| Claude 3 | Undisclosed | 200k | Early 2023 | 2024 | Proprietary |

**Open Source Models**

| LLaMA 2 | 7B-70B | 4k | June 2023 | 2023 | Custom |
| Mistral | 7B-8x7B | 32k | Sept 2023 | 2023 | Apache 2.0 |
| Falcon | 7B-180B | 2k | May 2023 | 2023 | Apache 2.0 |

Note: Context lengths shown are for base models. Many have extended context variants.`

      const result = simplifiedTableProcessor(input)
      
      expect(result.processed).toContain('| GPT-4 |')
      expect(result.processed).toContain('| Claude 3 |')
      expect(result.processed).toContain('**Open Source Models**')
      expect(result.processed).toContain('| LLaMA 2 |')
      expect(result.processed).toContain('Note: Context lengths')
    })
  })

  describe('Complex Stock Analysis Tables', () => {
    it('should handle detailed sentiment analysis tables', () => {
      const input = `**AAPL Sentiment Analysis - Q4 2023**

| Metric | Value | Change | Sentiment |
| --- | --- | --- | --- |
| Analyst Rating | 4.2/5 | +0.3 | Positive |
| Social Sentiment | 68% | +5% | Positive |

Recent Catalysts:
| Event | Date | Impact | Price Move |
| --- | --- | --- | --- |
| iPhone 15 Launch | Sept 2023 | High | +3.2% |
| Q3 Earnings Beat | Oct 2023 | Medium | +1.8% |
| Vision Pro Announce | Jan 2024 | High | +4.1% |

**Technical Indicators**
| Indicator | Current | Signal | Strength |
| --- | --- | --- | --- |
| RSI | 58 | Neutral | Medium |
| MACD | Bullish | Buy | Strong |
| 50-Day MA | Above | Buy | Medium |`

      const result = simplifiedTableProcessor(input)
      
      // Should preserve all three tables with their headers
      expect(result.processed).toContain('**AAPL Sentiment Analysis')
      expect(result.processed).toContain('| Analyst Rating |')
      expect(result.processed).toContain('Recent Catalysts:')
      expect(result.processed).toContain('| iPhone 15 Launch |')
      expect(result.processed).toContain('**Technical Indicators**')
      expect(result.processed).toContain('| RSI |')
      
      // Tables should maintain structure
      const separatorCount = (result.processed.match(/\| --- \|/g) || []).length
      expect(separatorCount).toBeGreaterThanOrEqual(3)
    })
  })

  describe('Streaming Table Reconstruction', () => {
    it('should handle tables arriving in chunks', () => {
      const reconstructor = new TableStreamReconstructor()
      
      // Simulate streaming chunks
      const chunks = [
        '| Country | Population',
        ' | GDP | HDI |\n',
        '| --- | --- |',
        ' --- | --- |\n',
        '| USA | 331M | $',
        '26.9T | 0.921 |\n',
        '| China | 1.4',
        '25B | $17.7T | 0.768 |\n'
      ]
      
      let result
      for (const chunk of chunks) {
        result = reconstructor.addChunk(chunk)
      }
      
      result = reconstructor.finalize()
      
      expect(result.raw).toContain('| Country | Population | GDP | HDI |')
      expect(result.raw).toContain('| USA |')
      expect(result.raw).toContain('| China |')
      
      // Should have complete table
      expect(result.hasIncompleteTable).toBe(false)
      expect(result.tables.length).toBeGreaterThan(0)
    })

    it('should handle interrupted tables with continuations', () => {
      const reconstructor = new TableStreamReconstructor()
      
      const content = `| Metric | Q1 | Q2 | Q3 | Q4 |
| --- | --- | --- | --- | --- |
| Revenue | $45M | $48M | $52M | $55M |

Year-end summary: Strong growth across all quarters

| Profit | $8M | $9M | $11M | $13M |
| Margin | 17.8% | 18.8% | 21.2% | 23.6% |`

      const lines = content.split('\n')
      for (const line of lines) {
        reconstructor.addChunk(line + '\n')
      }
      
      const result = reconstructor.finalize()
      
      // Should recognize this as a continued table
      expect(result.raw).toContain('| Revenue |')
      expect(result.raw).toContain('Year-end summary')
      expect(result.raw).toContain('| Profit |')
      expect(result.raw).toContain('| Margin |')
    })
  })

  describe('Edge Cases and Complex Patterns', () => {
    it('should handle tables with inline calculations and formulas', () => {
      const input = `| Calculation | Formula | Result |
| --- | --- | --- |
| ROI | (Gain - Cost) / Cost × 100 | 24.5% |
| P/E Ratio | Price / EPS | 28.3 |
| Debt/Equity | Total Debt / Total Equity | 0.45 |

* All calculations based on FY 2023 data

| Beta | Covariance(Stock, Market) / Variance(Market) | 1.12 |
| Sharpe Ratio | (Return - Risk Free) / Std Dev | 1.85 |`

      const result = simplifiedTableProcessor(input)
      
      expect(result.processed).toContain('| ROI |')
      expect(result.processed).toContain('(Gain - Cost) / Cost')
      expect(result.processed).toContain('* All calculations')
      expect(result.processed).toContain('| Beta |')
      expect(result.processed).toContain('Covariance(Stock, Market)')
    })

    it('should handle deeply nested country comparison data', () => {
      const input = `## Economic Indicators by Region

**G7 Countries**
| Country | GDP Growth | Inflation | Unemployment | Debt/GDP |
| --- | --- | --- | --- | --- |
| USA | 2.5% | 3.4% | 3.7% | 123% |
| Japan | 1.3% | 3.2% | 2.6% | 264% |
| Germany | 0.2% | 6.4% | 3.0% | 66% |

**BRICS Nations**
| Brazil | 2.9% | 4.6% | 8.0% | 88% |
| Russia | -2.1% | 5.9% | 3.3% | 17% |
| India | 7.2% | 5.7% | 7.8% | 84% |
| China | 5.2% | 0.7% | 5.0% | 77% |
| South Africa | 0.6% | 5.9% | 32.9% | 70% |

**Summary Statistics**
Average G7 Growth: 1.3%
Average BRICS Growth: 2.8%`

      const result = simplifiedTableProcessor(input)
      
      // Should keep all sections together
      expect(result.processed).toContain('**G7 Countries**')
      expect(result.processed).toContain('| USA |')
      expect(result.processed).toContain('**BRICS Nations**')
      expect(result.processed).toContain('| Brazil |')
      expect(result.processed).toContain('| South Africa |')
      expect(result.processed).toContain('**Summary Statistics**')
      expect(result.processed).toContain('Average G7 Growth')
    })
  })

  describe('Performance with Large Research Tables', () => {
    it('should efficiently process large financial datasets', () => {
      // Generate a large table with 100 rows
      const headers = '| Ticker | Company | Price | Change | Volume | Market Cap |'
      const separator = '| --- | --- | --- | --- | --- | --- |'
      
      const rows = Array(100).fill(0).map((_, i) => 
        `| TICK${i} | Company ${i} Inc | $${100 + i} | ${(Math.random() * 10 - 5).toFixed(2)}% | ${Math.floor(Math.random() * 1000000)} | $${(10 + i * 0.5).toFixed(1)}B |`
      )
      
      const input = [headers, separator, ...rows].join('\n')
      
      const start = performance.now()
      const result = simplifiedTableProcessor(input)
      const end = performance.now()
      
      expect(end - start).toBeLessThan(100) // Should process in under 100ms
      expect(result.processed).toContain('| TICK0 |')
      expect(result.processed).toContain('| TICK99 |')
      
      // Should maintain structure
      const processedLines = result.processed.split('\n')
      const tableRows = processedLines.filter(l => l.includes('| TICK'))
      expect(tableRows.length).toBe(100)
    })

    it('should handle complex multi-section reports efficiently', () => {
      const sections = Array(10).fill(0).map((_, i) => `
**Section ${i + 1}: Region Analysis**

| Country | Metric A | Metric B | Metric C |
| --- | --- | --- | --- |
| Country ${i * 3} | ${100 + i} | ${200 + i} | ${300 + i} |
| Country ${i * 3 + 1} | ${101 + i} | ${201 + i} | ${301 + i} |
| Country ${i * 3 + 2} | ${102 + i} | ${202 + i} | ${302 + i} |

Summary: Performance improved by ${5 + i}% year-over-year.
      `)
      
      const input = sections.join('\n')
      
      const start = performance.now()
      const result = simplifiedTableProcessor(input)
      const end = performance.now()
      
      expect(end - start).toBeLessThan(200) // Should process quickly
      
      // All sections should be preserved
      for (let i = 1; i <= 10; i++) {
        expect(result.processed).toContain(`**Section ${i}:`)
        expect(result.processed).toContain(`Country ${(i - 1) * 3}`)
      }
    })
  })
})