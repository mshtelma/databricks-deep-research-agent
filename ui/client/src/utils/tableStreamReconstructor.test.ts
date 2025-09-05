import { describe, it, expect } from 'vitest'
import { 
  TableStreamReconstructor, 
  processStreamingWithTableReconstruction,
  validateTableIntegrity 
} from './tableStreamReconstructor'

describe('TableStreamReconstructor', () => {
  describe('Basic Functionality', () => {
    it('should handle complete table in single chunk', () => {
      const reconstructor = new TableStreamReconstructor()
      const table = `| Header 1 | Header 2 |
| --- | --- |
| Data 1 | Data 2 |`
      
      const result = reconstructor.addChunk(table + '\n')
      expect(result.hasIncompleteTable).toBe(false)
      expect(result.raw).toContain('| Header 1 | Header 2 |')
      expect(result.tables).toHaveLength(0) // Table not finalized yet
      
      const final = reconstructor.finalize()
      expect(final.tables).toHaveLength(1)
      expect(final.tables[0].isComplete).toBe(true)
    })
    
    it('should detect incomplete tables during streaming', () => {
      const reconstructor = new TableStreamReconstructor()
      
      // Add partial table
      reconstructor.addChunk('| Header 1 | Header 2 |\n')
      let result = reconstructor.addChunk('| --- |')
      
      expect(result.hasIncompleteTable).toBe(true)
      expect(result.display).toContain('Table is being received')
      
      // Complete the table
      result = reconstructor.addChunk(' --- |\n')
      result = reconstructor.addChunk('| Data 1 | Data 2 |\n')
      
      const final = reconstructor.finalize()
      expect(final.hasIncompleteTable).toBe(false)
      expect(final.raw).toContain('| Data 1 | Data 2 |')
    })
  })
  
  describe('Tax Comparison Table Reconstruction', () => {
    it('should handle the tax comparison table format', () => {
      const reconstructor = new TableStreamReconstructor()
      
      // Simulate streaming the tax table
      const chunks = [
        '| Country | Gross Income',
        ' (€) | Income Tax (€)',
        ') | Net Income (€) |\n',
        '|---------|--------',
        '---------|----------',
        '------|----------------|\n',
        '| Spain | 70,000 |',
        ' 15,400 | 54,600 |\n',
        '| France | 70,000 | 12,600 | 57,400 |\n'
      ]
      
      let lastResult
      for (const chunk of chunks) {
        lastResult = reconstructor.addChunk(chunk)
      }
      
      const final = reconstructor.finalize()
      
      // Table should be complete and valid
      expect(final.hasIncompleteTable).toBe(false)
      expect(final.raw).toContain('Spain')
      expect(final.raw).toContain('France')
      expect(final.raw).not.toContain('||') // No double pipes
    })
    
    it('should handle wide tables with many columns', () => {
      const reconstructor = new TableStreamReconstructor()
      
      const wideTable = `| Country | Income Tax | Social Security | Child Benefit | Net Income | Effective Rate |
|---------|------------|-----------------|---------------|------------|----------------|
| Switzerland (Zug) | 7,000 | 3,500 | 0 | 59,500 | 15.0% |
| Germany | 14,000 | 11,900 | 0 | 44,100 | 37.0% |`
      
      // Stream in small chunks
      const chunkSize = 30
      for (let i = 0; i < wideTable.length; i += chunkSize) {
        reconstructor.addChunk(wideTable.slice(i, i + chunkSize))
      }
      
      const final = reconstructor.finalize()
      expect(final.raw).toContain('Switzerland')
      expect(final.raw).toContain('Germany')
      expect(final.tables).toHaveLength(1)
    })
  })
  
  describe('Stock Sentiment Table Handling', () => {
    it('should process stock analysis tables correctly', () => {
      const stockTable = `| Stock | Current Price | P/E Ratio | Sentiment | Recommendation |
|-------|---------------|-----------|-----------|----------------|
| AAPL | $195.42 | 32.5 | 7.8/10 | HOLD |
| MSFT | $378.91 | 35.2 | 8.5/10 | HOLD |`
      
      const result = processStreamingWithTableReconstruction(stockTable)
      
      expect(result.tables).toHaveLength(1)
      expect(result.tables[0].isComplete).toBe(true)
      expect(result.raw).toContain('AAPL')
      expect(result.raw).toContain('MSFT')
    })
  })
  
  describe('Malformed Table Detection and Fixing', () => {
    it('should detect and fix double pipes', () => {
      const malformed = `| Header 1 || Header 2 |
|---|---|
| Data 1 || Data 2 |`
      
      const result = processStreamingWithTableReconstruction(malformed)
      
      // Should fix double pipes
      expect(result.display).not.toContain('||')
      expect(result.display).toContain('| Header 1 | Header 2 |')
    })
    
    it('should detect inline separators', () => {
      const malformed = `| Header 1 | Header 2 |---|---|
| Data 1 | Data 2 |`
      
      const validation = validateTableIntegrity(malformed)
      
      expect(validation.valid).toBe(false)
      expect(validation.issues).toContain('Contains inline separator with content')
    })
    
    it('should detect broken separator patterns', () => {
      const broken = `| Header |
|---|---|---|---|---|
| Data |`
      
      const validation = validateTableIntegrity(broken)
      
      expect(validation.valid).toBe(false)
      expect(validation.issues.some(i => i.includes('broken separator'))).toBe(true)
    })
    
    it('should handle the exact malformed output from the prompt', () => {
      const malformedOutput = `1. Married couple without children| Country | Gross (€) | Income Tax (€) | Social‑Security (€) | Child‑Benefit (€) | Net Disposable (€)|---|---|---|---|---|
|

---    ---    ---    ---    ---
---    ---    ---    ---    ---
---    ---    ---    ---    ---
---    ---    ---    ---    ---    ---
| --- | --- | --- | --- | | --- | --- |`
      
      const validation = validateTableIntegrity(malformedOutput)
      
      expect(validation.valid).toBe(false)
      expect(validation.issues.length).toBeGreaterThan(0)
      
      // Try to reconstruct
      const result = processStreamingWithTableReconstruction(malformedOutput)
      
      // Should attempt to fix issues
      expect(result.display).not.toContain('|---|---|---|---|---|')
    })
  })
  
  describe('Streaming Edge Cases', () => {
    it('should handle chunks that split pipe characters', () => {
      const reconstructor = new TableStreamReconstructor()
      
      // Split right at a pipe
      reconstructor.addChunk('| Header 1 ')
      reconstructor.addChunk('|')
      reconstructor.addChunk(' Header 2 |\n')
      reconstructor.addChunk('|---')
      reconstructor.addChunk('|---|\n')
      reconstructor.addChunk('| Data 1 | Data 2 |\n')
      
      const final = reconstructor.finalize()
      
      expect(final.tables).toHaveLength(1)
      expect(final.raw).toContain('| Header 1 | Header 2 |')
    })
    
    it('should handle empty chunks gracefully', () => {
      const reconstructor = new TableStreamReconstructor()
      
      reconstructor.addChunk('')
      reconstructor.addChunk('| A | B |\n')
      reconstructor.addChunk('')
      reconstructor.addChunk('|---|---|\n')
      reconstructor.addChunk('')
      
      const final = reconstructor.finalize()
      expect(final.raw).toContain('| A | B |')
    })
    
    it('should handle tables with varying column counts', () => {
      const inconsistent = `| A | B | C |
|---|---|
| 1 | 2 | 3 | 4 |
| X | Y |`
      
      const validation = validateTableIntegrity(inconsistent)
      
      expect(validation.valid).toBe(false)
      expect(validation.issues.some(i => i.includes('Column mismatch'))).toBe(true)
    })
  })
  
  describe('Performance with Large Tables', () => {
    it('should handle large tables efficiently', () => {
      const reconstructor = new TableStreamReconstructor()
      
      // Generate a large table
      let largeTable = '| ' + Array(20).fill('Column').join(' | ') + ' |\n'
      largeTable += '|' + Array(20).fill('---').join('|') + '|\n'
      
      // Add 100 rows
      for (let i = 0; i < 100; i++) {
        largeTable += '| ' + Array(20).fill(`Data${i}`).join(' | ') + ' |\n'
      }
      
      // Stream in chunks
      const chunkSize = 100
      const startTime = Date.now()
      
      for (let i = 0; i < largeTable.length; i += chunkSize) {
        reconstructor.addChunk(largeTable.slice(i, i + chunkSize))
      }
      
      const final = reconstructor.finalize()
      const duration = Date.now() - startTime
      
      expect(final.tables).toHaveLength(1)
      expect(duration).toBeLessThan(1000) // Should process in under 1 second
    })
  })
  
  describe('Integration with UI', () => {
    it('should provide proper display content during streaming', () => {
      const reconstructor = new TableStreamReconstructor()
      
      // Start streaming
      let result = reconstructor.addChunk('Some text before table\n\n')
      expect(result.display).toContain('Some text before table')
      
      // Start table
      result = reconstructor.addChunk('| Header |\n')
      expect(result.hasIncompleteTable).toBe(true)
      
      // During table streaming, show placeholder
      expect(result.display).toContain('Some text before table')
      
      // Complete table
      result = reconstructor.addChunk('|---|\n| Data |\n\n')
      
      // More text after
      result = reconstructor.addChunk('Text after table\n')
      
      const final = reconstructor.finalize()
      expect(final.display).toContain('Some text before table')
      expect(final.display).toContain('| Header |')
      expect(final.display).toContain('Text after table')
    })
  })
})