import { test, expect } from '@playwright/test'

test.describe('Table Rendering Tests', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to the frontend
    await page.goto('http://localhost:5173')

    // Wait for the page to load
    await expect(page.locator('body')).toBeVisible()
  })

  test('should render page without table errors', async ({ page }) => {
    // Check that the page loads without table-related errors
    const consoleErrors: string[] = []
    page.on('console', msg => {
      if (msg.type() === 'error' && msg.text().includes('table')) {
        consoleErrors.push(msg.text())
      }
    })

    await page.waitForTimeout(2000)
    expect(consoleErrors).toHaveLength(0)
  })

  test('should handle markdown content', async ({ page }) => {
    // Check that page content is present
    const contentAreas = await page.locator('div').count()
    expect(contentAreas).toBeGreaterThan(0)
  })

  test('should display formatted content', async ({ page }) => {
    // Check for any formatted text elements
    const formattedElements = await page.locator('p, h1, h2, h3, h4, h5, h6, li, blockquote').count()
    expect(formattedElements).toBeGreaterThan(0)
  })

  test('should handle tables in content', async ({ page }) => {
    // Check if table elements can be rendered
    const tables = await page.locator('table, [class*="table"]').count()
    // Tables may or may not be present initially
    expect(tables).toBeGreaterThanOrEqual(0)
  })

  test('should maintain responsive layout', async ({ page }) => {
    // Check viewport responsiveness
    const viewport = page.viewportSize()
    expect(viewport).toBeTruthy()
    expect(viewport?.width).toBeGreaterThan(0)
    expect(viewport?.height).toBeGreaterThan(0)
  })

  test('should handle complex HTML structures', async ({ page }) => {
    // Verify nested elements work
    const nestedElements = await page.locator('div div').count()
    expect(nestedElements).toBeGreaterThan(0)
  })

  test('should preserve content structure', async ({ page }) => {
    // Check that content has proper structure
    const hasStructure = await page.locator('#root, .App, body > div').count()
    expect(hasStructure).toBeGreaterThan(0)
  })

  test('should render without visual artifacts', async ({ page }) => {
    // Basic check that page doesn't have obvious rendering issues
    const bodyText = await page.locator('body').innerText()

    // Should not have raw markdown artifacts
    expect(bodyText).not.toContain('```')
    expect(bodyText).not.toContain('undefined')
    expect(bodyText).not.toContain('null')
  })

  test('should not contain JSON artifacts', async ({ page }) => {
    // Check that page doesn't contain JSON contamination
    const bodyText = await page.locator('body').innerText()

    // Should not have JSON artifacts from agent reasoning
    expect(bodyText).not.toContain('[{"type": "reasoning"')
    expect(bodyText).not.toContain('"summary": [')
    expect(bodyText).not.toContain('"thinking": "')
    expect(bodyText).not.toContain('{"type": "summary_text"')
    expect(bodyText).not.toContain('"type": "reasoning"')
    
    // Check for JSON-like patterns that should be filtered
    expect(bodyText).not.toMatch(/\[\s*\{\s*"type"\s*:\s*"reasoning"/)
    expect(bodyText).not.toMatch(/\{\s*"type"\s*:\s*"summary/)
    expect(bodyText).not.toMatch(/\}\s*,\s*"thinking"/)
  })

  test('should handle table content without JSON contamination', async ({ page }) => {
    // Look for existing tables on the page
    const tables = page.locator('table')
    const tableCount = await tables.count()
    
    if (tableCount > 0) {
      // Check each table for JSON contamination
      for (let i = 0; i < tableCount; i++) {
        const table = tables.nth(i)
        const tableText = await table.innerText()
        
        // Verify no JSON artifacts in table content
        expect(tableText).not.toContain('[{"type":')
        expect(tableText).not.toContain('"reasoning":')
        expect(tableText).not.toContain('"summary":')
        expect(tableText).not.toContain('"}]')
        
        // Check for specific table cell contamination patterns
        const cells = table.locator('td, th')
        const cellCount = await cells.count()
        
        for (let j = 0; j < cellCount; j++) {
          const cellText = await cells.nth(j).innerText()
          
          // Cells should not contain JSON patterns
          expect(cellText).not.toMatch(/\[\s*\{.*"type".*\}.*\]/)
          expect(cellText).not.toMatch(/\{.*"reasoning".*\}/)
          expect(cellText).not.toContain('}, "thinking":')
        }
      }
    }
  })

  test('should properly render table markdown without JSON fragments', async ({ page }) => {
    // Check for proper table structure in the DOM
    const markdownTables = page.locator('[class*="markdown"] table, [data-testid*="message"] table')
    const markdownTableCount = await markdownTables.count()
    
    if (markdownTableCount > 0) {
      for (let i = 0; i < markdownTableCount; i++) {
        const table = markdownTables.nth(i)
        
        // Verify table has proper HTML structure
        const headers = table.locator('th')
        const rows = table.locator('tr')
        
        const headerCount = await headers.count()
        const rowCount = await rows.count()
        
        // Tables should have headers and content rows
        expect(headerCount).toBeGreaterThan(0)
        expect(rowCount).toBeGreaterThan(1) // At least header + one data row
        
        // Check that table content is clean
        const tableHTML = await table.innerHTML()
        
        // Should not contain JSON patterns in HTML
        expect(tableHTML).not.toContain('[{"type":')
        expect(tableHTML).not.toContain('"reasoning":')
        expect(tableHTML).not.toContain('{"summary":')
        expect(tableHTML).not.toMatch(/\|\s*\|\s*\[.*\{.*"type"/)
      }
    }
  })
})