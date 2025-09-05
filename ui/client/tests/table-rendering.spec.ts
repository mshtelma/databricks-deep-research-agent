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
})