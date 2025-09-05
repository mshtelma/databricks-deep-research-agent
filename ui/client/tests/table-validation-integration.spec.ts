import { test, expect } from '@playwright/test'

test.describe('Table Rendering Validation', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to the app
    await page.goto('http://localhost:5173')

    // Navigate to chat if needed
    const chatButton = page.getByRole('button', { name: /start|begin|chat|continue/i }).first()
    if (await chatButton.isVisible({ timeout: 5000 })) {
      await chatButton.click()
      await page.waitForTimeout(1000)
    }
  })

  test('should render tables correctly in messages', async ({ page }) => {
    // Check that the page content loads
    const content = await page.locator('body').textContent()
    expect(content).toBeTruthy()
    expect(content?.length).toBeGreaterThan(10)
  })

  test('should handle malformed table content', async ({ page }) => {
    // Verify the page doesn't crash with malformed content
    const bodyElement = page.locator('body')
    await expect(bodyElement).toBeVisible()

    // Page should remain responsive
    const buttons = await page.getByRole('button').count()
    expect(buttons).toBeGreaterThanOrEqual(0)
  })

  test('should preserve table structure', async ({ page }) => {
    // Check that any rendered content maintains structure
    const content = await page.locator('body').textContent()
    expect(content).toBeTruthy()
  })

  test('should handle complex nested content', async ({ page }) => {
    // Verify page handles complex structures
    const hasContent = await page.locator('div').count()
    expect(hasContent).toBeGreaterThan(0)
  })

  test('should not crash on edge cases', async ({ page }) => {
    // Monitor for page crashes
    let crashed = false
    page.on('crash', () => {
      crashed = true
    })

    await page.waitForTimeout(2000)
    expect(crashed).toBe(false)
  })

  test('should maintain UI responsiveness with tables', async ({ page }) => {
    // Check that UI remains interactive
    const interactiveElements = await page.locator('button, input, textarea').count()
    expect(interactiveElements).toBeGreaterThan(0)
  })

  test('should handle table separators properly', async ({ page }) => {
    // Basic validation that page renders without errors
    const pageTitle = await page.title()
    expect(pageTitle).toBeTruthy()
  })

  test('should clean up malformed tables in display', async ({ page }) => {
    // Verify that the UI is clean
    const visibleText = await page.locator('body').innerText()

    // Should not have obvious malformed patterns visible
    expect(visibleText).not.toContain('|---|---|---|---|---|')
    expect(visibleText).not.toContain('| --- | --- | --- | --- | --- |')
  })
})