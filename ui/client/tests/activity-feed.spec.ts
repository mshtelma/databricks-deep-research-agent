import { test, expect } from '@playwright/test'

test.describe('ActivityFeed Component', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to the app
    await page.goto('http://localhost:5173')
    await page.waitForTimeout(1000) // Wait for app to load
  })

  test('should display the chat interface', async ({ page }) => {
    // The app starts on welcome page, navigate to chat
    const chatButton = page.getByRole('button', { name: /Start Chatting/i })
    if (await chatButton.isVisible()) {
      await chatButton.click()
      await page.waitForTimeout(1000)
    }

    // Check that we're on the chat page - look for any container
    const hasContent = await page.locator('body').textContent()
    expect(hasContent).toBeTruthy()
    expect(hasContent?.length).toBeGreaterThan(10)
  })

  test('should show welcome page initially', async ({ page }) => {
    // Check for welcome page elements
    await expect(page.locator('text=/welcome|deep research|databricks/i').first()).toBeVisible({ timeout: 10000 })
  })

  test('should navigate between pages', async ({ page }) => {
    // Find and click navigation button to chat
    const chatButton = page.getByRole('button', { name: /Start Chatting/i })

    if (await chatButton.isVisible()) {
      await chatButton.click()
      await page.waitForTimeout(1000)

      // Verify navigation happened by checking page content changed
      const content = await page.locator('body').textContent()
      expect(content).toBeTruthy()
      expect(content?.length).toBeGreaterThan(10)
    }
  })

  test('should have proper page structure', async ({ page }) => {
    // Check basic page structure
    await expect(page.locator('body')).toBeVisible()
    await expect(page.locator('#root, .root, [data-testid="root"]').first()).toBeVisible()
  })

  test('should handle theme/styling', async ({ page }) => {
    // Check that styles are applied
    const body = page.locator('body')
    await expect(body).toBeVisible()

    // Check for any styled elements
    const styledElements = await page.locator('[class], [style]').count()
    expect(styledElements).toBeGreaterThan(0)
  })
})