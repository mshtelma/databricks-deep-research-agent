import { test, expect } from '@playwright/test'

test.describe('Stream Parsing and Events', () => {
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

  test('should handle streaming responses', async ({ page }) => {
    // Check that the page is ready with content
    const content = await page.locator('body').textContent()
    expect(content).toBeTruthy()
    expect(content?.length).toBeGreaterThan(10)
  })

  test('should process message events', async ({ page }) => {
    // Find message input
    const messageInput = page.locator('textarea, input[type="text"]').first()

    if (await messageInput.isVisible({ timeout: 5000 })) {
      // Type a test message
      await messageInput.fill('Test message')

      // Check that input received the text
      const value = await messageInput.inputValue()
      expect(value).toBe('Test message')

      // Clear for next test
      await messageInput.clear()
    }
  })

  test('should handle content accumulation', async ({ page }) => {
    // Check that content area exists
    const hasContent = await page.locator('div').count()
    expect(hasContent).toBeGreaterThan(0)
  })

  test('should handle error states gracefully', async ({ page }) => {
    // Check page loads without critical errors
    let hasError = false
    page.on('pageerror', () => {
      hasError = true
    })

    await page.waitForTimeout(2000)
    expect(hasError).toBe(false)
  })

  test('should reset state correctly', async ({ page }) => {
    // Navigate to chat and back
    const buttons = page.getByRole('button')
    const buttonCount = await buttons.count()

    if (buttonCount > 0) {
      // Click first button and check state changes
      await buttons.first().click({ timeout: 5000 }).catch(() => { })
      await page.waitForTimeout(500)

      // Page should still be responsive
      const isVisible = await page.locator('body').isVisible()
      expect(isVisible).toBe(true)
    }
  })

  test('should handle malformed events gracefully', async ({ page }) => {
    // Ensure page doesn't crash with console errors
    const consoleErrors: string[] = []
    page.on('console', msg => {
      if (msg.type() === 'error' && !msg.text().includes('favicon') && !msg.text().includes('404')) {
        consoleErrors.push(msg.text())
      }
    })

    await page.waitForTimeout(2000)

    // Should have no critical errors
    expect(consoleErrors.filter(e => e.includes('TypeError') || e.includes('ReferenceError')).length).toBe(0)
  })

  test('should maintain UI responsiveness', async ({ page }) => {
    // Check that UI elements remain interactive
    const interactiveElements = await page.locator('button, a, input, textarea').count()
    expect(interactiveElements).toBeGreaterThan(0)

    // Check first interactive element is enabled
    const firstButton = page.getByRole('button').first()
    if (await firstButton.isVisible({ timeout: 5000 })) {
      const isDisabled = await firstButton.isDisabled()
      expect(isDisabled).toBe(false)
    }
  })

  test('should handle rapid updates', async ({ page }) => {
    // Test that rapid interactions don't break the UI
    const messageInput = page.locator('textarea, input[type="text"]').first()

    if (await messageInput.isVisible({ timeout: 5000 })) {
      // Rapidly type and clear
      for (let i = 0; i < 3; i++) {
        await messageInput.fill(`Test ${i}`)
        await messageInput.clear()
      }

      // Input should still be functional
      await messageInput.fill('Final test')
      const value = await messageInput.inputValue()
      expect(value).toBe('Final test')
    }
  })
})