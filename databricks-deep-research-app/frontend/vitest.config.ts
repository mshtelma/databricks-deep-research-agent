import { defineConfig, mergeConfig } from 'vitest/config'
import viteConfig from './vite.config'

// Vitest config — reuses the Vite plugin/alias setup so tests resolve
// the same '@' aliases as the dev server.
export default mergeConfig(
  viteConfig,
  defineConfig({
    test: {
      environment: 'jsdom',
      globals: true,
      setupFiles: [],
      include: ['src/**/*.{test,spec}.{ts,tsx}'],
    },
  })
)
