import { fileURLToPath, URL } from 'node:url'

import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Dedicated build for the standalone shell-app UI (the deployed per-agent app).
// Separate `outDir` from the main build (which uses `../static` +
// `emptyOutDir`) so the two never clobber each other. The exporter bundles the
// output of this build into the shell-app zip's `static/`.
export default defineConfig({
  plugins: [react()],
  define: {
    // Build id carried on client error reports (see lib/clientErrors.ts).
    __BUILD_ID__: JSON.stringify(process.env.BUILD_ID ?? String(Date.now())),
  },
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
      // Inherited by the shell entry; the shell doesn't register plugins but
      // the alias must resolve for any transitive import.
      '@plugins/external': fileURLToPath(new URL('./src/plugins/external.ts', import.meta.url)),
    },
  },
  // The shell-app serves static assets under /static (app.py mounts
  // StaticFiles there) while index.html is served at /, so asset URLs must be
  // /static/assets/… — set the base accordingly.
  base: '/static/',
  build: {
    outDir: 'dist-shell',
    emptyOutDir: true,
    rollupOptions: {
      input: fileURLToPath(new URL('./shell.html', import.meta.url)),
    },
  },
})
