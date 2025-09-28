import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: 'static',
    emptyOutDir: true,
  },
  server: {
    port: 3000,
    proxy: {
      '/invocations': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
    }
  }
})