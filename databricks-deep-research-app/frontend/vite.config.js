var _a;
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { fileURLToPath, URL } from 'node:url';
// https://vitejs.dev/config/
export default defineConfig({
    plugins: [react()],
    define: {
        // Build id carried on client error reports (see lib/clientErrors.ts).
        __BUILD_ID__: JSON.stringify((_a = process.env.BUILD_ID) !== null && _a !== void 0 ? _a : String(Date.now())),
    },
    resolve: {
        alias: {
            '@': fileURLToPath(new URL('./src', import.meta.url)),
            // External plugin entry point - child projects override this alias
            '@plugins/external': fileURLToPath(new URL('./src/plugins/external.ts', import.meta.url)),
        },
    },
    build: {
        // Output to static folder for unified deployment
        outDir: '../static',
        emptyOutDir: true,
    },
    server: {
        port: 5173,
        proxy: {
            '/api': {
                target: 'http://localhost:8000',
                changeOrigin: true,
            },
        },
    },
});
