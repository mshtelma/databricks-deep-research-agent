import React from 'react'
import ReactDOM from 'react-dom/client'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import App from './App'
import './styles/globals.css'

// External plugin entry point - child projects override via Vite alias
import { registerExternalPlugins } from '@plugins/external'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 30, // 30 seconds (reduced from 5 minutes to prevent stale cache)
      retry: 1,
      refetchOnWindowFocus: true, // Refetch on tab switch to ensure fresh data
    },
  },
})

// Register external plugins BEFORE React renders
// This allows plugins to register output renderers, panels, etc.
try {
  registerExternalPlugins()
} catch (error) {
  console.error('[plugins] Failed to register external plugins:', error)
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </React.StrictMode>,
)
