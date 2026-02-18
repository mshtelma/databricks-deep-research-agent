import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { ErrorBoundary } from './components/common/ErrorBoundary'
import ChatPage from './pages/ChatPage'
import AgentsPage from './pages/AgentsPage'
import TemplatesPage from './pages/TemplatesPage'

function App() {
  return (
    <ErrorBoundary name="App">
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Navigate to="/chat" replace />} />
          {/* Use optional param to prevent component remount when navigating between /chat and /chat/:chatId */}
          <Route path="/chat/:chatId?" element={<ChatPage />} />
          <Route path="/data-sources" element={<Navigate to="/chat" replace />} />
          <Route path="/agents" element={<AgentsPage />} />
          <Route path="/templates" element={<TemplatesPage />} />
        </Routes>
      </BrowserRouter>
    </ErrorBoundary>
  )
}

export default App
