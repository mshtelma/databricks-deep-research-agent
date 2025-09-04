import { useState } from 'react'
import { WelcomePage } from "./pages/WelcomePage";
import { ChatPage } from "./pages/ChatPage";

function App() {
  const [currentPage, setCurrentPage] = useState<'welcome' | 'chat'>('chat')
  
  const navigateToChat = () => setCurrentPage('chat')
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const _navigateToWelcome = () => setCurrentPage('welcome') // Unused but kept for future use
  
  // Temporary fix: use inline styles to bypass potential CSS issues
  return (
    <div style={{ minHeight: '100vh', backgroundColor: 'white' }}>
      {currentPage === 'welcome' ? (
        <WelcomePage onNavigateToChat={navigateToChat} />
      ) : (
        <ChatPage />
      )}
    </div>
  );
}

export default App;
