import React, { useState } from 'react'
import { WelcomePage } from "./pages/WelcomePage";
import { ChatPage } from "./pages/ChatPage";

function App() {
  const [currentPage, setCurrentPage] = useState<'welcome' | 'chat'>('welcome')
  
  const navigateToChat = () => setCurrentPage('chat')
  const navigateToWelcome = () => setCurrentPage('welcome')
  
  return (
    <div className="min-h-screen bg-background">
      {currentPage === 'welcome' ? (
        <WelcomePage onNavigateToChat={navigateToChat} />
      ) : (
        <ChatPage />
      )}
    </div>
  );
}

export default App;
