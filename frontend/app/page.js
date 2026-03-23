'use client';

import React, { useState } from 'react';
import { PersonaProvider } from './context/PersonaContext';
import Sidebar from './components/Sidebar';
import Dashboard from './components/Dashboard';
import AskAI from './components/AskAI';
import RightsAwareness from './components/RightsAwareness';
import LegalWorkflows from './components/LegalWorkflows';
import Settings from './components/Settings';
import Profile from './components/Profile';

function AppContent() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [initialAskQuery, setInitialAskQuery] = useState('');
  const [activeChatId, setActiveChatId] = useState(null);

  // Sync Global Theme (V20)
  React.useEffect(() => {
    const saved = localStorage.getItem('lawmate_settings');
    if (saved) {
      try {
        const { theme } = JSON.parse(saved);
        if (theme) document.documentElement.setAttribute('data-theme', theme);
      } catch (e) {}
    }
  }, []);

  const handleAskAI = (query, chatId = null) => {
    setActiveChatId(chatId);
    setInitialAskQuery(chatId ? '' : query);
    setActiveTab('ask-ai');
  };

  const renderContent = () => {
    switch (activeTab) {
      case 'dashboard': return <Dashboard handleAskAI={handleAskAI} setActiveTab={setActiveTab} />;
      case 'ask-ai': return <AskAI initialQuery={initialAskQuery} setInitialQuery={setInitialAskQuery} activeChatId={activeChatId} setActiveChatId={setActiveChatId} />;
      case 'rights': return <RightsAwareness handleAskAI={handleAskAI} />;
      case 'workflows': return <LegalWorkflows handleAskAI={handleAskAI} />;
      case 'profile': return <Profile setActiveTab={setActiveTab} setActiveChatId={setActiveChatId} handleAskAI={handleAskAI} />;
      case 'settings': return <Settings />;
      default: return (
        <div className="flex-1 flex items-center justify-center">
          <p className="text-[var(--text-muted)]">Work in progress: {activeTab}</p>
        </div>
      );
    }
  };

  return (
    <div className="app-layout">
      <Sidebar 
        activeTab={activeTab} 
        setActiveTab={setActiveTab} 
        isCollapsed={isCollapsed} 
        setIsCollapsed={setIsCollapsed} 
        activeChatId={activeChatId}
        setActiveChatId={setActiveChatId}
        handleAskAI={handleAskAI}
      />
      <main className="flex-1 flex flex-col relative overflow-hidden bg-[var(--bg-app)]">
        {renderContent()}
      </main>
    </div>
  );
}

export default function Page() {
  return (
    <PersonaProvider>
      <AppContent />
    </PersonaProvider>
  );
}
