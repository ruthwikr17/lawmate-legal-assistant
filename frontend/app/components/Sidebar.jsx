'use client';

import React, { useState, useEffect, useRef } from 'react';
import { 
  MessageSquare, 
  LayoutDashboard, 
  ShieldCheck, 
  Workflow,
  Bookmark,
  User,
  Settings,
  ChevronRight,
  Menu,
  Plus,
  Trash2,
  Edit2
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { usePersona } from '../context/PersonaContext';

const navItems = [
  { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { id: 'ask-ai', label: 'Ask AI', icon: MessageSquare },
  { id: 'rights', label: 'Rights Awareness', icon: ShieldCheck },
  { id: 'workflows', label: 'Legal Workflows', icon: Workflow },
  { id: 'profile', label: 'Profile', icon: User },
  { id: 'settings', label: 'Settings', icon: Settings },
];

export default function Sidebar({ activeTab, setActiveTab, isCollapsed, setIsCollapsed, activeChatId, setActiveChatId, handleAskAI }) {
  const { activePersona } = usePersona();
  const [chatHistory, setChatHistory] = useState([]);
  const [sidebarWidth, setSidebarWidth] = useState(260);
  const isResizing = useRef(false);

  useEffect(() => {
    // Load history and width
    const loadState = () => {
      try {
        const history = JSON.parse(localStorage.getItem('lawmate_chat_history') || '[]');
        setChatHistory(history);
        const savedWidth = localStorage.getItem('lawmate_sidebar_width');
        if (savedWidth) setSidebarWidth(parseInt(savedWidth));
      } catch (e) {}
    };
    
    loadState();
    const interval = setInterval(loadState, 2000);
    return () => clearInterval(interval);
  }, []);

  const handleMouseDown = (e) => {
    e.preventDefault();
    isResizing.current = true;
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
    document.body.style.cursor = 'col-resize';
  };

  const handleMouseMove = (e) => {
    if (!isResizing.current) return;
    const newWidth = e.clientX;
    if (newWidth >= 200 && newWidth <= 450) {
      setSidebarWidth(newWidth);
      localStorage.setItem('lawmate_sidebar_width', newWidth.toString());
    }
  };

  const handleMouseUp = () => {
    isResizing.current = false;
    document.removeEventListener('mousemove', handleMouseMove);
    document.removeEventListener('mouseup', handleMouseUp);
    document.body.style.cursor = 'default';
  };

  const actualWidth = isCollapsed ? 80 : sidebarWidth;

  return (
    <motion.div 
      animate={{ width: actualWidth }}
      transition={{ duration: 0.1 }}
      className="h-full bg-[var(--bg-sidebar)] border-r border-[var(--border-light)] flex flex-col flex-shrink-0 relative z-20"
    >
      {/* Header */}
      <div className="flex items-center justify-between p-4 h-16 border-b border-[var(--border-light)]">
        <AnimatePresence>
          {!isCollapsed && (
            <motion.div 
              initial={{ opacity: 0 }} 
              animate={{ opacity: 1 }} 
              exit={{ opacity: 0 }}
              className="flex items-center gap-2 overflow-hidden whitespace-nowrap"
            >
              <div className="w-8 h-8 rounded bg-[var(--accent-teal)] flex items-center justify-center font-bold text-[var(--text-inverse)] flex-shrink-0">
                L
              </div>
              <span className="font-bold text-lg tracking-tight text-[var(--text-primary)]">LawMate</span>
            </motion.div>
          )}
        </AnimatePresence>
        <button 
          onClick={() => setIsCollapsed(!isCollapsed)}
          className={`p-1.5 rounded-md text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)] transition-colors ${isCollapsed ? 'mx-auto' : ''}`}
        >
          <Menu size={20} />
        </button>
      </div>

      {/* Navigation */}
      <nav className="flex-1 overflow-y-auto py-4 px-3 space-y-1 scrollbar-thin">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;
          return (
            <div 
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              title={isCollapsed ? item.label : ''}
              className={`flex items-center gap-3 px-3 py-2.5 rounded-md cursor-pointer transition-colors duration-200
                ${isActive 
                  ? 'bg-[var(--bg-hover)] text-[var(--text-primary)] border-l-2 border-[var(--text-primary)]' 
                  : 'text-[var(--text-secondary)] hover:bg-[var(--bg-hover)] hover:text-[var(--text-primary)]'
                }
              `}
            >
              <Icon size={20} className="flex-shrink-0" />
              {!isCollapsed && (
                <span className="text-sm font-medium whitespace-nowrap overflow-hidden text-ellipsis flex-1">
                  {item.label}
                </span>
              )}
            </div>
          );
        })}
        
        {/* Chat History Section */}
        {!isCollapsed && (
          <div className="pt-4 mt-2 border-t border-[var(--border-light)]">
            <div className="flex items-center justify-between mb-3 px-3">
              <h3 className="text-xs font-semibold text-[var(--text-muted)] uppercase tracking-wider">Recent Chats</h3>
              <button 
                onClick={() => handleAskAI('', null)}
                className="text-xs flex items-center gap-1 font-medium bg-[var(--text-primary)] text-[var(--bg-app)] px-2.5 py-1 rounded hover:opacity-80 transition-opacity shadow-sm"
              >
                <Plus size={12} strokeWidth={3} /> New
              </button>
            </div>
            
            <div className="space-y-0.5">
              {chatHistory.length === 0 ? (
                <div className="px-3 py-2 text-xs text-[var(--text-muted)]">No recent chats yet.</div>
              ) : (
                chatHistory.map((chat) => (
                  <div 
                    key={chat.id}
                    className={`group flex items-center justify-between px-3 py-2 rounded-md transition-colors text-sm
                      ${activeChatId === chat.id && activeTab === 'ask-ai'
                        ? 'bg-[var(--bg-hover)] text-[var(--text-primary)] font-medium'
                        : 'text-[var(--text-secondary)] hover:bg-[var(--bg-hover)] hover:text-[var(--text-primary)]'
                      }
                    `}
                  >
                    <div 
                      className="flex items-center gap-3 flex-1 overflow-hidden cursor-pointer"
                      onClick={() => handleAskAI(chat.title, chat.id)}
                    >
                      <MessageSquare size={16} className="flex-shrink-0 opacity-70" />
                      <span className="whitespace-nowrap overflow-hidden text-ellipsis flex-1">
                        {chat.title}
                      </span>
                    </div>
                    
                    <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0">
                      <button 
                        onClick={(e) => {
                          e.stopPropagation();
                          const newName = prompt("Rename chat:", chat.title);
                          if(newName && newName.trim()) {
                            const updated = chatHistory.map(c => c.id === chat.id ? {...c, title: newName.trim()} : c);
                            localStorage.setItem('lawmate_chat_history', JSON.stringify(updated));
                            setChatHistory(updated);
                          }
                        }}
                        className="p-1 hover:text-[var(--accent-teal)] rounded cursor-pointer"
                      >
                        <Edit2 size={12} />
                      </button>
                      <button 
                        onClick={(e) => {
                          e.stopPropagation();
                          if(confirm("Delete this chat?")) {
                            localStorage.removeItem(`lawmate_chat_${chat.id}`);
                            const updated = chatHistory.filter(c => c.id !== chat.id);
                            localStorage.setItem('lawmate_chat_history', JSON.stringify(updated));
                            setChatHistory(updated);
                            if(activeChatId === chat.id) setActiveChatId(null);
                          }
                        }}
                        className="p-1 hover:text-red-500 rounded cursor-pointer"
                      >
                        <Trash2 size={12} />
                      </button>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        )}
      </nav>

      {/* Persona Badge */}
      <div className="p-4 border-t border-[var(--border-light)]">
        <div 
          className={`flex items-center ${isCollapsed ? 'justify-center' : 'gap-3 px-3 py-2'} rounded-md cursor-pointer hover:bg-[var(--bg-hover)] transition-colors`}
          onClick={() => setActiveTab('settings')}
        >
          <div className="w-8 h-8 rounded-full bg-[var(--bg-panel)] border border-[var(--border-light)] flex items-center justify-center text-lg flex-shrink-0">
            {activePersona.icon}
          </div>
          {!isCollapsed && (
            <div className="flex flex-col overflow-hidden whitespace-nowrap">
              <span className="text-xs font-semibold text-[var(--text-primary)]">{activePersona.label}</span>
              <span className="text-[10px] text-[var(--accent-teal)] uppercase tracking-wider">Active Persona</span>
            </div>
          )}
        </div>
      </div>

      {/* Resize Handle */}
      {!isCollapsed && (
        <div 
          onMouseDown={handleMouseDown}
          className="absolute top-0 right-0 w-1 h-full cursor-col-resize hover:bg-[var(--accent-teal)] opacity-0 hover:opacity-20 transition-opacity z-30"
        />
      )}
    </motion.div>
  );
}
