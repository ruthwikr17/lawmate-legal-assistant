'use client';

import React, { useState, useEffect, useRef } from 'react';
import { Send, User, Bot, Trash2, StopCircle, Bookmark, RefreshCw, Sparkles, AlertCircle, Shield, Copy, ChevronRight, MessageSquare, Plus } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import ReactMarkdown from 'react-markdown';

export default function AskAI({ initialQuery, setInitialQuery, activeChatId, setActiveChatId }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [chatTitle, setChatTitle] = useState('New Analysis');
  const abortControllerRef = useRef(null);
  const scrollRef = useRef(null);
  const hasSentInitialRef = useRef(false);

  // Load chat history and title
  useEffect(() => {
    const loadChat = () => {
      if (activeChatId) {
        const savedMessages = localStorage.getItem(`lawmate_chat_${activeChatId}`);
        if (savedMessages) setMessages(JSON.parse(savedMessages));
        
        const history = JSON.parse(localStorage.getItem('lawmate_chat_history') || '[]');
        const currentChat = history.find(h => h.id === activeChatId);
        if (currentChat) setChatTitle(currentChat.title);
      } else {
        setMessages([]);
        setChatTitle('New Analysis');
      }
    };

    loadChat();
    window.addEventListener('storage', loadChat);
    return () => window.removeEventListener('storage', loadChat);
  }, [activeChatId]);

  // Handle Initial Query from Dashboard (V49 Guarded)
  useEffect(() => {
    if (initialQuery && !activeChatId && !hasSentInitialRef.current) {
      hasSentInitialRef.current = true;
      handleSend(initialQuery);
      setInitialQuery('');
    }
  }, [initialQuery, activeChatId]);

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const handleStop = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      setIsLoading(false);
      setMessages(prev => {
        const updated = [...prev, {
          role: 'assistant',
          content: '_Analysis halted by user._',
          isStopped: true
        }];
        if (activeChatId) localStorage.setItem(`lawmate_chat_${activeChatId}`, JSON.stringify(updated));
        return updated;
      });
    }
  };

  const handleSend = async (query = input) => {
    if (!query.trim() || isLoading) return;

    setError(null);
    const userMessage = { role: 'user', content: query, id: Date.now() };
    const newMessages = [...messages, userMessage];
    setMessages(newMessages);
    setInput('');
    setIsLoading(true);

    let currentChatId = activeChatId;
    if (!currentChatId) {
      currentChatId = Date.now();
      localStorage.setItem(`lawmate_chat_${currentChatId}`, JSON.stringify([userMessage]));
      setActiveChatId(currentChatId);
    }
    
    // Context
    const profile = JSON.parse(localStorage.getItem('lawmate_user_profile') || '{}');
    const settings = JSON.parse(localStorage.getItem('lawmate_settings') || '{}');
    const persona = JSON.parse(localStorage.getItem('lawmate_active_persona') || '{"id":"citizen", "label":"Citizen"}');

    const context = {
      persona: persona.id || 'citizen',
      jurisdiction: settings.jurisdiction || 'India',
      city: settings.city || 'Not Specified',
      gender: profile.gender || 'Not Specified',
      style: settings.aiStyle || 'Simple',
      depth: settings.aiDepth || 'Medium'
    };

    abortControllerRef.current = new AbortController();

    try {
      const response = await fetch('http://localhost:8080/api/legal/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          history: messages.map(m => ({ role: m.role, content: m.content })),
          context: context
        }),
        signal: abortControllerRef.current.signal
      });

      if (!response.ok) throw new Error('Network response failure');
      const data = await response.json();
      
      const assistantMessage = {
        role: 'assistant',
        content: data.answer || "I'm sorry, I couldn't process that.",
        sources: data.sources || []
      };
      
      const updatedMessages = [...newMessages, assistantMessage];
      setMessages(updatedMessages);
      localStorage.setItem(`lawmate_chat_${currentChatId}`, JSON.stringify(updatedMessages));
      
      const fullHistory = JSON.parse(localStorage.getItem('lawmate_chat_history') || '[]');
      const existingIdx = fullHistory.findIndex(h => h.id === currentChatId);
      
      if (existingIdx !== -1) {
        fullHistory[existingIdx].updatedAt = new Date().toISOString();
      } else {
        fullHistory.unshift({
          id: currentChatId,
          title: query.slice(0, 45) + (query.length > 45 ? '...' : ''),
          updatedAt: new Date().toISOString()
        });
      }
      
      localStorage.setItem('lawmate_chat_history', JSON.stringify(fullHistory.slice(0, 50)));
      window.dispatchEvent(new Event('storage'));

    } catch (err) {
      if (err.name !== 'AbortError') setError('LawMate is experiencing a momentary technical difficulty.');
    } finally {
      setIsLoading(false);
      abortControllerRef.current = null;
    }
  };

  return (
    <div className="flex flex-col h-full bg-[var(--bg-app)] relative overflow-hidden">
      
      {/* Dynamic Chat Header (V44) */}
      <div className="h-16 flex items-center justify-between px-8 border-b border-[var(--border-light)] bg-[var(--bg-app)]/50 backdrop-blur-md z-10">
        <div className="flex items-center gap-3 overflow-hidden">
          <div className="flex-shrink-0 w-8 h-8 rounded-lg bg-[var(--bg-panel)] border border-[var(--border-light)] flex items-center justify-center text-[var(--accent-teal)]">
             <MessageSquare size={16} />
          </div>
          <div className="flex flex-col overflow-hidden">
            <h2 className="text-sm font-bold text-[var(--text-primary)]">{chatTitle}</h2>
            <div className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse"></span>
              <span className="text-[10px] text-[var(--text-muted)] uppercase tracking-widest font-bold">Active Engine</span>
            </div>
          </div>
        </div>
        
        <div className="flex items-center gap-4">
           {activeChatId && (
             <button 
               onClick={() => setActiveChatId(null)}
               className="text-[10px] font-black uppercase tracking-widest text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors flex items-center gap-1"
             >
               <Plus size={14} /> New Chat
             </button>
           )}
        </div>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-10 md:px-20 space-y-12 hide-scrollbar pb-64">
        <AnimatePresence>
          {messages.length === 0 && !isLoading && (
            <motion.div 
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="h-full flex flex-col items-center justify-center text-center space-y-6 opacity-30 py-20"
            >
               <Bot size={80} strokeWidth={1} />
               <div>
                  <h3 className="text-xl font-black uppercase tracking-widest text-[var(--text-primary)]">Intelligence Ready</h3>
                  <p className="text-sm mt-1 max-w-sm mx-auto">LawMate is standing by for clinical legal auditing.</p>
               </div>
            </motion.div>
          )}

          {messages.map((msg, idx) => (
            <motion.div 
              initial={{ opacity: 0, x: msg.role === 'user' ? 10 : -10 }}
              animate={{ opacity: 1, x: 0 }}
              key={idx}
              className={`flex gap-4 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}
            >
              <div className={`w-12 h-12 rounded-xl shrink-0 flex items-center justify-center shadow-md ${
                msg.role === 'user' ? 'bg-[var(--text-primary)] text-[var(--bg-app)]' : 'bg-[var(--bg-panel)] border border-[var(--border-light)] text-[var(--accent-teal)]'
              }`}>
                {msg.role === 'user' ? <User size={20} /> : <Bot size={20} />}
              </div>
              
              <div className={`max-w-[85%] space-y-4 ${msg.role === 'user' ? 'text-right' : ''}`}>
                <div className={`p-6 md:p-7 rounded-[2rem] leading-relaxed text-[15px] ${
                  msg.role === 'user' 
                  ? 'bg-[var(--bg-panel)] border border-[var(--border-light)] rounded-tr-none text-left shadow-sm' 
                  : 'bg-[var(--bg-panel)]/30 border-l-4 border-[var(--border-light)] pl-8 py-5 rounded-none font-medium text-[var(--text-primary)] relative'
                }`}>
                  {msg.role === 'assistant' && (
                    <div className="absolute top-0 left-0 w-1 h-full bg-[var(--accent-teal)] opacity-40"></div>
                  )}
                  <ReactMarkdown 
                    components={{
                      p: ({node, ...props}) => <p className="mt-6 mb-6 last:mb-0 first:mt-0" {...props} />,
                      h2: ({node, ...props}) => <h2 className="text-lg font-black mt-8 mb-4 text-[var(--accent-teal)] uppercase tracking-wider" {...props} />,
                      h3: ({node, ...props}) => <h3 className="text-base font-black mt-6 mb-3 text-[var(--text-primary)]" {...props} />,
                      ul: ({node, ...props}) => <ul className="space-y-3 mb-14 list-none" {...props} />,
                      li: ({node, ...props}) => (
                        <li className="flex gap-3 items-start" {...props}>
                          <span className="w-1.5 h-1.5 rounded-full bg-[var(--accent-teal)] mt-2 shrink-0" />
                          <span>{props.children}</span>
                        </li>
                      ),
                      strong: ({node, ...props}) => <strong className="font-black text-[var(--accent-teal)]" {...props} />,
                    }}
                  >
                    {msg.content}
                  </ReactMarkdown>

                  {msg.role === 'assistant' && !msg.isStopped && (
                    <div className="flex items-center gap-6 mt-8 pt-5 border-t border-[var(--border-light)]/30">
                      <button 
                        onClick={() => navigator.clipboard.writeText(msg.content)}
                        className="flex items-center gap-2 text-[10px] uppercase font-black tracking-[0.2em] text-[var(--text-muted)] hover:text-[var(--accent-teal)] opacity-60 hover:opacity-100 transition-all active:scale-90"
                      >
                         <Copy size={13} /> Copy
                      </button>
                      {idx === messages.length - 1 && (
                        <button 
                          onClick={() => {
                            const lastUserMsg = [...messages].reverse().find(m => m.role === 'user');
                            if (lastUserMsg) handleSend(lastUserMsg.content);
                          }}
                          className="flex items-center gap-2 text-[10px] uppercase font-black tracking-[0.2em] text-[var(--text-muted)] hover:text-[var(--accent-teal)] opacity-60 hover:opacity-100 transition-all active:scale-90"
                        >
                           <RefreshCw size={13} /> Retry
                        </button>
                      )}
                    </div>
                  )}
                </div>
              </div>
            </motion.div>
          ))}

          {isLoading && (
            <motion.div 
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              className="flex gap-4"
            >
              <div className="w-8 h-8 rounded-lg glass-panel !p-0 border-[var(--border-light)] flex items-center justify-center text-[var(--accent-teal)] animate-pulse">
                <Sparkles size={16} />
              </div>
              <div className="glass-panel !p-4 rounded-xl border-[var(--border-light)] rounded-tl-none">
                 <div className="flex gap-1.5">
                    <div className="w-1.5 h-1.5 bg-[var(--text-muted)] rounded-full animate-bounce"></div>
                    <div className="w-1.5 h-1.5 bg-[var(--text-muted)] rounded-full animate-bounce [animation-delay:-0.15s]"></div>
                    <div className="w-1.5 h-1.5 bg-[var(--text-muted)] rounded-full animate-bounce [animation-delay:-0.3s]"></div>
                 </div>
              </div>
            </motion.div>
          )}

          {error && (
            <motion.div 
              initial={{ opacity: 0, y: 5 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex justify-center"
            >
              <div className="bg-red-500/10 border border-red-500/20 text-red-500 px-6 py-3 rounded-2xl flex items-center gap-2 text-[10px] font-bold uppercase tracking-widest">
                <AlertCircle size={14} /> {error}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
        <div ref={scrollRef} />
      </div>

      {/* Input Section */}
      <div className="absolute bottom-0 left-0 right-0 p-8 pb-10 bg-gradient-to-t from-[var(--bg-app)] via-[var(--bg-app)] to-transparent pointer-events-none">
        <div className="max-w-2xl mx-auto relative group pointer-events-auto">
          <div className="absolute -inset-0.5 bg-gradient-to-r from-[var(--accent-teal)] to-blue-600 rounded-[2.5rem] opacity-0 blur-md group-focus-within:opacity-10 transition-opacity"></div>
          <div className="relative border border-[var(--border-light)] flex items-center p-1.5 bg-[var(--bg-panel)] rounded-[2.5rem] shadow-2xl">
            <textarea 
              rows={1}
              placeholder="Ask anything about Indian Law..."
              className="flex-1 bg-transparent border-none py-3 px-6 outline-none text-[var(--text-primary)] font-medium placeholder:text-[var(--text-muted)]/30 resize-none max-h-32 text-sm leading-relaxed"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                }
              }}
            />
            <div className="flex gap-2 p-1">
              {isLoading ? (
                <button 
                  onClick={handleStop}
                  className="bg-red-500/10 text-red-500 p-3 rounded-full hover:bg-red-500/20 transition-all active:scale-95"
                >
                  <StopCircle size={20} />
                </button>
              ) : (
                <button 
                  onClick={() => handleSend()}
                  disabled={!input.trim()}
                  className="bg-[var(--text-primary)] text-[var(--bg-app)] p-3 rounded-full hover:opacity-90 active:scale-95 transition-all shadow-xl disabled:opacity-20"
                >
                  <Send size={20} />
                </button>
              )}
            </div>
          </div>
        </div>
      </div>

    </div>
  );
}
