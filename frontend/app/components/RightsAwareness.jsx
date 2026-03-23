'use client';

import React, { useState, useEffect, useMemo, useRef } from 'react';
import { usePersona } from '../context/PersonaContext';
import { ShieldCheck, BookOpen, Scale, AlertCircle, ChevronRight, X, Search, ChevronLeft, ArrowRight, Layers } from 'lucide-react';
import { motion, AnimatePresence, useScroll, useMotionValueEvent } from 'framer-motion';

export default function RightsAwareness({ handleAskAI }) {
  const { activePersona } = usePersona();
  const [rights, setRights] = useState([]);
  const [selectedRight, setSelectedRight] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [activeTab, setActiveTab] = useState('persona'); // 'persona', 'general', 'all'
  const [isLoading, setIsLoading] = useState(true);
  const scrollContainerRef = useRef(null);
  
  // DRIVING INDEX: The single source of truth for the 'active card'.
  const [activeIndex, setActiveIndex] = useState(0);

  useEffect(() => {
    fetch('/data/rights_ontology.json')
      .then(res => res.json())
      .then(data => {
        setRights(data || []);
        setIsLoading(false);
      })
      .catch(err => {
        console.error("Failed to load rights:", err);
        setIsLoading(false);
      });

    // Hard-lock all scrolls to prevent page-slip while in this module
    const mainContainer = document.querySelector('main');
    const originalBodyStyle = window.getComputedStyle(document.body).overflow;
    const originalMainStyle = mainContainer ? window.getComputedStyle(mainContainer).overflow : 'hidden';
    
    document.body.style.overflow = 'hidden';
    if (mainContainer) mainContainer.style.overflow = 'hidden';

    return () => {
      document.body.style.overflow = originalBodyStyle;
      if (mainContainer) mainContainer.style.overflow = originalMainStyle;
    };
  }, []);

  const filteredRights = useMemo(() => {
    if (!rights) return [];
    return rights.filter(r => {
      const q = searchQuery.toLowerCase();
      const matchesSearch = r.title.toLowerCase().includes(q) || 
                           r.summary.toLowerCase().includes(q) ||
                           r.category.toLowerCase().includes(q);
      
      const isPersonaMatch = activePersona && r.persona === activePersona.id;
      const isGeneralMatch = r.persona === 'citizen';

      if (activeTab === 'persona') return matchesSearch && isPersonaMatch;
      if (activeTab === 'general') return matchesSearch && isGeneralMatch;
      return matchesSearch; // 'all'
    });
  }, [rights, searchQuery, activeTab, activePersona]);

  const { scrollY } = useScroll({
    container: scrollContainerRef,
  });

  useMotionValueEvent(scrollY, "change", (latest) => {
    const newIndex = Math.floor(latest / 550);
    if (newIndex !== activeIndex && newIndex >= 0 && newIndex < (filteredRights?.length || 0)) {
      setActiveIndex(newIndex);
    }
  });

  const RightCard = ({ right, index, total, isActive, isNext, isStack }) => {
    const variants = {
      active: { y: 0, opacity: 1, scale: 1, zIndex: 100 },
      next: { y: 20, opacity: 0.4, scale: 0.95, zIndex: 50 },
      stack: { y: 40, opacity: 0.1, scale: 0.9, zIndex: 20 },
      exit: { y: -250, opacity: 0, scale: 1.05, rotate: -3, zIndex: 110 }
    };

    let status = 'stack';
    if (isActive) status = 'active';
    else if (isNext) status = 'next';
    else if (index < activeIndex) return null; 

    return (
      <motion.div 
        animate={status}
        variants={variants}
        initial={{ y: 250, opacity: 0, scale: 0.95 }}
        transition={{ type: "spring", damping: 25, stiffness: 180, mass: 1 }}
        style={{ 
          position: 'absolute',
          top: 0, left: 0, right: 0, bottom: 0,
          display: 'flex', alignItems: 'center', justifyContent: 'center'
        }}
        className="w-full h-full pointer-events-none"
      >
        <div 
          onClick={() => { if(isActive) setSelectedRight(right); }}
          className="glass-panel p-0 rounded-[2.5rem] cursor-pointer transition-all h-[380px] w-full max-w-4xl bg-[var(--bg-panel)] border border-[var(--border-light)] shadow-2xl relative overflow-hidden group flex pointer-events-auto"
        >
          {/* Deck Indexer */}
          {isActive && (
            <div className="absolute top-8 right-8 z-20 text-[10px] font-black text-[var(--text-muted)] opacity-20 uppercase tracking-[0.3em]">
              {index + 1} / {total}
            </div>
          )}

          {/* Left Visual Pillar */}
          <div className="w-1/3 bg-[var(--bg-panel)] border-r border-[var(--border-light)] p-8 flex flex-col justify-between relative overflow-hidden">
             <div className="absolute top-0 left-0 w-full h-full bg-gradient-to-br from-[var(--text-primary)] opacity-5 to-transparent pointer-events-none"></div>
             <div className="relative z-10">
                <div className="w-14 h-14 rounded-2xl bg-[var(--text-primary)] text-[var(--bg-app)] flex items-center justify-center shadow-lg mb-4 group-hover:scale-110 transition-transform duration-500">
                   <ShieldCheck size={28} />
                </div>
                <span className="text-[9px] uppercase font-black tracking-[0.4em] text-[var(--text-muted)] block mb-2">Statutory Index</span>
                <h3 className="text-3xl font-black text-[var(--text-primary)] leading-[1.1] tracking-tighter mb-4">
                  {right.title}
                </h3>
             </div>
             <div className="relative z-10">
                <div className="flex items-center gap-2 mb-4">
                   <div className="px-3 py-1 rounded-full bg-[var(--bg-app)] border border-[var(--border-light)] text-[8px] font-black uppercase text-[var(--text-muted)] tracking-widest">
                     {right.category}
                   </div>
                   <div className="px-3 py-1 rounded-full bg-[var(--bg-app)] border border-[var(--border-light)] text-[8px] font-black uppercase text-[var(--text-muted)] tracking-widest">
                     Certified
                   </div>
                </div>
                <div className="text-[10px] font-black text-[var(--text-muted)] opacity-20 uppercase tracking-[0.3em]">REF: {right.id}</div>
             </div>
          </div>

          {/* Right Content Dossier */}
          <div className="flex-1 p-10 flex flex-col justify-between">
            <div>
              <div className="flex items-center gap-4 mb-8">
                <div className="h-[1px] w-8 bg-[var(--border-light)]"></div>
                <span className="text-[10px] uppercase font-black tracking-[0.3em] text-[var(--text-muted)] opacity-40">Core Legal Provision</span>
              </div>
              <p className="text-base text-[var(--text-primary)]/70 leading-relaxed font-medium line-clamp-5 first-letter:text-2xl first-letter:font-black first-letter:text-[var(--text-primary)] first-letter:mr-1">
                {right.description}
              </p>
            </div>

            <div className="flex items-center justify-between pt-12 border-t border-[var(--border-light)]">
              <div className="flex items-center gap-8">
                <div className="flex flex-col gap-1">
                   <span className="text-[8px] uppercase font-black text-[var(--text-muted)] opacity-40 tracking-widest">Enforceability</span>
                   <div className="flex gap-1">
                     {[1,2,3,4,5].map(i => <div key={i} className={`w-4 h-1 rounded-full ${i <= 3 ? 'bg-[var(--accent-teal)]' : 'bg-[var(--border-light)]'}`}></div>)}
                   </div>
                </div>
              </div>
              
              <button 
                onClick={() => setSelectedRight(right)}
                className="flex items-center gap-2 px-8 py-3 rounded-full bg-[var(--text-primary)] text-[var(--bg-app)] font-black text-[10px] uppercase tracking-widest hover:opacity-90 transition-all shadow-xl group/btn"
              >
                Explore <ArrowRight size={14} className="group-hover:translate-x-1 transition-transform" />
              </button>
            </div>
          </div>
        </div>
      </motion.div>
    );
  };

  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden bg-[var(--bg-app)] relative">
      
      {/* FIXED TOP SECTION */}
      <div className="shrink-0 pt-12 pb-8 px-12 z-[120] bg-[var(--bg-app)]">
        <header className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-10 max-w-[1400px] mx-auto">
          <div>
            <div className="flex items-center gap-2.5 mb-2 opacity-30">
              <Scale size={20} className="text-[var(--text-primary)]" />
              <span className="text-[10px] uppercase font-black tracking-[0.5em] text-[var(--text-primary)]">Rights Defense Module</span>
            </div>
            <h2 className="text-4xl font-black tracking-tighter text-[var(--text-primary)]">
              Rights Awareness
            </h2>
          </div>

          <div className="flex flex-col sm:flex-row items-center gap-6">
            <div className="flex items-center gap-1 p-1 bg-[var(--bg-panel)] border border-[var(--border-light)] rounded-full">
              {[
                { id: 'persona', label: `Personalized` },
                { id: 'general', label: 'General' },
                { id: 'all', label: 'All' }
              ].map(tab => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`px-8 py-2.5 rounded-full text-[10px] font-black transition-all uppercase tracking-widest ${
                    activeTab === tab.id 
                    ? 'bg-[var(--text-primary)] text-[var(--bg-app)] shadow-lg' 
                    : 'text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)]'
                  }`}
                >
                  {tab.label}
                </button>
              ))}
            </div>

            <div className="relative group">
              <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-[var(--text-muted)] opacity-30 group-focus-within:opacity-100 transition-opacity" size={16} />
              <input 
                type="text"
                placeholder="Search repository..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full sm:w-80 pl-12 pr-6 py-3 bg-[var(--bg-panel)] border border-[var(--border-light)] rounded-full text-sm text-[var(--text-primary)] focus:outline-none focus:border-[var(--accent-teal)] transition-all placeholder:text-[var(--text-muted)] opacity-60"
              />
            </div>
          </div>
        </header>
      </div>

      <div className="flex-1 relative overflow-hidden">
        <div className="absolute inset-0 z-10 flex items-center justify-center p-12 overflow-hidden pointer-events-none">
          <div className="w-full max-w-4xl h-[380px] relative">
            <AnimatePresence mode="popLayout">
              {isLoading ? (
                <div key="loader" className="absolute inset-0 flex items-center justify-center">
                  <div className="w-10 h-10 border border-[var(--border-light)] border-t-[var(--accent-teal)] rounded-full animate-spin"></div>
                </div>
              ) : (filteredRights?.length || 0) > 0 ? (
                filteredRights.map((r, i) => (
                  <RightCard 
                    key={r.id} 
                    right={r} 
                    index={i} 
                    total={filteredRights.length}
                    isActive={i === activeIndex}
                    isNext={i === activeIndex + 1}
                    isStack={i > activeIndex + 1}
                  />
                )).slice(activeIndex, activeIndex + 3)
              ) : (
                <div key="empty" className="absolute inset-0 flex flex-col items-center justify-center text-center opacity-20">
                  <AlertCircle size={64} strokeWidth={1} className="mb-6 text-[var(--text-primary)]" />
                  <p className="text-sm font-black uppercase tracking-widest text-[var(--text-primary)]">No matching records</p>
                </div>
              )}
            </AnimatePresence>
          </div>
        </div>

        <div 
          ref={scrollContainerRef}
          onClick={() => {
            const activeRight = filteredRights[activeIndex];
            if (activeRight) setSelectedRight(activeRight);
          }}
          className="absolute inset-0 overflow-y-auto z-20 pointer-events-auto hide-scrollbar cursor-pointer"
        >
          <div style={{ height: `${(filteredRights?.length || 1) * 550 + 200}px` }}></div>
        </div>
      </div>

      {/* MODAL VIEW */}
      <AnimatePresence>
        {selectedRight && (
          <div className="fixed inset-0 z-[1000] flex items-center justify-center p-6 bg-black/60 backdrop-blur-xl">
            <motion.div 
              initial={{ opacity: 0, scale: 0.95, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 20 }}
              className="w-full max-w-2xl bg-[var(--bg-app)] border border-[var(--border-light)] rounded-[3rem] overflow-hidden shadow-2xl relative"
            >
               <button onClick={() => setSelectedRight(null)} className="absolute top-8 right-8 text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-all z-50">
                  <X size={24} />
               </button>
               
               <div className="p-10 space-y-8">
                  <div className="flex items-center gap-6">
                     <div className="w-16 h-16 rounded-2xl bg-[var(--bg-panel)] flex items-center justify-center text-[var(--accent-teal)]">
                        <BookOpen size={32} />
                     </div>
                     <div>
                        <h3 className="text-2xl font-bold text-[var(--text-primary)] leading-tight">{selectedRight.title}</h3>
                        <p className="text-[9px] font-bold uppercase tracking-[0.4em] text-[var(--accent-teal)] mt-1">{selectedRight.id}</p>
                     </div>
                  </div>

                  <div className="space-y-6">
                     <div className="p-8 bg-[var(--bg-panel)] rounded-2xl border border-[var(--border-light)]">
                        <h4 className="text-[10px] font-bold uppercase tracking-widest text-[var(--text-muted)] mb-3">Statutory Essence</h4>
                        <p className="text-base text-[var(--text-primary)] leading-relaxed">{selectedRight.description}</p>
                     </div>

                     <div className="p-6 bg-[var(--bg-panel)] rounded-2xl border border-[var(--border-light)] flex items-center justify-between">
                        <h4 className="text-[10px] font-bold uppercase tracking-widest text-[var(--text-muted)]">Legal Category</h4>
                        <div className="text-sm font-bold text-[var(--text-primary)] uppercase tracking-widest">{selectedRight.category}</div>
                     </div>
                  </div>

                  <button 
                    onClick={() => { handleAskAI(`I want to learn more about my right: ${selectedRight.title}. What are the landmark judgments or sections related to this?`); setSelectedRight(null); }}
                    className="w-full py-5 bg-[var(--text-primary)] text-[var(--bg-app)] rounded-2xl font-bold uppercase tracking-widest hover:opacity-90 active:scale-95 transition-all shadow-lg"
                  >
                    Discuss with AI Assistant
                  </button>
               </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </div>
  );
}
