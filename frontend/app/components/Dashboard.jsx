'use client';

import React, { useState, useEffect } from 'react';
import { Search, Sparkles, Clock, ArrowRight, Shield, Scale, Gavel, Briefcase, ChevronRight, Zap, TrendingUp, X, MessageSquare } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { usePersona } from '../context/PersonaContext';

const LEGAL_TIPS = [
  {
    title: "Transfer of Property Act, 1882 (Section 106)",
    content: "Absent a written lease, a 15-day termination notice is standard for monthly residential tenancies in India."
  },
  {
    title: "Consumer Protection Act, 2019",
    content: "Consumers can file complaints for 'deficiency in service' or 'unfair trade practices' at the District Commission for claims up to ₹50 Lakhs."
  },
  {
    title: "The Code of Criminal Procedure (Section 154)",
    content: "Police are duty-bound to register an FIR for any cognizable offense. Refusal can lead to action under Section 154(3) or 156(3)."
  },
  {
    title: "Registration Act, 1908 (Section 17)",
    content: "Any document creating an interest in immovable property worth more than ₹100 must be compulsorily registered to be legally valid."
  },
  {
    title: "RTI Act, 2005 (Section 6)",
    content: "You do not need to provide a reason for requesting information, only your contact details for the response."
  },
  {
    title: "Maintenance Act (Section 125 CrPC)",
    content: "Spouses, children, and parents can claim monthly maintenance from individuals with sufficient means who neglect them."
  },
  {
    title: "IT Act, 2000 (Section 66E)",
    content: "Intentionally capturing or publishing private images of a person without consent is a punishable offense with up to 3 years in jail."
  },
  {
    title: "UGC Fee Refund Policy",
    content: "Higher education institutions must provide 100% refund (minus max ₹1000) if a student withdraws within 15 days of the formal admission date."
  },
  {
    title: "Motor Vehicles Act (No-Fault Liability)",
    content: "In cases of death or permanent disablement in road accidents, compensation can be claimed even without proving the driver's fault."
  },
  {
    title: "Indian Evidence Act (Section 114A)",
    content: "In certain sexual offense cases, the court presumes absence of consent if the victim states so in her evidence."
  }
];

export default function Dashboard({ setActiveTab, handleAskAI }) {
  const { activePersona } = usePersona();
  const [searchQuery, setSearchQuery] = useState('');
  const [history, setHistory] = useState([]);
  const [userName, setUserName] = useState('Ruthvik');
  const [currentTip, setCurrentTip] = useState(LEGAL_TIPS[0]);
  const [isInsightOpen, setIsInsightOpen] = useState(false);

  useEffect(() => {
    try {
      const savedHistory = JSON.parse(localStorage.getItem('lawmate_chat_history') || '[]');
      setHistory(savedHistory.slice(0, 5));
      const profile = JSON.parse(localStorage.getItem('lawmate_user_profile') || '{}');
      if (profile.name) setUserName(profile.name.split(' ')[0]);
      
      const lastIndex = parseInt(localStorage.getItem('lawmate_tip_index') || '-1');
      const nextIndex = (lastIndex + 1) % LEGAL_TIPS.length;
      setCurrentTip(LEGAL_TIPS[nextIndex]);
      localStorage.setItem('lawmate_tip_index', nextIndex.toString());
    } catch(e) {}
  }, []);

  const personaActions = {
    citizen: [
      { id: 'police-complaint', title: 'Police Complaint Guide', desc: 'Step-by-step for filing FIR/GD', icon: Gavel },
      { id: 'consumer-rights', title: 'Consumer Protection', desc: 'Defective products or service delay', icon: Shield },
      { id: 'property-check', title: 'Property Verification', desc: 'Essential docs for buying land/flat', icon: Scale },
    ],
    student: [
      { id: 'fee-dispute', title: 'Fee Refund Policy', desc: 'Legal grounds for college refunds', icon: Scale },
      { id: 'ragging-complaint', title: 'Anti-Ragging Laws', desc: 'UGC guidelines and police reporting', icon: Shield },
      { id: 'intern-rights', title: 'Internship Rights', desc: 'Stipend and work-hour regulations', icon: Briefcase },
    ],
    tenant: [
      { id: 'deposit-recovery', title: 'Deposit Recovery', desc: 'Reclaiming security from landlord', icon: Shield },
      { id: 'eviction-defense', title: 'Eviction Defense', desc: 'Stay order and illegal lockout protection', icon: Scale },
      { id: 'agreement-review', title: 'Rent Agreement Audit', desc: 'Check for hidden "lock-in" clauses', icon: Gavel },
    ],
    employee: [
      { id: 'unpaid-salary', title: 'Salary Recovery', desc: 'Demand notice for unpaid wages', icon: Briefcase },
      { id: 'posh-complaint', title: 'Workplace Harassment', desc: 'ICC filing and POSH regulations', icon: Shield },
      { id: 'termination-pay', title: 'Severance Logic', desc: 'Calculate legal notice pay period', icon: Scale },
    ],
    senior: [
      { id: 'pension-claim', title: 'Pension Disbursement', desc: 'Resolving bank/govt delays', icon: Shield },
      { id: 'maintenance-act', title: 'Maintenance Claims', desc: 'Senior Citizen Welfare Act guide', icon: Scale },
      { id: 'will-drafting', title: 'Succession Planning', desc: 'Legal requirements for a valid Will', icon: Gavel },
    ],
    business: [
      { id: 'trademark-filing', title: 'IP Protection', desc: 'Trademark registration roadmap', icon: Shield },
      { id: 'gst-compliance', title: 'GST Dispute Logic', desc: 'Handling notice from tax department', icon: Scale },
      { id: 'contract-draft', title: 'Vendor Agreements', desc: 'Essential clauses to avoid litigation', icon: Briefcase },
    ]
  };

  const currentActions = personaActions[activePersona.id] || personaActions.citizen;

  return (
    <div className="flex-1 overflow-y-auto p-6 md:p-8 bg-[var(--bg-app)] hide-scrollbar">
      <div className="max-w-5xl mx-auto space-y-8">
        
        {/* Header Section */}
        <header className="flex flex-col md:flex-row md:items-end justify-between gap-8 pt-8">
          <div>
            <div className="inline-flex items-center gap-2 px-3 py-1 bg-[var(--bg-hover)] border border-[var(--border-light)] rounded-full text-[10px] font-bold uppercase tracking-widest text-[var(--accent-teal)] mb-4">
              <Sparkles size={12} /> Intelligence Center Active
            </div>
            <h1 className="text-2xl font-extrabold tracking-tight text-[var(--text-primary)] mb-1 leading-tight">
              Greetings, <span className="text-[var(--text-muted)]">{userName}</span>.
            </h1>
            <p className="text-[var(--text-muted)] text-[14px] max-w-xl leading-relaxed">
              Your AI-augmented legal workstation is ready. Analyze cases, draft documents, and verify jurisdictions in real-time.
            </p>
          </div>
          <div className="flex items-center gap-4 bg-[var(--bg-panel)] p-6 rounded-[2rem] border border-[var(--border-light)] shadow-sm">
             <div className="w-12 h-12 rounded-2xl bg-[var(--bg-app)] border border-[var(--border-light)] flex items-center justify-center text-[var(--accent-teal)]">
                <TrendingUp size={24} />
             </div>
             <div>
                <p className="text-[10px] font-bold uppercase tracking-widest text-[var(--text-muted)]">Platform Load</p>
                <p className="text-lg font-extrabold text-[var(--text-primary)]">Optimized</p>
             </div>
          </div>
        </header>

        {/* Hero Search Area */}
        <section className="relative group">
          <div className="absolute -inset-1 bg-gradient-to-r from-[var(--accent-teal)] to-blue-500 rounded-[2.5rem] opacity-20 blur-xl group-focus-within:opacity-40 transition duration-1000"></div>
          <div className="relative glass-panel rounded-[2.5rem] border-[var(--border-light)] p-2 shadow-2xl flex items-center">
            <div className="pl-8 text-[var(--text-muted)]">
              <Search size={24} />
            </div>
            <input 
              type="text" 
              placeholder="Describe your legal situation (e.g. 'My landlord refuses to return my deposit')" 
              className="flex-1 bg-transparent border-none py-4 px-5 text-base text-[var(--text-primary)] outline-none placeholder:text-[var(--text-muted)]/50 font-medium"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleAskAI(searchQuery)}
            />
            <button 
              onClick={() => handleAskAI(searchQuery)}
              className="mr-2 bg-[var(--text-primary)] text-[var(--bg-app)] px-6 py-3 rounded-[1.5rem] font-black text-xs uppercase tracking-widest hover:opacity-90 hover:scale-[1.02] active:scale-95 transition-all shadow-lg flex items-center gap-3"
            >
              Analyze Case <Zap size={16} fill="currentColor" />
            </button>
          </div>
        </section>

        {/* Dashboard Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-10">
          
          <div className="lg:col-span-2 space-y-10">
            {/* Daily Legal Tip (DYNAMIC) */}
            <div 
              className="glass-panel p-10 rounded-[2.5rem] border-2 border-dashed border-[var(--border-light)] flex items-center gap-10 hover:bg-[var(--bg-hover)] transition-all cursor-pointer group mb-10"
              onClick={() => setIsInsightOpen(true)}
            >
               <div className="w-20 h-20 shrink-0 rounded-[1.5rem] bg-[var(--text-primary)] flex items-center justify-center text-[var(--bg-app)] shadow-2xl group-hover:scale-105 transition-transform">
                  <Shield size={32} />
               </div>
               <div>
                  <h4 className="text-[10px] items-center gap-2 font-bold uppercase tracking-[0.2em] text-[var(--text-muted)] mb-3 flex">
                    <Clock size={12} /> Daily Jurisdictional Insight
                  </h4>
                  <p className="text-lg font-bold text-[var(--text-primary)] leading-tight mb-2">
                    {currentTip.title}
                  </p>
                  <p className="text-sm text-[var(--text-muted)] leading-relaxed line-clamp-1">
                    {currentTip.content}
                  </p>
               </div>
            </div>

            <div>
              <div className="flex items-center justify-between mb-8">
                <h3 className="text-[11px] font-bold uppercase tracking-[0.3em] text-[var(--text-muted)] flex items-center gap-3">
                  <Scale size={14} /> Persona-Tailored Discovery
                </h3>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {currentActions.map((action) => (
                  <button 
                    key={action.id}
                    onClick={() => handleAskAI(action.title)}
                    className="glass-panel text-left p-8 rounded-[2rem] border border-[var(--border-light)] hover:border-[var(--text-muted)] hover:bg-[var(--bg-hover)] transition-all group relative overflow-hidden shadow-sm"
                  >
                    <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
                       <action.icon size={80} />
                    </div>
                    <div className="w-12 h-12 rounded-2xl bg-[var(--bg-app)] border border-[var(--border-light)] flex items-center justify-center text-[var(--text-primary)] mb-6 group-hover:scale-110 transition-transform shadow-inner">
                      <action.icon size={20} />
                    </div>
                    <h4 className="text-lg font-bold text-[var(--text-primary)] mb-2">{action.title}</h4>
                    <p className="text-sm text-[var(--text-muted)] leading-relaxed mb-6">{action.desc}</p>
                    <div className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-widest text-[var(--text-primary)] opacity-0 group-hover:opacity-100 transition-all translate-x-[-10px] group-hover:translate-x-0">
                      Start workflow <ArrowRight size={12} />
                    </div>
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Right Column: Archive Log */}
          <div className="space-y-10">
            <div className="glass-panel p-8 rounded-[2.5rem] border border-[var(--border-light)] h-full shadow-lg bg-gradient-to-b from-[var(--bg-panel)] to-[var(--bg-sidebar)]">
              <div className="flex items-center justify-between mb-8">
                <h3 className="text-[11px] font-bold uppercase tracking-[0.3em] text-[var(--text-muted)]">Archive Log</h3>
                <button 
                  onClick={() => setActiveTab('profile')}
                  className="text-[10px] font-bold uppercase tracking-widest text-[var(--text-primary)] hover:underline"
                >
                  View All
                </button>
              </div>
              
              <div className="space-y-6">
                {history.length > 0 ? history.map((item) => (
                  <div 
                    key={item.id} 
                    onClick={() => handleAskAI(item.title, item.id)}
                    className="group cursor-pointer"
                  >
                    <div className="flex items-start gap-4 mb-2">
                       <div className="w-2 h-2 rounded-full mt-2 bg-[var(--accent-teal)] group-hover:scale-150 transition-transform"></div>
                       <div className="flex-1">
                          <p className="text-sm font-bold text-[var(--text-primary)] group-hover:text-[var(--accent-teal)] transition-colors line-clamp-1">{item.title}</p>
                          <p className="text-[10px] text-[var(--text-muted)] mt-1">{new Date(item.updatedAt || Date.now()).toLocaleDateString()}</p>
                       </div>
                    </div>
                    <div className="h-px bg-gradient-to-r from-[var(--border-light)] to-transparent ml-6"></div>
                  </div>
                )) : (
                  <div className="py-20 text-center opacity-30">
                    <Clock size={40} className="mx-auto mb-4" />
                    <p className="text-xs font-bold uppercase tracking-widest">No recent sessions</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Insight Modal (V49) */}
      <AnimatePresence>
        {isInsightOpen && (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-6">
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setIsInsightOpen(false)}
              className="absolute inset-0 bg-black/80 backdrop-blur-sm"
            />
            <motion.div 
              initial={{ opacity: 0, scale: 0.9, y: 20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.9, y: 20 }}
              className="relative w-full max-w-lg bg-[var(--bg-panel)] border border-[var(--border-light)] rounded-[3rem] p-10 shadow-2xl"
            >
              <button 
                onClick={() => setIsInsightOpen(false)}
                className="absolute top-8 right-8 text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors"
              >
                <X size={24} />
              </button>
              
              <div className="w-16 h-16 rounded-2xl bg-[var(--bg-app)] border border-[var(--border-light)] flex items-center justify-center text-[var(--accent-teal)] mb-8">
                <Shield size={32} />
              </div>
              
              <h4 className="text-[10px] font-bold uppercase tracking-[0.2em] text-[var(--text-muted)] mb-4">Jurisdictional Insight</h4>
              <h2 className="text-2xl font-black text-[var(--text-primary)] mb-6 leading-tight">{currentTip.title}</h2>
              <p className="text-lg text-[var(--text-muted)] leading-relaxed mb-10">
                {currentTip.content}
              </p>
              
              <div className="flex flex-col gap-3">
                <button 
                  onClick={() => {
                    setIsInsightOpen(false);
                    handleAskAI(`Help me understand: ${currentTip.title}. Specifically, why is this important for a ${activePersona.label || 'citizen'}?`);
                  }}
                  className="w-full bg-[var(--text-primary)] text-[var(--bg-app)] py-5 rounded-2xl font-black text-xs uppercase tracking-widest hover:opacity-90 shadow-xl flex items-center justify-center gap-3 transition-all"
                >
                  Analyze with LawMate AI <MessageSquare size={16} />
                </button>
                <button 
                   onClick={() => setIsInsightOpen(false)}
                   className="w-full py-4 text-[10px] font-bold uppercase tracking-widest text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors"
                >
                   Stay on Dashboard
                </button>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </div>
  );
}
