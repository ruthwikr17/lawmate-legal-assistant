'use client';

import React, { useState, useEffect, useMemo, useRef } from 'react';
import { usePersona } from '../context/PersonaContext';
import { 
  Briefcase, 
  ChevronRight, 
  Search, 
  Filter, 
  FileText, 
  ArrowRight, 
  ArrowLeft,
  Loader2,
  CheckCircle2,
  Clock,
  Plus,
  X,
  Sparkles,
  MessageSquare,
  Send,
  ShieldCheck,
  Scale
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import ReactMarkdown from 'react-markdown';

export default function LegalWorkflows() {
  const { activePersona } = usePersona();
  const [workflows, setWorkflows] = useState([]);
  const [activeWorkflow, setActiveWorkflow] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isArchitecting, setIsArchitecting] = useState(false);
  const [architectError, setArchitectError] = useState(null);
  const [generalSearch, setGeneralSearch] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('All');
  const [showCustomModal, setShowCustomModal] = useState(false);
  const [customRequest, setCustomRequest] = useState('');
  
  // Sidebar (Widget) State
  const [widgetInput, setWidgetInput] = useState('');
  const [widgetMessages, setWidgetMessages] = useState([
    { role: 'assistant', content: "Vault Link Active. Ask anything about your current manifest, or request specific document drafts." }
  ]);
  const [isWidgetLoading, setIsWidgetLoading] = useState(false);
  const widgetEndRef = useRef(null);

  useEffect(() => {
    fetch('/data/workflows.json')
      .then(res => res.json())
      .then(data => {
        setWorkflows(data);
        setIsLoading(false);
      })
      .catch(err => {
        console.error("Failed to load workflows:", err);
        setIsLoading(false);
      });
  }, []);

  useEffect(() => {
    if (widgetEndRef.current) {
      widgetEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [widgetMessages]);

  const personaWorkflows = useMemo(() => {
    return workflows.filter(w => w.persona === activePersona.id);
  }, [workflows, activePersona.id]);

  const filteredWorkflows = useMemo(() => {
    return workflows.filter(w => {
      const matchesSearch = w.title.toLowerCase().includes(generalSearch.toLowerCase()) || 
                           w.desc.toLowerCase().includes(generalSearch.toLowerCase());
      const matchesCategory = selectedCategory === 'All' || w.category === selectedCategory;
      return matchesSearch && matchesCategory;
    });
  }, [workflows, generalSearch, selectedCategory]);

  const groupedWorkflows = useMemo(() => {
    const groups = {};
    filteredWorkflows.forEach(w => {
      if (!groups[w.category]) groups[w.category] = [];
      groups[w.category].push(w);
    });
    return groups;
  }, [filteredWorkflows]);

  const personaCategories = useMemo(() => {
    const cats = new Set(['All']);
    workflows.forEach(w => cats.add(w.category));
    return Array.from(cats);
  }, [workflows]);

  // Unified Failover Engine (V25)
  const unifiedFetch = async (endpoint, options) => {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';
    const fallbackUrl = process.env.NEXT_PUBLIC_AI_API_URL || 'http://localhost:8000';
    const urls = [
      `${apiUrl}/api/legal${endpoint}`, 
      `${fallbackUrl}${endpoint}`              
    ];
    let lastError = null;
    for (const url of urls) {
      try {
        const res = await fetch(url, options);
        if (res.ok) return res;
        console.warn(`Bridge ${url} returned ${res.status}. Trying fallback...`);
      } catch (e) {
        lastError = e;
        console.warn(`Link ${url} unreachable. Trying fallback...`);
      }
    }
    throw lastError || new Error("Statutory connectivity lost.");
  };
  
  const startWorkflow = async (workflow) => {
    // [V29] HYBRID LOAD CORE: Prioritize baked manifests for instant, high-integrity logic
    if (workflow.steps && workflow.steps.length > 0) {
      console.log(`[V29] Loading Baked Manifest for: ${workflow.title}`);
      setActiveWorkflow(workflow);
      setWidgetMessages([{ 
        role: 'assistant', 
        content: `Vault Link Sync: ${workflow.title} (Registry Certified). Manifest loaded instantly with clinical statutory integrity.` 
      }]);
      return;
    }

    setIsArchitecting(true);
    setArchitectError(null);
    setActiveWorkflow({ ...workflow, steps: [] });
    setWidgetMessages([{ role: 'assistant', content: `Vault Link: ${workflow.title}. Architecting procedural manifest...` }]);
    
    const settings = JSON.parse(localStorage.getItem('lawmate_settings') || '{}');
    const architectPrompt = `ARCHITECT DOSSIER: For the following legal situation, provide a 3-phase procedural guide with STATUTORY PRECISION. 
    Workflow: ${workflow.title}
    Description: ${workflow.desc}
    Persona: ${activePersona.label}
    
    CRITICAL INSTRUCTIONS:
    1. USE SPECIFIC INDIAN ACTS AND SECTIONS (e.g., Section 154 CrPC, Consumer Protection Act 2019).
    2. USE NATURAL, PLAIN LANGUAGE to explain each step. Ensure the 'logic' field contains meaningful sentences that help the user understand the process.
    3. PROVIDE RIGOROUS ACTIONABLE STEPS. NO NURSERY ADVICE. 
    4. USE PROFESSIONAL BUT ACCESSIBLE LEGAL TERMINOLOGY.
    
    Format ONLY as JSON:
    {
      "steps": [
        {
          "title": "Clear, Meaningful Step Title",
          "logic": "Helpful, detailed natural language explanation of the legal rationale and process.",
          "objective": "CORE PROCEDURAL GOAL",
          "checklist": ["Actionable Item 1", "Actionable Item 2"],
          "docs": ["Specific Document Name"]
        }
      ]
    }`;

    try {
      const res = await unifiedFetch('/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: architectPrompt,
          history: [],
          context: {
             persona: activePersona?.id || 'citizen',
             style: settings.aiStyle || 'Detailed',
             depth: settings.aiDepth || 'Medium'
          }
        })
      });

      if (!res.ok) throw new Error(`Vault Link Error: ${res.status}`);

      const data = await res.json();
      let generated = null;
      try {
        const textToParse = data.answer || "";
        const jsonMatch = textToParse.match(/\{[\s\S]*\}/);
        if (jsonMatch) {
            const cleanJson = jsonMatch[0].replace(/```json|```/g, ''); 
            generated = JSON.parse(cleanJson);
        }

        // Fallback: Ultra-Tenacious Step Parser (V27)
        if (!generated || !generated.steps || generated.steps.length === 0) {
           const rawText = (generated?.answer || data.answer || "").replace(/```json|```/g, '').trim();
           const steps = [];
           const stepRegex = /(?:\d+\.|\bStep\s*\d+:?|\bPhase\s*\d+:?)\s*([^\n:]+)(?::|\.|\n)\s*([^]+?)(?=(?:\d+\.|\bStep\s*\d+:?|\bPhase\s*\d+:?)|$)/gi;
           let match;
           while ((match = stepRegex.exec(rawText)) !== null) {
              steps.push({
                 title: match[1].trim().toUpperCase(),
                 logic: match[2].trim(),
                 objective: "STATUTORY PROCEDURE",
                 checklist: ["Review Guidelines"],
                 docs: ["Procedural Manual"]
              });
           }
           
           // Ultimate Fallback: Single Step (V27)
           if (steps.length === 0 && rawText.length > 10) {
              steps.push({
                 title: "PROCEDURAL MANIFEST",
                 logic: rawText,
                 objective: "STATUTORY GOAL",
                 checklist: ["Follow Procedural Steps"],
                 docs: ["Unified Dossier"]
              });
           }
           
           if (steps.length > 0) {
              generated = { ...generated, steps };
           }
        }
      } catch (pe) { 
        console.error("Parse Error:", pe);
      }

      if (generated && generated.steps && generated.steps.length > 0) {
          setActiveWorkflow({ ...workflow, steps: generated.steps });
          setWidgetMessages([{ role: 'assistant', content: `### ⚖️ Strategy Manifested: ${workflow.title}\n\nI have successfully architected a unique procedural path based on your Profile and Jurisdiction.` }]);
          const count = parseInt(localStorage.getItem('lawmate_workflow_count') || '0');
          localStorage.setItem('lawmate_workflow_count', (count + 1).toString());
          window.dispatchEvent(new Event('storage'));
        } else {
          throw new Error("Vault Link Sync Error: Manifest inconsistent. Re-attempting sync recommended.");
      }
    } catch (err) {
      setArchitectError(err.message || "Vault analysis interrupted.");
    } finally {
      setIsArchitecting(false);
    }
  };

  const submitCustomRequest = async () => {
    if (!customRequest.trim()) return;
    setShowCustomModal(false);
    setIsArchitecting(true);
    const mockWorkflow = { id: 'custom', title: 'Custom Statutory Strategy', desc: customRequest, steps: [] };
    setActiveWorkflow(mockWorkflow);
    
    const settings = JSON.parse(localStorage.getItem('lawmate_settings') || '{}');

    try {
      const res = await unifiedFetch('/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: `ARCHITECT DOSSIER: For the following situation, provide a 3-phase procedural guide. \nSituation: ${customRequest}. \nInclude specific Indian Acts/Sections. \nFormat ONLY as JSON: { "steps": [{ "title": "...", "logic": "...", "objective": "...", "checklist": ["..."], "docs": ["..."] }] }`,
          history: [],
          context: {
             persona: activePersona?.id || 'citizen',
             style: settings.aiStyle || 'Detailed',
             depth: settings.aiDepth || 'Medium'
          }
        })
      });
      const data = await res.json();
      let generated = null;
      try {
        const textToParse = data.answer || "";
        const jsonMatch = textToParse.match(/\{[\s\S]*\}/);
        if (jsonMatch) {
            const cleanJson = jsonMatch[0].replace(/```json|```/g, '').trim(); 
            generated = JSON.parse(cleanJson);
        }

        if (!generated || !generated.steps || generated.steps.length === 0) {
           const rawText = (generated?.answer || data.answer || "").replace(/```json|```/g, '').trim();
           const steps = [];
           const stepRegex = /(?:\d+\.|\bStep\s*\d+:?|\bPhase\s*\d+:?)\s*([^\n:]+)(?::|\.|\n)\s*([^]+?)(?=(?:\d+\.|\bStep\s*\d+:?|\bPhase\s*\d+:?)|$)/gi;
           let match;
           while ((match = stepRegex.exec(rawText)) !== null) {
              steps.push({
                 title: match[1].trim().toUpperCase(),
                 logic: match[2].trim(),
                 objective: "STATUTORY PROCEDURE",
                 checklist: ["Verify Documentation"],
                 docs: ["Custom Manifest"]
              });
           }

           if (steps.length === 0 && rawText.length > 10) {
              steps.push({
                 title: "CUSTOM STATUTORY STRATEGY",
                 logic: rawText,
                 objective: "PROCEDURAL GOAL",
                 checklist: ["Analyze Manifest"],
                 docs: ["Detailed Dossier"]
              });
           }

           if (steps.length > 0) {
              generated = { ...generated, steps };
           }
        }
      } catch (pe) { 
        console.error("Parse Error:", pe);
      }

      if (generated && generated.steps && generated.steps.length > 0) {
        setActiveWorkflow({ ...mockWorkflow, steps: generated.steps });
        setWidgetMessages([{ role: 'assistant', content: `### ⚖️ Custom Manifest Generated\n\nI have architected a procedural strategy for: *${customRequest}*` }]);
      } else {
        throw new Error("Custom Manifest Error: Logic mapping failed. Please provide more detail.");
      }
    } catch (e) {
      setArchitectError(e.message || "Custom analysis failed.");
    } finally {
      setIsArchitecting(false);
    }
  };

  const handleWidgetSend = async (forcedText = null) => {
    const text = forcedText || widgetInput;
    if (!text.trim()) return;
    if (!forcedText) setWidgetInput('');
    
    setWidgetMessages(prev => [...prev, { role: 'user', content: text }]);
    setIsWidgetLoading(true);

    const settings = JSON.parse(localStorage.getItem('lawmate_settings') || '{}');

    try {
      const response = await unifiedFetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: text,
          history: widgetMessages.map(m => ({ role: m.role, content: m.content })),
          context: {
             persona: activePersona.id,
             style: settings.aiStyle || 'Detailed',
             depth: settings.aiDepth || 'Medium'
          }
        })
      });
      const data = await response.json();
      setWidgetMessages(prev => [...prev, { role: 'assistant', content: data.answer }]);
    } catch (e) { 
      setWidgetMessages(prev => [...prev, { role: 'assistant', content: "Service unavailable." }]); 
    } finally { 
      setIsWidgetLoading(false); 
    }
  };

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text);
  };

  const aggregatedDocs = useMemo(() => {
    if (!activeWorkflow || !activeWorkflow.steps) return [];
    const docsMap = new Map();
    activeWorkflow.steps.forEach((step, idx) => {
      if (step.docs) {
        step.docs.forEach(docName => {
          if (!docsMap.has(docName)) {
            docsMap.set(docName, { name: docName, phases: [`Phase ${idx + 1}`] });
          } else {
            docsMap.get(docName).phases.push(`Phase ${idx + 1}`);
          }
        });
      }
    });
    return Array.from(docsMap.values());
  }, [activeWorkflow]);

  if (isLoading) return <div className="flex-1 flex items-center justify-center bg-[var(--bg-app)]"><Loader2 size={24} className="text-[var(--accent-teal)] animate-spin opacity-40" /></div>;

  return (
    <div className="flex-1 flex overflow-hidden bg-[var(--bg-app)] text-[var(--text-primary)]">
      {!activeWorkflow ? (
        <div className="flex-1 overflow-y-auto p-10 hide-scrollbar">
          <header className="mb-12 max-w-6xl mx-auto flex items-center justify-between border-b border-[var(--border-light)] pb-8">
            <div>
               <h1 className="text-3xl font-black tracking-tight mb-1 text-[var(--text-primary)]">Legal Workflows</h1>
               <p className="text-[var(--text-muted)] text-[11px] font-black uppercase tracking-widest">{workflows.length}+ Professional Manifests</p>
            </div>
            <button onClick={() => setShowCustomModal(true)} className="px-8 py-3 bg-[var(--text-primary)] text-[var(--bg-app)] rounded-full flex items-center gap-3 hover:opacity-90 transition-all shadow-xl hover:scale-105 active:scale-95">
              <Plus size={18} />
              <span className="text-[10px] font-black uppercase tracking-widest">Custom Strategy</span>
            </button>
          </header>

          <div className="max-w-6xl mx-auto mb-16">
            <h3 className="text-[9px] font-black uppercase tracking-[0.4em] text-[var(--accent-teal)] mb-6 opacity-40 text-[var(--text-primary)]">RECOMMENDED FOR {activePersona.label.toUpperCase()}S</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
              {personaWorkflows.map(w => (
                <div key={w.id} onClick={() => startWorkflow(w)} className="bg-white/[0.03] backdrop-blur-3xl border border-white/10 p-6 rounded-3xl cursor-pointer hover:bg-white/[0.08] hover:border-white/20 transition-all group flex flex-col justify-between h-40 shadow-2xl shadow-black/20">
                  <div>
                    <h4 className="text-lg font-black mb-1.5 leading-tight tracking-tight text-white group-hover:text-[var(--accent-teal)] transition-colors">{w.title}</h4>
                    <p className="text-[12px] text-zinc-400 font-medium leading-relaxed line-clamp-2">{w.desc}</p>
                  </div>
                  <div className="flex items-center text-[10px] font-black uppercase tracking-[0.2em] text-zinc-600 group-hover:text-[var(--accent-teal)] transition-all pt-4 border-t border-white/5">
                    Launch Manifest <ArrowRight size={14} className="ml-auto group-hover:translate-x-1 transition-transform" />
                  </div>
                </div>
              ))}
            </div>
          </div>
          
          <div className="max-w-6xl mx-auto pb-40">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-6 mb-12 border-b border-[var(--border-light)] pb-6">
               <div className="flex-1">
                  <h3 className="text-[9px] font-black text-[var(--text-muted)] uppercase tracking-[0.4em] mb-3">GLOBAL PROCEDURAL VAULT</h3>
                  <div className="flex flex-wrap gap-2">
                    {personaCategories.map(cat => (
                      <button key={cat} onClick={() => setSelectedCategory(cat)} className={`px-4 py-1.5 rounded-full text-[8px] font-black uppercase tracking-widest transition-all ${selectedCategory === cat ? 'bg-[var(--text-primary)] text-[var(--bg-app)]' : 'bg-[var(--bg-panel)] text-[var(--text-muted)] border border-[var(--border-light)]'}`}>{cat}</button>
                    ))}
                  </div>
               </div>
               <div className="relative group min-w-[280px]">
                 <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-[var(--text-muted)] opacity-40" size={14} />
                 <input type="text" placeholder="Search Vault..." value={generalSearch} onChange={e => setGeneralSearch(e.target.value)} className="w-full bg-[var(--bg-panel)] border border-[var(--border-light)] rounded-full py-3 pl-12 pr-6 text-[11px] text-[var(--text-primary)] outline-none focus:border-[var(--accent-teal)] transition-all font-bold placeholder:text-[var(--text-muted)]" />
               </div>
            </div>

             <div className="space-y-20">
               {Object.entries(groupedWorkflows).sort().map(([category, items]) => (
                <section key={category} className="space-y-8">
                   <div className="flex items-center gap-6">
                      <h2 className="text-[11px] font-black uppercase tracking-[0.6em] text-[var(--accent-teal)]">{category}</h2>
                      <div className="h-px flex-1 bg-white/5" />
                      <span className="text-[9px] font-black text-zinc-600 uppercase tracking-widest">{items.length} PLAYBOOKS</span>
                   </div>
                   <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5">
                      {items.map(w => (
                        <div key={w.id} onClick={() => startWorkflow(w)} className="p-6 rounded-3xl bg-white/[0.02] border border-white/5 hover:bg-white/[0.06] hover:border-white/15 transition-all cursor-pointer group flex flex-col justify-between h-36 hover:-translate-y-1">
                           <div>
                              <h4 className="text-[15px] font-bold text-zinc-400 group-hover:text-white mb-2 tracking-tight line-clamp-1 transition-colors">{w.title}</h4>
                              <p className="text-[11px] text-zinc-600 group-hover:text-zinc-400 line-clamp-2 font-medium leading-relaxed transition-colors">{w.desc}</p>
                           </div>
                           <div className="flex items-center justify-between pt-3 border-t border-white/5">
                              <span className="text-[9px] font-black text-zinc-700 group-hover:text-[var(--accent-teal)] uppercase tracking-widest transition-colors">Manifest</span>
                              <ArrowRight size={10} className="text-zinc-800 group-hover:text-[var(--accent-teal)] transition-colors" />
                           </div>
                        </div>
                      ))}
                   </div>
                </section>
              ))}
            </div>
          </div>
        </div>
      ) : (
        <div className="flex-1 flex flex-col h-full bg-[var(--bg-app)]">
           <div className="h-28 shrink-0 border-b border-[var(--border-light)] px-8 flex items-center justify-between bg-[var(--bg-panel)]">
            <div className="flex items-center gap-6">
              <button onClick={() => setActiveWorkflow(null)} className="w-10 h-10 rounded-full border border-[var(--border-light)] hover:bg-[var(--bg-hover)] text-[var(--text-muted)] flex items-center justify-center transition-all bg-[var(--bg-app)]"><ArrowLeft size={16} /></button>
              <div>
                <span className="text-[8px] font-black uppercase tracking-[0.4em] text-[var(--accent-teal)] block mb-0.5 opacity-60 underline decoration-dotted">COMPLETE PROCEDURAL MANIFEST</span>
                <h2 className="text-xl font-black tracking-tight text-[var(--text-primary)] uppercase mb-1">{activeWorkflow.title}</h2>
                <p className="text-[14px] text-[var(--text-muted)] font-bold italic line-clamp-1">{activeWorkflow.desc}</p>
              </div>
            </div>
            <button onClick={() => setActiveWorkflow(null)} className="px-6 py-2.5 bg-[var(--text-primary)] text-[var(--bg-app)] text-[9px] font-black uppercase rounded-full hover:opacity-90 active:scale-95 transition-all shadow-xl">EXIT VAULT</button>
          </div>

          <div className="flex-1 flex overflow-hidden">
            <div className="flex-1 overflow-y-auto p-10 custom-scrollbar">
              <div className="max-w-2xl mx-auto space-y-12">
                <AnimatePresence mode="wait">
                  {isArchitecting ? (
                    <div className="flex flex-col items-center justify-center py-40 text-center">
                       <Loader2 size={32} className="text-[var(--accent-teal)] animate-spin mb-6 opacity-20" />
                       <h3 className="text-xl font-black text-[var(--text-primary)] uppercase tracking-tight mb-2">Architecting Manifest</h3>
                       <p className="text-[var(--text-muted)] text-[9px] font-black uppercase tracking-[0.4em] animate-pulse">Syncing Vault...</p>
                    </div>
                  ) : architectError ? (
                    <div className="flex flex-col items-center justify-center py-40 text-center space-y-6">
                       <div className="w-16 h-16 rounded-full bg-red-500/10 flex items-center justify-center text-red-500 mb-2">
                          <X size={32} />
                       </div>
                       <div>
                          <h3 className="text-xl font-black text-[var(--text-primary)] uppercase tracking-tight mb-2">Architecting Interrupted</h3>
                          <p className="text-[var(--text-muted)] text-[11px] font-bold uppercase tracking-widest max-w-xs mx-auto mb-8 opacity-60">{architectError}</p>
                       </div>
                       <button onClick={() => startWorkflow(activeWorkflow)} className="px-8 py-3 bg-[var(--text-primary)] text-[var(--bg-app)] text-[10px] font-black uppercase tracking-widest rounded-full hover:opacity-90 active:scale-95 transition-all shadow-xl">Re-attempt Sync</button>
                    </div>
                  ) : (
                    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-16 pb-40">
                       <section className="space-y-8">
                          <h3 className="text-[9px] font-black text-[var(--text-muted)] uppercase tracking-[0.5em] flex items-center gap-4 opacity-30">
                             <div className="h-px flex-1 bg-[var(--border-light)]" /> 01. STEPS <div className="h-px flex-1 bg-[var(--border-light)]" />
                          </h3>
                          <div className="space-y-8">
                             {activeWorkflow.steps?.map((step, idx) => (
                               <div key={idx} className="flex gap-6 items-start group">
                                  <div className="text-[14px] font-black text-[var(--accent-teal)] opacity-40 shrink-0 mt-0.5">{idx + 1}.</div>
                                  <div className="flex-1 space-y-3">
                                     <div>
                                        <h4 className="text-[18px] font-black text-[var(--text-primary)] tracking-tight mb-1">
                                           {step.title} <span className="text-[var(--text-muted)] opacity-20 mx-2">—</span> <span className="text-[var(--text-muted)] opacity-60 font-medium lowercase italic text-sm">{step.objective}</span>
                                        </h4>
                                        <p className="text-[14px] text-[var(--text-muted)] font-medium leading-relaxed max-w-xl">{step.logic}</p>
                                     </div>
                                     
                                     {step.checklist && step.checklist.length > 0 && (
                                       <div className="flex flex-wrap gap-2 pt-1">
                                          {step.checklist.map((item, cidx) => (
                                            <div key={cidx} className="flex items-center gap-2 px-3 py-1 rounded-full bg-[var(--bg-panel)] border border-[var(--border-light)] text-[10px] font-bold text-[var(--text-muted)]">
                                               <div className="w-1 h-1 rounded-full bg-[var(--accent-teal)] opacity-40" />
                                               {item}
                                            </div>
                                          ))}
                                       </div>
                                     )}
                                  </div>
                               </div>
                             ))}
                          </div>
                       </section>

                       <section className="space-y-8">
                          <h3 className="text-[9px] font-black text-[var(--text-muted)] uppercase tracking-[0.5em] flex items-center gap-4 opacity-30">
                             <div className="h-px flex-1 bg-[var(--border-light)]" /> 02. DOCUMENTS <div className="h-px flex-1 bg-[var(--border-light)]" />
                          </h3>
                          <div className="space-y-3">
                             {aggregatedDocs.length > 0 ? aggregatedDocs.map((doc, idx) => (
                               <div key={idx} className="flex items-center justify-between p-4 rounded-2xl bg-[var(--bg-panel)] border border-[var(--border-light)] group hover:bg-[var(--bg-hover)] transition-all">
                                  <div className="flex items-center gap-4">
                                     <FileText size={16} className="text-[var(--text-muted)] opacity-20 group-hover:text-[var(--accent-teal)] transition-colors" />
                                     <div>
                                        <h4 className="text-sm font-black text-[var(--text-primary)] uppercase tracking-tight inline-block mr-3">{doc.name}</h4>
                                        <span className="text-[9px] font-black text-[var(--text-muted)] opacity-40 uppercase tracking-widest italic">
                                           ({doc.phases.map(p => p.replace('Phase', 'Step')).join(', ')})
                                        </span>
                                     </div>
                                  </div>
                                  <button onClick={() => { handleWidgetSend(`DRAFT DOCUMENT: ${doc.name} \nSTATUTORY CONTEXT: ${activeWorkflow.title}`); }} className="px-5 py-2 hover:bg-[var(--text-primary)] hover:text-[var(--bg-app)] border border-[var(--border-light)] text-[var(--text-muted)] font-black uppercase text-[9px] tracking-widest rounded-full transition-all">DRAFT NOW</button>
                               </div>
                             )) : (
                               <div className="py-12 text-center border border-dashed border-[var(--border-light)] rounded-2xl">
                                  <p className="text-[var(--text-muted)] opacity-20 font-black uppercase tracking-widest text-[9px]">No documents required.</p>
                               </div>
                             )}
                          </div>
                       </section>

                       <div className="pt-12 border-t border-[var(--border-light)] text-center">
                          <div className="w-12 h-12 rounded-full bg-[var(--bg-panel)] mx-auto mb-6 flex items-center justify-center text-[var(--accent-teal)] opacity-40"><ShieldCheck size={24} /></div>
                          <h3 className="text-xl font-black text-[var(--text-primary)] uppercase tracking-tight mb-2">Statutory Conclusion</h3>
                          <p className="text-[var(--text-muted)] font-bold max-w-sm mx-auto leading-relaxed">Milestones for <span className="text-[var(--text-primary)]">{activeWorkflow.title}</span> completed. Proceed with filing as outlined.</p>
                       </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </div>

            <div className="w-[420px] border-l border-[var(--border-light)] bg-[var(--bg-panel)] flex flex-col shadow-2xl relative z-10">
              <div className="p-8 border-b border-[var(--border-light)] flex items-center gap-4 bg-[var(--bg-app)] opacity-60">
                <div className="w-9 h-9 rounded-xl bg-[var(--accent-teal)] text-black flex items-center justify-center shadow-lg"><MessageSquare size={18} /></div>
                <div>
                   <h3 className="text-[11px] font-black text-[var(--text-primary)] tracking-widest uppercase mb-0.5">SUPPORT</h3>
                   <p className="text-[9px] text-[var(--text-muted)] font-black uppercase tracking-widest">VAULT SYNC ACTIVE</p>
                </div>
              </div>

              <div className="flex-1 overflow-y-auto px-8 py-10 space-y-12 custom-scrollbar">
                {widgetMessages.map((m, i) => (
                  <div key={i} className={`flex flex-col ${m.role === 'user' ? 'items-end' : 'items-start'} group/msg relative`}>
                    <div className={`p-6 rounded-3xl text-[14px] leading-relaxed font-bold relative
                      ${m.role === 'user' ? 'bg-[var(--bg-app)] border border-[var(--border-light)] text-[var(--text-primary)] rounded-tr-none' : 'bg-transparent text-[var(--text-muted)] border-none rounded-tl-none prose prose-p:m-0 prose-p:leading-relaxed prose-headings:mb-6 prose-headings:text-[var(--text-primary)] max-w-full font-medium whitespace-pre-wrap'}
                    `}>
                      {m.role === 'assistant' && (
                        <button onClick={() => copyToClipboard(m.content)} className="absolute -top-4 -right-2 p-2 bg-[var(--bg-panel)] border border-[var(--border-light)] rounded-lg text-[var(--text-muted)] hover:text-[var(--text-primary)] opacity-0 group-hover/msg:opacity-100 transition-all">
                           <FileText size={14} />
                        </button>
                      )}
                      {m.role === 'assistant' ? <ReactMarkdown>{m.content}</ReactMarkdown> : m.content}
                    </div>
                  </div>
                ))}
                {isWidgetLoading && <div className="flex items-center gap-3 text-[var(--accent-teal)] opacity-60 font-black text-[9px] uppercase tracking-widest"><Loader2 size={14} className="animate-spin" /> CONSULTING...</div>}
                <div ref={widgetEndRef} />
              </div>

              <div className="p-8 bg-[var(--bg-panel)] border-t border-[var(--border-light)]">
                <div className="relative">
                  <input type="text" value={widgetInput} onChange={e => setWidgetInput(e.target.value)} onKeyDown={e => { if (e.key === 'Enter') handleWidgetSend(); }} placeholder="Ask tactical query..." className="w-full bg-[var(--bg-app)] border border-[var(--border-light)] rounded-full py-4 pl-8 pr-16 text-[12px] text-[var(--text-primary)] outline-none focus:border-[var(--accent-teal)] transition-all font-bold placeholder:text-[var(--text-muted)] shadow-inner" />
                  <button onClick={() => handleWidgetSend()} disabled={!widgetInput.trim() || isWidgetLoading} className="absolute right-2 top-2 bottom-2 w-10 h-10 bg-[var(--text-primary)] text-[var(--bg-app)] rounded-full disabled:opacity-30 flex items-center justify-center transition-all shadow-lg hover:scale-105 active:scale-95"><Send size={16} /></button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      <AnimatePresence>
        {showCustomModal && (
          <div className="fixed inset-0 z-[1000] flex items-center justify-center p-6 bg-black/60 backdrop-blur-xl">
            <motion.div initial={{ opacity: 0, scale: 0.95, y: 15 }} animate={{ opacity: 1, scale: 1, y: 0 }} exit={{ opacity: 0, scale: 0.95 }} className="w-full max-w-2xl bg-[var(--bg-app)] border border-[var(--border-light)] rounded-[4rem] p-12 shadow-2xl relative">
              <button onClick={() => setShowCustomModal(false)} className="absolute top-10 right-10 text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-all"><X size={32} /></button>
              
              <div className="mb-10 text-center">
                <div className="w-16 h-16 bg-[var(--bg-panel)] rounded-2xl flex items-center justify-center mx-auto mb-6 text-[var(--accent-teal)]"><Sparkles size={32} /></div>
                <h3 className="text-3xl font-black text-[var(--text-primary)] tracking-widest mb-4 uppercase">Custom Strategy</h3>
                <p className="text-[var(--text-muted)] text-md font-bold max-w-sm mx-auto leading-relaxed italic">Architect a complete statutory manifest for your situation.</p>
              </div>

              <div className="space-y-10">
                <textarea value={customRequest} onChange={e => setCustomRequest(e.target.value)} placeholder="Describe your case..." className="w-full h-56 bg-[var(--bg-panel)] border border-[var(--border-light)] rounded-[2rem] p-8 text-lg text-[var(--text-primary)] outline-none focus:border-[var(--accent-teal)] resize-none font-bold leading-relaxed shadow-inner placeholder:text-[var(--text-muted)]" />
                <button onClick={submitCustomRequest} className="w-full py-6 bg-[var(--text-primary)] text-[var(--bg-app)] rounded-full text-[13px] font-black uppercase tracking-widest hover:opacity-90 active:scale-95 transition-all shadow-xl">ARCHITECT MANIFEST</button>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </div>
  );
}
