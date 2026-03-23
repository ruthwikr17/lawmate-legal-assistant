'use client';

import React, { useState, useEffect } from 'react';
import { 
  Settings as SettingsIcon, 
  MapPin, 
  User, 
  Sparkles, 
  Shield, 
  Trash2, 
  Download, 
  AlertTriangle, 
  Check, 
  ChevronRight, 
  Globe, 
  Zap, 
  MessageSquare,
  Search,
  Crosshair
} from 'lucide-react';
import { motion } from 'framer-motion';
import { usePersona } from '../context/PersonaContext';

export default function Settings() {
  const { activePersona, changePersona, personas } = usePersona();
  
  // States & Cities Data
  const jurisdictions = {
    "Andhra Pradesh": ["Visakhapatnam", "Vijayawada", "Guntur", "Nellore", "Tirupati"],
    "Arunachal Pradesh": ["Itanagar", "Tawang", "Ziro"],
    "Assam": ["Guwahati", "Dibrugarh", "Silchar", "Tezpur"],
    "Bihar": ["Patna", "Gaya", "Bhagalpur", "Muzaffarpur"],
    "Chhattisgarh": ["Raipur", "Bhilai", "Bilaspur"],
    "Goa": ["Panaji", "Margao", "Vasco da Gama"],
    "Gujarat": ["Ahmedabad", "Surat", "Vadodara", "Rajkot"],
    "Haryana": ["Gurugram", "Faridabad", "Panipat", "Rohtak"],
    "Himachal Pradesh": ["Shimla", "Manali", "Dharamshala"],
    "Jharkhand": ["Ranchi", "Jamshedpur", "Dhanbad"],
    "Karnataka": ["Bengaluru", "Mysuru", "Hubballi", "Mangaluru"],
    "Kerala": ["Thiruvananthapuram", "Kochi", "Kozhikode"],
    "Madhya Pradesh": ["Bhopal", "Indore", "Gwalior", "Jabalpur"],
    "Maharashtra": ["Mumbai", "Pune", "Nagpur", "Nashik", "Thane"],
    "Manipur": ["Imphal"],
    "Meghalaya": ["Shillong"],
    "Mizoram": ["Aizawl"],
    "Nagaland": ["Kohima", "Dimapur"],
    "Odisha": ["Bhubaneswar", "Cuttack", "Rourkela"],
    "Punjab": ["Chandigarh", "Ludhiana", "Amritsar", "Jalandhar"],
    "Rajasthan": ["Jaipur", "Jodhpur", "Udaipur", "Kota"],
    "Sikkim": ["Gangtok"],
    "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai", "Trichy"],
    "Telangana": ["Hyderabad", "Warangal", "Nizamabad", "Khammam"],
    "Tripura": ["Agartala"],
    "Uttar Pradesh": ["Lucknow", "Kanpur", "Noida", "Varanasi", "Agra"],
    "Uttarakhand": ["Dehradun", "Haridwar", "Nainital"],
    "West Bengal": ["Kolkata", "Howrah", "Durgapur", "Siliguri"],
    "Delhi": ["New Delhi", "North Delhi", "South Delhi"],
    "Chandigarh": ["Chandigarh"],
    "Jammu & Kashmir": ["Srinagar", "Jammu"],
    "Ladakh": ["Leh", "Kargil"],
    "Puducherry": ["Puducherry"],
    "Andaman & Nicobar": ["Port Blair"],
    "Daman & Diu": ["Daman"],
    "Lakshadweep": ["Kavaratti"]
  };

  const [settings, setSettings] = useState({
    jurisdiction: 'Telangana',
    city: 'Hyderabad',
    theme: 'dark',
    aiStyle: 'Simple',
    aiDepth: 'Medium',
    autoDetect: false
  });

  const [isDetecting, setIsDetecting] = useState(false);

  useEffect(() => {
    const savedSettings = localStorage.getItem('lawmate_settings');
    if (savedSettings) {
      const parsed = JSON.parse(savedSettings);
      setSettings(parsed);
      // Sync theme with document (V18)
      document.documentElement.setAttribute('data-theme', parsed.theme || 'dark');
    }
  }, []);

  const updateSetting = (key, value) => {
    const newSettings = { ...settings, [key]: value };
    // Reset city if state changes
    if (key === 'jurisdiction') {
      newSettings.city = jurisdictions[value][0];
    }
    // Apply theme immediately (V18)
    if (key === 'theme') {
      document.documentElement.setAttribute('data-theme', value);
    }
    setSettings(newSettings);
    localStorage.setItem('lawmate_settings', JSON.stringify(newSettings));
  };

  const handleAutoDetect = () => {
    setIsDetecting(true);
    setTimeout(() => {
      updateSetting('jurisdiction', 'Telangana');
      updateSetting('city', 'Hyderabad');
      updateSetting('autoDetect', true);
      setIsDetecting(false);
    }, 1500);
  };

  const handleClearHistory = () => {
    if (confirm('Are you sure? This will permanently delete all chat history.')) {
      localStorage.removeItem('lawmate_chat_history');
      window.dispatchEvent(new Event('storage'));
      alert('History cleared successfully.');
    }
  };

  const handleExportData = () => {
    const data = {
      profile: JSON.parse(localStorage.getItem('lawmate_user_profile') || '{}'),
      settings: settings,
      history: JSON.parse(localStorage.getItem('lawmate_chat_history') || '[]')
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `lawmate_data_${new Date().toISOString().split('T')[0]}.json`;
    a.click();
  };

  const personaList = [
    { id: 'citizen', label: 'General Citizen', sub: 'Everyday legal protection' },
    { id: 'student', label: 'Student', sub: 'Education & career rights' },
    { id: 'tenant', label: 'Tenant', sub: 'Rent & housing disputes' },
    { id: 'employee', label: 'Employee', sub: 'Workplace & labor laws' },
    { id: 'senior', label: 'Senior Citizen', sub: 'Welfare & succession' },
    { id: 'business', label: 'Business Owner', sub: 'Corporate & IP compliance' }
  ];

  return (
    <div className="flex-1 overflow-y-auto bg-[var(--bg-app)] p-10 md:p-16 hide-scrollbar">
      <div className="max-w-4xl mx-auto space-y-16">
        
        <header>
          <h1 className="text-4xl font-extrabold text-[var(--text-primary)] mb-4 flex items-center gap-4 tracking-tight">
            <div className="p-3 bg-[var(--bg-panel)] rounded-2xl border border-[var(--border-light)] text-[var(--accent-teal)]">
               <SettingsIcon size={24} />
            </div>
            Preferences & System
          </h1>
          <p className="text-[var(--text-muted)] text-lg">Configure your legal workstation and identity parameters.</p>
        </header>

        {/* Persona Selection */}
        <section className="space-y-8">
          <div className="flex items-center justify-between">
             <h3 className="text-[11px] font-bold uppercase tracking-[0.3em] text-[var(--text-muted)] flex items-center gap-2">
               <User size={14} /> Active Persona
             </h3>
             <span className="text-[10px] font-bold bg-[var(--accent-teal)]/10 text-[var(--accent-teal)] px-3 py-1 rounded-full uppercase tracking-widest">Context Driver</span>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {personaList.map((p) => (
              <button 
                key={p.id}
                onClick={() => changePersona(p.id)}
                className={`p-6 rounded-[2rem] text-left border transition-all relative overflow-hidden group ${
                  activePersona.id === p.id 
                  ? 'border-[var(--accent-teal)] bg-[var(--accent-teal)]/5 shadow-lg' 
                  : 'border-[var(--border-light)] bg-[var(--bg-panel)] hover:border-[var(--text-muted)]'
                }`}
              >
                {activePersona.id === p.id && (
                  <div className="absolute top-4 right-4 text-[var(--accent-teal)]">
                    <Check size={18} />
                  </div>
                )}
                <p className="font-extrabold text-[var(--text-primary)] mb-1">{p.label}</p>
                <p className="text-[10px] uppercase font-bold tracking-widest text-[var(--text-muted)] group-hover:text-[var(--text-primary)] transition-colors">{p.sub}</p>
              </button>
            ))}
          </div>
        </section>

        {/* Jurisdiction & Location */}
        <section className="space-y-8 glass-panel p-10 rounded-[2.5rem] border border-[var(--border-light)]">
          <h3 className="text-[11px] font-bold uppercase tracking-[0.3em] text-[var(--text-muted)] flex items-center gap-2">
            <MapPin size={14} /> Regional Jurisdiction
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
             <div className="space-y-4">
                <label className="text-xs font-bold text-[var(--text-muted)] uppercase tracking-widest">Select State / UT</label>
                <div className="relative">
                   <select 
                     className="w-full bg-[var(--bg-app)] border border-[var(--border-light)] rounded-2xl py-4 px-6 outline-none text-[var(--text-primary)] font-bold appearance-none cursor-pointer focus:border-[var(--accent-teal)] transition-all"
                     value={settings.jurisdiction}
                     onChange={(e) => updateSetting('jurisdiction', e.target.value)}
                   >
                     {Object.keys(jurisdictions).map(state => (
                       <option key={state} value={state}>{state}</option>
                     ))}
                   </select>
                   <ChevronRight className="absolute right-6 top-1/2 -translate-y-1/2 rotate-90 text-[var(--text-muted)] pointer-events-none" size={16} />
                </div>
             </div>
             <div className="space-y-4">
                <label className="text-xs font-bold text-[var(--text-muted)] uppercase tracking-widest">District / City</label>
                <div className="relative">
                   <select 
                     className="w-full bg-[var(--bg-app)] border border-[var(--border-light)] rounded-2xl py-4 px-6 outline-none text-[var(--text-primary)] font-bold appearance-none cursor-pointer focus:border-[var(--accent-teal)] transition-all"
                     value={settings.city}
                     onChange={(e) => updateSetting('city', e.target.value)}
                   >
                     {jurisdictions[settings.jurisdiction].map(city => (
                       <option key={city} value={city}>{city}</option>
                     ))}
                   </select>
                   <ChevronRight className="absolute right-6 top-1/2 -translate-y-1/2 rotate-90 text-[var(--text-muted)] pointer-events-none" size={16} />
                </div>
             </div>
          </div>
          <button 
            onClick={handleAutoDetect}
            disabled={isDetecting}
            className="w-full py-4 bg-[var(--bg-panel)] border border-dashed border-[var(--border-light)] rounded-2xl text-[11px] font-bold uppercase tracking-widest text-[var(--text-primary)] hover:bg-[var(--bg-hover)] transition-all flex items-center justify-center gap-3"
          >
            {isDetecting ? (
              <div className="flex items-center gap-2">
                 <div className="w-4 h-4 border-2 border-[var(--accent-teal)] border-t-transparent rounded-full animate-spin"></div>
                 Scanning Network...
              </div>
            ) : (
              <><Crosshair size={14} /> Auto-Detect Current Location</>
            )}
          </button>
        </section>

        {/* AI Behavior */}
        <section className="space-y-10">
          <h3 className="text-[11px] font-bold uppercase tracking-[0.3em] text-[var(--text-muted)] flex items-center gap-2">
            <Sparkles size={14} /> AI Analysis Engine
          </h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {/* Visual Persona (V18) */}
            <div className="space-y-6">
               <div>
                  <h4 className="font-bold text-[var(--text-primary)] mb-1">Visual Persona</h4>
                  <p className="text-xs text-[var(--text-muted)]">Switch between High-Contrast Dark and Sunlight Light.</p>
               </div>
               <div className="flex p-2 bg-[var(--bg-panel)] rounded-2xl border border-[var(--border-light)]">
                  {['dark', 'light'].map(t => (
                    <button 
                      key={t}
                      onClick={() => updateSetting('theme', t)}
                      className={`flex-1 py-3 rounded-xl font-bold text-xs uppercase tracking-widest transition-all ${
                        settings.theme === t 
                        ? 'bg-[var(--bg-app)] text-[var(--accent-teal)] shadow-sm border border-[var(--border-light)]' 
                        : 'text-[var(--text-muted)] hover:text-[var(--text-primary)]'
                      }`}
                    >
                      {t}
                    </button>
                  ))}
               </div>
            </div>

            {/* Response Style */}
            <div className="space-y-6">
               <div>
                  <h4 className="font-bold text-[var(--text-primary)] mb-1">Response Tone</h4>
                  <p className="text-xs text-[var(--text-muted)]">Control the complexity of generated legal text.</p>
               </div>
               <div className="flex p-2 bg-[var(--bg-panel)] rounded-2xl border border-[var(--border-light)]">
                  {['Simple', 'Detailed'].map(style => (
                    <button 
                      key={style}
                      onClick={() => updateSetting('aiStyle', style)}
                      className={`flex-1 py-3 rounded-xl font-bold text-xs uppercase tracking-widest transition-all ${
                        settings.aiStyle === style 
                        ? 'bg-[var(--bg-app)] text-[var(--text-primary)] shadow-sm border border-[var(--border-light)]' 
                        : 'text-[var(--text-muted)] hover:text-[var(--text-primary)]'
                      }`}
                    >
                      {style}
                    </button>
                  ))}
               </div>
            </div>

            {/* Explanation Depth */}
            <div className="space-y-6">
               <div>
                  <h4 className="font-bold text-[var(--text-primary)] mb-1">Analytical Depth</h4>
                  <p className="text-xs text-[var(--text-muted)]">How much background context should AI include?</p>
               </div>
               <div className="flex p-2 bg-[var(--bg-panel)] rounded-2xl border border-[var(--border-light)]">
                  {['Concise', 'Medium', 'Lengthy'].map(depth => (
                    <button 
                      key={depth}
                      onClick={() => updateSetting('aiDepth', depth)}
                      className={`flex-1 py-3 rounded-xl font-bold text-xs uppercase tracking-widest transition-all ${
                        settings.aiDepth === depth 
                        ? 'bg-[var(--bg-app)] text-[var(--text-primary)] shadow-sm border border-[var(--border-light)]' 
                        : 'text-[var(--text-muted)] hover:text-[var(--text-primary)]'
                      }`}
                    >
                      {depth}
                    </button>
                  ))}
               </div>
            </div>
          </div>
        </section>

        {/* Privacy & Safety */}
        <section className="space-y-8 glass-panel p-10 rounded-[2.5rem] border border-red-500/10 bg-red-500/5">
          <h3 className="text-[11px] font-bold uppercase tracking-[0.3em] text-red-500 flex items-center gap-2">
            <Shield size={14} /> Privacy & Governance
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
             <button 
               onClick={handleClearHistory}
               className="p-6 bg-[var(--bg-app)] border border-[var(--border-light)] rounded-[1.5rem] text-left hover:border-red-500/50 transition-all group"
             >
                <div className="w-10 h-10 rounded-xl bg-red-500/10 flex items-center justify-center text-red-500 mb-4 group-hover:scale-110 transition-transform">
                   <Trash2 size={18} />
                </div>
                <h4 className="font-bold text-[var(--text-primary)] mb-1">Clear Consultation Log</h4>
                <p className="text-[10px] text-[var(--text-muted)] uppercase tracking-widest">Permanent Deletion</p>
             </button>
             <button 
               onClick={handleExportData}
               className="p-6 bg-[var(--bg-app)] border border-[var(--border-light)] rounded-[1.5rem] text-left hover:border-[var(--accent-teal)]/50 transition-all group"
             >
                <div className="w-10 h-10 rounded-xl bg-[var(--accent-teal)]/10 flex items-center justify-center text-[var(--accent-teal)] mb-4 group-hover:scale-110 transition-transform">
                   <Download size={18} />
                </div>
                <h4 className="font-bold text-[var(--text-primary)] mb-1">Export Personal Schema</h4>
                <p className="text-[10px] text-[var(--text-muted)] uppercase tracking-widest">Download JSON</p>
             </button>
          </div>
          
          <div className="p-6 bg-[var(--bg-app)] rounded-2xl border border-[var(--border-light)] flex items-start gap-4">
             <AlertTriangle className="text-amber-500 shrink-0" size={20} />
             <p className="text-[10px] text-[var(--text-muted)] leading-relaxed italic uppercase tracking-wider">
               LawMate utilizes 256-bit local encryption. Your data never leaves this device unless explicitly exported or transmitted via encrypted AI analysis tunnels.
             </p>
          </div>
        </section>

      </div>
    </div>
  );
}
