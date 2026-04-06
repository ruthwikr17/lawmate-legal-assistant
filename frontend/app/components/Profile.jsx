'use client';

import React, { useState, useEffect } from 'react';
import { User, Mail, MapPin, Calendar, Shield, Edit2, LogOut, Clock, Search, ChevronRight, Save, X, Camera, Info } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useAuth } from '../context/AuthContext';

export default function Profile({ setActiveTab, setActiveChatId, handleAskAI }) {
  const { user: authUser, logout } = useAuth();
  
  const [user, setUser] = useState({
    name: authUser?.fullName || 'Citizen',
    email: authUser?.email || '',
    location: 'India',
    joinedDate: authUser?.createdAt ? new Date(authUser.createdAt).toLocaleDateString() : 'Today',
    gender: 'N/A'
  });

  const [isEditModalOpen, setIsEditModalOpen] = useState(false);
  const [editForm, setEditForm] = useState({ ...user });
  const [history, setHistory] = useState([]);
  const [searchQuery, setSearchQuery] = useState('');

  useEffect(() => {
    const savedHistory = JSON.parse(localStorage.getItem('lawmate_chat_history') || '[]');
    setHistory(savedHistory);
  }, []);

  const handleSaveProfile = (e) => {
    e.preventDefault();
    setUser(editForm);
    localStorage.setItem('lawmate_user_profile', JSON.stringify(editForm));
    setIsEditModalOpen(false);
    // Dispatch custom event for other components (like Sidebar/Dashboard) to sync
    window.dispatchEvent(new Event('storage'));
  };

  const filteredHistory = history.filter(item => 
    item.title?.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const getInitials = (name) => {
    return name.split(' ').map(n => n[0]).join('').toUpperCase();
  };

  return (
    <div className="flex-1 overflow-y-auto bg-[var(--bg-app)] p-10 md:p-16 hide-scrollbar">
      <div className="max-w-5xl mx-auto space-y-12">
        
        {/* Profile Hero Card */}
        <section className="glass-panel rounded-[3rem] p-12 border border-[var(--border-light)] relative overflow-hidden shadow-2xl">
          <div className="absolute top-0 right-0 p-12 opacity-5 pointer-events-none">
             <User size={240} />
          </div>
          
          <div className="flex flex-col md:flex-row items-center gap-10 relative z-10">
            <div className="relative group">
              <div className="w-40 h-40 rounded-[2.5rem] bg-gradient-to-br from-[var(--accent-teal)] to-blue-600 flex items-center justify-center text-5xl font-black text-[var(--text-inverse)] shadow-2xl group-hover:scale-105 transition-transform duration-500">
                {getInitials(user.name)}
              </div>
              <button className="absolute -bottom-2 -right-2 p-3 bg-[var(--text-primary)] text-[var(--bg-app)] rounded-2xl shadow-xl hover:scale-110 active:scale-95 transition-all">
                <Camera size={18} />
              </button>
            </div>
            
            <div className="text-center md:text-left flex-1">
              <h2 className="text-4xl font-black text-[var(--text-primary)] mb-2 tracking-tight">{user.name}</h2>
              <p className="text-[var(--text-muted)] flex items-center justify-center md:justify-start gap-2 mb-6 font-medium">
                <Mail size={16} /> {user.email}
              </p>
              
              <div className="flex flex-wrap items-center justify-center md:justify-start gap-4">
                <button 
                  onClick={() => { setEditForm({...user}); setIsEditModalOpen(true); }}
                  className="bg-[var(--text-primary)] text-[var(--bg-app)] px-8 py-3 rounded-2xl font-bold text-sm flex items-center gap-2 hover:scale-105 transition-all"
                >
                  <Edit2 size={16} /> Edit Profile
                </button>
                <div className="px-6 py-3 bg-[var(--bg-panel)] border border-[var(--border-light)] rounded-2xl text-[11px] font-bold uppercase tracking-widest text-[var(--text-muted)] flex items-center gap-2">
                  <Shield size={14} className="text-[var(--accent-teal)]" /> Verified Citizen
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Info Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="glass-panel p-8 rounded-[2.5rem] border border-[var(--border-light)] space-y-6">
            <h3 className="text-[11px] font-bold uppercase tracking-[0.3em] text-[var(--text-muted)] flex items-center gap-2">
              <Info size={14} /> Personal Details
            </h3>
            <div className="space-y-4">
              <div className="flex items-center justify-between p-4 bg-[var(--bg-app)] rounded-2xl border border-[var(--border-light)]">
                 <span className="text-xs font-bold text-[var(--text-muted)] flex items-center gap-2 uppercase tracking-widest"><MapPin size={14} /> Location</span>
                 <span className="text-sm font-bold text-[var(--text-primary)]">{user.location}</span>
              </div>
              <div className="flex items-center justify-between p-4 bg-[var(--bg-app)] rounded-2xl border border-[var(--border-light)]">
                 <span className="text-xs font-bold text-[var(--text-muted)] flex items-center gap-2 uppercase tracking-widest"><User size={14} /> Gender</span>
                 <span className="text-sm font-bold text-[var(--text-primary)]">{user.gender}</span>
              </div>
              <div className="flex items-center justify-between p-4 bg-[var(--bg-app)] rounded-2xl border border-[var(--border-light)]">
                 <span className="text-xs font-bold text-[var(--text-muted)] flex items-center gap-2 uppercase tracking-widest"><Calendar size={14} /> Joined</span>
                 <span className="text-sm font-bold text-[var(--text-primary)]">{user.joinedDate}</span>
              </div>
            </div>
          </div>

          <div className="glass-panel p-8 rounded-[2.5rem] border border-[var(--border-light)] space-y-6">
            <h3 className="text-[11px] font-bold uppercase tracking-[0.3em] text-[var(--text-muted)] flex items-center gap-2">
              <Clock size={14} /> Analytics Snapshot
            </h3>
             <div className="grid grid-cols-2 gap-4">
               <div className="p-6 bg-[var(--bg-app)] rounded-3xl border border-[var(--border-light)] text-center">
                  <p className="text-3xl font-black text-[var(--text-primary)]">{history.length}</p>
                  <p className="text-[10px] font-bold uppercase tracking-widest text-[var(--text-muted)] mt-2">Consultations</p>
               </div>
               <div className="p-6 bg-[var(--bg-app)] rounded-3xl border border-[var(--border-light)] text-center">
                  <p className="text-3xl font-black text-[var(--accent-teal)]">{localStorage.getItem('lawmate_workflow_count') || '0'}</p>
                  <p className="text-[10px] font-bold uppercase tracking-widest text-[var(--text-muted)] mt-2">Legal Workflows</p>
               </div>
            </div>
          </div>
        </div>

        {/* History Section */}
        <section className="glass-panel rounded-[3rem] border border-[var(--border-light)] p-10 shadow-lg">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-6 mb-10">
            <div>
              <h3 className="text-2xl font-black text-[var(--text-primary)]">Consultation History</h3>
              <p className="text-sm text-[var(--text-muted)] mt-1">Review your past legal interactions</p>
            </div>
            <div className="relative">
              <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-[var(--text-muted)]" size={18} />
              <input 
                type="text" 
                placeholder="Search history..." 
                className="pl-12 pr-6 py-4 bg-[var(--bg-app)] border border-[var(--border-light)] rounded-2xl w-full md:w-80 outline-none focus:border-[var(--text-primary)] transition-all font-medium text-sm"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>
          </div>

          <div className="space-y-4">
            {filteredHistory.length > 0 ? filteredHistory.map((item) => (
              <div 
                key={item.id} 
                onClick={() => handleAskAI(item.title, item.id)}
                className="flex items-center justify-between p-6 bg-[var(--bg-app)] rounded-[2rem] border border-[var(--border-light)] hover:border-[var(--text-muted)] transition-all cursor-pointer group"
              >
                <div className="flex items-center gap-6">
                  <div className="w-12 h-12 rounded-2xl bg-[var(--bg-panel)] flex items-center justify-center text-[var(--text-muted)] group-hover:text-[var(--text-primary)] transition-colors">
                    <Clock size={20} />
                  </div>
                  <div>
                    <p className="font-bold text-[var(--text-primary)] group-hover:text-[var(--accent-teal)] transition-colors">{item.title}</p>
                    <p className="text-xs text-[var(--text-muted)] mt-1">{new Date(item.updatedAt).toLocaleDateString()}</p>
                  </div>
                </div>
                <ChevronRight size={20} className="text-[var(--text-muted)] group-hover:text-[var(--text-primary)] group-hover:translate-x-1 transition-all" />
              </div>
            )) : (
              <div className="py-20 text-center text-[var(--text-muted)] opacity-50 border-2 border-dashed border-[var(--border-light)] rounded-[2.5rem]">
                <Clock size={48} className="mx-auto mb-4 opacity-20" />
                <p className="font-bold uppercase tracking-[0.2em] text-xs">History is empty</p>
              </div>
            )}
          </div>
        </section>

        {/* Danger Zone */}
        <section className="flex justify-center pt-8">
           <button onClick={logout} className="flex items-center gap-3 text-red-500 font-bold uppercase tracking-widest text-xs hover:text-red-400 transition-colors p-4 rounded-2xl hover:bg-red-500/5">
              <LogOut size={16} /> Sign out of LawMate
           </button>
        </section>
      </div>

      {/* Edit Profile Modal */}
      <AnimatePresence>
        {isEditModalOpen && (
          <div className="fixed inset-0 z-[100] flex items-center justify-center p-6 md:p-10">
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setIsEditModalOpen(false)}
              className="absolute inset-0 bg-black/80 backdrop-blur-md"
            ></motion.div>
            
            <motion.div 
              initial={{ scale: 0.95, opacity: 0, y: 20 }}
              animate={{ scale: 1, opacity: 1, y: 0 }}
              exit={{ scale: 0.95, opacity: 0, y: 20 }}
              className="relative w-full max-w-xl bg-[var(--bg-app)] rounded-[3rem] border border-[var(--border-light)] shadow-2xl overflow-hidden"
            >
              <div className="p-10 space-y-8">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-3xl font-black text-[var(--text-primary)]">Update Identity</h3>
                    <p className="text-sm text-[var(--text-muted)] mt-1">Changes are synced across your workspace</p>
                  </div>
                  <button 
                    onClick={() => setIsEditModalOpen(false)}
                    className="p-3 bg-[var(--bg-panel)] text-[var(--text-muted)] rounded-2xl hover:text-[var(--text-primary)] transition-colors"
                  >
                    <X size={24} />
                  </button>
                </div>

                <form onSubmit={handleSaveProfile} className="space-y-6">
                  <div className="space-y-4">
                    <label className="text-[10px] font-bold uppercase tracking-[0.2em] text-[var(--text-muted)] ml-2">Full Legal Name</label>
                    <input 
                      type="text" 
                      className="w-full bg-[var(--bg-panel)] border border-[var(--border-light)] rounded-2xl py-5 px-6 outline-none text-[var(--text-primary)] font-bold focus:border-[var(--text-primary)] transition-all"
                      value={editForm.name}
                      onChange={(e) => setEditForm({...editForm, name: e.target.value})}
                      required
                    />
                  </div>

                  <div className="space-y-4">
                    <label className="text-[10px] font-bold uppercase tracking-[0.2em] text-[var(--text-muted)] ml-2">Communication Email</label>
                    <input 
                      type="email" 
                      className="w-full bg-[var(--bg-panel)] border border-[var(--border-light)] rounded-2xl py-5 px-6 outline-none text-[var(--text-primary)] font-bold focus:border-[var(--text-primary)] transition-all"
                      value={editForm.email}
                      onChange={(e) => setEditForm({...editForm, email: e.target.value})}
                      required
                    />
                  </div>

                  <div className="grid grid-cols-2 gap-6">
                    <div className="space-y-4">
                      <label className="text-[10px] font-bold uppercase tracking-[0.2em] text-[var(--text-muted)] ml-2">Gender Identification</label>
                      <select 
                        className="w-full bg-[var(--bg-panel)] border border-[var(--border-light)] rounded-2xl py-5 px-6 outline-none text-[var(--text-primary)] font-bold focus:border-[var(--text-primary)] transition-all appearance-none cursor-pointer"
                        value={editForm.gender}
                        onChange={(e) => setEditForm({...editForm, gender: e.target.value})}
                      >
                        <option value="Male">Male</option>
                        <option value="Female">Female</option>
                        <option value="Other">Other</option>
                      </select>
                    </div>
                    <div className="space-y-4">
                      <label className="text-[10px] font-bold uppercase tracking-[0.2em] text-[var(--text-muted)] ml-2">Active Location</label>
                      <input 
                        type="text" 
                        className="w-full bg-[var(--bg-panel)] border border-[var(--border-light)] rounded-2xl py-5 px-6 outline-none text-[var(--text-primary)] font-bold focus:border-[var(--text-primary)] transition-all"
                        value={editForm.location}
                        onChange={(e) => setEditForm({...editForm, location: e.target.value})}
                        placeholder="City, State"
                      />
                    </div>
                  </div>

                  <div className="flex gap-4 pt-10">
                    <button 
                      type="button"
                      onClick={() => setIsEditModalOpen(false)}
                      className="flex-1 py-5 rounded-3xl font-black text-sm uppercase tracking-widest border border-[var(--border-light)] text-[var(--text-muted)] hover:bg-[var(--bg-hover)] transition-all"
                    >
                      Discard
                    </button>
                    <button 
                      type="submit"
                      className="flex-1 py-5 rounded-3xl font-black text-sm uppercase tracking-widest bg-[var(--text-primary)] text-[var(--bg-app)] shadow-2xl hover:scale-[1.02] active:scale-95 transition-all"
                    >
                      Save Changes
                    </button>
                  </div>
                </form>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </div>
  );
}
