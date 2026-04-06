'use client';

import React, { useState } from 'react';
import { Mail, Lock, User as UserIcon, Shield, Loader, ArrowRight } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { motion, AnimatePresence } from 'framer-motion';

export default function Auth() {
  const { login } = useAuth();
  const [isLogin, setIsLogin] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const [formData, setFormData] = useState({
    email: '',
    password: '',
    fullName: '',
    primaryProfile: 'GENERAL'
  });

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    const endpoint = isLogin ? '/api/auth/login' : '/api/auth/register';
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';

    try {
      const response = await fetch(`${apiUrl}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Authentication failed');
      }

      // Login success
      login(data);

    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen w-full flex items-center justify-center bg-[var(--bg-app)] relative overflow-hidden p-6">
      {/* Background decorations */}
      <div className="absolute top-1/2 left-1/4 -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-[var(--accent-teal)] rounded-full blur-[120px] opacity-20 pointer-events-none"></div>
      <div className="absolute bottom-1/4 right-1/4 translate-x-1/2 translate-y-1/2 w-96 h-96 bg-blue-600 rounded-full blur-[120px] opacity-10 pointer-events-none"></div>

      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="w-full max-w-md"
      >
        <div className="glass-panel p-10 rounded-[3rem] border border-[var(--border-light)] shadow-2xl relative z-10">
          
          <div className="text-center mb-10">
            <div className="w-16 h-16 mx-auto bg-gradient-to-br from-[var(--text-primary)] to-[var(--text-muted)] rounded-2xl flex items-center justify-center shadow-lg mb-6">
               <Shield size={32} className="text-[var(--bg-app)]" />
            </div>
            <h1 className="text-3xl font-black text-[var(--text-primary)] tracking-tight">LawMate</h1>
            <p className="text-sm font-bold tracking-widest uppercase text-[var(--text-muted)] mt-2">
              {isLogin ? 'Secure Gateway' : 'Create Intelligence Profile'}
            </p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-5">
            <AnimatePresence mode="wait">
              {!isLogin && (
                <motion.div 
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="space-y-5 overflow-hidden"
                >
                  <div className="relative group">
                    <UserIcon size={18} className="absolute left-5 top-1/2 -translate-y-1/2 text-[var(--text-muted)] group-focus-within:text-[var(--accent-teal)] transition-colors" />
                    <input 
                      type="text" 
                      placeholder="Full Legal Name" 
                      required={!isLogin}
                      className="w-full bg-[var(--bg-panel)] border border-[var(--border-light)] rounded-2xl py-4 pl-14 pr-6 outline-none text-[var(--text-primary)] font-bold focus:border-[var(--text-primary)] transition-all"
                      value={formData.fullName}
                      onChange={(e) => setFormData({...formData, fullName: e.target.value})}
                    />
                  </div>
                  
                  <div className="relative group">
                    <Shield size={18} className="absolute left-5 top-1/2 -translate-y-1/2 text-[var(--text-muted)] group-focus-within:text-[var(--accent-teal)] transition-colors" />
                    <select 
                      className="w-full bg-[var(--bg-panel)] border border-[var(--border-light)] rounded-2xl py-4 pl-14 pr-6 outline-none text-[var(--text-primary)] font-bold focus:border-[var(--text-primary)] transition-all appearance-none cursor-pointer"
                      value={formData.primaryProfile}
                      onChange={(e) => setFormData({...formData, primaryProfile: e.target.value})}
                    >
                      <option value="GENERAL">General Citizen</option>
                      <option value="TENANT">Tenant</option>
                      <option value="STUDENT">Student</option>
                      <option value="EMPLOYEE">Employee</option>
                      <option value="CONSUMER">Consumer</option>
                    </select>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            <div className="relative group">
              <Mail size={18} className="absolute left-5 top-1/2 -translate-y-1/2 text-[var(--text-muted)] group-focus-within:text-[var(--text-primary)] transition-colors" />
              <input 
                type="email" 
                placeholder="Secure Email" 
                required
                className="w-full bg-[var(--bg-panel)] border border-[var(--border-light)] rounded-2xl py-4 pl-14 pr-6 outline-none text-[var(--text-primary)] font-bold focus:border-[var(--text-primary)] transition-all"
                value={formData.email}
                onChange={(e) => setFormData({...formData, email: e.target.value})}
              />
            </div>

            <div className="relative group">
              <Lock size={18} className="absolute left-5 top-1/2 -translate-y-1/2 text-[var(--text-muted)] group-focus-within:text-[var(--text-primary)] transition-colors" />
              <input 
                type="password" 
                placeholder="Master Password" 
                required
                className="w-full bg-[var(--bg-panel)] border border-[var(--border-light)] rounded-2xl py-4 pl-14 pr-6 outline-none text-[var(--text-primary)] font-bold focus:border-[var(--text-primary)] transition-all"
                value={formData.password}
                onChange={(e) => setFormData({...formData, password: e.target.value})}
              />
            </div>

            {error && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="p-4 bg-red-500/10 border border-red-500/20 rounded-2xl text-center">
                <p className="text-xs font-bold text-red-500 uppercase tracking-widest">{error}</p>
              </motion.div>
            )}

            <button 
              type="submit" 
              disabled={loading}
              className="w-full flex items-center justify-center gap-2 py-5 rounded-[1.5rem] bg-[var(--text-primary)] text-[var(--bg-app)] font-black uppercase text-sm tracking-widest shadow-2xl hover:scale-[1.02] active:scale-95 transition-all disabled:opacity-50 disabled:scale-100 mt-4"
            >
              {loading ? <Loader size={20} className="animate-spin" /> : (isLogin ? 'Authenticate' : 'Initialize Profile')}
              {!loading && <ArrowRight size={18} />}
            </button>
          </form>

          <div className="mt-8 text-center">
            <button 
              onClick={() => setIsLogin(!isLogin)}
              className="text-[var(--text-muted)] hover:text-[var(--text-primary)] font-bold text-sm transition-colors"
            >
              {isLogin ? "Don't have a profile? Create one" : "Already registered? Authenticate"}
            </button>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
