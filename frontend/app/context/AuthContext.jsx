'use client';

import React, { createContext, useContext, useState, useEffect } from 'react';

const AuthContext = createContext();

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  // Rehydrate on boot
  useEffect(() => {
    try {
      const stored = localStorage.getItem('lawmate_auth_token');
      if (stored) {
        setUser(JSON.parse(stored));
      }
    } catch (e) {
      localStorage.removeItem('lawmate_auth_token');
    }
    setLoading(false);
  }, []);

  const login = (userData) => {
    setUser(userData);
    localStorage.setItem('lawmate_auth_token', JSON.stringify(userData));
  };

  const logout = () => {
    setUser(null);
    localStorage.removeItem('lawmate_auth_token');
    // Clear other data
    localStorage.removeItem('lawmate_user_profile');
    localStorage.removeItem('lawmate_chat_history');
  };

  return (
    <AuthContext.Provider value={{ user, login, logout, loading }}>
      {!loading && children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  return useContext(AuthContext);
}
