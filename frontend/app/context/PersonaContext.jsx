'use client';

import React, { createContext, useContext, useState, useEffect } from 'react';
import { useAuth } from './AuthContext';

const PersonaContext = createContext();

export const personas = [
  { id: 'citizen', label: 'General Citizen' },
  { id: 'student', label: 'Student' },
  { id: 'tenant', label: 'Tenant' },
  { id: 'employee', label: 'Employee' },
  { id: 'senior', label: 'Senior Citizen' },
  { id: 'business', label: 'Business Owner' },
];

export function PersonaProvider({ children }) {
  const { user } = useAuth();
  const [activePersona, setActivePersona] = useState(personas[0]); // default
  
  // Sync with Auth user on load
  useEffect(() => {
    if (user?.primaryProfile) {
      const match = personas.find(p => p.id.toUpperCase() === user.primaryProfile.toUpperCase());
      if (match) setActivePersona(match);
    } else {
      const saved = localStorage.getItem('lawmate_persona');
      if (saved) {
        const found = personas.find(p => p.id === saved);
        if (found) setActivePersona(found);
      }
    }
  }, [user]);

  const changePersona = (personaId) => {
    const found = personas.find(p => p.id === personaId);
    if (found) {
      setActivePersona(found);
      localStorage.setItem('lawmate_persona', personaId);
    }
  };

  return (
    <PersonaContext.Provider value={{ activePersona, changePersona, personas }}>
      {children}
    </PersonaContext.Provider>
  );
}

export function usePersona() {
  const context = useContext(PersonaContext);
  if (!context) {
    throw new Error('usePersona must be used within a PersonaProvider');
  }
  return context;
}
