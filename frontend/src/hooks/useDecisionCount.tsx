
import { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { supabase } from '@/lib/supabase';

type DecisionCountContextType = {
  decisionCount: number;
  incrementDecisionCount: () => void;
  isLoading: boolean;
};

const DecisionCountContext = createContext<DecisionCountContextType | undefined>(undefined);

export const DecisionCountProvider = ({ children }: { children: React.ReactNode }) => {
  const [decisionCount, setDecisionCount] = useState(0);
  const [isLoading, setIsLoading] = useState(true);

  const loadDecisionCount = useCallback(async () => {
    setIsLoading(true);
    const { data: { session } } = await supabase.auth.getSession();
    const loggedIn = Boolean(session?.user);

    if (!loggedIn) {
      try {
        const raw = localStorage.getItem('guest_decisions_v1');
        const arr = raw ? JSON.parse(raw) : [];
        setDecisionCount(Array.isArray(arr) ? arr.length : 0);
      } catch {
        setDecisionCount(0);
      }
    } else {
      const { count } = await supabase
        .from('user_video_decisions')
        .select('*', { count: 'exact', head: true })
        .eq('user_id', session!.user!.id);
      setDecisionCount(count || 0);
    }
    setIsLoading(false);
  }, []);

  useEffect(() => {
    loadDecisionCount();
    const { data: listener } = supabase.auth.onAuthStateChange((_event, session) => {
      loadDecisionCount();
    });
    return () => {
      listener.subscription.unsubscribe();
    };
  }, [loadDecisionCount]);

  const incrementDecisionCount = () => {
    setDecisionCount(prevCount => prevCount + 1);
  };

  return (
    <DecisionCountContext.Provider value={{ decisionCount, incrementDecisionCount, isLoading }}>
      {children}
    </DecisionCountContext.Provider>
  );
};

export const useDecisionCount = () => {
  const context = useContext(DecisionCountContext);
  if (context === undefined) {
    throw new Error('useDecisionCount must be used within a DecisionCountProvider');
  }
  return context;
};
