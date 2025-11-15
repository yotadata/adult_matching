'use client';

import Header from '@/components/Header';
import DesktopSidebar from '@/components/DesktopSidebar';
import useMediaQuery from '@/hooks/useMediaQuery';
import GlobalModals from '@/components/GlobalModals';
import { usePathname, useRouter } from 'next/navigation';
import { useEffect, useState } from 'react';
import { supabase } from '@/lib/supabase';
import { DecisionCountProvider } from '@/hooks/useDecisionCount';

export default function ClientLayout({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const pathname = usePathname();
  const isHome = pathname === '/';
  const nonHomeGradient = 'linear-gradient(90deg, #ADB4E3 0%, #C8BAE3 33.333%, #F7BECE 66.666%, #F9B1C4 100%)';
  const homeGradient = 'linear-gradient(90deg, #C4C8E3 0%, #D7D1E3 33.333%, #F7D7E0 66.666%, #F9C9D6 100%)';
  const isMobile = useMediaQuery('(max-width: 639px)');
  const [isLoggedIn, setIsLoggedIn] = useState<boolean | null>(null);
  const [authInitialized, setAuthInitialized] = useState(false);

  useEffect(() => {
    let isMounted = true;

    const hydrateSession = async () => {
      try {
        const { data: { session } } = await supabase.auth.getSession();
        if (isMounted) setIsLoggedIn(Boolean(session?.user));
      } catch {
        if (isMounted) setIsLoggedIn(false);
      } finally {
        if (isMounted) setAuthInitialized(true);
      }
    };

    hydrateSession();
    const { data: listener } = supabase.auth.onAuthStateChange((_event, session) => {
      setIsLoggedIn(Boolean(session?.user));
      setAuthInitialized(true);
    });

    return () => {
      isMounted = false;
      listener.subscription.unsubscribe();
    };
  }, []);

  useEffect(() => {
    if (!authInitialized || isLoggedIn === null || !pathname) return;
    const isSwipePath = pathname.startsWith('/swipe');
    const isAboutPage = pathname === '/about';
    const isSearchPage = pathname === '/search';
    const requiresLogin = !(isSwipePath || isAboutPage || isSearchPage);

    if (requiresLogin && isLoggedIn === false) {
      router.replace('/swipe');
    }
  }, [authInitialized, isLoggedIn, pathname, router]);

  return (
    <DecisionCountProvider>
      <div className="min-h-screen w-full" style={{ background: isHome ? homeGradient : nonHomeGradient }}>
        {!isMobile && <DesktopSidebar />}
        <div className={!isMobile ? 'pl-56' : ''}>
          {isMobile ? <Header cardWidth={undefined} /> : <GlobalModals />}
          {children}
        </div>
      </div>
    </DecisionCountProvider>
  );
}
