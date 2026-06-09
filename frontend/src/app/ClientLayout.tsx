'use client';

import GlobalModals from '@/components/GlobalModals';
import BrowseTabBar from '@/components/BrowseTabBar';
import { usePathname, useRouter } from 'next/navigation';
import { useEffect, useState } from 'react';
import { supabase } from '@/lib/supabase';
import { DecisionCountProvider } from '@/hooks/useDecisionCount';
import { setGTagUserId } from '@/lib/analytics';

export default function ClientLayout({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const pathname = usePathname();
  const isHome = pathname === '/';
  const nonHomeGradient = '#0d1117';
  const homeGradient = 'linear-gradient(135deg, #1a0d2e 0%, #160d25 33%, #2a1020 66%, #1e0d1a 100%)';
  const [isLoggedIn, setIsLoggedIn] = useState<boolean | null>(null);
  const [authInitialized, setAuthInitialized] = useState(false);

  useEffect(() => {
    let isMounted = true;

    const hydrateSession = async () => {
      try {
        const { data: { session } } = await supabase.auth.getSession();
        if (isMounted) setIsLoggedIn(Boolean(session?.user));
        setGTagUserId(session?.user?.id ?? null);
      } catch {
        if (isMounted) setIsLoggedIn(false);
        setGTagUserId(null);
      } finally {
        if (isMounted) setAuthInitialized(true);
      }
    };

    hydrateSession();
    const { data: listener } = supabase.auth.onAuthStateChange((_event, session) => {
      setIsLoggedIn(Boolean(session?.user));
      setAuthInitialized(true);
      setGTagUserId(session?.user?.id ?? null);
    });

    return () => {
      isMounted = false;
      listener.subscription.unsubscribe();
    };
  }, []);

  const isQuizPath = pathname?.startsWith('/quiz') ?? false;

  useEffect(() => {
    if (!authInitialized || isLoggedIn === null || !pathname) return;
    if (pathname.startsWith('/quiz')) return;
    const publicPaths = ['/', '/swipe', '/grid', '/about', '/videos/', '/performers', '/tags', '/list', '/u/'];
    const isPublic = publicPaths.some((p) => pathname === p || pathname.startsWith(p));
    if (!isPublic && isLoggedIn === false) {
      router.replace('/grid');
    }
  }, [authInitialized, isLoggedIn, pathname, router]);

  // 診断ページは完全独立レイアウト
  if (isQuizPath) {
    return <>{children}</>;
  }

  return (
    <DecisionCountProvider>
      <div className="min-h-screen w-full" style={{ background: isHome ? homeGradient : nonHomeGradient }}>
        <BrowseTabBar />
        <GlobalModals />
        <div className="pt-16">
          {children}
        </div>
      </div>
    </DecisionCountProvider>
  );
}
