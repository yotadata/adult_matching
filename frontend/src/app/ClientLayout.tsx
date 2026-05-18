'use client';

import Header from '@/components/Header';
import DesktopSidebar from '@/components/DesktopSidebar';
import useMediaQuery from '@/hooks/useMediaQuery';
import GlobalModals from '@/components/GlobalModals';
import BrowseTabBar from '@/components/BrowseTabBar';
import { usePathname, useRouter } from 'next/navigation';
import { useEffect, useState } from 'react';
import { supabase } from '@/lib/supabase';
import { DecisionCountProvider } from '@/hooks/useDecisionCount';
import { setGTagUserId } from '@/lib/analytics';
import AgeGate from '@/app/quiz/AgeGate';

export default function ClientLayout({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const pathname = usePathname();
  const isHome = pathname === '/';
  const nonHomeGradient = '#0d1117';
  const homeGradient = 'linear-gradient(135deg, #1a0d2e 0%, #160d25 33%, #2a1020 66%, #1e0d1a 100%)';
  const isMobile = useMediaQuery('(max-width: 639px)');
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
  const isBrowsePath = (pathname?.startsWith('/grid') || pathname?.startsWith('/swipe') || pathname?.startsWith('/browse')) ?? false;

  useEffect(() => {
    if (!authInitialized || isLoggedIn === null || !pathname) return;
    if (pathname.startsWith('/quiz')) return; // 診断ページは認証不要・リダイレクトなし
    const isSwipePath = pathname.startsWith('/swipe');
    const isGridPath = pathname.startsWith('/grid');
    const isAboutPage = pathname === '/about';
    const isSearchPage = pathname === '/search';
    const isVideoPage = pathname.startsWith('/videos/');
    const isPerformerPage = pathname.startsWith('/performers');
    const isTagPage = pathname.startsWith('/tags');
    const requiresLogin = !(isSwipePath || isGridPath || isAboutPage || isSearchPage || isVideoPage || isPerformerPage || isTagPage);

    if (requiresLogin && isLoggedIn === false) {
      router.replace('/swipe');
    }
  }, [authInitialized, isLoggedIn, pathname, router]);

  // 診断ページは完全独立レイアウト（サイドバー・ヘッダーなし）
  if (isQuizPath) {
    return <>{children}</>;
  }

  return (
    <DecisionCountProvider>
      <AgeGate />
      <div className="min-h-screen w-full" style={{ background: isHome ? homeGradient : nonHomeGradient }}>
        {!isMobile && !isBrowsePath && <DesktopSidebar />}
        <div className={!isMobile && !isBrowsePath ? 'pl-56' : ''}>
          {!isBrowsePath && (isMobile ? <Header cardWidth={undefined} /> : <GlobalModals />)}
          {isBrowsePath && <BrowseTabBar />}
          {isBrowsePath && <GlobalModals />}
          {children}
        </div>
      </div>
    </DecisionCountProvider>
  );
}
