'use client';

import Header from '@/components/Header';
import DesktopSidebar from '@/components/DesktopSidebar';
import useMediaQuery from '@/hooks/useMediaQuery';
import GlobalModals from '@/components/GlobalModals';
import { usePathname } from 'next/navigation';
import { DecisionCountProvider } from '@/hooks/useDecisionCount';

export default function ClientLayout({ children }: { children: React.ReactNode }) {
  console.log('[DEBUG] ClientLayout: Rendering');
  const pathname = usePathname();
  const isPersonalityPage = pathname.startsWith('/personality');
  const isHome = pathname === '/';
  const nonHomeGradient = 'linear-gradient(90deg, #ADB4E3 0%, #C8BAE3 33.333%, #F7BECE 66.666%, #F9B1C4 100%)';
  const homeGradient = 'linear-gradient(90deg, #C4C8E3 0%, #D7D1E3 33.333%, #F7D7E0 66.666%, #F9C9D6 100%)';
  const isMobile = useMediaQuery('(max-width: 639px)');

  const background = isPersonalityPage ? nonHomeGradient : (isHome ? homeGradient : nonHomeGradient);

  return (
    <DecisionCountProvider>
      <div className="min-h-screen w-full" style={{ background }}>
        {isPersonalityPage ? (
          <main className="pt-10 pb-12">
            <div className="container mx-auto max-w-4xl px-4">
              <div className="bg-white/50 backdrop-blur-lg border border-white/50 rounded-2xl shadow-2xl overflow-hidden">
                {children}
              </div>
            </div>
          </main>
        ) : (
          <>
            {/* Desktop fixed left sidebar */}
            {!isMobile && <DesktopSidebar />}
            <div className={!isMobile ? 'pl-56' : ''}>
              {isMobile ? <Header cardWidth={undefined} /> : <GlobalModals />}
              {children}
            </div>
          </>
        )}
      </div>
    </DecisionCountProvider>
  );
}
