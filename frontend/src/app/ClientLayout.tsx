'use client';

import Header from '@/components/Header';
import DesktopSidebar from '@/components/DesktopSidebar';
import useMediaQuery from '@/hooks/useMediaQuery';
import GlobalModals from '@/components/GlobalModals';
import { usePathname } from 'next/navigation';
import { DecisionCountProvider } from '@/hooks/useDecisionCount.tsx';
import Link from 'next/link';
import Image from 'next/image';

export default function ClientLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const isPersonalityPage = pathname === '/personality';
  const isHome = pathname === '/';
  const nonHomeGradient = 'linear-gradient(90deg, #ADB4E3 0%, #C8BAE3 33.333%, #F7BECE 66.666%, #F9B1C4 100%)';
  const homeGradient = 'linear-gradient(90deg, #C4C8E3 0%, #D7D1E3 33.333%, #F7D7E0 66.666%, #F9C9D6 100%)';
  const isMobile = useMediaQuery('(max-width: 639px)');

  const background = isPersonalityPage ? nonHomeGradient : (isHome ? homeGradient : nonHomeGradient);

  return (
    <DecisionCountProvider>
      <div className="min-h-screen w-full" style={{ background }}>
        {isPersonalityPage ? (
          <>
            <header className="fixed top-0 left-0 w-full p-4 z-50 flex justify-center sm:justify-start">
              <Link href="/" aria-label="ホームへ移動" className="inline-flex">
                <Image src="/seiheki_lab.png" alt="Seiheki Lab Logo" width={120} height={40} priority draggable="false" />
              </Link>
            </header>
            <main className="pt-20">
              {children}
            </main>
          </>
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