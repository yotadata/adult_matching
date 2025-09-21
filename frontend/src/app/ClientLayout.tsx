'use client';

import Header from '@/components/Header';
import { usePathname } from 'next/navigation';

export default function ClientLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const isHome = pathname === '/';
  const nonHomeGradient = 'linear-gradient(90deg, #ADB4E3 0%, #C8BAE3 33.333%, #F7BECE 66.666%, #F9B1C4 100%)';
  const homeGradient = 'linear-gradient(90deg, #C4C8E3 0%, #D7D1E3 33.333%, #F7D7E0 66.666%, #F9C9D6 100%)';

  return (
    <div className="min-h-screen w-full" style={{ background: isHome ? homeGradient : nonHomeGradient }}>
      <Header cardWidth={undefined} />
      {children}
    </div>
  );
}
