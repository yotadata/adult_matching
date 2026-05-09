'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useState } from 'react';

export default function QuizHeader() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);

  const links = [
    { href: '/quiz', label: '診断する' },
    { href: '/quiz/characters', label: 'キャラ一覧' },
    { href: '/quiz/about', label: 'この診断とは' },
  ];

  return (
    <>
      <header className="w-full sticky top-0 z-50" style={{ background: 'rgba(13,11,8,0.96)', backdropFilter: 'blur(12px)', borderBottom: '1px solid rgba(180,150,80,0.25)' }}>
        <div className="max-w-lg mx-auto px-4 h-14 flex items-center justify-between">
          <Link href="/quiz" className="font-black text-[16px] leading-tight" style={{ color: '#e8d5a0' }}>
            偏愛16診断
          </Link>

          {/* PC: 通常ナビ */}
          <nav className="hidden sm:flex items-center gap-1">
            {links.map((link) => {
              const isActive = pathname === link.href;
              return (
                <Link
                  key={link.href}
                  href={link.href}
                  className="px-3 py-1.5 rounded-full text-[12px] font-bold transition-all"
                  style={isActive
                    ? { background: 'rgba(180,150,80,0.25)', color: '#e8d5a0', border: '1px solid rgba(180,150,80,0.5)' }
                    : { color: 'rgba(200,180,140,0.6)' }
                  }
                >
                  {link.label}
                </Link>
              );
            })}
          </nav>

          {/* スマホ: ハンバーガー */}
          <button
            onClick={() => setOpen((v) => !v)}
            aria-label="メニューを開く"
            className="flex sm:hidden flex-col justify-center items-center w-10 h-10 gap-1.5"
          >
            <span className={`block w-6 h-0.5 transition-transform duration-200 ${open ? 'translate-y-2 rotate-45' : ''}`} style={{ background: 'rgba(200,180,140,0.7)' }} />
            <span className={`block w-6 h-0.5 transition-opacity duration-200 ${open ? 'opacity-0' : ''}`} style={{ background: 'rgba(200,180,140,0.7)' }} />
            <span className={`block w-6 h-0.5 transition-transform duration-200 ${open ? '-translate-y-2 -rotate-45' : ''}`} style={{ background: 'rgba(200,180,140,0.7)' }} />
          </button>
        </div>
      </header>

      {/* スマホ用ドロワー */}
      {open && (
        <>
          <div className="fixed inset-0 z-40 sm:hidden" onClick={() => setOpen(false)} />
          <nav className="fixed top-14 right-0 z-50 w-48 rounded-bl-2xl shadow-lg py-2 sm:hidden" style={{ background: '#181410', border: '1px solid rgba(180,150,80,0.3)', borderTop: 'none', borderRight: 'none' }}>
            {links.map((link) => {
              const isActive = pathname === link.href;
              return (
                <Link
                  key={link.href}
                  href={link.href}
                  onClick={() => setOpen(false)}
                  className="block px-5 py-3 text-[14px] font-bold transition-colors"
                  style={isActive
                    ? { color: '#e8d5a0', background: 'rgba(180,150,80,0.15)' }
                    : { color: 'rgba(200,180,140,0.6)' }
                  }
                >
                  {link.label}
                </Link>
              );
            })}
          </nav>
        </>
      )}
    </>
  );
}
