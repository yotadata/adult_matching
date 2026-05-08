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
      <header className="w-full sticky top-0 z-50" style={{ background: 'rgba(253,246,232,0.92)', backdropFilter: 'blur(8px)', borderBottom: '2px dashed rgba(180,120,60,0.35)' }}>
        <div className="max-w-lg mx-auto px-4 h-14 flex items-center justify-between">
          <Link href="/quiz" className="font-black text-[16px] text-[#3d1a00] leading-tight">
            性癖16タイプ診断
          </Link>

          {/* PC: 通常ナビ */}
          <nav className="hidden sm:flex items-center gap-1">
            {links.map((link) => {
              const isActive = pathname === link.href;
              return (
                <Link
                  key={link.href}
                  href={link.href}
                  className={`px-3 py-1.5 rounded-full text-[12px] font-bold transition-all ${
                    isActive
                      ? 'text-white'
                      : 'text-[#b5541a] hover:bg-[#e8c9a0]/40'
                  }`}
                  style={isActive ? { background: '#c87941', boxShadow: '0 2px 0 #9e5a28' } : {}}
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
            <span className={`block w-6 h-0.5 bg-[#b5541a] transition-transform duration-200 ${open ? 'translate-y-2 rotate-45' : ''}`} />
            <span className={`block w-6 h-0.5 bg-[#b5541a] transition-opacity duration-200 ${open ? 'opacity-0' : ''}`} />
            <span className={`block w-6 h-0.5 bg-[#b5541a] transition-transform duration-200 ${open ? '-translate-y-2 -rotate-45' : ''}`} />
          </button>
        </div>
      </header>

      {/* スマホ用ドロワー */}
      {open && (
        <>
          <div
            className="fixed inset-0 z-40 sm:hidden"
            onClick={() => setOpen(false)}
          />
          <nav className="fixed top-14 right-0 z-50 w-48 rounded-bl-2xl shadow-lg py-2 sm:hidden" style={{ background: '#fdf6e8', border: '2px dashed rgba(180,120,60,0.35)', borderTop: 'none', borderRight: 'none' }}>
            {links.map((link) => {
              const isActive = pathname === link.href;
              return (
                <Link
                  key={link.href}
                  href={link.href}
                  onClick={() => setOpen(false)}
                  className={`block px-5 py-3 text-[14px] font-bold transition-colors ${
                    isActive
                      ? 'text-[#c87941] bg-[#c8946a]/15'
                      : 'text-[#3d1a00] hover:bg-[#c8946a]/10'
                  }`}
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
