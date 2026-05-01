'use client';

import Link from 'next/link';
import Image from 'next/image';
import { usePathname } from 'next/navigation';

export default function QuizHeader() {
  const pathname = usePathname();

  const links = [
    { href: '/quiz', label: '診断する' },
    { href: '/quiz/characters', label: 'キャラ一覧' },
    { href: '/quiz/about', label: 'この診断とは' },
  ];

  return (
    <header className="w-full border-b border-[#e8c9a0]/40 bg-[#fff8f0]/80 backdrop-blur-sm sticky top-0 z-50">
      <div className="max-w-lg mx-auto px-4 h-14 flex items-center justify-between">
        {/* ロゴ */}
        <Link href="/quiz" className="flex items-center gap-2">
          <Image src="/seiheki_lab_header.png" alt="性癖ラボ" width={100} height={28} className="object-contain" />
          <span className="font-black text-[13px] text-[#5c2e00] leading-tight border-l border-[#e8c9a0] pl-2">
            16タイプ診断
          </span>
        </Link>

        {/* ナビ */}
        <nav className="flex items-center gap-1">
          {links.map((link) => {
            const isActive = pathname === link.href;
            return (
              <Link
                key={link.href}
                href={link.href}
                className={`px-3 py-1.5 rounded-full text-[12px] font-bold transition-colors ${
                  isActive
                    ? 'bg-[#ffb347] text-white'
                    : 'text-[#b5541a] hover:bg-[#ffb347]/20'
                }`}
              >
                {link.label}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
