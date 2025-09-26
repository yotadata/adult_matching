'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import Image from 'next/image';

export default function PersonalityHeader() {
  const pathname = usePathname();

  const navLinkClasses = (href: string) => {
    const isActive = pathname.startsWith(href);
    return `px-4 py-2 rounded-md text-sm font-medium transition-colors ${
      isActive
        ? 'bg-white/40 text-gray-900'
        : 'text-gray-700 hover:bg-white/20 hover:text-gray-900'
    }`;
  };

  return (
    <header className="fixed top-0 left-0 w-full p-4 z-50">
      <div className="container mx-auto flex justify-between items-center">
        <Link href="/" aria-label="ホームへ移動" className="inline-flex">
          <Image src="/seiheki_lab.png" alt="Seiheki Lab Logo" width={120} height={40} priority draggable="false" />
        </Link>
        <nav className="flex items-center space-x-1 bg-white/30 backdrop-blur-lg border border-white/40 rounded-lg p-1">
          <Link href="/personality" className={navLinkClasses('/personality')}>
            診断テスト
          </Link>
          <Link href="#" className={navLinkClasses('/personality/types') + ' cursor-not-allowed opacity-50'}>
            性癖タイプ
          </Link>
        </nav>
      </div>
    </header>
  );
}
