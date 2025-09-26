'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import Image from 'next/image';
import { Menu, X } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

export default function PersonalityHeader() {
  const pathname = usePathname();
  const [resultType, setResultType] = useState<string | null>(null);
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  useEffect(() => {
    // クライアントサイドでのみlocalStorageから結果を取得
    const storedResult = localStorage.getItem('personalityResultType');
    if (storedResult) {
      setResultType(storedResult);
    }
  }, [pathname]); // パスが変わるたびに再チェック

  const navLinkClasses = (href: string, isMobile = false) => {
    let isActive = false;
    if (href === '/personality') {
      isActive = pathname === '/personality';
    } else if (href === '/personality/types') {
      isActive = pathname === '/personality/types';
    } else if (href === '/personality/result') {
      isActive = pathname.startsWith('/personality/result');
    }

    const baseClasses = isMobile ? 'block w-full text-left px-4 py-3 text-lg' : 'px-4 py-2 rounded-md text-sm font-medium';
    return `${baseClasses} transition-colors ${
      isActive
        ? 'bg-white/40 text-gray-900'
        : 'text-gray-700 hover:bg-white/20 hover:text-gray-900'
    }`;
  };

  const navLinks = (isMobile = false) => (
    <>
      <Link href="/personality" className={navLinkClasses('/personality', isMobile)}>
        診断テスト
      </Link>
      <Link href="/personality/types" className={navLinkClasses('/personality/types', isMobile)}>
        性癖タイプ
      </Link>
      {resultType && (
        <Link href={`/personality/result/${resultType}`} className={navLinkClasses('/personality/result', isMobile)}>
          あなたの結果
        </Link>
      )}
    </>
  );

  return (
    <header className="fixed top-0 left-0 w-full p-4 z-50 bg-white/30 backdrop-blur-lg border-b border-white/40">
      <div className="container mx-auto flex justify-between items-center">
        <Link href="/" aria-label="ホームへ移動" className="inline-flex">
          <Image src="/seiheki_lab.png" alt="Seiheki Lab Logo" width={120} height={40} priority draggable="false" />
        </Link>

        {/* Desktop Nav */}
        <nav className="hidden sm:flex items-center space-x-1 p-1">
          {navLinks()}
        </nav>

        {/* Mobile Nav Button */}
        <div className="sm:hidden">
          <button onClick={() => setIsMenuOpen(!isMenuOpen)} className="text-gray-800">
            {isMenuOpen ? <X size={28} /> : <Menu size={28} />}
          </button>
        </div>
      </div>

      {/* Mobile Menu Panel */}
      <AnimatePresence>
        {isMenuOpen && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="sm:hidden absolute top-full left-0 w-full bg-white/80 backdrop-blur-xl shadow-lg"
          >
            <nav className="flex flex-col space-y-2 p-4">
              {navLinks(true)}
            </nav>
          </motion.div>
        )}
      </AnimatePresence>
    </header>
  );
}