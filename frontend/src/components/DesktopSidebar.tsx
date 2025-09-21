'use client';

import Image from 'next/image';
import { useEffect, useState } from 'react';
import { useRouter, usePathname } from 'next/navigation';
import { Home as HomeIcon, Sparkles, BarChart2, Brain, Heart, User, UserPlus, LogOut } from 'lucide-react';
import { supabase } from '@/lib/supabase';

export default function DesktopSidebar() {
  const router = useRouter();
  const pathname = usePathname();
  const [isLoggedIn, setIsLoggedIn] = useState(false);

  useEffect(() => {
    (async () => {
      const { data: { session } } = await supabase.auth.getSession();
      setIsLoggedIn(Boolean(session?.user));
    })();
    const { data: listener } = supabase.auth.onAuthStateChange((_e, session) => {
      setIsLoggedIn(Boolean(session?.user));
    });
    return () => listener.subscription.unsubscribe();
  }, []);

  const openLogin = () => {
    try {
      window.dispatchEvent(new Event('open-auth-modal'));
    } catch {}
  };

  const handleLogout = async () => {
    try { await supabase.auth.signOut(); } catch {}
  };

  const NavButton = ({
    label,
    icon: Icon,
    href,
  }: { label: string; icon: React.ComponentType<{ size?: number; className?: string }>; href: string }) => (
    <button
      className={`w-full flex items-center gap-3 px-3 py-2 rounded-md text-sm transition-colors ${pathname === href ? 'bg-white text-gray-900' : 'text-gray-800 hover:bg-white/70'}`}
      onClick={() => router.push(href)}
    >
      <Icon size={18} className="shrink-0" />
      <span>{label}</span>
    </button>
  );

  return (
    <aside className="hidden sm:block fixed left-0 top-0 h-screen w-56 bg-white/80 backdrop-blur-md border-r border-white/30 shadow-md text-gray-800 z-40">
      <div className="h-full flex flex-col">
        <div className="px-4 py-4 border-b border-white/30">
          <div className="flex items-center gap-2">
            <Image src="/seiheki_lab.png" alt="Seiheki Lab Logo" width={120} height={40} priority draggable="false" />
          </div>
        </div>
        <nav className="flex-1 p-3 space-y-1">
          <NavButton label="ホーム" icon={HomeIcon} href="/" />
          <NavButton label="AIレコメンド" icon={Sparkles} href="/ai-recommend" />
          <NavButton label="性癖分析" icon={BarChart2} href="/analysis-results" />
          <NavButton label="性癖パーソナリティ診断" icon={Brain} href="/personality" />
          <NavButton label="お気に入り" icon={Heart} href="/likes" />
          {isLoggedIn && <NavButton label="アカウント管理" icon={User} href="/account-management" />}
        </nav>
        <div className="p-3 border-t border-white/30">
          {isLoggedIn ? (
            <button onClick={handleLogout} className="w-full flex items-center gap-2 px-3 py-2 text-left rounded-md text-sm text-gray-800 hover:bg-white/70">
              <LogOut size={18} />
              <span>ログアウト</span>
            </button>
          ) : (
            <button onClick={openLogin} className="w-full flex items-center gap-2 px-3 py-2 text-left rounded-md text-sm text-white bg-[#FF6B81] hover:brightness-95">
              <UserPlus size={18} />
              <span>ログイン / 新規登録</span>
            </button>
          )}
        </div>
      </div>
    </aside>
  );
}

