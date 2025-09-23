'use client';

import Image from 'next/image';
import { useEffect, useState } from 'react';
import { useRouter, usePathname } from 'next/navigation';
import { Home as HomeIcon, Sparkles, BarChart2, Brain, User, UserPlus, LogOut } from 'lucide-react';
import { supabase } from '@/lib/supabase';

export default function DesktopSidebar() {
  const router = useRouter();
  const pathname = usePathname();
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [decisionCount, setDecisionCount] = useState<number>(0);
  const personalizeTarget = Number(process.env.NEXT_PUBLIC_PERSONALIZE_TARGET || 20);
  const diagnosisTarget = Number(process.env.NEXT_PUBLIC_DIAGNOSIS_TARGET || 30);
  const isHome = pathname === '/';
  const nonHomeGradient = 'linear-gradient(90deg, #ADB4E3 0%, #C8BAE3 33.333%, #F7BECE 66.666%, #F9B1C4 100%)';
  const homeGradient = 'linear-gradient(90deg, #C4C8E3 0%, #D7D1E3 33.333%, #F7D7E0 66.666%, #F9C9D6 100%)';

  useEffect(() => {
    const loadAuthAndCount = async () => {
      const { data: { session } } = await supabase.auth.getSession();
      const loggedIn = Boolean(session?.user);
      setIsLoggedIn(loggedIn);
      if (!loggedIn) {
        try {
          const raw = localStorage.getItem('guest_decisions_v1');
          const arr = raw ? JSON.parse(raw) : [];
          setDecisionCount(Array.isArray(arr) ? arr.length : 0);
        } catch {
          setDecisionCount(0);
        }
      } else {
        const { count } = await supabase
          .from('user_video_decisions')
          .select('*', { count: 'exact', head: true })
          .eq('user_id', session!.user!.id);
        setDecisionCount(count || 0);
      }
    };

    loadAuthAndCount();

    const { data: listener } = supabase.auth.onAuthStateChange((_e, session) => {
      setIsLoggedIn(Boolean(session?.user));
      // Recompute count on auth state change
      loadAuthAndCount();
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
    disabled = false,
  }: { label: string; icon: React.ComponentType<{ size?: number; className?: string }>; href: string; disabled?: boolean }) => (
    <button
      disabled={disabled}
      className={`w-full min-w-0 flex items-center gap-3 px-3 py-2 rounded-md text-sm transition-colors ${disabled ? 'opacity-50 cursor-not-allowed text-gray-400' : pathname === href ? 'bg-white text-gray-900' : 'text-gray-800 hover:bg-white/70'}`}
      onClick={() => { if (!disabled) router.push(href); }}
      title={label}
    >
      <Icon size={18} className="shrink-0" />
      <span className="truncate">{label}</span>
    </button>
  );

  const NavButtonWithGauge = ({
    label,
    icon: Icon,
    href,
    disabled = false,
    progress,
    caption,
  }: {
    label: string;
    icon: React.ComponentType<{ size?: number; className?: string }>;
    href: string;
    disabled?: boolean;
    progress: number; // 0..1
    caption: string;
  }) => (
    <button
      disabled={disabled}
      className={`w-full text-left min-w-0 px-3 py-2 rounded-md text-sm transition-colors ${disabled ? 'opacity-50 cursor-not-allowed text-gray-400' : pathname === href ? 'bg-white text-gray-900' : 'text-gray-800 hover:bg-white/70'}`}
      onClick={() => { if (!disabled) router.push(href); }}
      title={label}
    >
      <div className="flex items-center gap-3">
        <Icon size={18} className="shrink-0" />
        <span className="truncate">{label}</span>
      </div>
      <div className="mt-2 pr-1">
        <div className="w-full h-2 bg-gray-200 rounded-full overflow-hidden">
          <div className="h-full rounded-full" style={{ width: `${Math.min(Math.max(progress, 0), 1) * 100}%`, background: 'linear-gradient(90deg, #ADB4E3 0%, #C8BAE3 33.333%, #F7BECE 66.666%, #F9B1C4 100%)' }} />
        </div>
        <div className="mt-1 text-[10px] text-gray-600 text-right">{caption}</div>
      </div>
    </button>
  );

  return (
    <aside className="hidden sm:block fixed left-0 top-0 h-screen w-56 bg-white/80 backdrop-blur-md border-r border-white/30 shadow-md text-gray-800 z-40">
      <div className="h-full flex flex-col">
        <div className="px-4 py-4 border-b border-white/30">
          <div className="flex items-center gap-2">
            <Image src="/seiheki_lab.png" alt="Seiheki Lab Logo" width={120} height={40} priority draggable="false" />
          </div>
          {!isLoggedIn && (
            <div className="mt-3">
              <button
                onClick={openLogin}
                className="w-full flex items-center gap-2 px-3 py-2 text-left rounded-md text-sm font-bold text-white shadow-md hover:shadow-lg transition-shadow"
                style={{ background: nonHomeGradient }}
              >
                <UserPlus size={18} />
                <span className="truncate">ログイン / 新規登録</span>
              </button>
            </div>
          )}
        </div>
        <nav className="flex-1 p-3 space-y-1">
          <NavButton label="ホーム" icon={HomeIcon} href="/" disabled={!isLoggedIn} />
          <NavButtonWithGauge
            label="AIレコメンド"
            icon={Sparkles}
            href="/ai-recommend"
            disabled={!isLoggedIn}
            progress={decisionCount / personalizeTarget}
            caption={`パーソナライズまであと${Math.max(personalizeTarget - decisionCount, 0)}枚`}
          />
          <NavButton label="性癖分析" icon={BarChart2} href="/analysis-results" disabled={!isLoggedIn} />
          {/* 性癖パーソナリティ診断: 準備中表記・disableを解除 */}
          <NavButton label="性癖パーソナリティ診断" icon={Brain} href="/personality" />
        </nav>
        {isLoggedIn && (
          <div className="p-3 border-t border-white/30">
            <button onClick={handleLogout} className="w-full flex items-center gap-2 px-3 py-2 text-left rounded-md text-sm text-gray-800 hover:bg-white/70 shadow-md hover:shadow-lg transition-shadow">
              <LogOut size={18} />
              <span className="truncate">ログアウト</span>
            </button>
          </div>
        )}
      </div>
    </aside>
  );
}
