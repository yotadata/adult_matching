'use client';

import Image from 'next/image';
import Link from 'next/link';
import { useEffect, useState } from 'react';
import { useRouter, usePathname } from 'next/navigation';
import { Home as HomeIcon, Sparkles, BarChart2, List, UserPlus } from 'lucide-react';
import { supabase } from '@/lib/supabase';
import { useDecisionCount } from '@/hooks/useDecisionCount';

export default function DesktopSidebar() {
  const router = useRouter();
  const pathname = usePathname();
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const { decisionCount } = useDecisionCount();
  const personalizeTarget = Number(process.env.NEXT_PUBLIC_PERSONALIZE_TARGET || 20);
  
  const nonHomeGradient = 'linear-gradient(90deg, #ADB4E3 0%, #C8BAE3 33.333%, #F7BECE 66.666%, #F9B1C4 100%)';

  useEffect(() => {
    const loadAuth = async () => {
      const { data: { session } } = await supabase.auth.getSession();
      setIsLoggedIn(Boolean(session?.user));
    };
    loadAuth();
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

  const NavButton = ({
    label,
    icon: Icon,
    href,
    disabled = false,
  }: { label: string; icon: React.ComponentType<{ size?: number; className?: string }>; href: string; disabled?: boolean }) => (
    <button
      disabled={disabled}
      className={`w-full min-w-0 flex items-center gap-3 px-3 py-2 rounded-md text-sm transition-colors ${disabled ? 'opacity-50 cursor-not-allowed text-gray-400' : pathname === href ? 'bg-gray-100 text-gray-900' : 'text-gray-800 hover:bg-gray-100'}`}
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
      className={`w-full text-left min-w-0 px-3 py-2 rounded-md text-sm transition-colors ${disabled ? 'opacity-50 cursor-not-allowed text-gray-400' : pathname === href ? 'bg-gray-100 text-gray-900' : 'text-gray-800 hover:bg-gray-100'}`}
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

  const remainingSwipes = Math.max(personalizeTarget - decisionCount, 0);
  const caption = remainingSwipes === 0 ? 'パーソナライズ完了' : `パーソナライズまであと${remainingSwipes}枚`;

  return (
    <aside className="hidden sm:block fixed left-0 top-0 h-screen w-56 bg-white border-r border-gray-100 shadow-md text-gray-800 z-40">
      <div className="h-full flex flex-col">
        <div className="px-4 py-4 border-b border-gray-100">
          <div className="flex items-center gap-2">
            <Link href="/swipe" aria-label="スワイプへ移動" className="inline-flex">
              <Image src="/seiheki_lab.png" alt="Seiheki Lab Logo" width={120} height={40} priority draggable="false" className="cursor-pointer" />
            </Link>
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
          <NavButton label="スワイプ" icon={HomeIcon} href="/swipe" disabled={false} />
          <NavButton label="気になるリスト" icon={List} href="/lists" disabled={!isLoggedIn} />
          <NavButtonWithGauge
            label="AIで探す"
            icon={Sparkles}
            href="/search"
            disabled={false}
            progress={decisionCount / personalizeTarget}
            caption={caption}
          />
          <NavButton label="あなたの性癖" icon={BarChart2} href="/insights" disabled={!isLoggedIn} />
        </nav>
        <div className="px-3 pt-2 pb-10">
          <div className="border-t border-gray-100 my-2" />
          <div className="space-y-1 text-sm text-gray-600">
            <NavButton label="お問い合わせ" icon={Sparkles} href="/contact" disabled={!isLoggedIn} />
            <NavButton label="アカウント設定" icon={UserPlus} href="/account-management" disabled={!isLoggedIn} />
            <NavButton label="このサイトについて" icon={BarChart2} href="/about" disabled={false} />
          </div>
        </div>
      </div>
    </aside>
  );
}
