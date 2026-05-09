'use client';

import Image from 'next/image';
import Link from 'next/link';
import { useEffect, useState } from 'react';
import { useRouter, usePathname } from 'next/navigation';
import { Home as HomeIcon, Sparkles, BarChart2, List, UserPlus, Settings, Info, Mail, FlaskConical } from 'lucide-react';
import { supabase } from '@/lib/supabase';
import { useDecisionCount } from '@/hooks/useDecisionCount';

export default function DesktopSidebar() {
  const router = useRouter();
  const pathname = usePathname();
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const { decisionCount } = useDecisionCount();
  const personalizeTarget = Number(process.env.NEXT_PUBLIC_PERSONALIZE_TARGET || 20);
  
  const accentGradient = 'linear-gradient(135deg, #7c3aed 0%, #8b5cf6 100%)';

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
    newTab = false,
  }: { label: string; icon: React.ComponentType<{ size?: number; className?: string }>; href: string; disabled?: boolean; newTab?: boolean }) => (
    <button
      disabled={disabled}
      className={`w-full min-w-0 flex items-center gap-3 px-3 py-2 rounded-md text-sm transition-colors ${disabled ? 'opacity-40 cursor-not-allowed text-[#656d76]' : pathname === href ? 'bg-[#21262d] text-[#e6edf3]' : 'text-[#8b949e] hover:bg-[#21262d] hover:text-[#e6edf3]'}`}
      onClick={() => { if (!disabled) { if (newTab) window.open(href, '_blank', 'noopener,noreferrer'); else router.push(href); } }}
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
      className={`w-full text-left min-w-0 px-3 py-2 rounded-md text-sm transition-colors ${disabled ? 'opacity-40 cursor-not-allowed text-[#656d76]' : pathname === href ? 'bg-[#21262d] text-[#e6edf3]' : 'text-[#8b949e] hover:bg-[#21262d] hover:text-[#e6edf3]'}`}
      onClick={() => { if (!disabled) router.push(href); }}
      title={label}
    >
      <div className="flex items-center gap-3">
        <Icon size={18} className="shrink-0" />
        <span className="truncate">{label}</span>
      </div>
      <div className="mt-2 pr-1">
        <div className="w-full h-2 bg-[#30363d] rounded-full overflow-hidden">
          <div className="h-full rounded-full" style={{ width: `${Math.min(Math.max(progress, 0), 1) * 100}%`, background: 'linear-gradient(90deg, #7c3aed 0%, #8b5cf6 50%, #a78bfa 100%)' }} />
        </div>
        <div className="mt-1 text-[10px] text-[#656d76] text-right">{caption}</div>
      </div>
    </button>
  );

  const remainingSwipes = Math.max(personalizeTarget - decisionCount, 0);
  const caption = remainingSwipes === 0 ? 'パーソナライズ完了' : `パーソナライズまであと${remainingSwipes}枚`;

  return (
    <aside className="hidden sm:block fixed left-0 top-0 h-screen w-56 bg-[#161b22] border-r border-[#30363d] z-40">
      <div className="h-full flex flex-col">
        <div className="px-4 py-4 border-b border-[#30363d]">
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
                style={{ background: accentGradient }}
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
            label="さがす"
            icon={Sparkles}
            href="/search"
            disabled={false}
            progress={decisionCount / personalizeTarget}
            caption={caption}
          />
          <NavButton label="あなたの性癖" icon={BarChart2} href="/insights" disabled={!isLoggedIn} />
        </nav>
        <div className="px-3 pt-2 pb-10">
          <div className="border-t border-[#30363d] my-2" />
          <div className="space-y-1 text-sm">
            <NavButton label="偏愛16診断" icon={FlaskConical} href="/quiz" disabled={false} newTab />
          </div>
          <div className="border-t border-[#30363d] my-2" />
          <div className="space-y-1 text-sm">
            <NavButton label="お問い合わせ" icon={Mail} href="/contact" disabled={!isLoggedIn} />
            <NavButton label="アカウント設定" icon={Settings} href="/account-management" disabled={!isLoggedIn} />
            <NavButton label="このサイトについて" icon={Info} href="/about" disabled={false} />
          </div>
        </div>
      </div>
    </aside>
  );
}
