'use client';

import Link from 'next/link';
import { usePathname, useRouter, useSearchParams } from 'next/navigation';
import { useState, useEffect, useRef, Suspense } from 'react';
import {
  LayoutGrid, Layers, Heart, Tag, Users, X, UserCircle,
  Snail, Rabbit, Cat, Dog, Bird, Crown,
  type LucideIcon,
} from 'lucide-react';
import LikedVideosDrawer from '@/components/LikedVideosDrawer';
import AccountManagementDrawer from '@/components/AccountManagementDrawer';
import Image from 'next/image';
import { supabase } from '@/lib/supabase';

type Level = {
  min: number;
  max: number;
  label: string;
  Icon: LucideIcon;
  color: string;
  iconColor: string;
  borderColor: string;
};

const LEVELS: Level[] = [
  { min: 0,   max: 1,   label: '未診断',       Icon: Snail,   color: 'from-gray-500 to-gray-400',     iconColor: 'text-gray-400',   borderColor: 'rgba(156,163,175,0.4)' },
  { min: 1,   max: 10,  label: '学習中',       Icon: Rabbit,  color: 'from-blue-500 to-cyan-400',     iconColor: 'text-cyan-400',   borderColor: 'rgba(34,211,238,0.4)'  },
  { min: 10,  max: 30,  label: '性癖覚醒中',   Icon: Cat,     color: 'from-yellow-500 to-amber-400',  iconColor: 'text-amber-400',  borderColor: 'rgba(251,191,36,0.4)'  },
  { min: 30,  max: 100, label: '性癖確立',     Icon: Dog,     color: 'from-orange-500 to-red-400',    iconColor: 'text-orange-400', borderColor: 'rgba(251,146,60,0.4)'  },
  { min: 100, max: 200, label: '性癖マスター', Icon: Bird,    color: 'from-violet-500 to-purple-400', iconColor: 'text-violet-400', borderColor: 'rgba(167,139,250,0.4)' },
  { min: 200, max: 400, label: '変態紳士',     Icon: Crown,   color: 'from-yellow-400 to-pink-400',   iconColor: 'text-yellow-400', borderColor: 'rgba(250,204,21,0.4)'  },
];

function getLevel(count: number): Level {
  return [...LEVELS].reverse().find((l) => count >= l.min) ?? LEVELS[0];
}

function getProgress(count: number): number {
  const lv = getLevel(count);
  const range = lv.max - lv.min;
  return Math.min(((count - lv.min) / range) * 100, 100);
}

function HeartParticle({ id, onDone }: { id: number; onDone: (id: number) => void }) {
  useEffect(() => {
    const t = setTimeout(() => onDone(id), 900);
    return () => clearTimeout(t);
  }, [id, onDone]);
  return (
    <div
      className="pointer-events-none fixed z-[9999] flex items-center gap-0.5 text-pink-400 text-sm font-bold select-none animate-float-heart"
      style={{ right: 72, bottom: 60 }}
    >
      <Heart size={14} fill="currentColor" />
      <span>+1</span>
    </div>
  );
}

type InsightTag = { id: string; name: string; cnt: number };
type InsightPerformer = { id: string; name: string; cnt: number };

function LevelUpOverlay({ level, likeCount, onDone, onOpenList }: { level: Level; likeCount: number; onDone: () => void; onOpenList: () => void }) {
  const { Icon } = level;
  const [tags, setTags] = useState<InsightTag[]>([]);
  const [performers, setPerformers] = useState<InsightPerformer[]>([]);

  useEffect(() => {
    (async () => {
      const [{ data: t }, { data: p }] = await Promise.all([
        supabase.rpc('get_user_liked_tags'),
        supabase.rpc('get_user_liked_performers'),
      ]);
      setTags(((t as InsightTag[] | null) ?? []).slice(0, 3));
      setPerformers(((p as InsightPerformer[] | null) ?? []).slice(0, 3));
    })();
  }, []);

  return (
    <div className="fixed inset-0 z-[9998] flex items-center justify-center bg-black/60 backdrop-blur-sm px-4">
      <div className="bg-[#0d1117] border border-violet-500/60 rounded-2xl w-full max-w-sm shadow-2xl shadow-violet-500/20 overflow-hidden">
        {/* ヘッダー */}
        <div className="relative px-6 pt-6 pb-4 text-center">
          <button onClick={onDone} className="absolute top-3 right-3 text-gray-500 hover:text-white transition-colors">
            <X size={18} />
          </button>
          <div className={`flex justify-center mb-2 ${level.iconColor}`}>
            <Icon size={40} strokeWidth={1.5} />
          </div>
          <div className="text-white text-lg font-extrabold tracking-wide">LEVEL UP!</div>
          <div className={`bg-gradient-to-r ${level.color} bg-clip-text text-transparent text-base font-bold`}>
            {level.label}
          </div>
        </div>

        {/* インサイト */}
        <div className="px-6 pb-4 space-y-4 border-t border-white/10 pt-4">
          <div className="text-center">
            <span className="text-gray-400 text-xs">累計いいね数</span>
            <div className="text-white text-3xl font-extrabold">{likeCount.toLocaleString('ja-JP')}</div>
          </div>
          {tags.length > 0 && (
            <div>
              <p className="text-gray-400 text-[11px] mb-2 flex items-center gap-1"><Tag size={10} />よく見るタグ</p>
              <div className="flex flex-wrap gap-1.5">
                {tags.map((tag) => (
                  <span key={tag.id} className="px-2.5 py-1 rounded-full bg-white/10 text-white text-xs font-semibold">
                    {tag.name} <span className="text-gray-400 font-normal">{tag.cnt}</span>
                  </span>
                ))}
              </div>
            </div>
          )}
          {performers.length > 0 && (
            <div>
              <p className="text-gray-400 text-[11px] mb-2 flex items-center gap-1"><Users size={10} />お気に入り女優</p>
              <div className="flex flex-wrap gap-1.5">
                {performers.map((p) => (
                  <span key={p.id} className="px-2.5 py-1 rounded-full bg-pink-500/20 text-pink-300 text-xs font-semibold">
                    {p.name} <span className="text-pink-400/60 font-normal">{p.cnt}</span>
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* CTA */}
        <div className="px-6 pb-6">
          <button
            onClick={() => { onDone(); onOpenList(); }}
            className="w-full py-3 rounded-xl bg-violet-600 hover:bg-violet-500 text-white text-sm font-bold transition-colors flex items-center justify-center gap-2"
          >
            <Heart size={14} fill="currentColor" />
            いいねした作品を見る
          </button>
        </div>
      </div>
    </div>
  );
}

function BrowseTabBarInner() {
  const pathname = usePathname();
  const router = useRouter();
  const searchParams = useSearchParams();
  const isGrid = pathname.startsWith('/grid');
  const isDebug = searchParams.get('debug') === '1';
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);
  const [isAccountOpen, setIsAccountOpen] = useState(false);
  const [likeCount, setLikeCount] = useState<number | null>(null);
  const [isLoggedIn, setIsLoggedIn] = useState<boolean | null>(null);
  const [particles, setParticles] = useState<number[]>([]);
  const [levelUpLevel, setLevelUpLevel] = useState<Level | null>(null);
  const prevLevelRef = useRef<Level | null>(null);
  const particleIdRef = useRef(0);

  useEffect(() => {
    const handler = () => setIsDrawerOpen(true);
    window.addEventListener('open-liked-drawer', handler);
    return () => window.removeEventListener('open-liked-drawer', handler);
  }, []);

  useEffect(() => {
    const fetchCount = async () => {
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) { setLikeCount(0); setIsLoggedIn(false); return; }
      setIsLoggedIn(true);
      const { count } = await supabase
        .from('user_video_decisions')
        .select('*', { count: 'exact', head: true })
        .eq('user_id', user.id)
        .in('decision_type', ['swipe_like', 'grid_like']);
      const c = count ?? 0;
      setLikeCount(c);
      prevLevelRef.current = getLevel(c);
    };
    fetchCount();

    const { data: authListener } = supabase.auth.onAuthStateChange((_e, session) => {
      setIsLoggedIn(Boolean(session?.user));
      if (!session?.user) setLikeCount(0);
    });

    const handler = () => {
      setParticles((prev) => [...prev, ++particleIdRef.current]);
      setLikeCount((prev) => {
        const next = (prev ?? 0) + 1;
        const newLevel = getLevel(next);
        if (prevLevelRef.current && newLevel.min !== prevLevelRef.current.min) {
          setLevelUpLevel(newLevel);
        }
        prevLevelRef.current = newLevel;
        return next;
      });
    };
    window.addEventListener('like-added', handler);
    return () => {
      window.removeEventListener('like-added', handler);
      authListener.subscription.unsubscribe();
    };
  }, []);

  void isDebug;
  void router;

  const lv = likeCount !== null ? getLevel(likeCount) : null;
  const progress = likeCount !== null ? getProgress(likeCount) : 0;
  const nextLv = lv ? LEVELS[LEVELS.indexOf(lv) + 1] : null;

  return (
    <>
      {particles.map((id) => (
        <HeartParticle key={id} id={id} onDone={(rid) => setParticles((p) => p.filter((x) => x !== rid))} />
      ))}
      {levelUpLevel && <LevelUpOverlay level={levelUpLevel} likeCount={likeCount ?? 0} onDone={() => setLevelUpLevel(null)} onOpenList={() => setIsDrawerOpen(true)} />}

      <div className="fixed top-0 left-0 right-0 z-40 bg-[#0d1117]/95 backdrop-blur flex flex-col">
        {/* メインヘッダー: 左(ロゴ+タブ) / 中央(レベル) / 右(アクション) */}
        <div className="grid grid-cols-3 items-center px-3 h-11">
          {/* 左: ロゴ＋タブ */}
          <div className="flex items-center gap-1.5">
            <Link href="/" className="flex-shrink-0">
              <Image src="/seiheki_lab.png" alt="性癖ラボ" width={72} height={22} className="h-5 w-auto" />
            </Link>
            <Link
              href="/grid"
              className={`flex items-center justify-center w-7 h-7 rounded-md transition-colors ${
                isGrid ? 'bg-violet-600 text-white' : 'text-[#8b949e] hover:text-[#e6edf3] hover:bg-[#161b22]'
              }`}
              title="グリッド"
            >
              <LayoutGrid size={14} />
            </Link>
            <Link
              href="/swipe"
              className={`flex items-center justify-center w-7 h-7 rounded-md transition-colors ${
                !isGrid ? 'bg-violet-600 text-white' : 'text-[#8b949e] hover:text-[#e6edf3] hover:bg-[#161b22]'
              }`}
              title="スワイプ"
            >
              <Layers size={14} />
            </Link>
          </div>

          {/* 中央: レベル表示 */}
          <div className="flex items-center justify-center">
            {lv && (
              <div
                className="flex items-center gap-1 px-2 py-0.5 rounded-full border bg-[#0d1117]/80"
                style={{ borderColor: lv.borderColor }}
              >
                <lv.Icon size={12} className={lv.iconColor} strokeWidth={2} />
                <span className={`text-[10px] font-extrabold bg-gradient-to-r ${lv.color} bg-clip-text text-transparent whitespace-nowrap`}>
                  {lv.label}
                </span>
                <span className="text-[10px] text-[#8b949e] whitespace-nowrap">
                  <Heart size={8} className="inline mr-0.5 text-pink-400" fill="currentColor" />
                  <span className="text-[#e6edf3] font-bold">{likeCount ?? 0}</span>
                  {nextLv && <span>/{nextLv.min}</span>}
                </span>
              </div>
            )}
          </div>

          {/* 右: ログイン時=リスト+設定、未ログイン時=ログインボタン */}
          <div className="flex justify-end items-center gap-0.5">
            {isLoggedIn === false ? (
              <button
                onClick={() => window.dispatchEvent(new Event('open-auth-modal'))}
                className="flex items-center gap-1 px-2.5 py-1 rounded-md bg-violet-600 hover:bg-violet-500 text-white transition-colors text-[11px] font-bold"
              >
                ログイン
              </button>
            ) : (
              <>
                <button
                  onClick={() => setIsDrawerOpen(true)}
                  className="flex items-center justify-center w-7 h-7 rounded-md text-[#8b949e] hover:text-pink-400 hover:bg-[#161b22] transition-colors"
                  title="気になるリスト"
                >
                  <Heart size={14} />
                </button>
                <button
                  onClick={() => setIsAccountOpen(true)}
                  className="flex items-center justify-center w-7 h-7 rounded-md text-[#8b949e] hover:text-[#e6edf3] hover:bg-[#161b22] transition-colors"
                  title="設定"
                >
                  <UserCircle size={14} />
                </button>
              </>
            )}
          </div>
        </div>

        {/* 光るゲージバー（ヘッダー下端ライン） */}
        <div className="relative h-[3px] w-full bg-[#30363d]">
          {lv && (
            <div
              className={`absolute inset-y-0 left-0 bg-gradient-to-r ${lv.color} transition-all duration-700 ease-out`}
              style={{ width: `${progress}%`, boxShadow: '0 0 10px 2px rgba(167,139,250,0.6)' }}
            />
          )}
        </div>
      </div>

      <LikedVideosDrawer isOpen={isDrawerOpen} onClose={() => setIsDrawerOpen(false)} />
      <AccountManagementDrawer isOpen={isAccountOpen} onClose={() => setIsAccountOpen(false)} />
    </>
  );
}

export default function BrowseTabBar() {
  return (
    <Suspense>
      <BrowseTabBarInner />
    </Suspense>
  );
}
