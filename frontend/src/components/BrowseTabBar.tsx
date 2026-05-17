'use client';

import Link from 'next/link';
import { usePathname, useRouter, useSearchParams } from 'next/navigation';
import { useState, useEffect, useRef, Suspense } from 'react';
import {
  LayoutGrid, Layers, Heart,
  Search, BookOpen, Lightbulb, Flame, Zap, Crown,
  type LucideIcon,
} from 'lucide-react';
import LikedVideosDrawer from '@/components/LikedVideosDrawer';
import Image from 'next/image';
import { supabase } from '@/lib/supabase';

type Level = {
  min: number;
  max: number;
  label: string;
  Icon: LucideIcon;
  color: string;
  iconColor: string;
};

const LEVELS: Level[] = [
  { min: 0,   max: 1,   label: '未診断',       Icon: Search,    color: 'from-gray-500 to-gray-400',    iconColor: 'text-gray-400' },
  { min: 1,   max: 10,  label: '学習中',       Icon: BookOpen,  color: 'from-blue-500 to-cyan-400',    iconColor: 'text-cyan-400' },
  { min: 10,  max: 30,  label: '性癖覚醒中',   Icon: Lightbulb, color: 'from-yellow-500 to-amber-400', iconColor: 'text-amber-400' },
  { min: 30,  max: 100, label: '性癖確立',     Icon: Flame,     color: 'from-orange-500 to-red-400',   iconColor: 'text-orange-400' },
  { min: 100, max: 200, label: '性癖マスター', Icon: Zap,       color: 'from-violet-500 to-purple-400', iconColor: 'text-violet-400' },
  { min: 200, max: 400, label: '変態紳士',     Icon: Crown,     color: 'from-yellow-400 to-pink-400',  iconColor: 'text-yellow-400' },
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

function LevelUpOverlay({ level, onDone }: { level: Level; onDone: () => void }) {
  useEffect(() => {
    const t = setTimeout(onDone, 2800);
    return () => clearTimeout(t);
  }, [onDone]);
  const { Icon } = level;
  return (
    <div className="pointer-events-none fixed inset-0 z-[9998] flex items-center justify-center">
      <div className="animate-level-up bg-[#0d1117]/90 border border-violet-500/60 rounded-2xl px-8 py-6 text-center shadow-2xl shadow-violet-500/20">
        <div className={`flex justify-center mb-3 ${level.iconColor}`}>
          <Icon size={48} strokeWidth={1.5} />
        </div>
        <div className="text-white text-xl font-extrabold tracking-wide">LEVEL UP!</div>
        <div className={`bg-gradient-to-r ${level.color} bg-clip-text text-transparent text-lg font-bold mt-1`}>
          {level.label}
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
  const [likeCount, setLikeCount] = useState<number | null>(null);
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
      if (!user) { setLikeCount(0); return; }
      const { count } = await supabase
        .from('user_video_decisions')
        .select('*', { count: 'exact', head: true })
        .eq('user_id', user.id)
        .eq('decision_type', 'like');
      const c = count ?? 0;
      setLikeCount(c);
      prevLevelRef.current = getLevel(c);
    };
    fetchCount();

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
    return () => window.removeEventListener('like-added', handler);
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
      {levelUpLevel && <LevelUpOverlay level={levelUpLevel} onDone={() => setLevelUpLevel(null)} />}

      <div className="fixed top-0 left-0 right-0 z-40 bg-[#0d1117]/95 backdrop-blur border-b border-[#30363d] flex flex-col">
        <div className="flex items-center px-3 h-12 gap-2">
          <Link href="/" className="flex-shrink-0">
            <Image src="/seiheki_lab.png" alt="性癖ラボ" width={80} height={24} className="h-6 w-auto" />
          </Link>

          <div className="flex gap-1 flex-1">
            <Link
              href="/grid"
              className={`flex items-center gap-1.5 px-4 py-1.5 rounded-lg text-sm font-bold transition-colors ${
                isGrid ? 'bg-violet-600 text-white' : 'text-[#8b949e] hover:text-[#e6edf3] hover:bg-[#161b22]'
              }`}
            >
              <LayoutGrid size={15} />
              グリッド
            </Link>
            <Link
              href="/swipe"
              className={`flex items-center gap-1.5 px-4 py-1.5 rounded-lg text-sm font-bold transition-colors ${
                !isGrid ? 'bg-violet-600 text-white' : 'text-[#8b949e] hover:text-[#e6edf3] hover:bg-[#161b22]'
              }`}
            >
              <Layers size={15} />
              スワイプ
            </Link>
          </div>

          {lv && (
            <div className="hidden sm:flex items-center gap-1.5 flex-shrink-0">
              <lv.Icon size={16} className={lv.iconColor} strokeWidth={2} />
              <div className="flex flex-col leading-none">
                <span className="text-[11px] font-bold text-[#e6edf3]">{lv.label}</span>
                <span className="text-[10px] text-[#8b949e]">
                  <Heart size={9} className="inline mr-0.5" fill="currentColor" />
                  {likeCount}{nextLv ? ` / ${nextLv.min}` : ''}
                </span>
              </div>
            </div>
          )}

          <button
            onClick={() => setIsDrawerOpen(true)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-bold text-[#8b949e] hover:text-pink-400 hover:bg-[#161b22] transition-colors flex-shrink-0"
          >
            <Heart size={15} />
            リスト
          </button>
        </div>

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
