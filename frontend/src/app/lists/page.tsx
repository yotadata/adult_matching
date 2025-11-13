'use client';

import Link from 'next/link';
import { useEffect, useState } from 'react';
import { supabase } from '@/lib/supabase';
import { Heart } from 'lucide-react';

type ListEntry = {
  id: string;
  title: string;
  description: string;
  href: string;
  icon: React.ComponentType<{ size?: number }>;
  accent: string;
  count?: number;
};

const GRADIENT = 'linear-gradient(90deg, #C4C8E3 0%, #D7D1E3 33.333%, #F7D7E0 66.666%, #F9C9D6 100%)';

export default function ListsPage() {
  const [likedCount, setLikedCount] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [isAuthenticated, setIsAuthenticated] = useState<boolean | null>(null);

  useEffect(() => {
    (async () => {
      setLoading(true);
      try {
        const { data: { user } } = await supabase.auth.getUser();
        if (!user) {
          setIsAuthenticated(false);
          setLikedCount(null);
          return;
        }
        setIsAuthenticated(true);
        const { count } = await supabase
          .from('user_video_decisions')
          .select('*', { count: 'exact', head: true })
          .eq('user_id', user.id)
          .eq('decision_type', 'like');
        setLikedCount(count ?? 0);
      } catch {
        setLikedCount(null);
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const lists: ListEntry[] = [
    {
      id: 'liked',
      title: 'いいねした作品',
      description: 'AIで探すやスワイプでLIKEした作品をまとめて見返せます。',
      href: '/lists/liked',
      icon: Heart,
      accent: 'from-rose-500/80 to-pink-400/80',
      count: likedCount ?? undefined,
    },
  ];

  return (
    <main className="min-h-screen px-4 sm:px-8 py-10 text-white" style={{ background: GRADIENT }}>
      <section className="w-full max-w-5xl mx-auto rounded-none sm:rounded-2xl bg-white/20 backdrop-blur-xl border border-white/30 shadow-[0_20px_60px_rgba(0,0,0,0.25)] px-4 sm:px-8 py-8 flex flex-col gap-8">
        <header className="space-y-3">
          <p className="text-sm uppercase tracking-[0.2em] text-white/60">Playlists</p>
          <h1 className="text-3xl sm:text-4xl font-bold">あなたの気になるリスト</h1>
          <p className="text-white/80 text-sm sm:text-base">
            いいね済みの作品や今後追加予定のオリジナル気になるリストをここから管理できます。
          </p>
          {isAuthenticated === false ? (
            <div className="rounded-xl border border-white/15 bg-white/5 px-4 py-3 text-xs text-white/70">
              ログインして気になるリストを作成すると、ここに一覧が表示されます。
            </div>
          ) : null}
        </header>

        <section className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {lists.map((list) => {
            const Icon = list.icon;
            return (
              <Link
                key={list.id}
                href={list.href}
                className="group rounded-2xl border border-white/30 bg-white/90 text-gray-900 hover:bg-white transition-all duration-200 p-6 flex flex-col gap-4 shadow-lg"
              >
                <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${list.accent} flex items-center justify-center text-white shadow-md`}>
                  <Icon size={22} />
                </div>
                <div className="space-y-2">
                  <div className="flex items-center justify-between gap-4">
                    <h2 className="text-xl font-semibold text-gray-900">{list.title}</h2>
                    {typeof list.count === 'number' ? (
                      <span className="text-sm text-gray-500">{loading ? '—' : `${list.count} 件`}</span>
                    ) : (
                      <span className="text-sm text-gray-500">{loading ? '—' : 'データなし'}</span>
                    )}
                  </div>
                  <p className="text-sm text-gray-600 leading-relaxed">
                    {list.description}
                  </p>
                </div>
                <div className="mt-auto text-sm font-semibold text-rose-500 inline-flex items-center gap-2">
                  <span>この気になるリストを開く</span>
                  <span className="transition-transform group-hover:translate-x-1">→</span>
                </div>
              </Link>
            );
          })}
        </section>
      </section>
    </main>
  );
}
