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
      description: 'あなたがLIKEした作品のアーカイブです。後から再チェックできます。',
      href: '/lists/liked',
      icon: Heart,
      accent: 'from-rose-500/80 to-pink-400/80',
      count: likedCount ?? undefined,
    },
  ];

  return (
    <main className="min-h-screen px-4 sm:px-8 py-10 bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white">
      <div className="max-w-5xl mx-auto flex flex-col gap-8">
        <header className="space-y-3">
          <p className="text-sm uppercase tracking-[0.2em] text-white/60">Lists</p>
          <h1 className="text-3xl sm:text-4xl font-bold">あなたのリスト</h1>
          <p className="text-white/80 text-sm sm:text-base">
            いいね済みの作品や今後追加予定のオリジナルリストをここから管理できます。
          </p>
          {isAuthenticated === false ? (
            <div className="rounded-xl border border-white/15 bg-white/5 px-4 py-3 text-xs text-white/70">
              ログインしてリストを作成すると、ここに一覧が表示されます。
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
                className="group rounded-2xl border border-white/10 bg-white/5 backdrop-blur hover:bg-white/10 transition-all duration-200 p-6 flex flex-col gap-4"
              >
                <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${list.accent} flex items-center justify-center text-white shadow-lg`}>
                  <Icon size={22} />
                </div>
                <div className="space-y-2">
                  <div className="flex items-center justify-between gap-4">
                    <h2 className="text-xl font-semibold">{list.title}</h2>
                    {typeof list.count === 'number' ? (
                      <span className="text-sm text-white/70">{loading ? '—' : `${list.count} 件`}</span>
                    ) : (
                      <span className="text-sm text-white/70">{loading ? '—' : 'データなし'}</span>
                    )}
                  </div>
                  <p className="text-sm text-white/70 leading-relaxed">
                    {list.description}
                  </p>
                </div>
                <div className="mt-auto text-sm font-semibold text-white/90 inline-flex items-center gap-2">
                  <span>このリストを開く</span>
                  <span className="transition-transform group-hover:translate-x-1">→</span>
                </div>
              </Link>
            );
          })}
        </section>
      </div>
    </main>
  );
}
