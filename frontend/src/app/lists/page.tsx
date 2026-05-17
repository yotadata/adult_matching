'use client';

import { useEffect, useState } from 'react';
import { supabase } from '@/lib/supabase';
import { Heart } from 'lucide-react';
import LikedVideosDrawer from '@/components/LikedVideosDrawer';

type ListEntry = {
  id: string;
  title: string;
  description: string;
  href?: string; // Make href optional
  onClick?: () => void; // Add onClick handler
  icon: React.ComponentType<{ size?: number }>;
  accent: string;
  count?: number;
};

export default function ListsPage() {
  const [likedCount, setLikedCount] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [isAuthenticated, setIsAuthenticated] = useState<boolean | null>(null);
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);

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
      description: 'AIで探すやスワイプで「気になる」とした作品をまとめて見返せます。',
      onClick: () => setIsDrawerOpen(true),
      icon: Heart,
      accent: 'from-rose-500/80 to-pink-400/80',
      count: likedCount ?? undefined,
    },
  ];

  const renderListItem = (list: ListEntry) => {
    const Icon = list.icon;
    const content = (
      <>
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
      </>
    );

    if (list.onClick) {
      return (
        <button
          key={list.id}
          onClick={list.onClick}
          className="group rounded-2xl border border-white/30 bg-white/90 text-gray-900 hover:bg-white transition-all duration-200 p-6 flex flex-col gap-4 shadow-lg text-left"
        >
          {content}
        </button>
      );
    }

    // Fallback to Link if href is provided
    return (
      <a
        key={list.id}
        href={list.href}
        className="group rounded-2xl border border-white/30 bg-white/90 text-gray-900 hover:bg-white transition-all duration-200 p-6 flex flex-col gap-4 shadow-lg"
      >
        {content}
      </a>
    );
  };

  return (
    <>
      <main className="w-full min-h-screen px-0 sm:px-4 py-8 text-white">
        <section className="w-full max-w-5xl mx-auto rounded-2xl bg-white/20 backdrop-blur-xl border border-white/30 shadow-[0_20px_60px_rgba(0,0,0,0.25)] p-4 sm:p-8 flex flex-col gap-8">
          <header className="space-y-3">
            <h1 className="text-xl sm:text-2xl font-extrabold tracking-tight">あなたの気になるリスト</h1>
            <p className="text-sm text-white/80 mt-2">
              「気になる」とした作品や今後追加予定のオリジナル気になるリストをここから管理できます。
            </p>
            {isAuthenticated === false ? (
              <div className="rounded-xl border border-white/15 bg-white/5 px-4 py-3 text-xs text-white/70">
                ログインして気になるリストを作成すると、ここに一覧が表示されます。
              </div>
            ) : null}
          </header>

          <section className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {lists.map(renderListItem)}
          </section>
        </section>
      </main>
      <LikedVideosDrawer isOpen={isDrawerOpen} onClose={() => setIsDrawerOpen(false)} />
    </>
  );
}
