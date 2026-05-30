'use client';

import { useEffect, useState } from 'react';
import { supabase } from '@/lib/supabase';
import { trackEvent } from '@/lib/analytics';

type Tag = { id: string; name: string };
type TagGroup = { id: string; name: string; tags: Tag[] };

const MIN_SELECT = 3;

interface Props {
  onComplete: (tagIds: string[]) => void;
}

export default function OnboardingModal({ onComplete }: Props) {
  const [groups, setGroups] = useState<TagGroup[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    (async () => {
      // show_in_ui=true のカテゴリとそのタグを取得
      const { data, error } = await supabase
        .from('tag_groups')
        .select('id, name, tags(id, name)')
        .eq('show_in_ui', true)
        .order('name');

      if (!error && data) {
        // タグが存在するカテゴリのみ表示
        const filtered = (data as unknown as TagGroup[]).filter(
          (g) => g.tags && g.tags.length > 0
        );
        setGroups(filtered);
      }
      setLoading(false);
    })();
  }, []);

  const toggle = (tagId: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(tagId)) next.delete(tagId);
      else next.add(tagId);
      return next;
    });
  };

  const handleComplete = () => {
    const tagIds = Array.from(selected);
    localStorage.setItem('onboarding_done', '1');
    localStorage.setItem('onboarding_tags', JSON.stringify(tagIds));
    trackEvent('onboarding_complete', { tag_count: tagIds.length });
    onComplete(tagIds);
  };

  const handleSkip = () => {
    localStorage.setItem('onboarding_done', '1');
    localStorage.setItem('onboarding_tags', JSON.stringify([]));
    trackEvent('onboarding_skip', {});
    onComplete([]);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ background: 'rgba(0,0,0,0.85)', backdropFilter: 'blur(4px)' }}>
      <div className="w-full max-w-lg max-h-[85vh] flex flex-col rounded-2xl border border-[#30363d] bg-[#0d1117] overflow-hidden">

        {/* ヘッダー */}
        <div className="px-6 pt-6 pb-4 border-b border-[#21262d] shrink-0">
          <p className="text-xs font-bold tracking-widest uppercase text-violet-400 mb-2">はじめに</p>
          <h2 className="text-xl font-black text-[#e6edf3] mb-1">好きなジャンルを選んでください</h2>
          <p className="text-sm text-[#8b949e]">
            選択した内容をもとに最初のおすすめを作ります。{MIN_SELECT}つ以上選んでください。
          </p>
        </div>

        {/* タグ一覧（スクロール可） */}
        <div className="flex-1 overflow-y-auto px-6 py-4 space-y-5">
          {loading ? (
            <div className="flex items-center justify-center py-12">
              <div className="w-6 h-6 border-2 border-violet-500 border-t-transparent rounded-full animate-spin" />
            </div>
          ) : (
            groups.map((group) => (
              <div key={group.id}>
                <p className="text-xs font-semibold text-[#656d76] uppercase tracking-wider mb-2">
                  {group.name}
                </p>
                <div className="flex flex-wrap gap-2">
                  {group.tags.map((tag) => {
                    const isSelected = selected.has(tag.id);
                    return (
                      <button
                        key={tag.id}
                        onClick={() => toggle(tag.id)}
                        className={`px-3 py-1.5 rounded-full text-sm font-medium transition-all ${
                          isSelected
                            ? 'bg-violet-600 text-white border border-violet-500'
                            : 'bg-[#161b22] text-[#8b949e] border border-[#30363d] hover:border-violet-500/50 hover:text-[#e6edf3]'
                        }`}
                      >
                        {tag.name}
                      </button>
                    );
                  })}
                </div>
              </div>
            ))
          )}
        </div>

        {/* フッター */}
        <div className="px-6 py-4 border-t border-[#21262d] shrink-0">
          <div className="flex items-center justify-between gap-3">
            <button
              onClick={handleSkip}
              className="text-sm text-[#656d76] hover:text-[#8b949e] transition-colors"
            >
              スキップ
            </button>
            <button
              onClick={handleComplete}
              disabled={selected.size < MIN_SELECT}
              className={`flex-1 py-3 rounded-full font-black text-sm transition-all ${
                selected.size >= MIN_SELECT
                  ? 'bg-violet-600 hover:bg-violet-500 text-white'
                  : 'bg-[#21262d] text-[#484f58] cursor-not-allowed'
              }`}
            >
              {selected.size >= MIN_SELECT
                ? `${selected.size}件で始める`
                : `あと${MIN_SELECT - selected.size}つ選んでください`}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
