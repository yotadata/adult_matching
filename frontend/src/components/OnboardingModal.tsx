'use client';

import { useEffect, useState } from 'react';
import { supabase } from '@/lib/supabase';
import { trackEvent } from '@/lib/analytics';

type Tag = { id: string; name: string; video_count: number };
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
      // ① タグとグループを取得
      const { data: tagData, error: tagError } = await supabase
        .from('tags')
        .select('id, name, tag_group_id, tag_groups!inner(id, name, show_in_ui)')
        .eq('tag_groups.show_in_ui', true);

      // ② tag_video_counts ビューからタグ別動画数を取得
      const { data: countData, error: countError } = await supabase
        .from('tag_video_counts')
        .select('tag_id, video_count');

      if (!tagError && !countError && tagData) {
        // カウントを Map に変換
        const countMap = new Map<string, number>();
        for (const row of (countData ?? []) as { tag_id: string; video_count: number }[]) {
          countMap.set(row.tag_id, row.video_count);
        }

        // グループごとに集約
        const groupMap = new Map<string, TagGroup>();
        for (const row of tagData as unknown as {
          id: string; name: string; tag_group_id: string;
          tag_groups: { id: string; name: string };
        }[]) {
          const gId = row.tag_groups.id;
          const gName = row.tag_groups.name;
          const count = countMap.get(row.id) ?? 0;
          if (!groupMap.has(gId)) groupMap.set(gId, { id: gId, name: gName, tags: [] });
          groupMap.get(gId)!.tags.push({ id: row.id, name: row.name, video_count: count });
        }

        // カテゴリ内を人気順・上位15件に絞り、グループをトップタグの動画数順に並べる
        const sorted = Array.from(groupMap.values())
          .map((g) => ({
            ...g,
            tags: g.tags.sort((a, b) => b.video_count - a.video_count),
          }))
          .filter((g) => g.tags.length > 0)
          .sort((a, b) => b.tags[0].video_count - a.tags[0].video_count);
        setGroups(sorted);
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
                        className={`px-3 py-1.5 rounded-full text-sm font-medium transition-all flex items-center gap-1.5 ${
                          isSelected
                            ? 'bg-violet-600 text-white border border-violet-500'
                            : 'bg-[#161b22] text-[#8b949e] border border-[#30363d] hover:border-violet-500/50 hover:text-[#e6edf3]'
                        }`}
                      >
                        {tag.name}
                        <span className={`text-[10px] ${isSelected ? 'text-violet-200' : 'text-[#484f58]'}`}>
                          {tag.video_count >= 1000
                            ? `${Math.floor(tag.video_count / 1000)}k`
                            : tag.video_count}
                        </span>
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
