'use client';

import { useEffect, useState, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { GripVertical, ChevronUp, ChevronDown, Save, X, Plus, Trash2, Loader2 } from 'lucide-react';
import { supabase } from '@/lib/supabase';

type Video = {
  id: string;
  title: string | null;
  thumbnail_url: string | null;
  sort_order: number | null;
  section_id: string | null;
};

type Section = {
  id: string;
  title: string | null;
  display_mode: 'ranked' | 'plain';
  sort_order: number;
};

interface Props {
  ownerUserId: string;
  listId: string;
  videos: Video[];
  sections: Section[];
  token: string;
}

export default function ListReorderMode({ ownerUserId, listId, videos, sections: initialSections, token }: Props) {
  const router = useRouter();
  const [isOwner, setIsOwner] = useState(false);
  const [isOpen, setIsOpen] = useState(false);
  const [saving, setSaving] = useState(false);

  // 編集中の状態
  const [orderedVideos, setOrderedVideos] = useState<Video[]>([]);
  const [sections, setSections] = useState<Section[]>([]);
  const [newSectionTitle, setNewSectionTitle] = useState('');
  const [addingSection, setAddingSection] = useState(false);

  useEffect(() => {
    supabase.auth.getUser().then(({ data: { user } }) => {
      setIsOwner(!!user && user.id === ownerUserId);
    });
  }, [ownerUserId]);

  const openReorder = useCallback(() => {
    // sort_order でソートして編集バッファに読み込む
    const sorted = [...videos].sort((a, b) => {
      if (a.sort_order === null && b.sort_order === null) return 0;
      if (a.sort_order === null) return 1;
      if (b.sort_order === null) return -1;
      return a.sort_order - b.sort_order;
    });
    setOrderedVideos(sorted);
    setSections([...initialSections].sort((a, b) => a.sort_order - b.sort_order));
    setIsOpen(true);
  }, [videos, initialSections]);

  const moveVideo = useCallback((index: number, dir: -1 | 1) => {
    setOrderedVideos(prev => {
      const next = [...prev];
      const target = index + dir;
      if (target < 0 || target >= next.length) return prev;
      [next[index], next[target]] = [next[target], next[index]];
      return next;
    });
  }, []);

  const assignSection = useCallback((videoId: string, sectionId: string | null) => {
    setOrderedVideos(prev => prev.map(v => v.id === videoId ? { ...v, section_id: sectionId } : v));
  }, []);

  const addSection = useCallback(async () => {
    if (!newSectionTitle.trim()) return;
    setAddingSection(true);
    const { data, error } = await supabase.rpc('upsert_list_section', {
      p_list_id: listId,
      p_section_id: null,
      p_title: newSectionTitle.trim(),
      p_sort_order: sections.length,
    });
    setAddingSection(false);
    if (error) { alert('セクション作成に失敗しました: ' + error.message); return; }
    setSections(prev => [...prev, {
      id: data as string,
      title: newSectionTitle.trim(),
      display_mode: 'ranked',
      sort_order: prev.length,
    }]);
    setNewSectionTitle('');
  }, [newSectionTitle, listId, sections.length]);

  const deleteSection = useCallback(async (sectionId: string) => {
    const { error } = await supabase.rpc('delete_list_section', { p_section_id: sectionId });
    if (error) { alert('セクション削除に失敗しました: ' + error.message); return; }
    setSections(prev => prev.filter(s => s.id !== sectionId));
    // そのセクションに属していた動画を未分類に戻す
    setOrderedVideos(prev => prev.map(v => v.section_id === sectionId ? { ...v, section_id: null } : v));
  }, []);

  const save = useCallback(async () => {
    setSaving(true);
    try {
      // 並び順を保存
      const videoIds = orderedVideos.map(v => v.id);
      const { error: reorderErr } = await supabase.rpc('reorder_list_videos', {
        p_list_id: listId,
        p_video_ids: videoIds,
      });
      if (reorderErr) throw reorderErr;

      // セクション割り当てを保存
      for (const video of orderedVideos) {
        const original = videos.find(v => v.id === video.id);
        if (original?.section_id !== video.section_id) {
          const { error } = await supabase.rpc('assign_video_section', {
            p_list_id: listId,
            p_video_id: video.id,
            p_section_id: video.section_id,
          });
          if (error) throw error;
        }
      }

      setIsOpen(false);
      router.refresh();
    } catch (err: unknown) {
      alert('保存に失敗しました: ' + (err instanceof Error ? err.message : String(err)));
    } finally {
      setSaving(false);
    }
  }, [orderedVideos, listId, videos, router]);

  if (!isOwner) return null;

  if (!isOpen) {
    return (
      <div className="mb-4 flex justify-end">
        <button
          onClick={openReorder}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-[#30363d] hover:border-[#8b949e] text-[#8b949e] hover:text-[#e6edf3] text-xs font-semibold transition-colors"
        >
          <GripVertical size={13} />
          並べ替え・セクション
        </button>
      </div>
    );
  }

  return (
    <div className="mb-6 rounded-xl border border-violet-500/40 bg-[#0d1117] overflow-hidden">
      {/* ヘッダー */}
      <div className="flex items-center justify-between px-4 py-3 bg-violet-500/10 border-b border-violet-500/30">
        <span className="text-sm font-bold text-violet-300">並べ替え・セクション管理</span>
        <div className="flex items-center gap-2">
          <button
            onClick={save}
            disabled={saving}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-violet-600 hover:bg-violet-500 disabled:opacity-60 text-white text-xs font-bold transition-colors"
          >
            {saving ? <Loader2 size={12} className="animate-spin" /> : <Save size={12} />}
            保存
          </button>
          <button
            onClick={() => setIsOpen(false)}
            className="p-1.5 rounded-lg hover:bg-[#21262d] text-[#656d76] hover:text-[#e6edf3] transition-colors"
          >
            <X size={14} />
          </button>
        </div>
      </div>

      <div className="p-4 space-y-6">
        {/* セクション管理 */}
        <div>
          <p className="text-xs font-semibold text-[#656d76] uppercase tracking-wider mb-3">セクション</p>
          <div className="space-y-2 mb-3">
            {sections.length === 0 && (
              <p className="text-xs text-[#484f58]">セクションなし（フラットなランキング）</p>
            )}
            {sections.map(section => (
              <div key={section.id} className="flex items-center gap-2 px-3 py-2 rounded-lg bg-[#161b22] border border-[#30363d]">
                <span className="text-sm text-[#e6edf3] flex-1">{section.title ?? '無題'}</span>
                <span className="text-xs text-[#484f58]">{section.display_mode === 'ranked' ? 'ランキング' : 'プレーン'}</span>
                <button
                  onClick={() => deleteSection(section.id)}
                  className="p-1 rounded hover:bg-red-500/20 text-[#656d76] hover:text-red-400 transition-colors"
                >
                  <Trash2 size={12} />
                </button>
              </div>
            ))}
          </div>
          {/* 新規セクション追加 */}
          <div className="flex gap-2">
            <input
              type="text"
              value={newSectionTitle}
              onChange={e => setNewSectionTitle(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && addSection()}
              placeholder="新しいセクション名"
              className="flex-1 px-3 py-1.5 rounded-lg bg-[#161b22] border border-[#30363d] focus:border-violet-500/60 outline-none text-xs text-[#e6edf3] placeholder-[#484f58]"
            />
            <button
              onClick={addSection}
              disabled={addingSection || !newSectionTitle.trim()}
              className="flex items-center gap-1 px-3 py-1.5 rounded-lg bg-[#161b22] border border-[#30363d] hover:border-violet-500/50 text-[#8b949e] hover:text-violet-300 disabled:opacity-40 text-xs font-semibold transition-colors"
            >
              {addingSection ? <Loader2 size={12} className="animate-spin" /> : <Plus size={12} />}
              追加
            </button>
          </div>
        </div>

        {/* 動画の並べ替え */}
        <div>
          <p className="text-xs font-semibold text-[#656d76] uppercase tracking-wider mb-3">動画の順序</p>
          <div className="space-y-1.5 max-h-[60vh] overflow-y-auto pr-1">
            {orderedVideos.map((video, i) => (
              <div key={video.id} className="flex items-center gap-2 px-3 py-2 rounded-lg bg-[#161b22] border border-[#21262d]">
                {/* ランク番号 */}
                <span className="text-xs font-black text-violet-400 w-6 text-right shrink-0">#{i + 1}</span>

                {/* タイトル */}
                <span className="text-xs text-[#8b949e] flex-1 line-clamp-1 min-w-0">
                  {video.title ?? '(タイトルなし)'}
                </span>

                {/* セクション割り当て */}
                {sections.length > 0 && (
                  <select
                    value={video.section_id ?? ''}
                    onChange={e => assignSection(video.id, e.target.value || null)}
                    className="text-xs bg-[#0d1117] border border-[#30363d] rounded px-1.5 py-0.5 text-[#8b949e] max-w-[100px]"
                  >
                    <option value="">未分類</option>
                    {sections.map(s => (
                      <option key={s.id} value={s.id}>{s.title ?? '無題'}</option>
                    ))}
                  </select>
                )}

                {/* 上下ボタン */}
                <div className="flex flex-col gap-0.5 shrink-0">
                  <button
                    onClick={() => moveVideo(i, -1)}
                    disabled={i === 0}
                    className="p-0.5 rounded hover:bg-[#30363d] text-[#484f58] hover:text-[#e6edf3] disabled:opacity-20 transition-colors"
                  >
                    <ChevronUp size={13} />
                  </button>
                  <button
                    onClick={() => moveVideo(i, 1)}
                    disabled={i === orderedVideos.length - 1}
                    className="p-0.5 rounded hover:bg-[#30363d] text-[#484f58] hover:text-[#e6edf3] disabled:opacity-20 transition-colors"
                  >
                    <ChevronDown size={13} />
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
