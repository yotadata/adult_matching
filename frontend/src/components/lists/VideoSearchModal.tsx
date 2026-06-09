'use client';

import { useState, useCallback, useRef } from 'react';
import { X, Search, Plus, Check, Loader2 } from 'lucide-react';
import { supabase } from '@/lib/supabase';

type SearchResult = {
  id: string;
  title: string | null;
  external_id: string | null;
  thumbnail_url: string | null;
  distribution_code: string | null;
};

interface VideoSearchModalProps {
  listId: string;
  onClose: () => void;
  onAdded?: () => void;
}

export default function VideoSearchModal({ listId, onClose, onAdded }: VideoSearchModalProps) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [searching, setSearching] = useState(false);
  const [addedIds, setAddedIds] = useState<Set<string>>(new Set());
  const [addingId, setAddingId] = useState<string | null>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const handleSearch = useCallback((q: string) => {
    setQuery(q);
    if (debounceRef.current) clearTimeout(debounceRef.current);
    if (!q.trim()) { setResults([]); return; }
    debounceRef.current = setTimeout(async () => {
      setSearching(true);
      const { data } = await supabase.rpc('search_videos', { p_query: q.trim(), p_limit: 20 });
      setResults((data as SearchResult[] | null) ?? []);
      setSearching(false);
    }, 400);
  }, []);

  const handleAdd = async (video: SearchResult) => {
    if (addedIds.has(video.id)) return;
    setAddingId(video.id);
    const { error } = await supabase
      .from('public_list_videos')
      .insert({ list_id: listId, video_id: video.id });
    if (!error) {
      setAddedIds((prev) => new Set(prev).add(video.id));
      onAdded?.();
    }
    setAddingId(null);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-end sm:items-center justify-center p-0 sm:p-4">
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />
      <div className="relative w-full sm:max-w-lg bg-[#161b22] rounded-t-2xl sm:rounded-2xl border border-[#30363d] flex flex-col max-h-[85vh]">

        {/* ヘッダー */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-[#30363d]">
          <h2 className="text-sm font-bold text-[#e6edf3]">動画を検索して追加</h2>
          <button onClick={onClose} className="text-[#8b949e] hover:text-white transition-colors">
            <X size={18} />
          </button>
        </div>

        {/* 検索ボックス */}
        <div className="px-4 py-3 border-b border-[#30363d]">
          <div className="flex items-center gap-2 bg-[#0d1117] rounded-lg px-3 py-2 border border-[#30363d] focus-within:border-violet-500/50">
            <Search size={14} className="text-[#656d76] shrink-0" />
            <input
              autoFocus
              type="text"
              value={query}
              onChange={(e) => handleSearch(e.target.value)}
              placeholder="タイトル・品番で検索（例: IPX-123、三上悠亜）"
              className="flex-1 bg-transparent text-sm text-[#e6edf3] placeholder-[#484f58] outline-none"
            />
            {searching && <Loader2 size={14} className="text-[#656d76] animate-spin shrink-0" />}
          </div>
        </div>

        {/* 検索結果 */}
        <div className="flex-1 overflow-y-auto">
          {results.length === 0 && query && !searching && (
            <p className="text-center text-[#656d76] text-sm py-10">見つかりませんでした</p>
          )}
          {results.length === 0 && !query && (
            <p className="text-center text-[#484f58] text-xs py-10">タイトルや品番で検索できます</p>
          )}
          {results.map((video) => {
            const added = addedIds.has(video.id);
            const adding = addingId === video.id;
            return (
              <div
                key={video.id}
                className="flex items-center gap-3 px-4 py-2.5 hover:bg-white/5 transition-colors border-b border-[#21262d] last:border-0"
              >
                {/* サムネイル */}
                <div className="w-12 h-16 shrink-0 rounded overflow-hidden bg-[#21262d]">
                  {video.thumbnail_url ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img src={video.thumbnail_url} alt="" className="w-full h-full object-cover" loading="lazy" />
                  ) : (
                    <div className="w-full h-full" />
                  )}
                </div>

                {/* タイトル */}
                <div className="flex-1 min-w-0">
                  <p className="text-xs text-[#e6edf3] line-clamp-2 leading-snug">{video.title}</p>
                  {video.distribution_code && (
                    <p className="text-[10px] text-[#484f58] mt-0.5">{video.distribution_code}</p>
                  )}
                </div>

                {/* 追加ボタン */}
                <button
                  onClick={() => handleAdd(video)}
                  disabled={added || adding}
                  className={`shrink-0 flex items-center justify-center w-8 h-8 rounded-full transition-all ${
                    added
                      ? 'bg-green-500/20 text-green-400 border border-green-500/30'
                      : 'bg-violet-600 hover:bg-violet-500 text-white'
                  } disabled:opacity-60`}
                >
                  {adding ? <Loader2 size={14} className="animate-spin" /> : added ? <Check size={14} /> : <Plus size={14} />}
                </button>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
