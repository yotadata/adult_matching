'use client';

import { useState, useEffect, useCallback, useRef, Suspense } from 'react';
import { Search, Heart, Plus, Check, Loader2, X, ChevronDown, Sparkles, Compass } from 'lucide-react';
import { useRouter } from 'next/navigation';
import { supabase } from '@/lib/supabase';

type Video = {
  id: string;
  title: string | null;
  external_id: string | null;
  thumbnail_url: string | null;
  thumbnail_vertical_url: string | null;
  product_url: string | null;
  distribution_code: string | null;
};

type Tag = { id: string; name: string };
type PublicList = { id: string; token: string; title: string | null; list_type: string };

const FANZA_AF_ID = process.env.NEXT_PUBLIC_FANZA_AFFILIATE_ID ?? 'yotadata2-001';

function toLgThumb(url: string | null | undefined) {
  if (!url) return null;
  return url.replace('ps.jpg', 'pl.jpg');
}

function toAffiliateUrl(raw?: string | null) {
  if (!raw) return '#';
  return `https://al.fanza.co.jp/?lurl=${encodeURIComponent(raw)}&af_id=${encodeURIComponent(FANZA_AF_ID)}&ch=link_tool&ch_id=link`;
}

// リストに追加するドロップダウン
function AddToListMenu({
  videoId,
  lists,
  onClose,
}: {
  videoId: string;
  lists: PublicList[];
  onClose: () => void;
}) {
  const [addedIds, setAddedIds] = useState<Set<string>>(new Set());
  const [loadingId, setLoadingId] = useState<string | null>(null);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) onClose();
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [onClose]);

  const handleAdd = async (listId: string) => {
    setLoadingId(listId);
    await supabase.from('public_list_videos').insert({ list_id: listId, video_id: videoId });
    setAddedIds((prev) => new Set(prev).add(listId));
    setLoadingId(null);
  };

  if (lists.length === 0) {
    return (
      <div ref={ref} className="absolute right-0 bottom-full mb-1 w-44 bg-[#161b22] border border-[#30363d] rounded-xl shadow-xl z-50 p-3 text-xs text-[#656d76]">
        リストがありません
      </div>
    );
  }

  return (
    <div ref={ref} className="absolute right-0 bottom-full mb-1 w-48 bg-[#161b22] border border-[#30363d] rounded-xl shadow-xl z-50 overflow-hidden">
      <p className="text-[10px] text-[#656d76] px-3 pt-2 pb-1 font-semibold">リストに追加</p>
      {lists.map((list) => {
        const added = addedIds.has(list.id);
        const loading = loadingId === list.id;
        return (
          <button
            key={list.id}
            onClick={() => !added && handleAdd(list.id)}
            className={`w-full flex items-center justify-between px-3 py-2 text-xs transition-colors ${added ? 'text-green-400' : 'text-[#e6edf3] hover:bg-white/5'}`}
          >
            <span className="truncate">{list.list_type === 'liked' ? 'お気に入りリスト' : (list.title ?? '無題')}</span>
            {loading ? <Loader2 size={12} className="animate-spin shrink-0" /> : added ? <Check size={12} className="shrink-0" /> : <Plus size={12} className="shrink-0 text-[#656d76]" />}
          </button>
        );
      })}
    </div>
  );
}

function VideoCard({
  video,
  likedIds,
  onLike,
  lists,
}: {
  video: Video;
  likedIds: Set<string>;
  onLike: (video: Video) => void;
  lists: PublicList[];
}) {
  const [showMenu, setShowMenu] = useState(false);
  const thumb = toLgThumb(video.thumbnail_url) ?? toLgThumb(video.thumbnail_vertical_url);
  const liked = likedIds.has(video.id);

  return (
    <div className="break-inside-avoid mb-3 relative group">
      <a
        href={toAffiliateUrl(video.product_url)}
        target="_blank"
        rel="noopener noreferrer"
        className="block rounded-xl overflow-hidden border border-[#21262d] group-hover:border-violet-500/40 transition-colors bg-[#161b22]"
      >
        {thumb ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={thumb} alt={video.title ?? ''} className="w-full h-auto" loading="lazy" />
        ) : (
          <div className="w-full aspect-[3/4] bg-[#21262d]" />
        )}
        <div className="p-2">
          <p className="text-[11px] text-[#8b949e] line-clamp-2 leading-snug">{video.title}</p>
        </div>
      </a>

      {/* アクションボタン */}
      <div className="absolute top-2 right-2 flex flex-col gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
        {/* いいね */}
        <button
          onClick={(e) => { e.preventDefault(); onLike(video); }}
          className={`flex items-center justify-center w-8 h-8 rounded-full shadow-lg transition-all ${liked ? 'bg-pink-500 text-white' : 'bg-[#0d1117]/80 text-[#8b949e] hover:text-pink-400 hover:bg-[#0d1117]'}`}
        >
          <Heart size={14} fill={liked ? 'currentColor' : 'none'} />
        </button>

        {/* リストに追加 */}
        {lists.length > 0 && (
          <div className="relative">
            <button
              onClick={(e) => { e.preventDefault(); setShowMenu((v) => !v); }}
              className="flex items-center justify-center w-8 h-8 rounded-full bg-[#0d1117]/80 text-[#8b949e] hover:text-violet-400 shadow-lg transition-all"
            >
              <Plus size={14} />
            </button>
            {showMenu && (
              <AddToListMenu
                videoId={video.id}
                lists={lists}
                onClose={() => setShowMenu(false)}
              />
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function ExplorePageInner() {
  const router = useRouter();

  const [videos, setVideos] = useState<Video[]>([]);
  const [loading, setLoading] = useState(false);
  const [query, setQuery] = useState('');
  const [selectedTag, setSelectedTag] = useState<Tag | null>(null);
  const [popularTags, setPopularTags] = useState<Tag[]>([]);
  const [likedIds, setLikedIds] = useState<Set<string>>(new Set());
  const [lists, setLists] = useState<PublicList[]>([]);
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [offset, setOffset] = useState(0);
  const [hasMore, setHasMore] = useState(true);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const LIMIT = 30;

  // 認証・リスト・いいね済みIDを取得
  useEffect(() => {
    (async () => {
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) return;
      setIsLoggedIn(true);

      const [{ data: listData }, { data: likedData }] = await Promise.all([
        supabase.from('public_lists').select('id, token, title, list_type').eq('user_id', user.id).eq('is_active', true),
        supabase.from('user_video_decisions').select('video_id').eq('user_id', user.id).in('decision_type', ['swipe_like', 'grid_like']),
      ]);
      setLists((listData as PublicList[]) ?? []);
      setLikedIds(new Set((likedData ?? []).map((d: { video_id: string }) => d.video_id)));
    })();
  }, []);

  // 人気タグ取得
  useEffect(() => {
    supabase
      .from('tags')
      .select('id, name, tag_group_id')
      .limit(20)
      .then(({ data }) => setPopularTags((data as Tag[]) ?? []));
  }, []);

  const fetchVideos = useCallback(async (q: string, tag: Tag | null, newOffset: number, append = false) => {
    setLoading(true);
    const { data } = await supabase.rpc('explore_videos', {
      p_query: q.trim() || null,
      p_tag_id: tag?.id ?? null,
      p_performer_id: null,
      p_limit: LIMIT,
      p_offset: newOffset,
    });
    const results = (data as Video[] | null) ?? [];
    setVideos((prev) => append ? [...prev, ...results] : results);
    setHasMore(results.length === LIMIT);
    setLoading(false);
  }, []);

  // 初回・フィルター変更時
  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      setOffset(0);
      fetchVideos(query, selectedTag, 0, false);
    }, 400);
  }, [query, selectedTag, fetchVideos]);

  const handleLoadMore = () => {
    const next = offset + LIMIT;
    setOffset(next);
    fetchVideos(query, selectedTag, next, true);
  };

  const handleLike = async (video: Video) => {
    if (!isLoggedIn) { window.dispatchEvent(new Event('open-auth-modal')); return; }
    if (likedIds.has(video.id)) return;
    await supabase.from('user_video_decisions').insert({
      video_id: video.id,
      decision_type: 'grid_like',
      surface: 'grid',
    });
    setLikedIds((prev) => new Set(prev).add(video.id));
    window.dispatchEvent(new Event('like-added'));
  };

  return (
    <div className="min-h-screen bg-[#0d1117] text-[#e6edf3]">
      {/* タブ */}
      <div className="flex gap-1 px-4 pt-3 pb-0">
        <button
          onClick={() => router.push('/grid')}
          className="flex items-center gap-1.5 px-4 py-2 rounded-full text-sm font-bold transition-colors text-[#8b949e] hover:text-[#e6edf3] hover:bg-white/5"
        >
          <Sparkles size={14} />
          おすすめ
        </button>
        <button
          className="flex items-center gap-1.5 px-4 py-2 rounded-full text-sm font-bold transition-colors bg-violet-600 text-white"
        >
          <Compass size={14} />
          さがす
        </button>
      </div>

      {/* フィルターバー */}
      <div className="sticky top-[43px] sm:top-[43px] z-30 bg-[#0d1117]/95 backdrop-blur border-b border-[#21262d] px-4 py-3 space-y-2">
        {/* 検索 */}
        <div className="flex items-center gap-2 bg-[#161b22] border border-[#30363d] rounded-lg px-3 py-2 focus-within:border-violet-500/50">
          <Search size={14} className="text-[#656d76] shrink-0" />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="タイトル・品番で検索"
            className="flex-1 bg-transparent text-sm text-[#e6edf3] placeholder-[#484f58] outline-none"
          />
          {query && (
            <button onClick={() => setQuery('')} className="text-[#656d76] hover:text-white">
              <X size={14} />
            </button>
          )}
        </div>

        {/* タグフィルター */}
        <div className="flex gap-1.5 overflow-x-auto pb-0.5 scrollbar-hide">
          <button
            onClick={() => setSelectedTag(null)}
            className={`flex-shrink-0 px-3 py-1 rounded-full text-xs font-bold transition-colors ${!selectedTag ? 'bg-violet-600 text-white' : 'bg-white/5 text-[#8b949e] hover:bg-white/10'}`}
          >
            すべて
          </button>
          {popularTags.map((tag) => (
            <button
              key={tag.id}
              onClick={() => setSelectedTag(selectedTag?.id === tag.id ? null : tag)}
              className={`flex-shrink-0 px-3 py-1 rounded-full text-xs font-bold transition-colors ${selectedTag?.id === tag.id ? 'bg-violet-600 text-white' : 'bg-white/5 text-[#8b949e] hover:bg-white/10'}`}
            >
              {tag.name}
            </button>
          ))}
        </div>
      </div>

      {/* グリッド */}
      <div className="px-3 pt-4">
        {videos.length === 0 && !loading ? (
          <p className="text-center text-[#656d76] py-20 text-sm">見つかりませんでした</p>
        ) : (
          <div className="[column-count:2] sm:[column-count:3] lg:[column-count:4] [column-gap:12px]">
            {videos.map((video) => (
              <VideoCard
                key={video.id}
                video={video}
                likedIds={likedIds}
                onLike={handleLike}
                lists={lists}
              />
            ))}
          </div>
        )}

        {/* もっと見る */}
        {hasMore && (
          <div className="flex justify-center py-8">
            <button
              onClick={handleLoadMore}
              disabled={loading}
              className="flex items-center gap-2 px-6 py-2.5 rounded-full bg-[#161b22] border border-[#30363d] hover:border-violet-500/50 text-sm font-bold text-[#8b949e] hover:text-[#e6edf3] transition-all disabled:opacity-50"
            >
              {loading ? <Loader2 size={14} className="animate-spin" /> : <ChevronDown size={14} />}
              もっと見る
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

export default function ExplorePage() {
  return (
    <Suspense>
      <ExplorePageInner />
    </Suspense>
  );
}
