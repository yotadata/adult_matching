'use client';

import { useState, useEffect, useCallback, useRef, Suspense } from 'react';
import { Search, Heart, Plus, Check, Loader2, X, ChevronDown, Sparkles, Compass, ExternalLink, Play, ArrowLeft } from 'lucide-react';
import { useRouter, useSearchParams } from 'next/navigation';
import { supabase } from '@/lib/supabase';
import { resolveThumbnail } from '@/utils/thumbnail';

type Video = {
  id: string;
  title: string | null;
  external_id: string | null;
  thumbnail_url: string | null;
  thumbnail_vertical_url: string | null;
  product_url: string | null;
  distribution_code: string | null;
  source: string | null;
  image_urls: string[] | null;
};

type Tag = { id: string; name: string; cnt?: number };
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

function toFanzaEmbedUrl(externalId: string | null) {
  if (!externalId) return '';
  return `https://www.dmm.co.jp/litevideo/-/part/=/affi_id=${FANZA_AF_ID}/cid=${externalId}/size=1280_720/`;
}

// リストに追加するドロップダウン
function AddToListMenu({ videoId, lists, onClose }: { videoId: string; lists: PublicList[]; onClose: () => void }) {
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

// 動画カード
function VideoCard({ video, likedIds, onLike, lists, onClick, addToListId, addedToListIds, onAddToList }: {
  video: Video;
  likedIds: Set<string>;
  onLike: (video: Video) => void;
  lists: PublicList[];
  onClick: (video: Video) => void;
  addToListId?: string | null;
  addedToListIds?: Set<string>;
  onAddToList?: (videoId: string) => void;
}) {
  const [showMenu, setShowMenu] = useState(false);
  const { primary: thumbPrimary, fallback: thumbFallback } = resolveThumbnail({
    source: video.source,
    thumbnail_url: video.thumbnail_url,
    image_urls: video.image_urls,
  });
  const thumb = thumbPrimary ?? toLgThumb(video.thumbnail_url) ?? toLgThumb(video.thumbnail_vertical_url);
  const [thumbSrc, setThumbSrc] = useState(thumb);
  const liked = likedIds.has(video.id);
  const addedToList = addToListId ? (addedToListIds?.has(video.id) ?? false) : false;

  return (
    <div className="break-inside-avoid mb-2 relative group">
      <div
        className={`block rounded-lg overflow-hidden border transition-colors bg-[#161b22] cursor-pointer ${addedToList ? 'border-green-500/60' : 'border-[#21262d] group-hover:border-violet-500/40'}`}
        onClick={() => onClick(video)}
      >
        {thumbSrc ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={thumbSrc}
            alt={video.title ?? ''}
            className={`w-full h-auto ${addedToList ? 'opacity-60' : ''}`}
            loading="lazy"
            onError={() => { if (thumbFallback && thumbSrc !== thumbFallback) setThumbSrc(thumbFallback); }}
          />
        ) : (
          <div className="w-full aspect-[3/4] bg-[#21262d]" />
        )}
        <div className="p-1.5">
          <p className="text-[10px] text-[#8b949e] line-clamp-2 leading-snug">{video.title}</p>
        </div>
      </div>

      {/* アクションボタン */}
      <div className="absolute top-1.5 right-1.5 flex flex-col gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
        {/* リスト追加モード: 直接追加ボタン */}
        {addToListId ? (
          <button
            onClick={(e) => { e.stopPropagation(); if (!addedToList) onAddToList?.(video.id); }}
            className={`flex items-center justify-center w-7 h-7 rounded-full shadow-lg transition-all ${addedToList ? 'bg-green-500 text-white' : 'bg-violet-600 hover:bg-violet-500 text-white'}`}
          >
            {addedToList ? <Check size={12} /> : <Plus size={12} />}
          </button>
        ) : (
          <>
            <button
              onClick={(e) => { e.stopPropagation(); onLike(video); }}
              className={`flex items-center justify-center w-7 h-7 rounded-full shadow-lg transition-all ${liked ? 'bg-pink-500 text-white' : 'bg-[#0d1117]/80 text-[#8b949e] hover:text-pink-400'}`}
            >
              <Heart size={12} fill={liked ? 'currentColor' : 'none'} />
            </button>
            {lists.length > 0 && (
              <div className="relative">
                <button
                  onClick={(e) => { e.stopPropagation(); setShowMenu((v) => !v); }}
                  className="flex items-center justify-center w-7 h-7 rounded-full bg-[#0d1117]/80 text-[#8b949e] hover:text-violet-400 shadow-lg transition-all"
                >
                  <Plus size={12} />
                </button>
                {showMenu && <AddToListMenu videoId={video.id} lists={lists} onClose={() => setShowMenu(false)} />}
              </div>
            )}
          </>
        )}
      </div>

      {/* 追加済みオーバーレイ */}
      {addedToList && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div className="w-8 h-8 rounded-full bg-green-500 flex items-center justify-center shadow-lg">
            <Check size={16} className="text-white" />
          </div>
        </div>
      )}
    </div>
  );
}

// 動画モーダル
function VideoModal({ video, likedIds, onLike, onClose }: {
  video: Video;
  likedIds: Set<string>;
  onLike: (video: Video) => void;
  onClose: () => void;
}) {
  const [showVideo, setShowVideo] = useState(false);
  const liked = likedIds.has(video.id);
  const OVERLAY_HIDE_DELAY_MS = 700;
  const overlayHideTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  return (
    <div className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-4" onClick={onClose}>
      <div className="relative w-full max-w-2xl bg-white rounded-2xl border border-gray-200 shadow-xl flex flex-col overflow-hidden" onClick={(e) => e.stopPropagation()}>
        <button onClick={onClose} className="absolute top-2 right-2 z-20 w-8 h-8 flex items-center justify-center rounded-full bg-black/60 text-white hover:bg-black/80">
          <X size={16} />
        </button>
        <div className="relative w-full aspect-[4/3] bg-black/90 flex items-center justify-center rounded-t-2xl overflow-hidden">
          {!showVideo && (
            <div
              className="absolute inset-0 w-full h-full bg-contain bg-no-repeat bg-center flex items-center justify-center z-10 cursor-pointer"
              style={{ backgroundImage: video.thumbnail_url ? `url(${video.thumbnail_url})` : undefined, backgroundColor: video.thumbnail_url ? undefined : '#1f2937' }}
              onClick={() => setShowVideo(true)}
            >
              <div className="absolute inset-0 bg-black bg-opacity-40 flex items-center justify-center">
                <Play className="text-white w-16 h-16 opacity-80" fill="white" />
              </div>
            </div>
          )}
          <iframe
            scrolling="no"
            referrerPolicy="no-referrer"
            src={toFanzaEmbedUrl(video.external_id)}
            frameBorder="0"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share; fullscreen"
            loading="eager"
            onLoad={() => {
              if (overlayHideTimer.current) clearTimeout(overlayHideTimer.current);
              overlayHideTimer.current = setTimeout(() => setShowVideo(true), OVERLAY_HIDE_DELAY_MS);
            }}
            className="absolute top-0 left-0 w-full h-full overflow-hidden"
          />
        </div>
        <div className="flex flex-col text-gray-800 px-4 py-3 gap-2">
          <h2 className="text-base font-extrabold tracking-tight line-clamp-2">{video.title}</h2>
          <div className="flex gap-2">
            <button
              onClick={() => onLike(video)}
              className={`flex items-center gap-1.5 justify-center flex-1 py-2.5 rounded-lg text-sm font-bold transition-colors border ${
                liked ? 'bg-pink-500 border-pink-500 text-white' : 'bg-white border-gray-300 text-gray-700 hover:bg-pink-50 hover:border-pink-300'
              }`}
            >
              <Heart size={15} fill={liked ? 'white' : 'none'} />
              {liked ? 'いいね済み' : 'いいね'}
            </button>
          </div>
          {liked && video.product_url && (
            <a
              href={toAffiliateUrl(video.product_url)}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1.5 justify-center w-full py-2 rounded-lg bg-[#f0f0f0] hover:bg-[#e0e0e0] text-gray-500 text-xs font-medium transition-colors"
            >
              <ExternalLink size={12} />本編を見る
            </a>
          )}
        </div>
      </div>
    </div>
  );
}

function ExplorePageInner() {
  const router = useRouter();

  const [videos, setVideos] = useState<Video[]>([]);
  const [loading, setLoading] = useState(false);
  const [query, setQuery] = useState('');
  const searchParams = useSearchParams();
  const addToListId = searchParams.get('add_to_list');
  const addToListTitle = searchParams.get('list_title') ?? 'リスト';
  const returnUrl = searchParams.get('return_url') ?? null;

  const [selectedTag, setSelectedTag] = useState<Tag | null>(null);
  const [popularTags, setPopularTags] = useState<Tag[]>([]);
  const [likedIds, setLikedIds] = useState<Set<string>>(new Set());
  const [lists, setLists] = useState<PublicList[]>([]);
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [offset, setOffset] = useState(0);
  const [hasMore, setHasMore] = useState(true);
  const [modalVideo, setModalVideo] = useState<Video | null>(null);
  const [addedToListIds, setAddedToListIds] = useState<Set<string>>(new Set());
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const LIMIT = 30;

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

  // 人気タグ取得（video_tagsの件数順）
  useEffect(() => {
    supabase.rpc('get_popular_tags', { p_limit: 20 }).then(({ data }) => {
      if (data) setPopularTags(data as Tag[]);
      else {
        // fallback
        supabase.from('tags').select('id, name').limit(20).then(({ data: d }) => setPopularTags((d as Tag[]) ?? []));
      }
    });
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

  const handleAddToList = useCallback(async (videoId: string) => {
    if (!addToListId) return;
    await supabase.from('public_list_videos').insert({ list_id: addToListId, video_id: videoId });
    setAddedToListIds((prev) => new Set(prev).add(videoId));
  }, [addToListId]);

  const handleLike = async (video: Video) => {
    if (!isLoggedIn) { window.dispatchEvent(new Event('open-auth-modal')); return; }
    if (likedIds.has(video.id)) return;
    await supabase.from('user_video_decisions').insert({ video_id: video.id, decision_type: 'grid_like', surface: 'grid' });
    setLikedIds((prev) => new Set(prev).add(video.id));
    window.dispatchEvent(new Event('like-added'));
  };

  return (
    <div className="min-h-screen bg-[#0d1117] text-[#e6edf3]">
      {/* リスト追加モードバナー */}
      {addToListId && (
        <div className="sticky top-0 z-40 flex items-center justify-between gap-3 px-4 py-2.5 bg-violet-600 text-white shadow-lg">
          <div className="flex items-center gap-2 min-w-0">
            <Plus size={14} className="shrink-0" />
            <span className="text-sm font-bold truncate">「{decodeURIComponent(addToListTitle)}」に追加中</span>
            {addedToListIds.size > 0 && (
              <span className="shrink-0 text-xs bg-white/20 px-2 py-0.5 rounded-full">{addedToListIds.size}件追加</span>
            )}
          </div>
          <button
            onClick={() => router.push(returnUrl ?? '/')}
            className="shrink-0 flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white/20 hover:bg-white/30 text-sm font-bold transition-colors"
          >
            <ArrowLeft size={13} />
            完了
          </button>
        </div>
      )}

      {/* タブ */}
      <div className="flex gap-1 px-4 pt-3 pb-0">
        <button onClick={() => router.push('/grid')} className="flex items-center gap-1.5 px-4 py-2 rounded-full text-sm font-bold transition-colors text-[#8b949e] hover:text-[#e6edf3] hover:bg-white/5">
          <Sparkles size={14} />おすすめ
        </button>
        <button className="flex items-center gap-1.5 px-4 py-2 rounded-full text-sm font-bold bg-violet-600 text-white">
          <Compass size={14} />さがす
        </button>
      </div>

      {/* フィルターバー */}
      <div className="sticky top-[43px] z-30 bg-[#0d1117]/95 backdrop-blur border-b border-[#21262d] px-4 py-3 space-y-2">
        <div className="flex items-center gap-2 bg-[#161b22] border border-[#30363d] rounded-lg px-3 py-2 focus-within:border-violet-500/50">
          <Search size={14} className="text-[#656d76] shrink-0" />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="タイトル・品番で検索"
            className="flex-1 bg-transparent text-sm text-[#e6edf3] placeholder-[#484f58] outline-none"
          />
          {query && <button onClick={() => setQuery('')} className="text-[#656d76] hover:text-white"><X size={14} /></button>}
        </div>
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

      {/* グリッド（5列） */}
      <div className="px-2 pt-3">
        {videos.length === 0 && !loading ? (
          <p className="text-center text-[#656d76] py-20 text-sm">見つかりませんでした</p>
        ) : (
          <div className="[column-count:2] sm:[column-count:3] lg:[column-count:5] [column-gap:8px]">
            {videos.map((video) => (
              <VideoCard
                key={video.id}
                video={video}
                likedIds={likedIds}
                onLike={handleLike}
                lists={lists}
                onClick={setModalVideo}
                addToListId={addToListId}
                addedToListIds={addedToListIds}
                onAddToList={handleAddToList}
              />
            ))}
          </div>
        )}
        {loading && (
          <div className="flex justify-center py-8"><Loader2 size={20} className="text-violet-400 animate-spin" /></div>
        )}
        {hasMore && !loading && (
          <div className="flex justify-center py-8">
            <button
              onClick={handleLoadMore}
              className="flex items-center gap-2 px-6 py-2.5 rounded-full bg-[#161b22] border border-[#30363d] hover:border-violet-500/50 text-sm font-bold text-[#8b949e] hover:text-[#e6edf3] transition-all"
            >
              <ChevronDown size={14} />もっと見る
            </button>
          </div>
        )}
      </div>

      {/* 動画モーダル */}
      {modalVideo && (
        <VideoModal
          video={modalVideo}
          likedIds={likedIds}
          onLike={handleLike}
          onClose={() => setModalVideo(null)}
        />
      )}
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
