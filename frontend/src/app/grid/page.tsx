'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { createClient } from '@/lib/supabase';
import { X, ExternalLink } from 'lucide-react';

type VideoItem = {
  id: string;
  title: string | null;
  external_id: string | null;
  thumbnail_url: string | null;
  thumbnail_vertical_url: string | null;
  sample_video_url: string | null;
  embed_url: string | null;
  product_url: string | null;
  product_released_at: string | null;
  performers: { id: string; name: string }[];
  tags: { id: string; name: string }[];
  source: string;
};

const SUPABASE_URL = process.env.NEXT_PUBLIC_SUPABASE_URL!;

export default function GridPage() {
  const [videos, setVideos] = useState<VideoItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [hasMore, setHasMore] = useState(true);
  const [selected, setSelected] = useState<VideoItem | null>(null);
  const loaderRef = useRef<HTMLDivElement>(null);
  const loadedIds = useRef<Set<string>>(new Set());

  const fetchVideos = useCallback(async () => {
    if (loading || !hasMore) return;
    setLoading(true);
    try {
      const supabase = createClient();
      const { data: { session } } = await supabase.auth.getSession();
      const headers: Record<string, string> = { 'Content-Type': 'application/json' };
      if (session?.access_token) {
        headers['Authorization'] = `Bearer ${session.access_token}`;
      }
      const res = await fetch(`${SUPABASE_URL}/functions/v1/videos-grid`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          limit: 30,
          exclude_ids: Array.from(loadedIds.current),
        }),
      });
      if (!res.ok) return;
      const data = await res.json();
      const newVideos: VideoItem[] = (data.videos ?? []).filter(
        (v: VideoItem) => !loadedIds.current.has(v.id)
      );
      if (newVideos.length === 0) {
        setHasMore(false);
        return;
      }
      newVideos.forEach((v) => loadedIds.current.add(v.id));
      setVideos((prev) => [...prev, ...newVideos]);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  }, [loading, hasMore]);

  // 初回ロード
  useEffect(() => {
    fetchVideos();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // 無限スクロール
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => { if (entries[0].isIntersecting) fetchVideos(); },
      { threshold: 0.1 }
    );
    if (loaderRef.current) observer.observe(loaderRef.current);
    return () => observer.disconnect();
  }, [fetchVideos]);

  const toAffiliateUrl = (raw?: string | null) => {
    const AF_ID = 'yotadata2-001';
    if (!raw) return '';
    if (raw.startsWith('https://al.fanza.co.jp/')) {
      try {
        const u = new URL(raw);
        u.searchParams.set('af_id', AF_ID);
        return u.toString();
      } catch {}
    }
    return `https://al.fanza.co.jp/?lurl=${encodeURIComponent(raw)}&af_id=${encodeURIComponent(AF_ID)}&ch=link_tool&ch_id=link`;
  };

  return (
    <div className="min-h-screen bg-[#0d1117] px-3 py-4">
      <h1 className="text-[#e6edf3] text-lg font-bold mb-4 px-1">おすすめ</h1>

      {/* Pinterest グリッド */}
      <div className="columns-2 sm:columns-3 md:columns-4 gap-2 space-y-2">
        {videos.map((video) => (
          <div
            key={video.id}
            className="break-inside-avoid rounded-xl overflow-hidden cursor-pointer bg-[#161b22] border border-[#30363d] hover:border-violet-500/60 transition-colors"
            onClick={() => setSelected(video)}
          >
            <div className="w-full bg-black">
              <img
                src={video.thumbnail_vertical_url || video.thumbnail_url || ''}
                alt={video.title ?? ''}
                className="w-full object-cover"
                loading="lazy"
              />
            </div>
            <div className="px-2 py-1.5">
              <p className="text-[11px] text-[#e6edf3] line-clamp-2 leading-snug">{video.title}</p>
            </div>
          </div>
        ))}
      </div>

      {/* ローダー */}
      <div ref={loaderRef} className="h-16 flex items-center justify-center">
        {loading && <div className="w-6 h-6 border-2 border-violet-500 border-t-transparent rounded-full animate-spin" />}
        {!hasMore && <p className="text-[#8b949e] text-xs">すべて表示しました</p>}
      </div>

      {/* 動画モーダル */}
      {selected && (
        <div
          className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-4"
          onClick={() => setSelected(null)}
        >
          <div
            className="relative w-full max-w-lg bg-[#0d1117] rounded-2xl overflow-hidden border border-[#30363d]"
            onClick={(e) => e.stopPropagation()}
          >
            {/* 閉じるボタン */}
            <button
              onClick={() => setSelected(null)}
              className="absolute top-2 right-2 z-10 w-8 h-8 flex items-center justify-center rounded-full bg-black/60 text-white hover:bg-black/80"
            >
              <X size={16} />
            </button>

            {/* iframe */}
            <div className="relative w-full aspect-video bg-black">
              <iframe
                src={selected.embed_url ?? ''}
                className="absolute inset-0 w-full h-full"
                frameBorder="0"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; fullscreen"
                scrolling="no"
                referrerPolicy="no-referrer"
              />
            </div>

            {/* 情報 */}
            <div className="px-4 py-3">
              <p className="text-[#e6edf3] text-sm font-bold line-clamp-2">{selected.title}</p>
              {selected.performers && (selected.performers as { id: string; name: string }[]).length > 0 && (
                <p className="text-[#8b949e] text-xs mt-1">
                  {(selected.performers as { id: string; name: string }[]).map((p) => p.name).join(' / ')}
                </p>
              )}
              <a
                href={toAffiliateUrl(selected.product_url)}
                target="_blank"
                rel="noopener noreferrer"
                className="mt-3 flex items-center gap-1.5 justify-center w-full py-2 rounded-lg bg-violet-600 hover:bg-violet-500 text-white text-sm font-bold transition-colors"
              >
                <ExternalLink size={14} />
                FANZAで見る
              </a>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
