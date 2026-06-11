'use client';

import { useEffect, useState, useCallback } from 'react';
import { X, Trash2, Loader2 } from 'lucide-react';
import { supabase } from '@/lib/supabase';
import { resolveThumbnail } from '@/utils/thumbnail';

type ListVideo = {
  id: string;
  video_id: string;
  title: string | null;
  thumbnail_url: string | null;
  distribution_code: string | null;
  source: string | null;
  image_urls: string[] | null;
};

interface ListVideosModalProps {
  listId: string;
  listTitle: string;
  onClose: () => void;
  onChanged?: () => void;
}

export default function ListVideosModal({ listId, listTitle, onClose, onChanged }: ListVideosModalProps) {
  const [videos, setVideos] = useState<ListVideo[]>([]);
  const [loading, setLoading] = useState(true);
  const [removingId, setRemovingId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fetchVideos = useCallback(async () => {
    setLoading(true);
    const { data, error } = await supabase
      .from('public_list_videos')
      .select('id, video_id, videos(title, thumbnail_url, distribution_code, source, image_urls)')
      .eq('list_id', listId)
      .order('added_at', { ascending: false });

    if (error) {
      setError('動画の取得に失敗しました');
      setLoading(false);
      return;
    }

    const rows = ((data ?? []) as unknown as Array<{
      id: string;
      video_id: string;
      videos: { title: string | null; thumbnail_url: string | null; distribution_code: string | null; source: string | null; image_urls: string[] | null } | null;
    }>).map((row) => ({
      id: row.id,
      video_id: row.video_id,
      title: row.videos?.title ?? null,
      thumbnail_url: row.videos?.thumbnail_url ?? null,
      distribution_code: row.videos?.distribution_code ?? null,
      source: row.videos?.source ?? null,
      image_urls: row.videos?.image_urls ?? null,
    }));

    setVideos(rows);
    setLoading(false);
  }, [listId]);

  useEffect(() => { fetchVideos(); }, [fetchVideos]);

  const handleRemove = async (rowId: string) => {
    setRemovingId(rowId);
    const { error } = await supabase
      .from('public_list_videos')
      .delete()
      .eq('id', rowId);

    if (error) {
      alert('削除に失敗しました: ' + error.message);
      setRemovingId(null);
      return;
    }

    setVideos((prev) => prev.filter((v) => v.id !== rowId));
    setRemovingId(null);
    onChanged?.();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-end sm:items-center justify-center p-0 sm:p-4">
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />
      <div className="relative w-full sm:max-w-lg bg-[#161b22] rounded-t-2xl sm:rounded-2xl border border-[#30363d] flex flex-col max-h-[85vh]">

        <div className="flex items-center justify-between px-4 py-3 border-b border-[#30363d] shrink-0">
          <div>
            <h2 className="text-sm font-bold text-[#e6edf3]">動画を管理</h2>
            <p className="text-[10px] text-[#656d76] mt-0.5 line-clamp-1">{listTitle}</p>
          </div>
          <button onClick={onClose} className="text-[#8b949e] hover:text-white transition-colors">
            <X size={18} />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto">
          {loading && (
            <div className="flex items-center justify-center py-16">
              <Loader2 size={20} className="text-violet-400 animate-spin" />
            </div>
          )}
          {error && (
            <p className="text-center text-red-400 text-sm py-10">{error}</p>
          )}
          {!loading && !error && videos.length === 0 && (
            <p className="text-center text-[#484f58] text-xs py-10">このリストにはまだ動画がありません</p>
          )}
          {!loading && videos.map((video) => (
            <div
              key={video.id}
              className="flex items-center gap-3 px-4 py-2.5 border-b border-[#21262d] last:border-0"
            >
              <div className="w-12 h-16 shrink-0 rounded overflow-hidden bg-[#21262d]">
                {(() => { const { primary } = resolveThumbnail({ source: video.source, thumbnail_url: video.thumbnail_url, image_urls: video.image_urls }); const thumb = primary ?? video.thumbnail_url; return thumb ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img src={thumb} alt="" className="w-full h-full object-cover" loading="lazy" />
                ) : (
                  <div className="w-full h-full" />
                ); })()}
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-xs text-[#e6edf3] line-clamp-2 leading-snug">{video.title}</p>
                {video.distribution_code && (
                  <p className="text-[10px] text-[#484f58] mt-0.5">{video.distribution_code}</p>
                )}
              </div>
              <button
                onClick={() => handleRemove(video.id)}
                disabled={removingId === video.id}
                className="shrink-0 flex items-center justify-center w-8 h-8 rounded-full bg-red-500/10 hover:bg-red-500/20 text-red-400 transition-colors disabled:opacity-50"
                aria-label="リストから削除"
              >
                {removingId === video.id
                  ? <Loader2 size={14} className="animate-spin" />
                  : <Trash2 size={14} />}
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
