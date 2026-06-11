'use client';

import { useEffect, useState, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { Search, Trash2, Loader2 } from 'lucide-react';
import { supabase } from '@/lib/supabase';

interface Props {
  ownerUserId: string;
  listId: string;
  listType: 'liked' | 'custom';
  listTitle?: string | null;
  token: string;
}

export default function ListEditMode({ ownerUserId, listId, listType, listTitle, token }: Props) {
  const router = useRouter();
  const [isOwner, setIsOwner] = useState(false);

  useEffect(() => {
    supabase.auth.getUser().then(({ data: { user } }) => {
      setIsOwner(!!user && user.id === ownerUserId);
    });
  }, [ownerUserId]);

  if (!isOwner || listType !== 'custom') return null;

  const exploreUrl = `/explore?add_to_list=${listId}&list_title=${encodeURIComponent(listTitle ?? 'リスト')}&return_url=${encodeURIComponent(`/list/${token}`)}`;

  return (
    <div className="mb-6 flex items-center justify-between gap-3 px-4 py-2.5 rounded-xl border border-violet-500/40 bg-violet-500/10">
      <span className="text-xs text-violet-300 font-semibold">編集モード</span>
      <button
        onClick={() => router.push(exploreUrl)}
        className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-violet-600 hover:bg-violet-500 text-white text-xs font-bold transition-colors"
      >
        <Search size={13} />
        動画を追加
      </button>
    </div>
  );
}

export function VideoDeleteButton({
  ownerUserId,
  listId,
  videoId,
}: {
  ownerUserId: string;
  listId: string;
  videoId: string;
}) {
  const router = useRouter();
  const [isOwner, setIsOwner] = useState(false);
  const [deleting, setDeleting] = useState(false);

  useEffect(() => {
    supabase.auth.getUser().then(({ data: { user } }) => {
      setIsOwner(!!user && user.id === ownerUserId);
    });
  }, [ownerUserId]);

  const handleDelete = useCallback(async (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDeleting(true);
    const { error } = await supabase
      .from('public_list_videos')
      .delete()
      .eq('list_id', listId)
      .eq('video_id', videoId);
    if (error) { alert('削除に失敗しました: ' + error.message); setDeleting(false); return; }
    router.refresh();
  }, [listId, videoId, router]);

  if (!isOwner) return null;

  return (
    <button
      onClick={handleDelete}
      disabled={deleting}
      className="absolute top-1.5 right-1.5 flex items-center justify-center w-7 h-7 rounded-lg bg-black/60 backdrop-blur hover:bg-red-500/80 text-white transition-colors opacity-0 group-hover:opacity-100 disabled:opacity-60 shadow"
      aria-label="リストから削除"
    >
      {deleting ? <Loader2 size={12} className="animate-spin" /> : <Trash2 size={12} />}
    </button>
  );
}
