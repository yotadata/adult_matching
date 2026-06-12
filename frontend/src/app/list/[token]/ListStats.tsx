'use client';

import { useEffect, useState } from 'react';
import { Eye, Heart } from 'lucide-react';
import { supabase } from '@/lib/supabase';

function getFingerprint(): string {
  const key = 'shl_fp';
  let fp = localStorage.getItem(key);
  if (!fp) {
    fp = crypto.randomUUID();
    localStorage.setItem(key, fp);
  }
  return fp;
}

type Props = {
  token: string;
  initialViewCount: number;
  initialLikeCount: number;
};

export default function ListStats({ token, initialViewCount, initialLikeCount }: Props) {
  const [likeCount, setLikeCount] = useState(initialLikeCount);
  const [liked, setLiked] = useState(false);
  const [likeLoading, setLikeLoading] = useState(false);

  useEffect(() => {
    const fp = getFingerprint();

    supabase.rpc('increment_list_view', { p_token: token }).then(() => {});

    supabase.rpc('get_list_like_status', { p_token: token, p_fingerprint: fp }).then(({ data }) => {
      if (data) {
        setLiked(data.liked);
        setLikeCount(data.count);
      }
    });
  }, [token]);

  const handleLike = async () => {
    if (likeLoading) return;
    setLikeLoading(true);
    const fp = getFingerprint();
    const { data } = await supabase.rpc('toggle_list_like', { p_token: token, p_fingerprint: fp });
    if (data && !data.error) {
      setLiked(data.liked);
      setLikeCount(data.count);
    }
    setLikeLoading(false);
  };

  return (
    <div className="flex items-center gap-4 text-sm text-[#656d76]">
      <span className="flex items-center gap-1.5">
        <Eye size={15} />
        <span>{initialViewCount.toLocaleString()}</span>
      </span>
      <button
        type="button"
        onClick={handleLike}
        disabled={likeLoading}
        className={`flex items-center gap-1.5 transition-colors cursor-pointer select-none rounded-full px-2.5 py-1 border ${
          liked
            ? 'text-rose-400 border-rose-400 bg-rose-400/10'
            : 'border-[#444c56] hover:text-rose-400 hover:border-rose-400 hover:bg-rose-400/10'
        }`}
      >
        <Heart size={14} className={liked ? 'fill-rose-400' : ''} />
        <span>{likeCount.toLocaleString()}</span>
      </button>
    </div>
  );
}
