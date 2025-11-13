'use client';

import { useCallback, useEffect, useState } from 'react';
import { supabase } from '@/lib/supabase';

export type LikedVideo = {
  id?: string;
  external_id: string;
  title: string;
  description: string | null;
  thumbnail_url: string | null;
  product_url: string | null;
  price: number | null;
  product_released_at: string | null;
  tags: { id: string; name: string }[] | null;
  performers: { id: string; name: string }[] | null;
};

export function useLikedVideos(limit = 40) {
  const [videos, setVideos] = useState<LikedVideo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isAuthenticated, setIsAuthenticated] = useState<boolean | null>(null);

  const fetchLikedVideos = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) {
        setIsAuthenticated(false);
        setVideos([]);
        return;
      }
      setIsAuthenticated(true);

      const { data, error: rpcError } = await supabase.rpc('get_user_likes', {
        p_search: null,
        p_sort: 'liked_at',
        p_order: 'desc',
        p_limit: limit,
        p_offset: 0,
        p_price_min: null,
        p_price_max: null,
        p_release_gte: null,
        p_release_lte: null,
        p_tag_ids: null,
        p_performer_ids: null,
      });
      if (rpcError) throw rpcError;
      setVideos((data as LikedVideo[]) ?? []);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'いいね一覧の取得に失敗しました');
    } finally {
      setLoading(false);
    }
  }, [limit]);

  useEffect(() => {
    fetchLikedVideos();
  }, [fetchLikedVideos]);

  return { videos, loading, error, isAuthenticated, refetch: fetchLikedVideos };
}

