'use client';

import VideoList from '@/components/VideoList';
import { useState, useEffect, useMemo } from 'react';
import { supabase } from '@/lib/supabase';
import { Video } from '@/types/video';

const useMyListVideos = (limit: number) => {
  const [videos, setVideos] = useState<Video[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isAuthenticated, setIsAuthenticated] = useState<boolean | null>(null);

  useEffect(() => {
    const fetchMyList = async () => {
      setLoading(true);
      setError(null);

      const { data: { user } } = await supabase.auth.getUser();
      if (!user) {
        setIsAuthenticated(false);
        setLoading(false);
        return;
      }
      setIsAuthenticated(true);

      try {
        // 1. Fetch the user's "My List" playlist
        const { data: playlist, error: playlistError } = await supabase
          .from('ai_recommend_playlists')
          .select('items')
          .eq('user_id', user.id)
          .order('created_at', { ascending: true })
          .limit(1)
          .single();

        if (playlistError) {
          if (playlistError.code === 'PGRST116') { // PostgREST error for "single() row not found"
            // This is not an error, it just means the user doesn't have a list yet.
            setVideos([]);
          } else {
            throw playlistError;
          }
        }

        if (playlist && Array.isArray(playlist.items)) {
          // 2. Extract video external_ids from the playlist items
          const videoIds = playlist.items.map((item: any) => item.video_external_id).filter(Boolean);

          if (videoIds.length > 0) {
            // 3. Fetch video details for the extracted IDs
            const { data: videoData, error: videoError } = await supabase
              .from('videos')
              .select(`
                external_id,
                title,
                thumbnail_url,
                price,
                product_released_at,
                product_url,
                tags ( id, name ),
                performers ( id, name )
              `)
              .in('external_id', videoIds)
              .limit(limit);

            if (videoError) throw videoError;

            // The fetched data needs to be transformed to match the Video type if necessary
            setVideos(videoData as Video[]);
          } else {
            setVideos([]);
          }
        } else {
          setVideos([]);
        }
      } catch (e: any) {
        setError(`エラーが発生しました: ${e.message}`);
        console.error(e);
      } finally {
        setLoading(false);
      }
    };

    fetchMyList();
  }, [limit]);

  return { videos, loading, error, isAuthenticated };
};

export default function MyListPage() {
  const { videos, loading, error, isAuthenticated } = useMyListVideos(80);

  const description = useMemo(() => {
    if (loading) return '読み込み中...';
    if (!isAuthenticated) return 'ログインして、気になるリストを確認しましょう。';
    if (videos.length === 0) return '気になるリストに作品がありません。';
    return `${videos.length} 件の作品`;
  }, [loading, isAuthenticated, videos.length]);

  return (
    <VideoList
      title="気になるリスト"
      description={description}
      videos={videos}
      loading={loading}
      error={error}
      isAuthenticated={isAuthenticated}
    />
  );
}
