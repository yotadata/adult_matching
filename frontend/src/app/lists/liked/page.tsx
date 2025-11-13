'use client';

import { useLikedVideos } from '@/hooks/useLikedVideos';
import VideoList from '@/components/VideoList';

export default function LikedListPage() {
  const { videos, loading, error, isAuthenticated } = useLikedVideos(80);

  return (
    <VideoList
      title="いいねした作品"
      description={`${videos.length} 件の作品`}
      videos={videos}
      loading={loading}
      error={error}
      isAuthenticated={isAuthenticated}
    />
  );
}
