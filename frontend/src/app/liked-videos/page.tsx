'use client';

import { useState, useEffect } from 'react';
import { supabase } from '@/lib/supabase';
import Header from '@/components/Header';
import { VideoRecord } from '@/../scripts/fanza_ingest'; // 型定義をインポート

export default function LikedVideosPage() {
  const [likedVideos, setLikedVideos] = useState<VideoRecord[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchLikedVideos = async () => {
      setLoading(true);
      const { data: { user } } = await supabase.auth.getUser();

      if (user) {
        const { data, error } = await supabase
          .from('likes')
          .select('videos (*)') // videosテーブルの全カラムを取得
          .eq('user_id', user.id);

        if (error) {
          console.error('Error fetching liked videos:', error);
        } else {
          // dataは { videos: VideoRecord }[] の形式なので、videosプロパティを抽出
          const videos = data.map(item => item.videos).filter(Boolean) as VideoRecord[];
          setLikedVideos(videos);
        }
      }
      setLoading(false);
    };

    fetchLikedVideos();
  }, []);

  return (
    <div className="min-h-screen bg-gray-100">
      <Header />
      <main className="container mx-auto px-4 py-8">
        <h1 className="text-3xl font-bold text-gray-800 mb-8">いいねした動画</h1>
        
        {loading ? (
          <p className="text-center text-gray-500">読み込み中...</p>
        ) : likedVideos.length > 0 ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
            {likedVideos.map((video) => (
              <div key={video.external_id} className="bg-white rounded-lg shadow-md overflow-hidden transform hover:-translate-y-1 transition-transform duration-300">
                <a href={video.product_url || '#'} target="_blank" rel="noopener noreferrer">
                  {video.thumbnail_url && (
                    <img src={video.thumbnail_url} alt={video.title} className="w-full h-48 object-cover" />
                  )}
                  <div className="p-4">
                    <h2 className="text-lg font-semibold text-gray-800 truncate" title={video.title}>
                      {video.title}
                    </h2>
                  </div>
                </a>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-center text-gray-500">いいねした動画はありません。</p>
        )}
      </main>
    </div>
  );
}
