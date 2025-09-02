'use client';

import { Dialog, Transition } from '@headlessui/react';
<<<<<<< HEAD
import { Fragment, useEffect, useState } from 'react';
import { X, Trash2 } from 'lucide-react';
import Link from 'next/link';
import { createClient } from '@supabase/supabase-js';

interface LikedVideo {
  id: string;
  title: string;
  description: string;
  thumbnail_url: string;
  preview_video_url: string;
  maker: string;
  genre: string;
  price: number;
  sample_video_url: string;
  image_urls: string[];
  performers: string[];
  tags: string[];
  liked_at: string;
  purchased: boolean;
=======
import { Fragment, useState, useEffect } from 'react';
import { X } from 'lucide-react';
import Link from 'next/link';
import Image from 'next/image';
import { supabase } from '@/lib/supabase';
import { PostgrestError } from '@supabase/supabase-js';

interface VideoRecord {
  external_id: string;
  title: string;
  description?: string;
  duration_seconds?: number;
  thumbnail_url?: string;
  preview_video_url?: string;
  distribution_code?: string;
  maker_code?: string;
  director?: string;
  series?: string;
  maker?: string;
  label?: string;
  price?: number;
  distribution_started_at?: string;
  product_released_at?: string;
  sample_video_url?: string;
  image_urls?: string[];
  source: string;
  published_at?: string;
  product_url?: string;
>>>>>>> origin/main
}

interface LikedVideosDrawerProps {
  isOpen: boolean;
  onClose: () => void;
}

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!;
const supabase = createClient(supabaseUrl, supabaseKey);

const LikedVideosDrawer: React.FC<LikedVideosDrawerProps> = ({ isOpen, onClose }) => {
<<<<<<< HEAD
  const [likedVideos, setLikedVideos] = useState<LikedVideo[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchLikedVideos = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const { data: { session } } = await supabase.auth.getSession();
      if (!session) {
        setError('ログインが必要です');
        return;
      }

      const { data, error } = await supabase.functions.invoke('likes');
      
      if (error) {
        console.error('API Error:', error);
        setError('いいねした動画の取得に失敗しました');
        return;
      }
      
      if (data?.likes) {
        setLikedVideos(data.likes);
      }
    } catch (err) {
      console.error('Fetch error:', err);
      setError('動画の取得中にエラーが発生しました');
    } finally {
      setLoading(false);
    }
  };

  const removeLike = async (videoId: string) => {
    try {
      const { data: { session } } = await supabase.auth.getSession();
      if (!session) return;

      const { error } = await supabase.functions.invoke('likes', {
        method: 'DELETE',
        body: { video_id: videoId },
      });
      
      if (error) {
        console.error('Remove like error:', error);
        return;
      }
      
      // ローカル状態からも削除
      setLikedVideos(prev => prev.filter(video => video.id !== videoId));
    } catch (err) {
      console.error('Remove like error:', err);
    }
  };

  useEffect(() => {
    if (isOpen) {
      fetchLikedVideos();
    }
=======
  const [likedVideos, setLikedVideos] = useState<VideoRecord[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const fetchLikedVideos = async () => {
      if (!isOpen) return;
      setLoading(true);
      const { data: { user } } = await supabase.auth.getUser();

      if (user) {
        const { data, error } = await supabase
          .from('user_video_decisions')
          .select('videos (*)')
          .eq('user_id', user.id)
          .eq('decision_type', 'like')
          .order('created_at', { ascending: false }) as { data: { videos: VideoRecord }[] | null, error: PostgrestError | null };

        if (error) {
          console.error('Error fetching liked videos:', error);
        } else {
          const videos = (data || []).map(item => item.videos).filter(Boolean) as VideoRecord[];
          setLikedVideos(videos);
        }
      }
      setLoading(false);
    };

    fetchLikedVideos();
>>>>>>> origin/main
  }, [isOpen]);

  return (
    <Transition appear show={isOpen} as={Fragment}>
      <Dialog as="div" className="relative z-50" onClose={onClose}>
        <Transition.Child
          as={Fragment}
          enter="ease-out duration-300"
          enterFrom="opacity-0"
          enterTo="opacity-100"
          leave="ease-in duration-200"
          leaveFrom="opacity-100"
          leaveTo="opacity-0"
        >
          <div className="fixed inset-0 bg-black/30 backdrop-blur-sm" />
        </Transition.Child>

        <div className="fixed inset-0 overflow-hidden">
          <div className="absolute inset-0 overflow-hidden">
            <div className="pointer-events-none fixed inset-y-0 right-0 flex max-w-full pl-10 sm:pl-16">
              <Transition.Child
                as={Fragment}
                enter="transform transition ease-in-out duration-500 sm:duration-700"
                enterFrom="translate-x-full"
                enterTo="translate-x-0"
                leave="transform transition ease-in-out duration-500 sm:duration-700"
                leaveFrom="translate-x-0"
                leaveTo="translate-x-full"
              >
                <Dialog.Panel className="pointer-events-auto w-screen max-w-md">
                  <div className="flex h-full flex-col bg-white shadow-xl">
                    <div className="p-6">
                      <div className="flex items-start justify-between">
                        <Dialog.Title className="text-xl font-bold text-gray-900">
                          いいねした動画
                        </Dialog.Title>
                        <div className="ml-3 flex h-7 items-center">
                          <button
                            type="button"
                            className="relative rounded-md bg-white text-gray-400 hover:text-gray-500 focus:outline-none focus:ring-2 focus:ring-amber-500"
                            onClick={onClose}
                          >
                            <span className="absolute -inset-2.5" />
                            <span className="sr-only">Close panel</span>
                            <X className="h-6 w-6" aria-hidden="true" />
                          </button>
                        </div>
                      </div>
                    </div>
                    
<<<<<<< HEAD
                    {/* Content */}
                    <div className="flex-1 px-4 sm:px-6 overflow-y-auto">
                      {loading ? (
                        <div className="flex items-center justify-center py-8">
                          <p className="text-gray-500">読み込み中...</p>
                        </div>
                      ) : error ? (
                        <div className="flex items-center justify-center py-8">
                          <p className="text-red-500">{error}</p>
                        </div>
                      ) : likedVideos.length === 0 ? (
                        <div className="flex items-center justify-center py-8">
                          <p className="text-gray-500">いいねした動画がありません</p>
                        </div>
                      ) : (
                        <div className="space-y-4">
                          {likedVideos.map((video) => (
                            <div key={video.id} className="bg-gray-50 rounded-lg shadow-sm overflow-hidden flex items-center p-3">
                              <div className="relative w-24 h-24 flex-shrink-0 rounded-md overflow-hidden">
                                {video.thumbnail_url ? (
                                  <img 
                                    src={video.thumbnail_url} 
                                    alt={video.title}
                                    className="w-full h-full object-cover"
                                  />
                                ) : (
                                  <div className="w-full h-full bg-gray-200 flex items-center justify-center">
                                    <span className="text-gray-400 text-xs">画像</span>
=======
                    <div className="flex-1 px-4 sm:px-6 overflow-y-auto">
                      {loading ? (
                        <p className="text-center text-gray-500">読み込み中...</p>
                      ) : likedVideos.length > 0 ? (
                        <div className="space-y-4">
                          {likedVideos.map((video) => (
                            <div key={video.external_id} className="bg-gray-50 rounded-lg shadow-sm overflow-hidden flex items-center p-3">
                              <div className="relative w-28 flex-shrink-0 rounded-md overflow-hidden bg-gray-200" style={{ height: 0, paddingBottom: '78px' }}>
                                {video.thumbnail_url ? (
                                  <Image src={video.thumbnail_url} alt={video.title} fill className="object-cover" />
                                ) : (
                                  <div className="w-full h-full flex items-center justify-center">
                                    <span className="text-gray-400 text-xs">画像なし</span>
>>>>>>> origin/main
                                  </div>
                                )}
                              </div>
                              <div className="pl-4 flex-grow">
                                <h3 className="text-sm font-semibold text-gray-800 line-clamp-2">{video.title}</h3>
<<<<<<< HEAD
                                <div className="mt-1">
                                  <p className="text-xs text-gray-500">{video.maker}</p>
                                  <div className="flex flex-wrap gap-1 mt-1">
                                    {video.tags.slice(0, 3).map((tag, index) => (
                                      <span key={index} className="text-xs px-2 py-0.5 bg-gray-200 rounded-full">
                                        {tag}
                                      </span>
                                    ))}
                                  </div>
                                </div>
                                <div className="mt-2 flex justify-between items-center">
                                  <p className="text-md font-bold text-amber-500">
                                    ¥{video.price.toLocaleString()}
                                  </p>
                                  <div className="flex gap-2">
                                    <button
                                      onClick={() => removeLike(video.id)}
                                      className="text-red-400 hover:text-red-600 p-1"
                                      title="いいねを削除"
                                    >
                                      <Trash2 className="h-4 w-4" />
                                    </button>
                                    <Link href="#" passHref>
                                      <button className="bg-transparent border border-amber-400 text-amber-400 hover:bg-amber-400 hover:text-white font-bold py-1 px-3 rounded-md transition-all duration-300 text-sm">
                                        見る
                                      </button>
                                    </Link>
                                  </div>
=======
                                <div className="mt-2 flex justify-between items-center">
                                  <p className="text-md font-bold text-amber-500">
                                    {video.price ? `￥${video.price.toLocaleString()}~` : '価格情報なし'}
                                  </p>
                                  <Link href={video.product_url || '#'} passHref target="_blank" rel="noopener noreferrer">
                                    <button className="bg-transparent border border-amber-400 text-amber-400 hover:bg-amber-400 hover:text-white font-bold py-1 px-3 rounded-md transition-all duration-300 text-sm">
                                      見る
                                    </button>
                                  </Link>
>>>>>>> origin/main
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
<<<<<<< HEAD
=======
                      ) : (
                        <p className="text-center text-gray-500">いいねした動画はありません。</p>
>>>>>>> origin/main
                      )}
                    </div>
                  </div>
                </Dialog.Panel>
              </Transition.Child>
            </div>
          </div>
        </div>
      </Dialog>
    </Transition>
  );
};

export default LikedVideosDrawer;