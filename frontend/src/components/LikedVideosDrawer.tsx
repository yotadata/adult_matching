'use client';

import { Dialog, Transition } from '@headlessui/react';
import { Fragment, useState, useEffect } from 'react';
import { X } from 'lucide-react';
import Link from 'next/link';
import Image from 'next/image';
import { supabase } from '@/lib/supabase';
import { VideoRecord } from '../../types';

interface LikedVideosDrawerProps {
  isOpen: boolean;
  onClose: () => void;
}

const LikedVideosDrawer: React.FC<LikedVideosDrawerProps> = ({ isOpen, onClose }) => {
  const [likedVideos, setLikedVideos] = useState<VideoRecord[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const fetchLikedVideos = async () => {
      if (!isOpen) return;
      setLoading(true);
      const { data: { user } } = await supabase.auth.getUser();

      if (user) {
        const { data, error } = await supabase
          .from('likes')
          .select('videos (*)')
          .eq('user_id', user.id)
          .order('created_at', { ascending: false });

        if (error) {
          console.error('Error fetching liked videos:', error);
        } else {
          const videos = data.map(item => item.videos).filter(Boolean) as VideoRecord[];
          setLikedVideos(videos);
        }
      }
      setLoading(false);
    };

    fetchLikedVideos();
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
                    
                    <div className="flex-1 px-4 sm:px-6 overflow-y-auto">
                      {loading ? (
                        <p className="text-center text-gray-500">読み込み中...</p>
                      ) : likedVideos.length > 0 ? (
                        <div className="space-y-4">
                          {likedVideos.map((video) => (
                            <div key={video.external_id} className="bg-gray-50 rounded-lg shadow-sm overflow-hidden flex items-center p-3">
                              <div className="relative w-24 flex-shrink-0 rounded-md overflow-hidden bg-gray-200 aspect-w-16 aspect-h-9">
                                {video.thumbnail_url ? (
                                  <Image src={video.thumbnail_url} alt={video.title} fill className="object-cover" />
                                ) : (
                                  <div className="w-full h-full flex items-center justify-center">
                                    <span className="text-gray-400 text-xs">画像なし</span>
                                  </div>
                                )}
                              </div>
                              <div className="pl-4 flex-grow">
                                <h3 className="text-sm font-semibold text-gray-800 line-clamp-2">{video.title}</h3>
                                <div className="mt-2 flex justify-between items-center">
                                  <p className="text-md font-bold text-amber-500">
                                    {video.price ? `￥${video.price.toLocaleString()}~` : '価格情報なし'}
                                  </p>
                                  <Link href={video.product_url || '#'} passHref target="_blank" rel="noopener noreferrer">
                                    <button className="bg-transparent border border-amber-400 text-amber-400 hover:bg-amber-400 hover:text-white font-bold py-1 px-3 rounded-md transition-all duration-300 text-sm">
                                      見る
                                    </button>
                                  </Link>
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <p className="text-center text-gray-500">いいねした動画はありません。</p>
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
