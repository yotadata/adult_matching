'use client';

import { Dialog, Transition } from '@headlessui/react';
import { Fragment, useState, useEffect, useMemo } from 'react';
import { X } from 'lucide-react';
import Link from 'next/link';
import Image from 'next/image';
import { supabase } from '@/lib/supabase';
import { PostgrestError } from '@supabase/supabase-js';

interface VideoRecord {
  id?: string;
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
}

interface LikedVideosDrawerProps {
  isOpen: boolean;
  onClose: () => void;
}

const LikedVideosDrawer: React.FC<LikedVideosDrawerProps> = ({ isOpen, onClose }) => {
  const [likedVideos, setLikedVideos] = useState<VideoRecord[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Filters and sorting (in-drawer)
  const [sort, setSort] = useState<'liked_at' | 'released' | 'price' | 'title'>('liked_at');
  const [order, setOrder] = useState<'desc' | 'asc'>('desc');
  const [tagOptions, setTagOptions] = useState<{ id: string; name: string }[]>([]);
  const [performerOptions, setPerformerOptions] = useState<{ id: string; name: string }[]>([]);
  const [selectedTagIds, setSelectedTagIds] = useState<string[]>([]);
  const [selectedPerformerIds, setSelectedPerformerIds] = useState<string[]>([]);
  const pageSize = 40;

  // DMM/FANZA アフィリエイトリンクへ変換（既にal.fanza.co.jpならそのまま）
  const toFanzaAffiliate = (raw: string | undefined | null): string | undefined => {
    if (!raw) return undefined;
    if (raw.startsWith('https://al.fanza.co.jp/')) return raw;
    const afId = process.env.NEXT_PUBLIC_FANZA_AFFILIATE_ID || '';
    if (!afId) return raw; // 環境変数未設定時は生URLを返す
    const lurl = encodeURIComponent(raw);
    const aid = encodeURIComponent(afId);
    return `https://al.fanza.co.jp/?lurl=${lurl}&af_id=${aid}&ch=link_tool&ch_id=link`;
  };

  const depsKey = useMemo(
    () => ({ isOpen, sort, order, selectedTagIds, selectedPerformerIds }),
    [isOpen, sort, order, selectedTagIds, selectedPerformerIds]
  );

  // Fetch filter options (tags/performers) when opening
  useEffect(() => {
    if (!isOpen) return;
    (async () => {
      try {
        const [{ data: tags }, { data: perfs }] = await Promise.all([
          supabase.from('tags').select('id, name').order('name', { ascending: true }).limit(200),
          supabase.from('performers').select('id, name').order('name', { ascending: true }).limit(200),
        ]);
        setTagOptions((tags as any[])?.filter(Boolean) || []);
        setPerformerOptions((perfs as any[])?.filter(Boolean) || []);
      } catch (e) {
        // ignore
      }
    })();
  }, [isOpen]);

  useEffect(() => {
    const fetchLikedVideos = async () => {
      if (!isOpen) return;
      setLoading(true);
      setError(null);
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) {
        setLikedVideos([]);
        setLoading(false);
        return;
      }

      // Prefer RPC for filtering/sorting performance
      const { data, error } = await supabase.rpc('get_user_likes', {
        p_search: null,
        p_sort: sort,
        p_order: order,
        p_limit: pageSize,
        p_offset: 0,
        p_price_min: null,
        p_price_max: null,
        p_release_gte: null,
        p_release_lte: null,
        p_tag_ids: selectedTagIds.length ? selectedTagIds : null,
        p_performer_ids: selectedPerformerIds.length ? selectedPerformerIds : null,
      });
      if (error) {
        console.error('Error fetching liked videos:', error);
        setError(error.message);
      } else {
        setLikedVideos((data as any[]) || []);
      }
      setLoading(false);
    };

    fetchLikedVideos();
  }, [depsKey]);

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
                <Dialog.Panel className="pointer-events-auto w-screen max-w-3xl md:max-w-4xl">
                  <div className="flex h-full flex-col bg-white shadow-xl">
                    <div className="p-6 sticky top-0 bg-white z-10 border-b">
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
                      {/* Inline filters */}
                      <div className="mt-4 grid grid-cols-1 md:grid-cols-12 gap-2">
                        <div className="md:col-span-5 flex items-center gap-2">
                          <select className="border rounded-lg px-2 py-2" value={sort} onChange={(e) => setSort(e.target.value as any)}>
                            <option value="liked_at">いいね日時</option>
                            <option value="released">発売日</option>
                            <option value="price">価格</option>
                            <option value="title">タイトル</option>
                          </select>
                          <select className="border rounded-lg px-2 py-2" value={order} onChange={(e) => setOrder(e.target.value as any)}>
                            <option value="desc">降順</option>
                            <option value="asc">昇順</option>
                          </select>
                        </div>
                        <div className="md:col-span-7" />
                        {/* Tags */}
                        <div className="md:col-span-6">
                          <label className="block text-xs text-gray-600 mb-1">タグで絞り込み</label>
                          <select
                            multiple
                            className="w-full border rounded-lg px-2 py-2 h-28"
                            value={selectedTagIds}
                            onChange={(e) => {
                              const opts = Array.from(e.target.selectedOptions).map(o => o.value);
                              setSelectedTagIds(opts);
                            }}
                          >
                            {tagOptions.map((t) => (
                              <option key={t.id} value={t.id}>{t.name}</option>
                            ))}
                          </select>
                        </div>
                        {/* Performers */}
                        <div className="md:col-span-6">
                          <label className="block text-xs text-gray-600 mb-1">出演で絞り込み</label>
                          <select
                            multiple
                            className="w-full border rounded-lg px-2 py-2 h-28"
                            value={selectedPerformerIds}
                            onChange={(e) => {
                              const opts = Array.from(e.target.selectedOptions).map(o => o.value);
                              setSelectedPerformerIds(opts);
                            }}
                          >
                            {performerOptions.map((p) => (
                              <option key={p.id} value={p.id}>{p.name}</option>
                            ))}
                          </select>
                        </div>
                        <div className="md:col-span-12 flex justify-end gap-2">
                          <button
                            className="px-3 py-2 text-xs rounded-lg border"
                            onClick={() => { setSelectedTagIds([]); setSelectedPerformerIds([]); setSort('liked_at'); setOrder('desc'); }}
                          >
                            条件クリア
                          </button>
                        </div>
                      </div>
                    </div>
                    
                    <div className="flex-1 px-4 sm:px-6 overflow-y-auto">
                      {error && (
                        <div className="mb-2 text-sm text-red-600">{error}</div>
                      )}
                      {loading ? (
                        <p className="text-center text-gray-500">読み込み中...</p>
                      ) : likedVideos.length > 0 ? (
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                          {likedVideos.map((video) => (
                            <div key={video.external_id} className="bg-gray-50 rounded-lg shadow-sm overflow-hidden flex items-center p-3">
                              <div className="relative w-32 flex-shrink-0 rounded-md overflow-hidden bg-gray-200" style={{ height: 0, paddingBottom: '90px' }}>
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
                                  <div className="flex gap-2">
                                    {video.id && (
                                      <Link href={`/videos/${video.id}`} passHref>
                                        <button className="bg-transparent border border-violet-400 text-violet-500 hover:bg-violet-500 hover:text-white font-bold py-1 px-3 rounded-md transition-all duration-300 text-sm">
                                          詳細を見る
                                        </button>
                                      </Link>
                                    )}
                                    <Link href={toFanzaAffiliate(video.product_url) || '#'} passHref target="_blank" rel="noopener noreferrer">
                                      <button className="bg-transparent border border-amber-400 text-amber-500 hover:bg-amber-400 hover:text-white font-bold py-1 px-3 rounded-md transition-all duration-300 text-sm">
                                        見る
                                      </button>
                                    </Link>
                                  </div>
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
