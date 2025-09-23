'use client';

import { Dialog, Transition } from '@headlessui/react';
import { Fragment, useState, useEffect, useMemo } from 'react';
import { X, Filter } from 'lucide-react';
import Link from 'next/link';
import Image from 'next/image';
import { supabase } from '@/lib/supabase';
import VideoDetailModal from './VideoDetailModal';

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
  const [tagOptions, setTagOptions] = useState<{ id: string; name: string; cnt?: number }[]>([]);
  const [performerOptions, setPerformerOptions] = useState<{ id: string; name: string; cnt?: number }[]>([]);
  const [selectedTagIds, setSelectedTagIds] = useState<string[]>([]);
  const [selectedPerformerIds, setSelectedPerformerIds] = useState<string[]>([]);
  const pageSize = 40;
  const [detailOpen, setDetailOpen] = useState(false);
  const [detailVideoId, setDetailVideoId] = useState<string | null>(null);
  const [mobileShowFilters, setMobileShowFilters] = useState(false);

  // DMM/FANZA アフィリエイトリンクへ変換（既にal.fanza.co.jpならそのまま）
  const toFanzaAffiliate = (raw: string | undefined | null): string | undefined => {
    if (!raw) return undefined;
    const AF_ID = 'yotadata2-001';
    try {
      if (raw.startsWith('https://al.fanza.co.jp/')) {
        const url = new URL(raw);
        url.searchParams.set('af_id', AF_ID);
        return url.toString();
      }
    } catch {}
    const lurl = encodeURIComponent(raw);
    return `https://al.fanza.co.jp/?lurl=${lurl}&af_id=${encodeURIComponent(AF_ID)}&ch=link_tool&ch_id=link`;
  };

  const depsKey = useMemo(
    () => ({ isOpen, sort, order, selectedTagIds, selectedPerformerIds }),
    [isOpen, sort, order, selectedTagIds, selectedPerformerIds]
  );

  // Fetch facet options (tags/performers across ALL likes) when opening
  useEffect(() => {
    if (!isOpen) return;
    (async () => {
      try {
        const [{ data: tags, error: te }, { data: perfs, error: pe }] = await Promise.all([
          supabase.rpc('get_user_liked_tags'),
          supabase.rpc('get_user_liked_performers'),
        ]);
        if (te) console.warn('facet tags error', te.message);
        if (pe) console.warn('facet performers error', pe.message);
        const tagRows = (tags as { id: string; name: string; cnt?: number }[] | null)?.filter(Boolean) || [];
        const perfRows = (perfs as { id: string; name: string; cnt?: number }[] | null)?.filter(Boolean) || [];
        setTagOptions(tagRows);
        setPerformerOptions(perfRows);
      } catch {
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
        const rows = (data as VideoRecord[]) || [];
        setLikedVideos(rows);
      }
      setLoading(false);
    };

    fetchLikedVideos();
  }, [depsKey]);

  // ここではオーバーレイは常時クリック可能とし、パネル側で stopPropagation して外側判定を防ぐ。

  // オーバーレイ押下で閉じないポリシー（Xボタンのみで閉じる）
  const handleDialogClose = () => {};
  return (
    <Transition appear show={isOpen} as={Fragment}>
      <Dialog as="div" className="relative z-50" onClose={handleDialogClose}>
        <Transition.Child
          as={Fragment}
          enter="ease-out duration-300"
          enterFrom="opacity-0"
          enterTo="opacity-100"
          leave="ease-in duration-200"
          leaveFrom="opacity-100"
          leaveTo="opacity-0"
        >
          <div
            className="fixed inset-0 bg-black/40 z-40 pointer-events-auto"
          />
        </Transition.Child>

        <div className="fixed inset-0 overflow-hidden z-[60] isolate">
          <div className="absolute inset-0 overflow-hidden">
            {/* Mobile bottom sheet */}
            <div className="pointer-events-none fixed inset-x-0 bottom-0 flex justify-center sm:hidden z-[60]">
              <Transition.Child
                as={Fragment}
                enter="transform transition ease-in-out duration-500"
                enterFrom="translate-y-full"
                enterTo="translate-y-0"
                leave="transform transition ease-in-out duration-500"
                leaveFrom="translate-y-0"
                leaveTo="translate-y-full"
              >
                <Dialog.Panel className="pointer-events-auto w-screen relative z-[70]">
                  <div className="w-full rounded-t-2xl bg-white shadow-2xl">
                    <div className="mx-auto mt-2 mb-1 h-1.5 w-12 rounded-full bg-gray-300" />
                    <div className="h-[90dvh] max-h-[90svh] overflow-hidden flex flex-col">
                      {/* Header: left X, center title, right filter toggle */}
                      <div className="sticky top-0 z-10 bg-white px-3 py-2 grid grid-cols-[auto_1fr_auto] items-center border-b">
                        <div className="flex justify-start">
                          <button
                            type="button"
                            className="rounded-md text-gray-500 hover:text-gray-700 focus:outline-none focus:ring-2 focus:ring-amber-500"
                            onClick={onClose}
                            aria-label="閉じる"
                          >
                            <X className="h-6 w-6" />
                          </button>
                        </div>
                        <div className="text-center">
                          <Dialog.Title className="text-base font-bold text-gray-900 whitespace-nowrap">いいねしたものリスト</Dialog.Title>
                        </div>
                        <div className="flex justify-end">
                          <button
                            type="button"
                            className={`rounded-md p-2 ${mobileShowFilters ? 'text-violet-600' : 'text-gray-600'} hover:text-violet-600`}
                            onClick={() => setMobileShowFilters(v => !v)}
                            aria-label="フィルタを切り替え"
                          >
                            <Filter className="h-5 w-5" />
                          </button>
                        </div>
                      </div>

                      {/* Filters (fixed, not scrollable) */}
                      {mobileShowFilters && (
                        <div className="p-3 border-b bg-gray-50">
                          {/* 並び替え */}
                          <div className="flex items-center gap-2 mb-3">
                            <select className="flex-1 border border-gray-300 rounded-lg px-2 py-2 bg-white text-gray-800" value={sort} onChange={(e) => setSort(e.target.value as 'liked_at' | 'released' | 'price' | 'title')}>
                              <option value="liked_at">いいね日時</option>
                              <option value="released">発売日</option>
                              <option value="price">価格</option>
                              <option value="title">タイトル</option>
                            </select>
                            <select className="w-28 border border-gray-300 rounded-lg px-2 py-2 bg-white text-gray-800" value={order} onChange={(e) => setOrder(e.target.value as 'desc' | 'asc')}>
                              <option value="desc">降順</option>
                              <option value="asc">昇順</option>
                            </select>
                          </div>

                          {/* 選択中のフィルタ（取り外し可能なチップ） */}
                          {(selectedTagIds.length > 0 || selectedPerformerIds.length > 0) && (
                            <div className="mb-2">
                              <div className="flex items-center justify-between mb-1">
                                <span className="text-xs text-gray-600">選択中のフィルタ</span>
                                <button
                                  className="text-xs text-gray-600 underline"
                                  onClick={() => { setSelectedTagIds([]); setSelectedPerformerIds([]); }}
                                >すべてクリア</button>
                              </div>
                              <div className="flex flex-wrap gap-2">
                                {selectedTagIds.map((id) => {
                                  const name = tagOptions.find(t => t.id === id)?.name || id;
                                  return (
                                    <button
                                      key={`tag-${id}`}
                                      onClick={() => setSelectedTagIds(selectedTagIds.filter(x => x !== id))}
                                      className="px-2 py-1 rounded-full bg-white border border-gray-300 text-xs text-gray-700"
                                    >
                                      {name} ×
                                    </button>
                                  );
                                })}
                                {selectedPerformerIds.map((id) => {
                                  const name = performerOptions.find(p => p.id === id)?.name || id;
                                  return (
                                    <button
                                      key={`perf-${id}`}
                                      onClick={() => setSelectedPerformerIds(selectedPerformerIds.filter(x => x !== id))}
                                      className="px-2 py-1 rounded-full bg-white border border-gray-300 text-xs text-gray-700"
                                    >
                                      {name} ×
                                    </button>
                                  );
                                })}
                              </div>
                            </div>
                          )}

                          {/* セレクター（選択・解除がしやすいよう残す） */}
                          <div className="mb-3">
                            <label className="block text-xs text-gray-600 mb-1">タグで絞り込み</label>
                            <select
                              multiple
                              className="w-full border border-gray-300 rounded-lg px-2 py-2 h-28 bg-white text-gray-800"
                              value={selectedTagIds}
                              onChange={(e) => {
                                const opts = Array.from(e.target.selectedOptions).map(o => o.value);
                                setSelectedTagIds(opts);
                              }}
                            >
                              {tagOptions.map((t) => (
                                <option key={t.id} value={t.id}>{t.name}{typeof t.cnt === 'number' ? ` (${t.cnt})` : ''}</option>
                              ))}
                            </select>
                          </div>
                          <div>
                            <label className="block text-xs text-gray-600 mb-1">出演で絞り込み</label>
                            <select
                              multiple
                              className="w-full border border-gray-300 rounded-lg px-2 py-2 h-28 bg-white text-gray-800"
                              value={selectedPerformerIds}
                              onChange={(e) => {
                                const opts = Array.from(e.target.selectedOptions).map(o => o.value);
                                setSelectedPerformerIds(opts);
                              }}
                            >
                              {performerOptions.map((p) => (
                                <option key={p.id} value={p.id}>{p.name}{typeof p.cnt === 'number' ? ` (${p.cnt})` : ''}</option>
                              ))}
                            </select>
                          </div>
                        </div>
                      )}

                      {/* Scrollable content: list only */}
                      <div className="flex-1 overflow-y-auto overscroll-contain touch-pan-y" style={{ WebkitOverflowScrolling: 'touch' as any }}>
                        <div className="p-4 pb-[calc(8px+env(safe-area-inset-bottom,0px))]">
                          {error && (
                            <div className="mb-2 text-sm text-red-600">{error}</div>
                          )}
                          {loading ? (
                            <p className="text-center text-gray-500">読み込み中...</p>
                          ) : likedVideos.length > 0 ? (
                            <div className="grid grid-cols-1 gap-4">
                              {likedVideos.map((video) => (
                                <div key={video.external_id} className="bg-gray-50 rounded-lg shadow-sm overflow-hidden flex items-center p-3">
                                  <div className="relative w-28 flex-shrink-0 rounded-md overflow-hidden bg-gray-200" style={{ height: 0, paddingBottom: '78px' }}>
                                    {video.thumbnail_url ? (
                                      <Image src={video.thumbnail_url} alt={video.title} fill className="object-cover" />
                                    ) : (
                                      <div className="w-full h-full flex items-center justify-center">
                                        <span className="text-gray-400 text-xs">画像なし</span>
                                      </div>
                                    )}
                                  </div>
                                  <div className="pl-3 flex-grow">
                                    <h3 className="text-sm font-semibold text-gray-800 line-clamp-2">{video.title}</h3>
                                    <div className="mt-2 flex justify-between items-center">
                                      <p className="text-sm font-bold text-[#FF6B81]">
                                        {video.price ? `￥${video.price.toLocaleString()}~` : '価格情報なし'}
                                      </p>
                                      <div className="flex gap-2">
                                        <Link href={toFanzaAffiliate(video.product_url) || '#'} passHref target="_blank" rel="noopener noreferrer">
                                          <button className="bg-transparent border border-[#FF6B81] text-[#FF6B81] hover:bg-[#FF6B81] hover:text-white font-bold py-1 px-3 rounded-md transition-all duration-300 text-xs">
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
                    </div>
                  </div>
                </Dialog.Panel>
              </Transition.Child>
            </div>

            {/* Desktop/Tablet side drawer */}
            <div className="pointer-events-none fixed inset-y-0 right-0 hidden sm:flex max-w-full pl-10 sm:pl-16 z-[60]">
              <Transition.Child
                as={Fragment}
                enter="transform transition ease-in-out duration-500 sm:duration-700"
                enterFrom="translate-x-full"
                enterTo="translate-x-0"
                leave="transform transition ease-in-out duration-500 sm:duration-700"
                leaveFrom="translate-x-0"
                leaveTo="translate-x-full"
              >
                <Dialog.Panel
                  className="pointer-events-auto w-screen max-w-3xl md:max-w-4xl relative z-[70]"
                >
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
                          <select className="border border-gray-300 rounded-lg px-2 py-2 bg-white text-gray-800" value={sort} onChange={(e) => setSort(e.target.value as 'liked_at' | 'released' | 'price' | 'title')}>
                            <option value="liked_at">いいね日時</option>
                            <option value="released">発売日</option>
                            <option value="price">価格</option>
                            <option value="title">タイトル</option>
                          </select>
                          <select className="border border-gray-300 rounded-lg px-2 py-2 bg-white text-gray-800" value={order} onChange={(e) => setOrder(e.target.value as 'desc' | 'asc')}>
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
                            className="w-full border border-gray-300 rounded-lg px-2 py-2 h-28 bg-white text-gray-800"
                            value={selectedTagIds}
                            onChange={(e) => {
                              const opts = Array.from(e.target.selectedOptions).map(o => o.value);
                              setSelectedTagIds(opts);
                            }}
                          >
                            {tagOptions.map((t) => (
                              <option key={t.id} value={t.id}>{t.name}{typeof t.cnt === 'number' ? ` (${t.cnt})` : ''}</option>
                            ))}
                          </select>
                        </div>
                        {/* Performers */}
                        <div className="md:col-span-6">
                          <label className="block text-xs text-gray-600 mb-1">出演で絞り込み</label>
                          <select
                            multiple
                            className="w-full border border-gray-300 rounded-lg px-2 py-2 h-28 bg-white text-gray-800"
                            value={selectedPerformerIds}
                            onChange={(e) => {
                              const opts = Array.from(e.target.selectedOptions).map(o => o.value);
                              setSelectedPerformerIds(opts);
                            }}
                          >
                            {performerOptions.map((p) => (
                              <option key={p.id} value={p.id}>{p.name}{typeof p.cnt === 'number' ? ` (${p.cnt})` : ''}</option>
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
                                  <p className="text-md font-bold text-[#FF6B81]">
                                    {video.price ? `￥${video.price.toLocaleString()}~` : '価格情報なし'}
                                  </p>
                                  <div className="flex gap-2">
                                    {video.id && (
                                      <button
                                        onClick={() => { setDetailVideoId(video.id!); setDetailOpen(true); }}
                                        className="bg-transparent border border-violet-400 text-violet-500 hover:bg-violet-500 hover:text-white font-bold py-1 px-3 rounded-md transition-all duration-300 text-sm"
                                      >
                                        詳細を見る
                                      </button>
                                    )}
                                    <Link href={toFanzaAffiliate(video.product_url) || '#'} passHref target="_blank" rel="noopener noreferrer">
                                      <button className="bg-transparent border border-[#FF6B81] text-[#FF6B81] hover:bg-[#FF6B81] hover:text-white font-bold py-1 px-3 rounded-md transition-all duration-300 text-sm">
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
                    <VideoDetailModal isOpen={detailOpen} onClose={() => setDetailOpen(false)} videoId={detailVideoId} />
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
