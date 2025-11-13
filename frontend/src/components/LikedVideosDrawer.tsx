'use client';

import { Dialog, Transition, Combobox, DialogBackdrop } from '@headlessui/react';
import { Fragment, useState, useEffect, useMemo, useRef } from 'react';
import type { SyntheticEvent, MouseEvent, PointerEvent } from 'react';
import { X, Filter } from 'lucide-react';
import Link from 'next/link';
import Image from 'next/image';
import { supabase } from '@/lib/supabase';

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

type TagFilterOption = { id: string; name: string; cnt?: number };
type PerformerFilterOption = { id: string; name: string; cnt?: number };

const LikedVideosDrawer: React.FC<LikedVideosDrawerProps> = ({ isOpen, onClose }) => {
  const [likedVideos, setLikedVideos] = useState<VideoRecord[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Filters and sorting (in-drawer)
  const [sort, setSort] = useState<'liked_at' | 'released' | 'price' | 'title'>('liked_at');
  const [order, setOrder] = useState<'desc' | 'asc'>('desc');
  const [tagOptions, setTagOptions] = useState<TagFilterOption[]>([]);
  const [performerOptions, setPerformerOptions] = useState<PerformerFilterOption[]>([]);
  const [selectedTagIds, setSelectedTagIds] = useState<string[]>([]);
  const [selectedPerformerIds, setSelectedPerformerIds] = useState<string[]>([]);
  const [tagSearch, setTagSearch] = useState('');
  const [performerSearch, setPerformerSearch] = useState('');
  const pageSize = 40;
  const [mobileShowFilters, setMobileShowFilters] = useState(false);
  const [tagComboboxOpen, setTagComboboxOpen] = useState(false);
  const [performerComboboxOpen, setPerformerComboboxOpen] = useState(false);
  const tagBlurTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const performerBlurTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const filteredTagOptions = useMemo(() => {
    const keyword = tagSearch.trim().toLowerCase();
    return tagOptions
      .filter((tag) => !selectedTagIds.includes(tag.id))
      .filter((tag) => {
        if (!keyword) return true;
        return tag.name.toLowerCase().includes(keyword);
      });
  }, [tagOptions, tagSearch, selectedTagIds]);

  const filteredPerformerOptions = useMemo(() => {
    const keyword = performerSearch.trim().toLowerCase();
    return performerOptions
      .filter((perf) => !selectedPerformerIds.includes(perf.id))
      .filter((perf) => {
        if (!keyword) return true;
        return perf.name.toLowerCase().includes(keyword);
      });
  }, [performerOptions, performerSearch, selectedPerformerIds]);

  const toggleTag = (id: string) => {
    setSelectedTagIds((prev) => (prev.includes(id) ? prev.filter((value) => value !== id) : [...prev, id]));
  };

  const togglePerformer = (id: string) => {
    setSelectedPerformerIds((prev) => (prev.includes(id) ? prev.filter((value) => value !== id) : [...prev, id]));
  };

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

  useEffect(() => {
    return () => {
      if (tagBlurTimerRef.current) clearTimeout(tagBlurTimerRef.current);
      if (performerBlurTimerRef.current) clearTimeout(performerBlurTimerRef.current);
    };
  }, []);

  const showTagDropdown = () => {
    if (tagBlurTimerRef.current) clearTimeout(tagBlurTimerRef.current);
    setTagComboboxOpen(true);
  };

  const hideTagDropdown = () => {
    tagBlurTimerRef.current = setTimeout(() => setTagComboboxOpen(false), 150);
  };

  const showPerformerDropdown = () => {
    if (performerBlurTimerRef.current) clearTimeout(performerBlurTimerRef.current);
    setPerformerComboboxOpen(true);
  };

  const hidePerformerDropdown = () => {
    performerBlurTimerRef.current = setTimeout(() => setPerformerComboboxOpen(false), 150);
  };

  const buildOptionsClass = (isOpen: boolean, extra: string) =>
    `absolute inset-x-0 top-full mt-2 ${extra} ${isOpen ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'} transition-opacity duration-150`;


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
        const tagRows = (tags as TagFilterOption[] | null)?.filter(Boolean) || [];
        const perfRows = (perfs as PerformerFilterOption[] | null)?.filter(Boolean) || [];
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
  }, [isOpen, sort, order, selectedTagIds, selectedPerformerIds]);

  const isDev = process.env.NODE_ENV === 'development';

  const logInteraction = (label: string, event: React.SyntheticEvent) => {
    if (!isDev) return;
    console.log('[LikedVideosDrawer]', label, event.type, {
      target: event.target,
      currentTarget: event.currentTarget,
    });
  };

  // ここではオーバーレイは常時クリック可能とし、パネル側で stopPropagation して外側判定を防ぐ。
  const handleDialogClose = () => {
    logInteraction('Dialog onClose', { type: 'close' } as any);
    onClose();
  };
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
          <DialogBackdrop className="fixed inset-0 bg-black/40" />
        </Transition.Child>

        <div className="fixed inset-0 flex items-end sm:items-stretch justify-center sm:justify-end">
          <Transition.Child
            as={Fragment}
            enter="transform transition ease-in-out duration-500"
            enterFrom="translate-y-full sm:translate-y-0 sm:translate-x-full"
            enterTo="translate-y-0 sm:translate-x-0"
            leave="transform transition ease-in-out duration-500"
            leaveFrom="translate-y-0 sm:translate-x-0"
            leaveTo="translate-y-full sm:translate-y-0 sm:translate-x-full"
          >
            <Dialog.Panel className="pointer-events-auto w-screen max-w-4xl h-[90dvh] sm:h-full bg-white/70 backdrop-blur-xl shadow-2xl sm:border border-white/60 sm:rounded-tl-3xl sm:rounded-bl-3xl overflow-hidden">
              {/* --- Unified Header --- */}
              <div className="sm:px-6 sm:py-5 p-3 grid grid-cols-[auto_1fr_auto] sm:flex items-center sm:justify-between border-b bg-white/90 sticky top-0 z-10">
                <div className="flex justify-start sm:hidden">
                  <button
                    type="button"
                    className="rounded-md text-gray-500 hover:text-gray-700 focus:outline-none focus:ring-2 focus:ring-amber-500"
                    onClick={onClose}
                    aria-label="閉じる"
                  >
                    <X className="h-6 w-6" />
                  </button>
                </div>
                <div className="text-center sm:text-left">
                  <Dialog.Title className="text-base sm:text-xl font-bold text-gray-900 whitespace-nowrap">
                    いいねした動画
                  </Dialog.Title>
                </div>
                <div className="flex justify-end sm:hidden">
                  <button
                    type="button"
                    className={`rounded-md p-2 ${mobileShowFilters ? 'text-violet-600' : 'text-gray-600'} hover:text-violet-600`}
                    onClick={() => setMobileShowFilters((v) => !v)}
                    aria-label="フィルタを切り替え"
                  >
                    <Filter className="h-5 w-5" />
                  </button>
                </div>
                <div className="hidden sm:block">
                  <button
                    type="button"
                    className="rounded-full bg-white/80 p-2 text-gray-500 hover:text-gray-700 shadow-xs focus:outline-none focus:ring-2 focus:ring-amber-500"
                    onClick={onClose}
                    aria-label="閉じる"
                  >
                    <X className="h-5 w-5" />
                  </button>
                </div>
              </div>

              {/* --- Unified Content Area --- */}
              <div className="grid h-full sm:h-[calc(100vh-81px)] grid-cols-1 lg:grid-cols-[270px_1fr] gap-6">
                {/* --- Filters --- */}
                <aside
                  className={`
                    border-r border-gray-200 bg-white/80 px-6 py-4 space-y-4 overflow-y-auto flex-col
                    ${mobileShowFilters ? 'flex' : 'hidden'} sm:flex
                  `}
                >
                  <div className="space-y-2">
                    <div className="rounded-2xl border border-gray-200 bg-gray-50 px-3 py-2 mb-4 text-[11px] font-medium text-gray-600">
                      検索結果: {likedVideos.length} 件
                    </div>
                    <div className="mt-2">
                      <span className="mt-2 text-xs font-semibold text-gray-500">検索 / 絞り込み</span>
                    </div>
                    <div className="space-y-3">
                      <div className="space-y-1">
                        <label className="text-[11px] text-gray-600">タグ名で検索</label>
                        {selectedTagIds.length > 0 && (
                          <div className="flex flex-wrap gap-2 text-[11px]">
                            {selectedTagIds.map((id) => {
                              const name = tagOptions.find((t) => t.id === id)?.name || id;
                              return (
                                <button
                                  key={`desktop-selected-tag-${id}`}
                                  onClick={() => toggleTag(id)}
                                  className="px-2 py-1 rounded-full border border-rose-500/60 bg-rose-100 text-rose-600 shadow-sm"
                                >
                                  {name} ×
                                </button>
                              );
                            })}
                          </div>
                        )}
                        <Combobox<TagFilterOption | null>
                          value={null}
                          onChange={(tag) => {
                            if (tag) toggleTag(tag.id);
                            setTagSearch('');
                          }}
                        >
                          <div className="relative">
                            <Combobox.Input
                              className="w-full rounded-2xl border border-gray-200 bg-white px-3 py-2 text-sm text-gray-900 focus:border-rose-400 focus:outline-none"
                              placeholder="タグ名で検索"
                              onFocus={showTagDropdown}
                              onBlur={hideTagDropdown}
                              onChange={(e) => setTagSearch(e.target.value)}
                              displayValue={() => tagSearch}
                            />
                            <Combobox.Button className="absolute inset-y-0 right-0 flex items-center pr-3 text-gray-400">
                              <Filter className="h-4 w-4" />
                            </Combobox.Button>
                            <Combobox.Options
                              static
                              className={buildOptionsClass(
                                tagComboboxOpen,
                                'max-h-56 overflow-y-auto rounded-2xl border border-gray-200 bg-white shadow-lg z-30'
                              )}
                            >
                              {filteredTagOptions.length === 0 ? (
                                <div className="px-3 py-2 text-xs text-gray-400">該当するタグがありません</div>
                              ) : (
                                filteredTagOptions.map((tag) => (
                                  <Combobox.Option
                                    key={tag.id}
                                    value={tag}
                                    className={({ active }) =>
                                      `flex cursor-pointer items-center justify-between px-3 py-2 text-sm transition ${
                                        active ? 'bg-gray-100 text-gray-900' : 'bg-white text-gray-800'
                                      }`
                                    }
                                  >
                                    <span>#{tag.name}</span>
                                    {tag.cnt ? <span className="text-xs text-gray-400">({tag.cnt})</span> : null}
                                  </Combobox.Option>
                                ))
                              )}
                            </Combobox.Options>
                          </div>
                        </Combobox>
                      </div>
                      <div className="space-y-1">
                        <label className="text-[11px] text-gray-600">出演者名で検索</label>
                        {selectedPerformerIds.length > 0 && (
                          <div className="flex flex-wrap gap-2 text-[11px]">
                            {selectedPerformerIds.map((id) => {
                              const name = performerOptions.find((p) => p.id === id)?.name || id;
                              return (
                                <button
                                  key={`desktop-selected-perf-${id}`}
                                  onClick={() => togglePerformer(id)}
                                  className="px-2 py-1 rounded-full border border-indigo-500/60 bg-indigo-100 text-indigo-600 shadow-sm"
                                >
                                  {name} ×
                                </button>
                              );
                            })}
                          </div>
                        )}
                        <Combobox<PerformerFilterOption | null>
                          value={null}
                          onChange={(perf) => {
                            if (perf) togglePerformer(perf.id);
                            setPerformerSearch('');
                          }}
                        >
                          <div className="relative">
                            <Combobox.Input
                              className="w-full rounded-2xl border border-gray-200 bg-white px-3 py-2 text-sm text-gray-900 focus:border-indigo-400 focus:outline-none"
                              placeholder="出演者名で検索"
                              onFocus={showPerformerDropdown}
                              onBlur={hidePerformerDropdown}
                              onChange={(e) => setPerformerSearch(e.target.value)}
                              displayValue={() => performerSearch}
                            />
                            <Combobox.Button className="absolute inset-y-0 right-0 flex items-center pr-3 text-gray-400">
                              <Filter className="h-4 w-4" />
                            </Combobox.Button>
                            <Combobox.Options
                              static
                              className={buildOptionsClass(
                                performerComboboxOpen,
                                'max-h-56 overflow-y-auto rounded-2xl border border-gray-200 bg-white shadow-lg z-30'
                              )}
                            >
                              {filteredPerformerOptions.length === 0 ? (
                                <div className="px-3 py-2 text-xs text-gray-400">該当する出演者がいません</div>
                              ) : (
                                filteredPerformerOptions.map((perf) => (
                                  <Combobox.Option
                                    key={perf.id}
                                    value={perf}
                                    className={({ active }) =>
                                      `flex cursor-pointer items-center justify-between px-3 py-2 text-sm transition ${
                                        active ? 'bg-gray-100 text-gray-900' : 'bg-white text-gray-800'
                                      }`
                                    }
                                  >
                                    <span>{perf.name}</span>
                                    {perf.cnt ? <span className="text-xs text-gray-400">({perf.cnt})</span> : null}
                                  </Combobox.Option>
                                ))
                              )}
                            </Combobox.Options>
                          </div>
                        </Combobox>
                      </div>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <span className="text-xs font-semibold text-gray-500">並び替え</span>
                    <div className="flex gap-2">
                      <select
                        className="flex-1 rounded-2xl border border-gray-300 bg-white px-3 py-2 text-sm text-gray-800"
                        value={sort}
                        onChange={(e) => setSort(e.target.value as 'liked_at' | 'released' | 'price' | 'title')}
                      >
                        <option value="liked_at">いいね日時</option>
                        <option value="released">発売日</option>
                        <option value="price">価格</option>
                        <option value="title">タイトル</option>
                      </select>
                      <select
                        className="w-28 rounded-2xl border border-gray-300 bg-white px-3 py-2 text-sm text-gray-800"
                        value={order}
                        onChange={(e) => setOrder(e.target.value as 'desc' | 'asc')}
                      >
                        <option value="desc">降順</option>
                        <option value="asc">昇順</option>
                      </select>
                    </div>
                  </div>
                </aside>

                {/* --- Video List --- */}
                <section className="px-6 py-6 overflow-y-auto space-y-4">
                  {error && <div className="text-sm text-red-600">{error}</div>}
                  {loading ? (
                    <p className="text-center text-gray-500">読み込み中...</p>
                  ) : likedVideos.length > 0 ? (
                    <div className="space-y-4">
                      {likedVideos.map((video) => (
                        <article
                          key={video.external_id}
                          className="grid gap-4 rounded-3xl border border-white/60 bg-white/80 p-4 shadow-[0_35px_60px_-40px_rgba(0,0,0,0.5)] backdrop-blur-xl lg:grid-cols-[220px_1fr]"
                        >
                          <div
                            className="relative w-full rounded-2xl overflow-hidden bg-gray-200"
                            style={{ paddingBottom: '70%' }}
                          >
                            {video.thumbnail_url ? (
                              <Image src={video.thumbnail_url} alt={video.title} fill className="object-cover" />
                            ) : (
                              <div className="flex h-full items-center justify-center text-xs text-gray-400">
                                画像なし
                              </div>
                            )}
                          </div>
                          <div className="flex flex-col justify-between gap-3">
                            <div>
                              <h3 className="text-base font-semibold text-gray-900 line-clamp-3">{video.title}</h3>
                              <p className="mt-1 text-sm text-gray-500">
                                {video.series || video.maker || 'シリーズ情報なし'}
                              </p>
                              {video.product_url ? (
                                <p className="text-xs text-gray-400 mt-1">{new URL(video.product_url).hostname}</p>
                              ) : null}
                            </div>
                            <div className="flex flex-wrap items-center justify-between gap-3">
                              <span className="text-lg font-bold text-[#FF6B81]">
                                {video.price ? `￥${video.price.toLocaleString()}~` : '価格情報なし'}
                              </span>
                              <Link
                                href={toFanzaAffiliate(video.product_url) || '#'}
                                passHref
                                target="_blank"
                                rel="noopener noreferrer"
                              >
                                <button className="rounded-full border border-[#FF6B81] px-4 py-2 text-sm font-bold text-[#FF6B81] transition hover:bg-[#FF6B81] hover:text-white">
                                  見る
                                </button>
                              </Link>
                            </div>
                          </div>
                        </article>
                      ))}
                    </div>
                  ) : (
                    <p className="text-center text-gray-500">いいねした動画はありません。</p>
                  )}
                </section>
              </div>
            </Dialog.Panel>
          </Transition.Child>
        </div>
      </Dialog>
    </Transition>
  );
};

export default LikedVideosDrawer;