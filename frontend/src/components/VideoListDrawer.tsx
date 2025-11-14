'use client';

import { Dialog, Transition, Combobox, DialogBackdrop } from '@headlessui/react';
import Image from 'next/image';
import Link from 'next/link';
import { Fragment, useEffect, useMemo, useRef, useState } from 'react';
import { X, Filter, Tag, Users, ExternalLink } from 'lucide-react';

export interface VideoRecord {
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

export type TagFilterOption = { id: string; name: string; cnt?: number };
export type PerformerFilterOption = { id: string; name: string; cnt?: number };
export type SortKey = 'liked_at' | 'released' | 'price' | 'title';
export type SortOrder = 'asc' | 'desc';

const formatDate = (value?: string | null) => {
  if (!value) return '—';
  try {
    return new Date(value).toLocaleDateString('ja-JP');
  } catch {
    return '—';
  }
};

const formatPrice = (price?: number | null) => {
  if (!price || price <= 0) return '価格情報なし';
  return `￥${price.toLocaleString()}`;
};

const defaultAffiliateBuilder = (raw?: string | null) => {
  if (!raw) return '#';
  const AF_ID = 'yotadata2-001';
  try {
    if (raw.startsWith('https://al.fanza.co.jp/')) {
      const url = new URL(raw);
      url.searchParams.set('af_id', AF_ID);
      return url.toString();
    }
  } catch {
    /* noop */
  }
  const lurl = encodeURIComponent(raw);
  return `https://al.fanza.co.jp/?lurl=${lurl}&af_id=${encodeURIComponent(AF_ID)}&ch=link_tool&ch_id=link`;
};

export interface VideoListDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  title?: string;
  videos: VideoRecord[];
  loading: boolean;
  error: string | null;
  sort: SortKey;
  order: SortOrder;
  onChangeSort: (value: SortKey) => void;
  onChangeOrder: (value: SortOrder) => void;
  tagOptions: TagFilterOption[];
  performerOptions: PerformerFilterOption[];
  selectedTagIds: string[];
  selectedPerformerIds: string[];
  onToggleTag: (id: string) => void;
  onTogglePerformer: (id: string) => void;
  buildAffiliateHref?: (url?: string | null) => string | undefined;
  totalCount?: number | null;
}

export default function VideoListDrawer({
  isOpen,
  onClose,
  title = '作品リスト',
  videos,
  loading,
  error,
  sort,
  order,
  onChangeSort,
  onChangeOrder,
  tagOptions,
  performerOptions,
  selectedTagIds,
  selectedPerformerIds,
  onToggleTag,
  onTogglePerformer,
  buildAffiliateHref = defaultAffiliateBuilder,
  totalCount,
}: VideoListDrawerProps) {
  const [tagSearch, setTagSearch] = useState('');
  const [performerSearch, setPerformerSearch] = useState('');
  const [mobileShowFilters, setMobileShowFilters] = useState(false);
  const [tagComboboxOpen, setTagComboboxOpen] = useState(false);
  const [performerComboboxOpen, setPerformerComboboxOpen] = useState(false);
  const tagBlurTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const performerBlurTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => {
      if (tagBlurTimerRef.current) clearTimeout(tagBlurTimerRef.current);
      if (performerBlurTimerRef.current) clearTimeout(performerBlurTimerRef.current);
    };
  }, []);

  const filteredTagOptions = useMemo(() => {
    const keyword = tagSearch.trim().toLowerCase();
    return tagOptions
      .filter((tag) => !selectedTagIds.includes(tag.id))
      .filter((tag) => (keyword ? tag.name.toLowerCase().includes(keyword) : true));
  }, [tagOptions, selectedTagIds, tagSearch]);

  const filteredPerformerOptions = useMemo(() => {
    const keyword = performerSearch.trim().toLowerCase();
    return performerOptions
      .filter((perf) => !selectedPerformerIds.includes(perf.id))
      .filter((perf) => (keyword ? perf.name.toLowerCase().includes(keyword) : true));
  }, [performerOptions, selectedPerformerIds, performerSearch]);

  const buildOptionsClass = (isOpenDropdown: boolean, extra: string) =>
    `absolute inset-x-0 top-full mt-2 z-20 ${extra} ${
      isOpenDropdown ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'
    } transition-opacity duration-150`;

  const handleClose = () => {
    setMobileShowFilters(false);
    onClose();
  };

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

  return (
    <Transition appear show={isOpen} as={Fragment}>
      <Dialog as="div" className="relative z-50" onClose={handleClose}>
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
            <Dialog.Panel className="pointer-events-auto w-screen max-w-4xl h-[90dvh] sm:h-full bg-white/70 backdrop-blur-xl shadow-2xl sm:border border-white/60 overflow-hidden">
              <div className="sm:px-6 sm:py-5 p-3 grid grid-cols-[auto_1fr_auto] sm:flex items-center sm:justify-between border-b bg-white/90 sticky top-0 z-10">
                <div className="flex justify-start sm:hidden">
                  <button
                    type="button"
                    className="rounded-md text-gray-500 hover:text-gray-700 focus:outline-none focus:ring-2 focus:ring-amber-500"
                    onClick={handleClose}
                    aria-label="閉じる"
                  >
                    <X className="h-6 w-6" />
                  </button>
                </div>
                <div className="text-center sm:text-left">
                  <Dialog.Title className="text-base sm:text-xl font-bold text-gray-900 whitespace-nowrap">
                    {title}
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
                    onClick={handleClose}
                    aria-label="閉じる"
                  >
                    <X className="h-5 w-5" />
                  </button>
                </div>
              </div>

              <div className="grid h-full sm:h-[calc(100vh-81px)] grid-cols-1 lg:grid-cols-[270px_1fr]">
                {/* Filters */}
                <aside
                  className={`border-r border-gray-200 bg-white/80 px-6 py-4 space-y-4 overflow-y-auto flex-col ${
                    mobileShowFilters ? 'flex' : 'hidden'
                  } sm:flex`}
                >
                  <div className="space-y-2">
                    <div>
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
                                  key={`selected-tag-${id}`}
                                  onClick={() => onToggleTag(id)}
                                  className="px-2 py-1 rounded-full border border-rose-500/60 bg-rose-100 text-rose-600"
                                >
                                  #{name} ×
                                </button>
                              );
                            })}
                          </div>
                        )}
                        <div className="relative">
                          <Combobox value="" onChange={(val: string) => onToggleTag(val)}>
                            <Combobox.Input
                              onFocus={showTagDropdown}
                              onBlur={hideTagDropdown}
                              placeholder="タグを検索"
                              value={tagSearch}
                              onChange={(event) => setTagSearch(event.target.value)}
                              className="w-full rounded-xl border border-gray-200 bg-white px-3 py-1.5 text-sm text-gray-900 shadow-sm focus:outline-none focus:ring-2 focus:ring-violet-500"
                            />
                            <div className={buildOptionsClass(tagComboboxOpen, 'rounded-2xl border border-gray-200 bg-white shadow-lg max-h-48 overflow-auto')}>
                              <Combobox.Options static className="py-2 text-sm text-gray-700">
                                {filteredTagOptions.length === 0 ? (
                                  <p className="px-3 py-2 text-xs text-gray-400">該当するタグはありません</p>
                                ) : (
                                  filteredTagOptions.map((tag) => (
                                    <Combobox.Option
                                      key={`tag-${tag.id}`}
                                      value={tag.id}
                                      className="px-3 py-1.5 cursor-pointer hover:bg-gray-50 flex justify-between text-sm"
                                    >
                                      <span>#{tag.name}</span>
                                      {tag.cnt ? <span className="text-xs text-gray-400">{tag.cnt}</span> : null}
                                    </Combobox.Option>
                                  ))
                                )}
                              </Combobox.Options>
                            </div>
                          </Combobox>
                        </div>
                      </div>

                      <div className="space-y-1">
                        <label className="text-[11px] text-gray-600">出演者で検索</label>
                        {selectedPerformerIds.length > 0 && (
                          <div className="flex flex-wrap gap-2 text-[11px]">
                            {selectedPerformerIds.map((id) => {
                              const name = performerOptions.find((p) => p.id === id)?.name || id;
                              return (
                                <button
                                  key={`perf-${id}`}
                                  onClick={() => onTogglePerformer(id)}
                                  className="px-2 py-1 rounded-full border border-indigo-500/60 bg-indigo-100 text-indigo-600"
                                >
                                  {name} ×
                                </button>
                              );
                            })}
                          </div>
                        )}
                        <div className="relative">
                          <Combobox value="" onChange={(val: string) => onTogglePerformer(val)}>
                            <Combobox.Input
                              onFocus={showPerformerDropdown}
                              onBlur={hidePerformerDropdown}
                              placeholder="出演者を検索"
                              value={performerSearch}
                              onChange={(event) => setPerformerSearch(event.target.value)}
                              className="w-full rounded-xl border border-gray-200 bg-white px-3 py-1.5 text-sm text-gray-900 shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
                            />
                            <div className={buildOptionsClass(performerComboboxOpen, 'rounded-2xl border border-gray-200 bg-white shadow-lg max-h-48 overflow-auto')}>
                              <Combobox.Options static className="py-2 text-sm text-gray-700">
                                {filteredPerformerOptions.length === 0 ? (
                                  <p className="px-3 py-2 text-xs text-gray-400">該当する出演者がありません</p>
                                ) : (
                                  filteredPerformerOptions.map((performer) => (
                                    <Combobox.Option
                                      key={`performer-${performer.id}`}
                                      value={performer.id}
                                      className="px-3 py-1.5 cursor-pointer hover:bg-gray-50 flex justify-between text-sm"
                                    >
                                      <span>{performer.name}</span>
                                      {performer.cnt ? <span className="text-xs text-gray-400">{performer.cnt}</span> : null}
                                    </Combobox.Option>
                                  ))
                                )}
                              </Combobox.Options>
                            </div>
                          </Combobox>
                        </div>
                      </div>

                      <div className="space-y-1 text-xs text-gray-600">
                        <label className="font-semibold text-gray-500">並び替え</label>
                        <div className="flex gap-2">
                          <select
                            value={sort}
                            onChange={(e) => onChangeSort(e.target.value as SortKey)}
                            className="flex-1 rounded-xl border border-gray-200 bg-white px-3 py-1.5 text-sm text-gray-900 shadow-sm focus:outline-none focus:ring-2 focus:ring-violet-500"
                          >
                            <option value="liked_at">LIKE日時</option>
                            <option value="released">発売日</option>
                            <option value="price">価格</option>
                            <option value="title">タイトル</option>
                          </select>
                          <select
                            value={order}
                            onChange={(e) => onChangeOrder(e.target.value as SortOrder)}
                            className="w-28 rounded-xl border border-gray-200 bg-white px-2 py-1.5 text-sm text-gray-900 shadow-sm focus:outline-none focus:ring-2 focus:ring-violet-500"
                          >
                            <option value="desc">降順</option>
                            <option value="asc">昇順</option>
                          </select>
                        </div>
                      </div>
                    </div>
                  </div>
                </aside>

                <section className="flex flex-col overflow-y-auto">
                  <div className="sticky top-0 z-10 bg-white/50 backdrop-blur px-4 py-3 border-b border-white/40 text-xs sm:text-sm text-gray-700 flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
                    <span>
                      表示中: <strong>{videos.length.toLocaleString('ja-JP')}</strong> 件
                    </span>
                    <span>
                      検索結果:{' '}
                      <strong>{typeof totalCount === 'number' ? totalCount.toLocaleString('ja-JP') : '—'}</strong> 件
                    </span>
                  </div>
                  {error ? (
                    <div className="m-4 rounded-xl bg-red-100 border border-red-300 text-sm text-red-700 px-4 py-3">
                      {error}
                    </div>
                  ) : null}
                  {loading ? (
                    <div className="p-6 grid grid-cols-1 md:grid-cols-2 gap-4">
                      {Array.from({ length: 4 }).map((_, idx) => (
                        <div key={`skeleton-${idx}`} className="h-40 rounded-2xl bg-gray-100 animate-pulse" />
                      ))}
                    </div>
                  ) : (
                    <div className="p-4 space-y-4 overflow-y-auto rounded-3xl">
                      {videos.map((video) => (
                        <article
                          key={video.external_id}
                          className="rounded-2xl border border-gray-200 bg-white text-gray-900 overflow-hidden flex flex-row gap-3 p-3 shadow-sm"
                        >
                          <div className="relative w-60 aspect-[5/4] rounded-xl overflow-hidden bg-slate-200">
                            {video.thumbnail_url ? (
                              <Image src={video.thumbnail_url} alt={video.title} fill className="object-cover" />
                            ) : (
                              <div className="w-full h-full flex items-center justify-center text-sm text-gray-500">No Image</div>
                            )}
                          </div>
                          <div className="flex flex-col gap-3 flex-1 min-w-0">
                            <h2 className="text-base font-semibold leading-tight line-clamp-2">{video.title}</h2>
                            <div className="text-xs text-gray-600 space-y-1">
                              <p>{formatPrice(video.price)}</p>
                              <p>リリース日: {formatDate(video.product_released_at)}</p>
                            </div>
                            <div className="flex flex-wrap gap-1.5 text-[11px] text-gray-600">
                              {video.tags?.slice(0, 3).map((tag) => (
                                <span key={tag.id} className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-gray-100 border border-gray-200">
                                  <Tag size={12} />
                                  {tag.name}
                                </span>
                              ))}
                              {video.performers?.slice(0, 2).map((perf) => (
                                <span key={perf.id} className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-gray-100 border border-gray-200">
                                  <Users size={12} />
                                  {perf.name}
                                </span>
                              ))}
                            </div>
                            <div className="mt-auto flex gap-2">
                              <Link
                                href={buildAffiliateHref(video.product_url) ?? '#'}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="inline-flex items-center justify-center gap-2 px-4 py-2 rounded-full bg-rose-500 text-white text-sm font-semibold hover:bg-rose-400 transition"
                              >
                                作品ページへ
                                <ExternalLink size={14} />
                              </Link>
                            </div>
                          </div>
                        </article>
                      ))}
                      {videos.length === 0 && !loading ? (
                        <div className="rounded-2xl border border-dashed border-gray-300 bg-white text-gray-600 p-8 text-center">
                          条件に一致する作品がありませんでした。
                        </div>
                      ) : null}
                    </div>
                  )}
                </section>
              </div>
            </Dialog.Panel>
          </Transition.Child>
        </div>
      </Dialog>
    </Transition>
  );
}
