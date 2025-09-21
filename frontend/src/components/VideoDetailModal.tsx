'use client';

import { Dialog, Transition } from '@headlessui/react';
import { Fragment, useEffect, useState } from 'react';
import { X } from 'lucide-react';
import { supabase } from '@/lib/supabase';
import Link from 'next/link';

type Performer = { id: string; name: string };
type Tag = { id: string; name: string };

type VideoRow = {
  id: string;
  external_id: string;
  title: string;
  description?: string | null;
  thumbnail_url?: string | null;
  product_url?: string | null;
  price?: number | null;
  product_released_at?: string | null;
  director?: string | null;
  series?: string | null;
  maker?: string | null;
  label?: string | null;
};

function toFanzaAffiliate(raw: string | null | undefined): string | undefined {
  if (!raw) return undefined;
  if (raw.startsWith('https://al.fanza.co.jp/')) return raw;
  const afId = process.env.NEXT_PUBLIC_FANZA_AFFILIATE_ID || '';
  if (!afId) return raw;
  return `https://al.fanza.co.jp/?lurl=${encodeURIComponent(raw)}&af_id=${encodeURIComponent(afId)}&ch=link_tool&ch_id=link`;
}

export default function VideoDetailModal({ isOpen, onClose, videoId }: { isOpen: boolean; onClose: () => void; videoId: string | null; }) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [video, setVideo] = useState<VideoRow | null>(null);
  const [performers, setPerformers] = useState<Performer[]>([]);
  const [tags, setTags] = useState<Tag[]>([]);

  useEffect(() => {
    if (!isOpen || !videoId) return;
    const run = async () => {
      setLoading(true);
      setError(null);
      try {
        const { data: v, error: ve } = await supabase
          .from('videos')
          .select('*')
          .eq('id', videoId)
          .maybeSingle();
        if (ve) throw ve;
        if (!v) {
          setError('見つかりませんでした');
          return;
        }
        const [{ data: perf }, { data: tg }] = await Promise.all([
          supabase.from('video_performers').select('performers(id, name)').eq('video_id', videoId),
          supabase.from('video_tags').select('tags(id, name)').eq('video_id', videoId),
        ]);
        setVideo(v as VideoRow);
        setPerformers(((perf || []) as { performers: Performer | null }[])
          .map(r => r.performers)
          .filter((p): p is Performer => Boolean(p))
        );
        setTags(((tg || []) as { tags: Tag | null }[])
          .map(r => r.tags)
          .filter((t): t is Tag => Boolean(t))
        );
      } catch (e: unknown) {
        let message = '読み込みに失敗しました';
        if (typeof e === 'string') message = e;
        else if (e && typeof e === 'object' && 'message' in e && typeof (e as { message?: unknown }).message === 'string') {
          message = (e as { message: string }).message;
        }
        setError(message);
      } finally {
        setLoading(false);
      }
    };
    run();
  }, [isOpen, videoId]);

  const fanzaEmbedUrl = video ? `https://www.dmm.co.jp/litevideo/-/part/=/affi_id=${process.env.NEXT_PUBLIC_FANZA_AFFILIATE_ID}/cid=${video.external_id}/size=1280_720/` : '';

  return (
    <Transition appear show={isOpen} as={Fragment}>
      <Dialog as="div" className="relative z-[60]" onClose={onClose}>
        <Transition.Child as={Fragment} enter="ease-out duration-200" enterFrom="opacity-0" enterTo="opacity-100" leave="ease-in duration-150" leaveFrom="opacity-100" leaveTo="opacity-0">
          <div className="fixed inset-0 bg-black/50" />
        </Transition.Child>
        <div className="fixed inset-0 overflow-y-auto p-4">
          <div className="mx-auto w-full max-w-4xl">
            <Transition.Child as={Fragment} enter="ease-out duration-200" enterFrom="opacity-0 scale-95" enterTo="opacity-100 scale-100" leave="ease-in duration-150" leaveFrom="opacity-100 scale-100" leaveTo="opacity-0 scale-95">
              <Dialog.Panel className="relative bg-white rounded-2xl shadow-2xl overflow-hidden">
                <button onClick={onClose} className="absolute top-3 right-3 text-gray-600 hover:text-gray-900" aria-label="閉じる">
                  <X size={22} />
                </button>
                <div className="w-full bg-black relative" style={{ paddingBottom: '56%' }}>
                  {video && (
                    <iframe
                      src={fanzaEmbedUrl}
                      title="Embedded Video Player"
                      frameBorder="0"
                      allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share; fullscreen"
                      loading="eager"
                      className="absolute inset-0 w-full h-full"
                    />
                  )}
                </div>
                <div className="p-4">
                  {loading ? (
                    <div className="py-6 text-center text-gray-500">読み込み中...</div>
                  ) : error ? (
                    <div className="py-6 text-center text-red-600">{error}</div>
                  ) : video ? (
                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                      <div className="sm:col-span-2 space-y-3">
                        <h2 className="text-lg font-bold">{video.title}</h2>
                        {video.product_released_at && (
                          <div className="text-sm text-gray-600">発売日: {new Date(video.product_released_at).toLocaleDateString('ja-JP')}</div>
                        )}
                        {typeof video.price === 'number' && (
                          <div className="text-sm text-gray-600">価格: ￥{Number(video.price).toLocaleString()}</div>
                        )}
                        {video.description && (
                          <p className="text-sm text-gray-700 whitespace-pre-wrap">{video.description}</p>
                        )}
                        {performers.length > 0 && (
                          <div className="text-sm">
                            <div className="text-gray-600 mb-1">出演:</div>
                            <div className="flex flex-wrap gap-2">
                              {performers.map(p => (
                                <span key={p.id} className="px-2 py-0.5 rounded-full bg-pink-300 text-white text-xs font-bold">{p.name}</span>
                              ))}
                            </div>
                          </div>
                        )}
                        {tags.length > 0 && (
                          <div className="text-sm">
                            <div className="text-gray-600 mb-1">タグ:</div>
                            <div className="flex flex-wrap gap-2">
                              {tags.map(t => (
                                <span key={t.id} className="px-2 py-0.5 rounded-full bg-purple-300 text-white text-xs font-bold">{t.name}</span>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                      <div>
                        <div className="border rounded-lg p-3">
                          <div className="text-sm text-gray-700 mb-2">外部サイトで見る</div>
                          <Link href={toFanzaAffiliate(video.product_url) || '#'} target="_blank" className="block w-full text-center bg-amber-500 text-white font-bold rounded-lg py-2">
                            商品ページへ
                          </Link>
                        </div>
                      </div>
                    </div>
                  ) : null}
                </div>
              </Dialog.Panel>
            </Transition.Child>
          </div>
        </div>
      </Dialog>
    </Transition>
  );
}
