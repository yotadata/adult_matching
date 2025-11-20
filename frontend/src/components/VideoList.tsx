'use client';

import Image from 'next/image';
import Link from 'next/link';
import { Tag, Users, ExternalLink } from 'lucide-react';
import { Video } from '@/types/video';
import { isUpcomingRelease } from '@/lib/videoMeta';

const GRADIENT = 'linear-gradient(90deg, #C4C8E3 0%, #D7D1E3 33.333%, #F7D7E0 66.666%, #F9C9D6 100%)';

const formatPrice = (price?: number | null) => {
  if (!price || price <= 0) return '価格情報なし';
  return `￥${price.toLocaleString()}`;
};

const formatDate = (value?: string | null) => {
  if (!value) return '—';
  try {
    return new Date(value).toLocaleDateString('ja-JP');
  } catch {
    return '—';
  }
};

const toAffiliateUrl = (raw?: string | null) => {
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
  return `https://al.fanza.co.jp/?lurl=${encodeURIComponent(raw)}&af_id=${encodeURIComponent(AF_ID)}&ch=link_tool&ch_id=link`;
};

interface VideoListProps {
  title: string;
  description: string;
  videos: Video[];
  loading: boolean;
  error: string | null;
  isAuthenticated: boolean | null;
}

export default function VideoList({ title, description, videos, loading, error, isAuthenticated }: VideoListProps) {
  return (
    <main className="min-h-screen px-4 sm:px-8 py-10 text-white" style={{ background: GRADIENT }}>
      <section className="w-full max-w-6xl mx-auto rounded-none sm:rounded-2xl bg-white/20 backdrop-blur-xl border border-white/30 shadow-[0_20px_60px_rgba(0,0,0,0.25)] px-4 sm:px-10 py-8 flex flex-col gap-8">
        <header className="space-y-3">
          <p className="text-sm uppercase tracking-[0.2em] text-white/60">Playlist</p>
          <h1 className="text-3xl sm:text-4xl font-bold">{title}</h1>
          <p className="text-white/80 text-sm sm:text-base">{description}</p>
        </header>

        {error ? (
          <div className="rounded-2xl bg-red-500/15 border border-red-400/40 text-sm text-red-200 px-4 py-3">
            {error}
          </div>
        ) : null}

        {!isAuthenticated && !loading ? (
          <div className="rounded-2xl bg-white/10 border border-white/15 p-8 text-center space-y-4">
            <p className="text-sm text-white/80">ログインすると、「気になる」とした作品を自動的に気になるリストへ保存します。</p>
            <button
              type="button"
              onClick={() => {
                const event = new CustomEvent('open-auth-modal');
                window.dispatchEvent(event);
              }}
              className="inline-flex items-center justify-center px-6 py-3 rounded-full bg-white text-gray-900 font-semibold hover:bg-gray-200 transition"
            >
              ログインして確認する
            </button>
          </div>
        ) : (
          <section className="grid grid-cols-1 md:grid-cols-2 gap-5">
            {videos.map((video) => (
              <article
                key={video.external_id}
                className="rounded-2xl border border-white/30 bg-white/95 text-gray-900 overflow-hidden flex flex-row gap-4 p-4 shadow-lg"
              >
                <div className="relative w-48 aspect-video rounded-xl overflow-hidden bg-slate-200">
                  {video.thumbnail_url ? (
                    <Image
                      src={video.thumbnail_url}
                      alt={video.title}
                      fill
                      className="object-cover"
                    />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center text-sm text-gray-500">
                      No Image
                    </div>
                  )}
                </div>
                <div className="flex flex-col gap-3 flex-1 min-w-0">
                  <h2 className="text-lg font-semibold line-clamp-2">{video.title}</h2>
                  <div className="text-sm text-gray-600 space-y-1">
                    <p>{formatPrice(video.price)}</p>
                    <div className="flex flex-wrap items-center gap-2">
                      <p>発売日: {formatDate(video.product_released_at)}</p>
                      {isUpcomingRelease(video.product_released_at) ? (
                        <span className="inline-flex items-center gap-1 rounded-full bg-amber-100/80 border border-amber-200 px-2 py-0.5 text-xs text-amber-700">
                          予約作品
                        </span>
                      ) : null}
                    </div>
                  </div>
                  <div className="flex flex-wrap gap-2 text-xs text-gray-600">
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
                      href={toAffiliateUrl(video.product_url)}
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
              <div className="col-span-full rounded-2xl border border-white/30 bg-white/80 text-gray-600 p-8 text-center">
                まだ「気になる」とした作品がありません。
              </div>
            ) : null}
          </section>
        )}
      </section>
    </main>
  );
}
