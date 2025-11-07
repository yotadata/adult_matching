'use client';

import { useMemo } from 'react';
import Image from 'next/image';
import Link from 'next/link';
import { Tag, Users, ExternalLink } from 'lucide-react';
import { useLikedVideos } from '@/hooks/useLikedVideos';

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
    /* ignore */
  }
  return `https://al.fanza.co.jp/?lurl=${encodeURIComponent(raw)}&af_id=${encodeURIComponent(AF_ID)}&ch=link_tool&ch_id=link`;
};

export default function LikedListPage() {
  const { videos, loading, error, isAuthenticated } = useLikedVideos(80);

  const summaryText = useMemo(() => {
    if (loading) return '読み込み中...';
    if (!isAuthenticated) return 'ログインして、ここにリストを作りましょう';
    return `${videos.length} 件の作品`;
  }, [loading, isAuthenticated, videos.length]);

  return (
    <main className="min-h-screen px-4 sm:px-8 py-10 bg-slate-950 text-white">
      <div className="max-w-6xl mx-auto flex flex-col gap-6">
        <header className="flex flex-col gap-2">
          <p className="text-xs uppercase tracking-[0.4em] text-white/60">List</p>
          <h1 className="text-3xl sm:text-4xl font-bold">いいねした作品</h1>
          <p className="text-white/80 text-sm sm:text-base">{summaryText}</p>
        </header>

        {error ? (
          <div className="rounded-2xl bg-red-500/15 border border-red-400/40 text-sm text-red-200 px-4 py-3">
            {error}
          </div>
        ) : null}

        {!isAuthenticated && !loading ? (
          <div className="rounded-2xl bg-white/10 border border-white/10 p-8 text-center space-y-4">
            <p className="text-sm text-white/80">ログインすると、LIKE した作品を自動的にリストへ保存します。</p>
            <button
              type="button"
              onClick={async () => {
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
                className="rounded-2xl border border-white/10 bg-white/5 backdrop-blur-sm overflow-hidden flex flex-col sm:flex-row gap-4 p-4"
              >
                <div className="relative w-full sm:w-48 aspect-video rounded-xl overflow-hidden bg-slate-900/40">
                  {video.thumbnail_url ? (
                    <Image
                      src={video.thumbnail_url}
                      alt={video.title}
                      fill
                      className="object-cover"
                    />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center text-sm text-white/50">
                      No Image
                    </div>
                  )}
                </div>
                <div className="flex flex-col gap-3 flex-1 min-w-0">
                  <h2 className="text-lg font-semibold line-clamp-2">{video.title}</h2>
                  <div className="text-sm text-white/70 space-y-1">
                    <p>{formatPrice(video.price)}</p>
                    <p>リリース日: {formatDate(video.product_released_at)}</p>
                  </div>
                  <div className="flex flex-wrap gap-2 text-xs text-white/70">
                    {video.tags?.slice(0, 3).map((tag) => (
                      <span key={tag.id} className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-white/10 border border-white/20">
                        <Tag size={12} />
                        {tag.name}
                      </span>
                    ))}
                    {video.performers?.slice(0, 2).map((perf) => (
                      <span key={perf.id} className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-white/10 border border-white/20">
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
                      className="inline-flex items-center justify-center gap-2 px-4 py-2 rounded-full bg-white text-gray-900 text-sm font-semibold hover:bg-gray-200 transition"
                    >
                      作品ページへ
                      <ExternalLink size={14} />
                    </Link>
                  </div>
                </div>
              </article>
            ))}
            {videos.length === 0 && !loading ? (
              <div className="col-span-full rounded-2xl border border-white/10 bg-white/5 p-8 text-center text-white/70">
                まだ LIKE した作品がありません。
              </div>
            ) : null}
          </section>
        )}
      </div>
    </main>
  );
}
