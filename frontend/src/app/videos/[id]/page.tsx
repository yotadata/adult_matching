'use client';

import { useEffect, useState } from 'react';
import { useParams } from 'next/navigation';
import { supabase } from '@/lib/supabase';
import Image from 'next/image';
import Link from 'next/link';

type Performer = { id: string; name: string };
type Tag = { id: string; name: string };

type Video = {
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
  performers: Performer[];
  tags: Tag[];
};

export default function VideoDetailPage() {
  const params = useParams<{ id: string }>();
  const id = params?.id;
  const [video, setVideo] = useState<Video | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!id) return;
    const run = async () => {
      setLoading(true);
      setError(null);
      try {
        // Fetch video
        const { data: v, error: ve } = await supabase
          .from('videos')
          .select('*')
          .eq('id', id)
          .maybeSingle();
        if (ve) throw ve;
        if (!v) {
          setError('見つかりませんでした');
          return;
        }
        // Fetch performers
        const { data: perf } = await supabase
          .from('video_performers')
          .select('performers(id, name)')
          .eq('video_id', id);
        const performersRows = (perf || []) as { performers: Performer | Performer[] | null }[];
        const performers: Performer[] = performersRows.flatMap((r) => {
          const p = r.performers;
          if (Array.isArray(p)) return p.filter(Boolean);
          return p ? [p] : [];
        });
        // Fetch tags
        const { data: tg } = await supabase
          .from('video_tags')
          .select('tags(id, name)')
          .eq('video_id', id);
        const tagRows = (tg || []) as { tags: Tag | Tag[] | null }[];
        const tags: Tag[] = tagRows.flatMap((r) => {
          const t = r.tags;
          if (Array.isArray(t)) return t.filter(Boolean);
          return t ? [t] : [];
        });
        setVideo({ ...v, performers, tags });
      } catch (e: unknown) {
        let message = '読み込みエラー';
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
  }, [id]);

  if (loading) return <div className="max-w-4xl mx-auto px-4 py-8">読み込み中...</div>;
  if (error) return <div className="max-w-4xl mx-auto px-4 py-8 text-red-600">{error}</div>;
  if (!video) return null;

<<<<<<< HEAD
  const toFanzaAffiliate = (raw: string | null | undefined): string | undefined => {
    if (!raw) return undefined;
    const AF_ID = 'yotadata2-001';
    try {
      if (raw.startsWith('https://al.fanza.co.jp/')) {
        const url = new URL(raw);
        url.searchParams.set('af_id', AF_ID);
        return url.toString();
      }
    } catch {}
    return `https://al.fanza.co.jp/?lurl=${encodeURIComponent(raw)}&af_id=${encodeURIComponent(AF_ID)}&ch=link_tool&ch_id=link`;
  };

=======
>>>>>>> ea197fe (feat: 田中リファクタリング (LFS対応版))
  const fanzaEmbedUrl = `https://www.dmm.co.jp/litevideo/-/part/=/affi_id=${process.env.NEXT_PUBLIC_FANZA_AFFILIATE_ID}/cid=${video.external_id}/size=1280_720/`;

  return (
    <div className="max-w-4xl mx-auto px-4 py-6">
      <h1 className="text-lg font-bold mb-3">{video.title}</h1>
      <div className="relative w-full" style={{ paddingBottom: '56%' }}>
        <iframe
          src={fanzaEmbedUrl}
          title="Embedded Video Player"
          frameBorder="0"
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share; fullscreen"
          loading="eager"
          className="absolute inset-0 w-full h-full"
        />
      </div>

      <div className="mt-4 grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="sm:col-span-2 space-y-3">
          {video.product_released_at && (
            <div className="text-sm text-gray-600">発売日: {new Date(video.product_released_at).toLocaleDateString('ja-JP')}</div>
          )}
          {video.price != null && (
            <div className="text-sm text-gray-600">価格: ￥{Number(video.price).toLocaleString()}</div>
          )}
          {video.description && (
            <p className="text-sm text-gray-700 whitespace-pre-wrap">{video.description}</p>
          )}

          {video.performers?.length ? (
            <div className="text-sm">
              <div className="text-gray-600 mb-1">出演:</div>
              <div className="flex flex-wrap gap-2">
                {video.performers.map((p) => (
                  <span key={p.id} className="px-2 py-0.5 rounded-full bg-pink-300 text-white text-xs font-bold">{p.name}</span>
                ))}
              </div>
            </div>
          ) : null}

          {video.tags?.length ? (
            <div className="text-sm">
              <div className="text-gray-600 mb-1">タグ:</div>
              <div className="flex flex-wrap gap-2">
                {video.tags.map((t) => (
                  <span key={t.id} className="px-2 py-0.5 rounded-full bg-purple-300 text-white text-xs font-bold">{t.name}</span>
                ))}
              </div>
            </div>
          ) : null}
        </div>
        <div>
          <div className="border rounded-lg p-3">
            <div className="text-sm text-gray-700 mb-2">外部サイトで見る</div>
<<<<<<< HEAD
            <Link href={toFanzaAffiliate(video.product_url) || '#'} target="_blank" className="block w-full text-center bg-amber-500 text-white font-bold rounded-lg py-2">
=======
            <Link href={video.product_url || '#'} target="_blank" className="block w-full text-center bg-amber-500 text-white font-bold rounded-lg py-2">
>>>>>>> ea197fe (feat: 田中リファクタリング (LFS対応版))
              商品ページへ
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
