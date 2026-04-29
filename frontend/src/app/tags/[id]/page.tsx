import { Metadata } from 'next';
import Image from 'next/image';
import Link from 'next/link';
import { notFound } from 'next/navigation';
import { supabase } from '@/lib/supabase';

const SITE_URL = 'https://seihekilab.com';
const SITE_NAME = '性癖ラボ';
const PAGE_SIZE = 24;

type TagVideo = {
  id: string;
  title: string;
  thumbnail_url: string | null;
  affiliate_url: string | null;
  product_url: string | null;
  product_released_at: string | null;
  external_id: string;
};

type Tag = {
  id: string;
  name: string;
};

async function getTag(id: string): Promise<Tag | null> {
  const { data, error } = await supabase
    .from('tags')
    .select('id, name')
    .eq('id', id)
    .maybeSingle();
  if (error || !data) return null;
  return data;
}

async function getTagVideos(id: string): Promise<{ videos: TagVideo[]; total: number }> {
  const { data, error, count } = await supabase
    .from('video_tags')
    .select('videos(id, title, thumbnail_url, affiliate_url, product_url, product_released_at, external_id)', { count: 'exact' })
    .eq('tag_id', id)
    .order('created_at', { referencedTable: 'videos', ascending: false })
    .limit(PAGE_SIZE);

  if (error || !data) return { videos: [], total: 0 };

  const videos = (data as unknown as { videos: TagVideo | null }[])
    .flatMap((r) => (r.videos ? [r.videos] : []));

  return { videos, total: count ?? videos.length };
}

export async function generateMetadata(
  { params }: { params: Promise<{ id: string }> }
): Promise<Metadata> {
  const { id } = await params;
  const tag = await getTag(id);
  if (!tag) return { title: `ジャンルが見つかりません | ${SITE_NAME}` };

  const title = `${tag.name} のAV動画一覧 | ${SITE_NAME}`;
  const description = `「${tag.name}」ジャンルのアダルト動画一覧。最新作から人気作まで、${SITE_NAME}で${tag.name}の作品をまとめてチェックできます。`;
  const canonicalUrl = `${SITE_URL}/tags/${id}`;

  return {
    title,
    description,
    alternates: { canonical: canonicalUrl },
    openGraph: {
      title,
      description,
      url: canonicalUrl,
      siteName: SITE_NAME,
      type: 'website',
      locale: 'ja_JP',
    },
    twitter: {
      card: 'summary',
      title,
      description,
    },
  };
}

export default async function TagPage(
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const [tag, { videos, total }] = await Promise.all([
    getTag(id),
    getTagVideos(id),
  ]);

  if (!tag) notFound();

  const jsonLd = {
    '@context': 'https://schema.org',
    '@type': 'CollectionPage',
    name: `${tag.name} 動画一覧`,
    description: `「${tag.name}」ジャンルの動画一覧`,
    url: `${SITE_URL}/tags/${id}`,
  };

  return (
    <>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
      />

      <div className="max-w-5xl mx-auto px-4 py-6">
        {/* パンくず */}
        <nav className="text-xs text-gray-400 mb-4 flex flex-wrap gap-1">
          <Link href="/" className="hover:text-gray-600">ホーム</Link>
          <span>/</span>
          <span className="text-gray-700">{tag.name}</span>
        </nav>

        {/* 見出し */}
        <div className="mb-6">
          <h1 className="text-xl font-bold text-gray-900 mb-1">
            {tag.name} の動画一覧
          </h1>
          <p className="text-sm text-gray-500">
            全 {total.toLocaleString()} 作品
            {total > PAGE_SIZE && `（最新 ${PAGE_SIZE} 件を表示）`}
          </p>
        </div>

        {/* 動画グリッド */}
        {videos.length > 0 ? (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4 mb-10">
            {videos.map((v) => (
              <Link key={v.id} href={`/videos/${v.id}`} className="group block">
                <div
                  className="relative w-full rounded-lg overflow-hidden bg-gray-100"
                  style={{ paddingBottom: '65%' }}
                >
                  {v.thumbnail_url ? (
                    <Image
                      src={v.thumbnail_url}
                      alt={v.title}
                      fill
                      sizes="(max-width: 640px) 50vw, (max-width: 768px) 33vw, 25vw"
                      className="object-cover group-hover:scale-105 transition-transform duration-200"
                    />
                  ) : (
                    <div className="absolute inset-0 flex items-center justify-center text-gray-300 text-xs">
                      No image
                    </div>
                  )}
                </div>
                <p className="mt-1.5 text-xs text-gray-700 line-clamp-2 leading-snug group-hover:text-purple-600 transition-colors">
                  {v.title}
                </p>
                {v.product_released_at && (
                  <p className="text-xs text-gray-400 mt-0.5">
                    {new Date(v.product_released_at).toLocaleDateString('ja-JP')}
                  </p>
                )}
              </Link>
            ))}
          </div>
        ) : (
          <p className="text-sm text-gray-500 my-10">作品が見つかりませんでした。</p>
        )}

        {/* CTA */}
        <div className="py-6 border-t text-center space-y-2">
          <p className="text-sm text-gray-500">
            スワイプするだけで、あなた好みの作品を AI が発掘します
          </p>
          <Link
            href="/swipe"
            className="inline-block bg-purple-500 hover:bg-purple-600 text-white font-bold rounded-full px-8 py-3 text-sm transition-colors"
          >
            無料で試してみる
          </Link>
        </div>
      </div>
    </>
  );
}
