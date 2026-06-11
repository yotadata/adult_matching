import { Metadata } from 'next';
import Link from 'next/link';
import VideoThumbnail from '@/components/VideoThumbnail';
import { notFound } from 'next/navigation';
import { supabase } from '@/lib/supabase';

const SITE_URL = 'https://www.seihekilab.com';
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
  source: string | null;
  image_urls: string[] | null;
};

type Tag = { id: string; name: string };

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
    .select('videos(id, title, thumbnail_url, affiliate_url, product_url, product_released_at, external_id, source, image_urls)', { count: 'exact' })
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
    twitter: { card: 'summary', title, description },
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

  const tagUrl = `${SITE_URL}/tags/${id}`;

  const jsonLd = [
    {
      '@context': 'https://schema.org',
      '@type': 'CollectionPage',
      name: `${tag.name} 動画一覧`,
      description: `「${tag.name}」ジャンルの動画一覧`,
      url: tagUrl,
      ...(videos.length > 0 && {
        mainEntity: {
          '@type': 'ItemList',
          itemListElement: videos.slice(0, 10).map((v, i) => ({
            '@type': 'ListItem',
            position: i + 1,
            url: `${SITE_URL}/videos/${v.id}`,
            name: v.title,
          })),
        },
      }),
    },
    {
      '@context': 'https://schema.org',
      '@type': 'BreadcrumbList',
      itemListElement: [
        { '@type': 'ListItem', position: 1, name: 'ホーム', item: SITE_URL },
        { '@type': 'ListItem', position: 2, name: 'ジャンル一覧', item: `${SITE_URL}/tags` },
        { '@type': 'ListItem', position: 3, name: tag.name, item: tagUrl },
      ],
    },
  ];

  return (
    <>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
      />

      <div className="max-w-5xl mx-auto px-4 py-6">
        {/* パンくず */}
        <nav className="text-xs text-[#656d76] mb-4 flex flex-wrap gap-1">
          <Link href="/" className="hover:text-[#8b949e]">ホーム</Link>
          <span>/</span>
          <Link href="/tags" className="hover:text-[#8b949e]">ジャンル一覧</Link>
          <span>/</span>
          <span className="text-[#8b949e]">{tag.name}</span>
        </nav>

        <div className="mb-6">
          <h1 className="text-xl font-bold text-[#e6edf3] mb-1">
            {tag.name} の動画一覧
          </h1>
          <p className="text-sm text-[#8b949e]">
            全 {total.toLocaleString()} 作品
            {total > PAGE_SIZE && `（最新 ${PAGE_SIZE} 件を表示）`}
          </p>
        </div>

        {videos.length > 0 ? (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4 mb-10">
            {videos.map((v) => (
              <Link key={v.id} href={`/videos/${v.id}`} className="group block">
                <div
                  className="relative w-full rounded-lg overflow-hidden bg-[#21262d]"
                  style={{ paddingBottom: '65%' }}
                >
                  <VideoThumbnail
                    source={v.source}
                    thumbnailUrl={v.thumbnail_url}
                    imageUrls={v.image_urls}
                    alt={v.title}
                    fill
                    sizes="(max-width: 640px) 50vw, (max-width: 768px) 33vw, 25vw"
                    className="object-cover group-hover:scale-105 transition-transform duration-200"
                  />
                </div>
                <p className="mt-1.5 text-xs text-[#8b949e] line-clamp-2 leading-snug group-hover:text-violet-400 transition-colors">
                  {v.title}
                </p>
                {v.product_released_at && (
                  <p className="text-xs text-[#656d76] mt-0.5">
                    {new Date(v.product_released_at).toLocaleDateString('ja-JP')}
                  </p>
                )}
              </Link>
            ))}
          </div>
        ) : (
          <p className="text-sm text-[#8b949e] my-10">作品が見つかりませんでした。</p>
        )}

        <div className="py-6 border-t border-[#30363d] text-center space-y-2">
          <p className="text-sm text-[#8b949e]">
            スワイプするだけで、あなた好みの作品を AI が発掘します
          </p>
          <Link
            href="/swipe"
            className="inline-block bg-violet-600 hover:bg-violet-500 text-white font-bold rounded-full px-8 py-3 text-sm transition-colors"
          >
            無料で試してみる
          </Link>
        </div>
      </div>
    </>
  );
}
