import { Metadata } from 'next';
import Image from 'next/image';
import Link from 'next/link';
import { notFound } from 'next/navigation';
import { supabase } from '@/lib/supabase';

const SITE_URL = 'https://seihekilab.com';
const SITE_NAME = '性癖ラボ';

type Performer = { id: string; name: string };
type Tag = { id: string; name: string };

type Video = {
  id: string;
  external_id: string;
  title: string;
  description?: string | null;
  thumbnail_url?: string | null;
  thumbnail_vertical_url?: string | null;
  product_url?: string | null;
  affiliate_url?: string | null;
  price?: number | null;
  product_released_at?: string | null;
  duration_seconds?: number | null;
  director?: string | null;
  series?: string | null;
  maker?: string | null;
  label?: string | null;
  performers: Performer[];
  tags: Tag[];
};

type SimilarVideo = {
  id: string;
  title: string;
  thumbnail_url: string | null;
  thumbnail_vertical_url: string | null;
  affiliate_url: string | null;
  product_url: string | null;
  external_id: string;
};

async function getVideo(id: string): Promise<Video | null> {
  const { data: v, error } = await supabase
    .from('videos')
    .select('*')
    .eq('id', id)
    .maybeSingle();
  if (error || !v) return null;

  const { data: perf } = await supabase
    .from('video_performers')
    .select('performers(id, name)')
    .eq('video_id', id);
  const performers: Performer[] = ((perf ?? []) as unknown as { performers: Performer | null }[])
    .flatMap((r) => (r.performers ? [r.performers] : []));

  const { data: tg } = await supabase
    .from('video_tags')
    .select('tags(id, name)')
    .eq('video_id', id);
  const tags: Tag[] = ((tg ?? []) as unknown as { tags: Tag | null }[])
    .flatMap((r) => (r.tags ? [r.tags] : []));

  return { ...v, performers, tags };
}

async function getSimilarVideos(id: string): Promise<SimilarVideo[]> {
  const { data } = await supabase.rpc('get_similar_videos', {
    p_video_id: id,
    p_limit: 8,
  });
  return (data as SimilarVideo[]) ?? [];
}

function formatDuration(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  if (h > 0) return `${h}時間${m}分`;
  return `${m}分`;
}

function resolveUrl(video: { affiliate_url?: string | null; product_url?: string | null }): string {
  return video.affiliate_url ?? video.product_url ?? '#';
}

// ── メタデータ ────────────────────────────────────────────────
export async function generateMetadata(
  { params }: { params: Promise<{ id: string }> }
): Promise<Metadata> {
  const { id } = await params;
  const video = await getVideo(id);
  if (!video) return { title: `作品が見つかりません | ${SITE_NAME}` };

  const performers = video.performers.map((p) => p.name).join('・');
  const tags = video.tags.map((t) => t.name).join(' ');
  const title = `${video.title} | ${SITE_NAME}`;
  const description = video.description
    ? video.description.slice(0, 120)
    : `${performers ? performers + '出演。' : ''}${tags ? 'ジャンル: ' + tags : ''}${SITE_NAME}でこの作品に似た作品を探せます。`;
  const ogImage = video.thumbnail_url ?? video.thumbnail_vertical_url;
  const canonicalUrl = `${SITE_URL}/videos/${id}`;

  return {
    title,
    description,
    alternates: { canonical: canonicalUrl },
    openGraph: {
      title,
      description,
      url: canonicalUrl,
      siteName: SITE_NAME,
      images: ogImage ? [{ url: ogImage, width: 800, height: 450 }] : [],
      type: 'website',
      locale: 'ja_JP',
    },
    twitter: {
      card: 'summary_large_image',
      title,
      description,
      images: ogImage ? [ogImage] : [],
    },
  };
}

// ── ページ本体 ────────────────────────────────────────────────
export default async function VideoDetailPage(
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const [video, similar] = await Promise.all([
    getVideo(id),
    getSimilarVideos(id),
  ]);

  if (!video) notFound();

  const productUrl = resolveUrl(video);
  const fanzaEmbedUrl = video.external_id
    ? `https://www.dmm.co.jp/litevideo/-/part/=/affi_id=${process.env.NEXT_PUBLIC_FANZA_AFFILIATE_ID}/cid=${video.external_id}/size=1280_720/`
    : null;

  const performers = video.performers.map((p) => p.name).join('・');
  const tags = video.tags.map((t) => t.name).join(' ');
  const description = video.description
    ? video.description.slice(0, 120)
    : `${performers ? performers + '出演。' : ''}${tags ? 'ジャンル: ' + tags : ''}`;

  // JSON-LD 構造化データ
  const jsonLd = {
    '@context': 'https://schema.org',
    '@type': 'Movie',
    name: video.title,
    description,
    image: video.thumbnail_url ?? video.thumbnail_vertical_url,
    url: `${SITE_URL}/videos/${id}`,
    ...(video.product_released_at && {
      datePublished: video.product_released_at.slice(0, 10),
    }),
    ...(video.maker && { productionCompany: { '@type': 'Organization', name: video.maker } }),
    ...(video.performers.length > 0 && {
      actor: video.performers.map((p) => ({ '@type': 'Person', name: p.name })),
    }),
  };

  const isUpcoming = video.product_released_at
    ? new Date(video.product_released_at) > new Date()
    : false;

  return (
    <>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
      />

      <div className="max-w-4xl mx-auto px-4 py-6">
        {/* パンくず */}
        <nav className="text-xs text-gray-400 mb-3 flex flex-wrap gap-1">
          <Link href="/" className="hover:text-gray-600">ホーム</Link>
          <span>/</span>
          {video.maker && (
            <>
              <span className="text-gray-600">{video.maker}</span>
              <span>/</span>
            </>
          )}
          <span className="text-gray-700 line-clamp-1">{video.title}</span>
        </nav>

        <h1 className="text-lg font-bold mb-3 leading-snug">{video.title}</h1>

        {/* 動画プレイヤー */}
        {fanzaEmbedUrl && (
          <div className="relative w-full mb-4" style={{ paddingBottom: '56.25%' }}>
            <iframe
              src={fanzaEmbedUrl}
              title={video.title}
              frameBorder="0"
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share; fullscreen"
              loading="eager"
              className="absolute inset-0 w-full h-full rounded-lg"
            />
          </div>
        )}

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
          {/* 作品情報 */}
          <div className="sm:col-span-2 space-y-3">
            {/* メタ情報 */}
            <div className="flex flex-wrap gap-x-4 gap-y-1 text-sm text-gray-500">
              {video.product_released_at && (
                <span className="flex items-center gap-1">
                  発売日: {new Date(video.product_released_at).toLocaleDateString('ja-JP')}
                  {isUpcoming && (
                    <span className="ml-1 inline-flex rounded-full bg-amber-100 border border-amber-200 px-2 py-0.5 text-xs text-amber-700">
                      予約作品
                    </span>
                  )}
                </span>
              )}
              {video.duration_seconds != null && (
                <span>収録時間: {formatDuration(video.duration_seconds)}</span>
              )}
              {video.price != null && (
                <span>価格: ¥{Number(video.price).toLocaleString()}</span>
              )}
              {video.maker && <span>メーカー: {video.maker}</span>}
              {video.series && <span>シリーズ: {video.series}</span>}
              {video.director && <span>監督: {video.director}</span>}
            </div>

            {/* 説明文 */}
            {video.description && (
              <p className="text-sm text-gray-700 whitespace-pre-wrap leading-relaxed">
                {video.description}
              </p>
            )}

            {/* 出演者 */}
            {video.performers.length > 0 && (
              <div className="text-sm">
                <div className="text-gray-500 mb-1.5">出演者</div>
                <div className="flex flex-wrap gap-2">
                  {video.performers.map((p) => (
                    <Link
                      key={p.id}
                      href={`/performers/${p.id}`}
                      className="px-2.5 py-1 rounded-full bg-pink-100 text-pink-700 text-xs font-medium hover:bg-pink-200 transition-colors"
                    >
                      {p.name}
                    </Link>
                  ))}
                </div>
              </div>
            )}

            {/* タグ */}
            {video.tags.length > 0 && (
              <div className="text-sm">
                <div className="text-gray-500 mb-1.5">ジャンル</div>
                <div className="flex flex-wrap gap-2">
                  {video.tags.map((t) => (
                    <Link
                      key={t.id}
                      href={`/tags/${t.id}`}
                      className="px-2.5 py-1 rounded-full bg-purple-100 text-purple-700 text-xs font-medium hover:bg-purple-200 transition-colors"
                    >
                      {t.name}
                    </Link>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* CTA サイドバー */}
          <div className="space-y-3">
            <div className="border rounded-xl p-4 bg-gray-50 sticky top-4">
              {video.thumbnail_url && (
                <div className="relative w-full mb-3 rounded-lg overflow-hidden" style={{ paddingBottom: '56.25%' }}>
                  <Image
                    src={video.thumbnail_url}
                    alt={video.title}
                    fill
                    sizes="(max-width: 640px) 100vw, 33vw"
                    className="object-cover"
                  />
                </div>
              )}
              <Link
                href={productUrl}
                target="_blank"
                rel="noopener noreferrer nofollow"
                className="block w-full text-center bg-amber-500 hover:bg-amber-600 text-white font-bold rounded-lg py-3 text-sm transition-colors"
              >
                FANZA で視聴する
              </Link>
              <p className="text-xs text-gray-400 mt-2 text-center">外部サイトに移動します</p>
            </div>

            {/* 性癖ラボで探す CTA */}
            <div className="border rounded-xl p-4 bg-purple-50">
              <p className="text-xs text-gray-600 mb-2">
                この作品が好きなら、あなたの好みに合った作品を AI が提案します。
              </p>
              <Link
                href="/swipe"
                className="block w-full text-center bg-purple-500 hover:bg-purple-600 text-white font-bold rounded-lg py-2.5 text-sm transition-colors"
              >
                好みの作品を探す →
              </Link>
            </div>
          </div>
        </div>

        {/* 類似作品 */}
        {similar.length > 0 && (
          <section className="mt-10">
            <h2 className="text-base font-bold mb-4 text-gray-800">
              この作品に似た作品
            </h2>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {similar.map((s) => (
                <Link
                  key={s.id}
                  href={`/videos/${s.id}`}
                  className="group block"
                >
                  <div className="relative w-full rounded-lg overflow-hidden bg-gray-100" style={{ paddingBottom: '65%' }}>
                    {s.thumbnail_url ? (
                      <Image
                        src={s.thumbnail_url}
                        alt={s.title}
                        fill
                        sizes="(max-width: 640px) 50vw, 25vw"
                        className="object-cover group-hover:scale-105 transition-transform duration-200"
                      />
                    ) : (
                      <div className="absolute inset-0 flex items-center justify-center text-gray-300 text-xs">
                        No image
                      </div>
                    )}
                  </div>
                  <p className="mt-1.5 text-xs text-gray-700 line-clamp-2 leading-snug group-hover:text-purple-600 transition-colors">
                    {s.title}
                  </p>
                </Link>
              ))}
            </div>
          </section>
        )}

        {/* 内部誘導フッター */}
        <div className="mt-10 py-6 border-t text-center space-y-2">
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
