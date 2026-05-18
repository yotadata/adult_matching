import { Metadata } from 'next';
import Image from 'next/image';
import Link from 'next/link';
import { notFound } from 'next/navigation';
import { supabase } from '@/lib/supabase';
import { isUpcomingRelease } from '@/lib/videoMeta';

const SITE_URL = 'https://seihekilab.com';
const SITE_NAME = '性癖ラボ';

type Performer = { id: string; name: string };
type Tag = { id: string; name: string };

type VideoDetail = {
  id: string;
  external_id: string;
  title: string;
  description: string | null;
  thumbnail_url: string | null;
  product_url: string | null;
  affiliate_url: string | null;
  price: number | null;
  product_released_at: string | null;
  director: string | null;
  series: string | null;
  maker: string | null;
  label: string | null;
};

async function getVideo(id: string): Promise<VideoDetail | null> {
  const { data, error } = await supabase
    .from('videos')
    .select('id, external_id, title, description, thumbnail_url, product_url, affiliate_url, price, product_released_at, director, series, maker, label')
    .eq('id', id)
    .maybeSingle();
  if (error || !data) return null;
  return data as VideoDetail;
}

async function getPerformers(videoId: string): Promise<Performer[]> {
  const { data } = await supabase
    .from('video_performers')
    .select('performers(id, name)')
    .eq('video_id', videoId);
  return ((data || []) as { performers: Performer | null }[])
    .flatMap(r => r.performers ? [r.performers] : []);
}

async function getTags(videoId: string): Promise<Tag[]> {
  const { data } = await supabase
    .from('video_tags')
    .select('tags(id, name)')
    .eq('video_id', videoId);
  return ((data || []) as { tags: Tag | null }[])
    .flatMap(r => r.tags ? [r.tags] : []);
}

export async function generateMetadata(
  { params }: { params: Promise<{ id: string }> }
): Promise<Metadata> {
  const { id } = await params;
  const video = await getVideo(id);
  if (!video) return { title: `動画が見つかりません | ${SITE_NAME}` };

  const title = `${video.title} | ${SITE_NAME}`;
  const description = video.description
    ? video.description.slice(0, 120)
    : `${video.title}の詳細・出演者・タグ情報。${SITE_NAME}でお気に入りの作品を見つけよう。`;
  const canonicalUrl = `${SITE_URL}/videos/${id}`;
  const image = video.thumbnail_url ?? `${SITE_URL}/opengraph-image.png`;

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
      images: [{ url: image }],
    },
    twitter: { card: 'summary_large_image', title, description, images: [image] },
  };
}

export default async function VideoPage(
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const [video, performers, tags] = await Promise.all([
    getVideo(id),
    getPerformers(id),
    getTags(id),
  ]);

  if (!video) notFound();

  const affiliateUrl = video.affiliate_url ?? video.product_url ?? null;
  const canonicalUrl = `${SITE_URL}/videos/${id}`;
  const fanzaEmbedUrl = `https://www.dmm.co.jp/litevideo/-/part/=/affi_id=${process.env.NEXT_PUBLIC_FANZA_AFFILIATE_ID}/cid=${video.external_id}/size=1280_720/`;

  const jsonLd = [
    {
      '@context': 'https://schema.org',
      '@type': 'Product',
      name: video.title,
      description: video.description ?? undefined,
      image: video.thumbnail_url ?? undefined,
      url: canonicalUrl,
      ...(video.price != null && {
        offers: {
          '@type': 'Offer',
          price: video.price,
          priceCurrency: 'JPY',
          url: affiliateUrl ?? canonicalUrl,
        },
      }),
    },
    {
      '@context': 'https://schema.org',
      '@type': 'BreadcrumbList',
      itemListElement: [
        { '@type': 'ListItem', position: 1, name: 'ホーム', item: SITE_URL },
        { '@type': 'ListItem', position: 2, name: video.title, item: canonicalUrl },
      ],
    },
  ];

  return (
    <>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
      />

      <div className="max-w-4xl mx-auto px-4 py-6 text-[#c9d1d9]">
        {/* パンくず */}
        <nav className="text-xs text-[#656d76] mb-4 flex flex-wrap gap-1">
          <Link href="/" className="hover:text-[#8b949e]">ホーム</Link>
          <span>/</span>
          <span className="text-[#8b949e] line-clamp-1">{video.title}</span>
        </nav>

        {/* 動画プレーヤー */}
        <div className="w-full bg-black rounded-lg overflow-hidden relative mb-6" style={{ paddingBottom: '56.25%' }}>
          <iframe
            src={fanzaEmbedUrl}
            title={video.title}
            frameBorder="0"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share; fullscreen"
            loading="eager"
            className="absolute inset-0 w-full h-full"
          />
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
          <div className="sm:col-span-2 space-y-4">
            <h1 className="text-xl font-bold leading-snug">{video.title}</h1>

            {/* メタ情報 */}
            <div className="text-sm text-[#8b949e] space-y-1">
              {video.product_released_at && (
                <div className="flex items-center gap-2">
                  <span>発売日: {new Date(video.product_released_at).toLocaleDateString('ja-JP')}</span>
                  {isUpcomingRelease(video.product_released_at) && (
                    <span className="inline-flex items-center rounded-full bg-amber-100/20 border border-amber-400/40 px-2 py-0.5 text-xs text-amber-400">
                      予約作品
                    </span>
                  )}
                </div>
              )}
              {video.maker && <div>メーカー: {video.maker}</div>}
              {video.label && <div>レーベル: {video.label}</div>}
              {video.series && <div>シリーズ: {video.series}</div>}
              {video.director && <div>監督: {video.director}</div>}
              {video.price != null && <div>価格: ¥{Number(video.price).toLocaleString()}</div>}
            </div>

            {/* 説明文 */}
            {video.description && (
              <p className="text-sm text-[#8b949e] whitespace-pre-wrap leading-relaxed">
                {video.description}
              </p>
            )}

            {/* 出演者 */}
            {performers.length > 0 && (
              <div>
                <div className="text-xs text-[#656d76] mb-2">出演者</div>
                <div className="flex flex-wrap gap-2">
                  {performers.map(p => (
                    <span
                      key={p.id}
                      className="px-3 py-1 rounded-full bg-[#21262d] border border-[#30363d] text-xs text-[#c9d1d9]"
                    >
                      {p.name}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* タグ */}
            {tags.length > 0 && (
              <div>
                <div className="text-xs text-[#656d76] mb-2">ジャンル</div>
                <div className="flex flex-wrap gap-2">
                  {tags.map(t => (
                    <Link
                      key={t.id}
                      href={`/tags/${t.id}`}
                      className="px-3 py-1 rounded-full bg-[#21262d] border border-[#30363d] text-xs text-[#c9d1d9] hover:border-[#8b949e] transition-colors"
                    >
                      {t.name}
                    </Link>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* サイドバー */}
          <div className="space-y-3">
            {video.thumbnail_url && (
              <div className="relative w-full aspect-video rounded-lg overflow-hidden border border-[#30363d]">
                <Image
                  src={video.thumbnail_url}
                  alt={video.title}
                  fill
                  className="object-cover"
                  sizes="(max-width: 640px) 100vw, 300px"
                />
              </div>
            )}
            {affiliateUrl && (
              <Link
                href={affiliateUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="block w-full text-center bg-amber-500 hover:bg-amber-400 text-white font-bold rounded-lg py-3 text-sm transition-colors"
              >
                FANZAで見る
              </Link>
            )}
            <Link
              href="/swipe"
              className="block w-full text-center bg-[#21262d] hover:bg-[#30363d] border border-[#30363d] text-[#c9d1d9] font-bold rounded-lg py-3 text-sm transition-colors"
            >
              好みの動画を探す
            </Link>
          </div>
        </div>
      </div>
    </>
  );
}
