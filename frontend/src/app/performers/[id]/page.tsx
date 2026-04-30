import { Metadata } from 'next';
import Image from 'next/image';
import Link from 'next/link';
import { notFound } from 'next/navigation';
import { supabase } from '@/lib/supabase';

const SITE_URL = 'https://seihekilab.com';
const SITE_NAME = '性癖ラボ';
const PAGE_SIZE = 24;

type PerformerVideo = {
  id: string;
  title: string;
  thumbnail_url: string | null;
  affiliate_url: string | null;
  product_url: string | null;
  product_released_at: string | null;
  external_id: string;
};

type Performer = {
  id: string;
  name: string;
  fanza_actress_id: string | null;
};

async function getPerformer(id: string): Promise<Performer | null> {
  const { data, error } = await supabase
    .from('performers')
    .select('id, name, fanza_actress_id')
    .eq('id', id)
    .maybeSingle();
  if (error || !data) return null;
  return data;
}

async function getPerformerVideos(id: string): Promise<{ videos: PerformerVideo[]; total: number }> {
  const { data, error, count } = await supabase
    .from('video_performers')
    .select('videos(id, title, thumbnail_url, affiliate_url, product_url, product_released_at, external_id)', { count: 'exact' })
    .eq('performer_id', id)
    .order('created_at', { referencedTable: 'videos', ascending: false })
    .limit(PAGE_SIZE);

  if (error || !data) return { videos: [], total: 0 };

  const videos = (data as unknown as { videos: PerformerVideo | null }[])
    .flatMap((r) => (r.videos ? [r.videos] : []));

  return { videos, total: count ?? videos.length };
}

export async function generateMetadata(
  { params }: { params: Promise<{ id: string }> }
): Promise<Metadata> {
  const { id } = await params;
  const performer = await getPerformer(id);
  if (!performer) return { title: `出演者が見つかりません | ${SITE_NAME}` };

  const title = `${performer.name} 出演AV動画一覧 | ${SITE_NAME}`;
  const description = `${performer.name}が出演するアダルト動画の一覧です。最新作から人気作まで、${SITE_NAME}で${performer.name}の出演作をまとめてチェックできます。`;
  const canonicalUrl = `${SITE_URL}/performers/${id}`;

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

export default async function PerformerPage(
  { params }: { params: Promise<{ id: string }> }
) {
  const { id } = await params;
  const [performer, { videos, total }] = await Promise.all([
    getPerformer(id),
    getPerformerVideos(id),
  ]);

  if (!performer) notFound();

  const jsonLd = {
    '@context': 'https://schema.org',
    '@type': 'Person',
    name: performer.name,
    url: `${SITE_URL}/performers/${id}`,
  };

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
          <Link href="/performers" className="hover:text-[#8b949e]">出演者一覧</Link>
          <span>/</span>
          <span className="text-[#8b949e]">{performer.name}</span>
        </nav>

        <div className="mb-6">
          <h1 className="text-xl font-bold text-[#e6edf3] mb-1">
            {performer.name} 出演作品一覧
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
                  {v.thumbnail_url ? (
                    <Image
                      src={v.thumbnail_url}
                      alt={v.title}
                      fill
                      sizes="(max-width: 640px) 50vw, (max-width: 768px) 33vw, 25vw"
                      className="object-cover group-hover:scale-105 transition-transform duration-200"
                    />
                  ) : (
                    <div className="absolute inset-0 flex items-center justify-center text-[#656d76] text-xs">
                      No image
                    </div>
                  )}
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
