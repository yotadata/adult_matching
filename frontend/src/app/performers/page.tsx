import { Metadata } from 'next';
import Link from 'next/link';
import { notFound } from 'next/navigation';
import { supabase } from '@/lib/supabase';

const SITE_URL = 'https://seihekilab.com';
const SITE_NAME = '性癖ラボ';
const PAGE_SIZE = 60;

type PerformerWithCount = {
  id: string;
  name: string;
  video_count: number;
};

export async function generateMetadata(
  { searchParams }: { searchParams: Promise<{ page?: string }> }
): Promise<Metadata> {
  const { page: pageStr } = await searchParams;
  const page = Math.max(1, parseInt(pageStr ?? '1', 10) || 1);
  const suffix = page > 1 ? ` (${page}ページ目)` : '';
  const title = `AV女優・出演者 一覧${suffix} | ${SITE_NAME}`;
  const description = `アダルト動画に出演するAV女優・出演者の一覧です。人気女優から新人まで${SITE_NAME}で出演作品をまとめてチェックできます。`;
  const canonicalUrl = page > 1 ? `${SITE_URL}/performers?page=${page}` : `${SITE_URL}/performers`;

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
  };
}

export default async function PerformersPage(
  { searchParams }: { searchParams: Promise<{ page?: string }> }
) {
  const { page: pageStr } = await searchParams;
  const page = Math.max(1, parseInt(pageStr ?? '1', 10) || 1);
  const offset = (page - 1) * PAGE_SIZE;

  const [performersResult, countResult] = await Promise.all([
    supabase.rpc('get_performers_with_count', { p_limit: PAGE_SIZE, p_offset: offset }),
    supabase.rpc('get_performers_count'),
  ]);

  if (performersResult.error) notFound();

  const performers = (performersResult.data ?? []) as PerformerWithCount[];
  const total = Number(countResult.data ?? 0);
  const totalPages = Math.ceil(total / PAGE_SIZE);

  if (page > totalPages && totalPages > 0) notFound();

  const jsonLd = {
    '@context': 'https://schema.org',
    '@type': 'CollectionPage',
    name: 'AV女優・出演者一覧',
    description: 'アダルト動画出演者の一覧',
    url: `${SITE_URL}/performers`,
  };

  // ページ番号リスト（省略あり）
  const pageNumbers = Array.from({ length: totalPages }, (_, i) => i + 1)
    .filter((p) => p === 1 || p === totalPages || Math.abs(p - page) <= 2)
    .reduce<(number | '...')[]>((acc, p, idx, arr) => {
      if (idx > 0 && p - (arr[idx - 1] as number) > 1) acc.push('...');
      acc.push(p);
      return acc;
    }, []);

  return (
    <>
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
      />

      <div className="max-w-5xl mx-auto px-4 py-6">
        {/* パンくず */}
        <nav className="text-xs text-gray-500 mb-4 flex flex-wrap gap-1">
          <Link href="/" className="hover:text-gray-700">ホーム</Link>
          <span>/</span>
          <span className="text-gray-700">出演者一覧</span>
          {page > 1 && (
            <>
              <span>/</span>
              <span className="text-gray-700">{page}ページ目</span>
            </>
          )}
        </nav>

        {/* ヘッダー */}
        <div className="mb-6">
          <h1 className="text-xl font-bold text-gray-900 mb-1">
            AV女優・出演者 一覧
          </h1>
          <p className="text-sm text-gray-600">
            全 {total.toLocaleString()} 人
            {totalPages > 1 && `（${page} / ${totalPages} ページ）`}
          </p>
        </div>

        {/* 出演者グリッド */}
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3 mb-8">
          {performers.map((p) => (
            <Link
              key={p.id}
              href={`/performers/${p.id}`}
              className="group flex flex-col items-center justify-center rounded-xl p-3 bg-white/80 backdrop-blur-sm border border-white/60 hover:bg-pink-50 hover:border-pink-300 transition-colors text-center shadow-sm"
            >
              <span className="text-sm font-medium text-gray-800 group-hover:text-pink-700 leading-snug">
                {p.name}
              </span>
              {p.video_count > 0 && (
                <span className="mt-1 text-xs text-gray-500">
                  {p.video_count} 作品
                </span>
              )}
            </Link>
          ))}
        </div>

        {/* ページネーション */}
        {totalPages > 1 && (
          <nav className="flex justify-center items-center gap-1.5 mb-10 flex-wrap">
            {page > 1 && (
              <Link
                href={page - 1 === 1 ? '/performers' : `/performers?page=${page - 1}`}
                className="px-3 py-2 rounded-lg bg-white/80 border border-gray-300 text-gray-700 text-sm font-medium hover:bg-white transition-colors shadow-sm"
              >
                ← 前へ
              </Link>
            )}

            {pageNumbers.map((item, idx) =>
              item === '...' ? (
                <span key={`ellipsis-${idx}`} className="px-2 py-2 text-gray-500 text-sm">…</span>
              ) : (
                <Link
                  key={item}
                  href={item === 1 ? '/performers' : `/performers?page=${item}`}
                  className={`min-w-[2.25rem] px-3 py-2 rounded-lg border text-sm font-medium text-center transition-colors shadow-sm ${
                    item === page
                      ? 'bg-pink-500 text-white border-pink-500'
                      : 'bg-white/80 border-gray-300 text-gray-700 hover:bg-white'
                  }`}
                >
                  {item}
                </Link>
              )
            )}

            {page < totalPages && (
              <Link
                href={`/performers?page=${page + 1}`}
                className="px-3 py-2 rounded-lg bg-white/80 border border-gray-300 text-gray-700 text-sm font-medium hover:bg-white transition-colors shadow-sm"
              >
                次へ →
              </Link>
            )}
          </nav>
        )}

        {/* CTA */}
        <div className="py-6 border-t border-white/40 text-center space-y-2">
          <p className="text-sm text-gray-600">
            スワイプするだけで、あなた好みの作品を AI が発掘します
          </p>
          <Link
            href="/swipe"
            className="inline-block bg-purple-500 hover:bg-purple-600 text-white font-bold rounded-full px-8 py-3 text-sm transition-colors shadow"
          >
            無料で試してみる
          </Link>
        </div>
      </div>
    </>
  );
}
