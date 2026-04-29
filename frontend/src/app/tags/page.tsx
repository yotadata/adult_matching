import { Metadata } from 'next';
import Link from 'next/link';
import { supabase } from '@/lib/supabase';

const SITE_URL = 'https://seihekilab.com';
const SITE_NAME = '性癖ラボ';

export const metadata: Metadata = {
  title: `AV動画 ジャンル一覧 | ${SITE_NAME}`,
  description: `アダルト動画のジャンル・カテゴリ一覧。中出し・巨乳・人妻・素人など275ジャンルの作品を${SITE_NAME}でまとめてチェックできます。`,
  alternates: { canonical: `${SITE_URL}/tags` },
  openGraph: {
    title: `AV動画 ジャンル一覧 | ${SITE_NAME}`,
    description: `アダルト動画のジャンル・カテゴリ一覧。275ジャンルの作品をチェック。`,
    url: `${SITE_URL}/tags`,
    siteName: SITE_NAME,
    type: 'website',
    locale: 'ja_JP',
  },
};

type TagWithCount = {
  id: string;
  name: string;
  video_count: number;
};

async function getTagsWithCount(): Promise<TagWithCount[]> {
  const { data, error } = await supabase.rpc('get_tags_with_count');
  if (error || !data) {
    // フォールバック: 件数なしで全件取得
    const { data: tags } = await supabase.from('tags').select('id, name').order('name');
    return (tags ?? []).map((t) => ({ ...t, video_count: 0 }));
  }
  return data as TagWithCount[];
}

export default async function TagsPage() {
  const tags = await getTagsWithCount();

  const jsonLd = {
    '@context': 'https://schema.org',
    '@type': 'CollectionPage',
    name: 'AV動画 ジャンル一覧',
    description: 'アダルト動画のジャンル・カテゴリ一覧',
    url: `${SITE_URL}/tags`,
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
          <span className="text-gray-700">ジャンル一覧</span>
        </nav>

        <div className="mb-6">
          <h1 className="text-xl font-bold text-gray-900 mb-1">
            AV動画 ジャンル一覧
          </h1>
          <p className="text-sm text-gray-500">全 {tags.length} ジャンル</p>
        </div>

        <div className="flex flex-wrap gap-2 mb-10">
          {tags.map((tag) => (
            <Link
              key={tag.id}
              href={`/tags/${tag.id}`}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-purple-50 hover:bg-purple-100 border border-purple-200 text-purple-800 text-sm transition-colors"
            >
              {tag.name}
              {tag.video_count > 0 && (
                <span className="text-xs text-purple-500 font-medium">
                  {tag.video_count.toLocaleString()}
                </span>
              )}
            </Link>
          ))}
        </div>

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
