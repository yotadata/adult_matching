import { Metadata } from 'next';
import Link from 'next/link';
import { supabase } from '@/lib/supabase';

const SITE_URL = 'https://www.seihekilab.com';
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

type TagWithCount = { id: string; name: string; video_count: number };

async function getTagsWithCount(): Promise<TagWithCount[]> {
  const { data, error } = await supabase.rpc('get_tags_with_count');
  if (error || !data) {
    const { data: tags } = await supabase.from('tags').select('id, name').order('name');
    return (tags ?? []).map((t) => ({ ...t, video_count: 0 }));
  }
  return data as TagWithCount[];
}

function tagSize(count: number, max: number): string {
  if (count >= max * 0.5) return 'text-base font-semibold';
  if (count >= max * 0.1) return 'text-sm font-medium';
  return 'text-xs font-normal';
}

export default async function TagsPage() {
  const tags = await getTagsWithCount();
  const maxCount = tags[0]?.video_count ?? 1;

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
        <nav className="text-xs text-[#656d76] mb-4 flex flex-wrap gap-1">
          <Link href="/" className="hover:text-[#8b949e]">ホーム</Link>
          <span>/</span>
          <span className="text-[#8b949e]">ジャンル一覧</span>
        </nav>

        <div className="mb-6">
          <h1 className="text-xl font-bold text-[#e6edf3] mb-1">AV動画 ジャンル一覧</h1>
          <p className="text-sm text-[#8b949e]">全 {tags.length} ジャンル</p>
        </div>

        <div className="bg-[#161b22] rounded-2xl border border-[#30363d] p-5 mb-8">
          <div className="flex flex-wrap gap-2">
            {tags.map((tag) => (
              <Link
                key={tag.id}
                href={`/tags/${tag.id}`}
                className={`inline-flex items-center gap-1 px-3 py-1.5 rounded-full bg-violet-900/20 hover:bg-violet-900/40 border border-violet-700/30 hover:border-violet-600/60 text-violet-300 hover:text-violet-200 transition-colors ${tagSize(tag.video_count, maxCount)}`}
              >
                {tag.name}
                {tag.video_count > 0 && (
                  <span className="text-violet-500 font-normal text-xs">
                    {tag.video_count.toLocaleString()}
                  </span>
                )}
              </Link>
            ))}
          </div>
        </div>

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
