import { Metadata } from 'next';
import Link from 'next/link';
import { supabase } from '@/lib/supabase';

const SITE_URL = 'https://www.seihekilab.com';
const SITE_NAME = '性癖ラボ';
const PAGE_SIZE = 60;

export const metadata: Metadata = {
  title: `AV女優一覧 | ${SITE_NAME}`,
  description: `7,000人以上のAV女優一覧。波多野結衣・葵つかさ・三上悠亜など人気女優の出演作品を${SITE_NAME}でまとめてチェックできます。`,
  alternates: { canonical: `${SITE_URL}/performers` },
  openGraph: {
    title: `AV女優一覧 | ${SITE_NAME}`,
    description: `7,000人以上のAV女優の出演作品一覧。`,
    url: `${SITE_URL}/performers`,
    siteName: SITE_NAME,
    type: 'website',
    locale: 'ja_JP',
  },
};

type PerformerRow = { id: string; name: string; video_performers: { count: number }[] };

const ROW_MAP: Record<string, string[]> = {
  'ア': ['ア','イ','ウ','エ','オ','あ','い','う','え','お'],
  'カ': ['カ','キ','ク','ケ','コ','ガ','ギ','グ','ゲ','ゴ','か','き','く','け','こ','が','ぎ','ぐ','げ','ご'],
  'サ': ['サ','シ','ス','セ','ソ','ザ','ジ','ズ','ゼ','ゾ','さ','し','す','せ','そ','ざ','じ','ず','ぜ','ぞ'],
  'タ': ['タ','チ','ツ','テ','ト','ダ','ヂ','ヅ','デ','ド','た','ち','つ','て','と','だ','ぢ','づ','で','ど'],
  'ナ': ['ナ','ニ','ヌ','ネ','ノ','な','に','ぬ','ね','の'],
  'ハ': ['ハ','ヒ','フ','ヘ','ホ','バ','ビ','ブ','ベ','ボ','パ','ピ','プ','ペ','ポ','は','ひ','ふ','へ','ほ','ば','び','ぶ','べ','ぼ','ぱ','ぴ','ぷ','ぺ','ぽ'],
  'マ': ['マ','ミ','ム','メ','モ','ま','み','む','め','も'],
  'ヤ': ['ヤ','ユ','ヨ','や','ゆ','よ'],
  'ラ': ['ラ','リ','ル','レ','ロ','ら','り','る','れ','ろ'],
  'ワ': ['ワ','ヲ','ン','わ','を','ん'],
};

const ROW_LABELS = ['ア','カ','サ','タ','ナ','ハ','マ','ヤ','ラ','ワ','他'];

async function getPerformers(page: number, row: string | null): Promise<{ performers: { id: string; name: string; video_count: number }[]; total: number }> {
  const from = (page - 1) * PAGE_SIZE;
  const to = from + PAGE_SIZE - 1;

  const { data, error, count } = await supabase
    .from('performers')
    .select('id, name, video_performers(count)', { count: 'exact' })
    .order('name')
    .range(from, to);

  if (error || !data) return { performers: [], total: 0 };

  let performers = (data as unknown as PerformerRow[]).map((p) => ({
    id: p.id,
    name: p.name,
    video_count: p.video_performers?.[0]?.count ?? 0,
  }));

  if (row && row !== '他') {
    const chars = ROW_MAP[row] ?? [];
    performers = performers.filter((p) => chars.includes(p.name.charAt(0)));
  } else if (row === '他') {
    const allChars = Object.values(ROW_MAP).flat();
    performers = performers.filter((p) => !allChars.includes(p.name.charAt(0)));
  }

  return { performers, total: count ?? 0 };
}

export default async function PerformersPage({
  searchParams,
}: {
  searchParams: Promise<{ page?: string; row?: string }>;
}) {
  const { page: pageStr, row } = await searchParams;
  const page = Math.max(1, parseInt(pageStr ?? '1', 10));
  const activeRow = row ?? null;

  const { performers, total } = await getPerformers(page, activeRow);
  const totalPages = Math.ceil(total / PAGE_SIZE);

  const jsonLd = {
    '@context': 'https://schema.org',
    '@type': 'CollectionPage',
    name: 'AV女優一覧',
    description: '7,000人以上のAV女優の出演作品一覧',
    url: `${SITE_URL}/performers`,
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
          <span className="text-[#8b949e]">AV女優一覧</span>
        </nav>

        <div className="mb-6">
          <h1 className="text-xl font-bold text-[#e6edf3] mb-1">AV女優一覧</h1>
          <p className="text-sm text-[#8b949e]">全 {total.toLocaleString()} 人</p>
        </div>

        {/* 50音インデックス */}
        <div className="flex flex-wrap gap-1.5 mb-6">
          <Link
            href="/performers"
            className={`px-3 py-1 rounded-full text-xs font-semibold transition-colors ${!activeRow ? 'bg-violet-600 text-white' : 'bg-[#21262d] text-[#8b949e] hover:text-[#e6edf3]'}`}
          >
            全て
          </Link>
          {ROW_LABELS.map((r) => (
            <Link
              key={r}
              href={`/performers?row=${r}`}
              className={`px-3 py-1 rounded-full text-xs font-semibold transition-colors ${activeRow === r ? 'bg-violet-600 text-white' : 'bg-[#21262d] text-[#8b949e] hover:text-[#e6edf3]'}`}
            >
              {r}行
            </Link>
          ))}
        </div>

        {/* 女優一覧 */}
        {performers.length > 0 ? (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-2 mb-8">
            {performers.map((p) => (
              <Link
                key={p.id}
                href={`/performers/${p.id}`}
                className="flex flex-col px-3 py-2.5 rounded-lg bg-[#161b22] border border-[#30363d] hover:border-violet-500/50 hover:bg-[#21262d] transition-colors group"
              >
                <span className="text-sm text-[#e6edf3] font-medium group-hover:text-violet-400 transition-colors truncate">
                  {p.name}
                </span>
                <span className="text-xs text-[#656d76] mt-0.5">{p.video_count} 作品</span>
              </Link>
            ))}
          </div>
        ) : (
          <p className="text-sm text-[#8b949e] my-10">女優が見つかりませんでした。</p>
        )}

        {/* ページネーション */}
        {totalPages > 1 && (
          <div className="flex items-center justify-center gap-2 flex-wrap mb-8">
            {page > 1 && (
              <Link
                href={`/performers?page=${page - 1}${activeRow ? `&row=${activeRow}` : ''}`}
                className="px-4 py-2 rounded-lg bg-[#21262d] text-[#8b949e] hover:text-[#e6edf3] text-sm transition-colors"
              >
                ← 前へ
              </Link>
            )}
            <span className="text-sm text-[#656d76]">{page} / {totalPages}</span>
            {page < totalPages && (
              <Link
                href={`/performers?page=${page + 1}${activeRow ? `&row=${activeRow}` : ''}`}
                className="px-4 py-2 rounded-lg bg-[#21262d] text-[#8b949e] hover:text-[#e6edf3] text-sm transition-colors"
              >
                次へ →
              </Link>
            )}
          </div>
        )}

        <div className="py-8 border-t border-[#30363d] text-center space-y-2">
          <p className="text-sm text-[#8b949e]">スワイプするだけで、あなた好みの作品を AI が発掘します</p>
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
