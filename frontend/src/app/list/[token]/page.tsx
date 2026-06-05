import { Metadata } from 'next';
import { notFound } from 'next/navigation';
import Link from 'next/link';
import { supabase } from '@/lib/supabase';
import CopyLinkButton from './CopyLinkButton';

const SITE_URL = process.env.NEXT_PUBLIC_SITE_URL ?? 'https://www.seihekilab.com';
const AF_ID = process.env.NEXT_PUBLIC_FANZA_AFFILIATE_ID ?? 'yotadata2-001';

type Video = {
  id: string;
  title: string | null;
  external_id: string | null;
  thumbnail_url: string | null;
  thumbnail_vertical_url: string | null;
  product_url: string | null;
  liked_at: string;
};

type TagStat = { tag_name: string; cnt: number };
type PerformerStat = { performer_name: string; cnt: number };

type ListData = {
  display_name: string | null;
  title: string | null;
  videos: Video[];
  tags: TagStat[];
  performers: PerformerStat[];
};

function toAffiliateUrl(raw?: string | null): string {
  if (!raw) return '';
  if (raw.startsWith('https://al.fanza.co.jp/')) {
    try {
      const u = new URL(raw);
      u.searchParams.set('af_id', AF_ID);
      return u.toString();
    } catch { /* ignore */ }
  }
  return `https://al.fanza.co.jp/?lurl=${encodeURIComponent(raw)}&af_id=${encodeURIComponent(AF_ID)}&ch=link_tool&ch_id=link`;
}

function toLgThumb(url: string | null | undefined): string | null {
  if (!url) return null;
  return url.replace('ps.jpg', 'pl.jpg');
}

// ランク(0始まり)に応じたチップサイズ
const CHIP_SIZES = [
  { text: 'text-base', px: 'px-4',   py: 'py-2' },
  { text: 'text-sm',   px: 'px-3.5', py: 'py-1.5' },
  { text: 'text-sm',   px: 'px-3',   py: 'py-1.5' },
  { text: 'text-xs',   px: 'px-3',   py: 'py-1' },
  { text: 'text-xs',   px: 'px-2.5', py: 'py-1' },
];
function chipSize(i: number) { return CHIP_SIZES[Math.min(i, CHIP_SIZES.length - 1)]; }

const TAG_COLOR  = { bg: 'bg-violet-500/20', border: 'border-violet-500/40', text: 'text-violet-200', num: 'text-violet-400' };
const PERF_COLOR = { bg: 'bg-pink-500/20',   border: 'border-pink-500/40',   text: 'text-pink-200',   num: 'text-pink-400' };

// 技術的・流通系タグ（嗜好を表さないもの）は除外
const EXCLUDED_TAGS = new Set([
  '独占配信', 'ハイビジョン', '単体作品', '4K', '8K', 'VR', 'Ultra HD',
  '無修正', '高画質', 'フルHD', 'DVD', 'Blu-ray', '収録時間4時間以上',
]);
function filterTags(tags: TagStat[]): TagStat[] {
  return tags.filter((t) => !EXCLUDED_TAGS.has(t.tag_name));
}

async function fetchListData(token: string): Promise<ListData | null> {
  const { data, error } = await supabase.rpc('get_public_list_data', { p_token: token });
  if (error || !data || data.error === 'not_found') return null;
  return data as ListData;
}

export async function generateMetadata(
  { params }: { params: Promise<{ token: string }> }
): Promise<Metadata> {
  const { token } = await params;
  const data = await fetchListData(token);
  if (!data) return { title: 'リストが見つかりません | 性癖ラボ' };

  const name = data.display_name ?? 'あなた';
  const topTags = data.tags.slice(0, 3).map((t) => t.tag_name).join('・');
  const description = topTags
    ? `好きなジャンル: ${topTags}。${data.videos.length}作品のいいねリスト。`
    : `${data.videos.length}作品のいいねリスト。`;

  return {
    title: `${name}のお気に入りリスト | 性癖ラボ`,
    description,
    openGraph: {
      title: `${name}のお気に入りリスト | 性癖ラボ`,
      description,
      url: `${SITE_URL}/list/${token}`,
      siteName: '性癖ラボ',
      type: 'website',
    },
    robots: { index: false, follow: false },
  };
}

export default async function PublicListPage(
  { params }: { params: Promise<{ token: string }> }
) {
  const { token } = await params;
  const data = await fetchListData(token);
  if (!data) notFound();

  const name = data.display_name;
  const pageUrl = `${SITE_URL}/list/${token}`;
  const filteredTags = filterTags(data.tags);

  return (
    <div className="min-h-screen bg-[#0d1117] text-[#e6edf3]">
      <div className="max-w-4xl mx-auto px-4 py-10">

        {/* ナビ */}
        <Link href="/grid" className="text-xs text-[#656d76] hover:text-[#8b949e] transition-colors mb-6 inline-block">
          ← 性癖ラボへ戻る
        </Link>

        {/* ヘッダー */}
        <div className="mb-8 flex items-start justify-between gap-4">
          <div>
            <h1 className="text-2xl font-black text-[#e6edf3] mb-1">
              {name ? (
                <><span className="text-violet-400">{name}</span>のお気に入りリスト</>
              ) : 'お気に入りリスト'}
            </h1>
            <p className="text-sm text-[#656d76]">いいね {data.videos.length}作品</p>
          </div>
          <CopyLinkButton url={pageUrl} />
        </div>

        {/* 好きなジャンルランキング */}
        {filteredTags.length > 0 && (
          <div className="mb-8">
            <p className="text-xs font-semibold text-[#656d76] uppercase tracking-wider mb-3">好きなジャンルランキング</p>
            <div className="flex flex-wrap gap-2 items-end">
              {filteredTags.slice(0, 8).map((t, i) => {
                const sz = chipSize(i);
                return (
                  <span
                    key={t.tag_name}
                    className={`inline-flex items-center gap-1.5 ${sz.px} ${sz.py} rounded-full border font-semibold ${sz.text} ${TAG_COLOR.bg} ${TAG_COLOR.border}`}
                  >
                    <span className={`font-black ${TAG_COLOR.num} text-[10px]`}>{i + 1}</span>
                    {t.tag_name}
                  </span>
                );
              })}
            </div>
          </div>
        )}

        {/* 推し女優ランキング */}
        {(data.performers ?? []).length > 0 && (
          <div className="mb-8">
            <p className="text-xs font-semibold text-[#656d76] uppercase tracking-wider mb-3">推し女優ランキング</p>
            <div className="flex flex-wrap gap-2 items-end">
              {(data.performers ?? []).slice(0, 8).map((p, i) => {
                const sz = chipSize(i);
                return (
                  <span
                    key={p.performer_name}
                    className={`inline-flex items-center gap-1.5 ${sz.px} ${sz.py} rounded-full border font-semibold ${sz.text} ${PERF_COLOR.bg} ${PERF_COLOR.border}`}
                  >
                    <span className={`font-black ${PERF_COLOR.num} text-[10px]`}>{i + 1}</span>
                    {p.performer_name}
                  </span>
                );
              })}
            </div>
          </div>
        )}

        {/* 作品グリッド */}
        {data.videos.length === 0 ? (
          <p className="text-center text-[#656d76] py-20">まだいいねした作品がありません。</p>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            {data.videos.map((video) => {
              const affiliateUrl = toAffiliateUrl(video.product_url);
              const thumb = toLgThumb(video.thumbnail_url);
              return (
                <a
                  key={video.id}
                  href={affiliateUrl || '#'}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="group block rounded-lg overflow-hidden border border-[#21262d] hover:border-violet-500/50 transition-colors bg-[#161b22]"
                >
                  {thumb ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img
                      src={thumb}
                      alt={video.title ?? ''}
                      className="w-full aspect-video object-cover group-hover:opacity-90 transition-opacity"
                      loading="lazy"
                    />
                  ) : (
                    <div className="w-full aspect-video bg-[#21262d] flex items-center justify-center">
                      <span className="text-[#484f58] text-xs">No Image</span>
                    </div>
                  )}
                  <div className="p-2">
                    <p className="text-xs text-[#8b949e] leading-tight line-clamp-2">
                      {video.title ?? ''}
                    </p>
                  </div>
                </a>
              );
            })}
          </div>
        )}

        {/* フッター */}
        <div className="mt-12 pt-6 border-t border-[#21262d] text-center">
          <p className="text-sm text-[#656d76] mb-3">このリストは性癖ラボで作成されました</p>
          <Link
            href="/grid"
            className="inline-block px-6 py-2.5 rounded-full bg-violet-600 hover:bg-violet-500 text-white text-sm font-bold transition-colors"
          >
            自分のリストを作る
          </Link>
        </div>
      </div>
    </div>
  );
}
