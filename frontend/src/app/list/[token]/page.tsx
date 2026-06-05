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

type ListData = {
  display_name: string | null;
  title: string | null;
  videos: Video[];
  tags: TagStat[];
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

const TYPE_MAP: { keywords: string[]; typeName: string; color: string }[] = [
  { keywords: ['美少女', 'ロリ', '制服', '女子校生', '妹'], typeName: '清楚ロリ系', color: '#f9a8d4' },
  { keywords: ['巨乳', '爆乳', 'お姉さん', 'むちむち', '痴女'], typeName: '巨乳お姉さん系', color: '#fbbf24' },
  { keywords: ['人妻', '不倫', '寝取られ', 'NTR', '禁断'], typeName: '背徳NTR系', color: '#a78bfa' },
  { keywords: ['ギャル', '日焼け', 'ビッチ', 'パリピ'], typeName: 'ギラギラギャル系', color: '#34d399' },
  { keywords: ['SM', '拘束', '調教', '支配', '主従'], typeName: '刺激スパイス系', color: '#f87171' },
  { keywords: ['単体作品', '王道', 'ノーマル'], typeName: '王道単体派', color: '#60a5fa' },
];

function deriveType(tags: TagStat[]): { typeName: string; color: string } {
  const tagNames = tags.map((t) => t.tag_name);
  for (const profile of TYPE_MAP) {
    if (profile.keywords.some((kw) => tagNames.some((n) => n.includes(kw)))) {
      return { typeName: profile.typeName, color: profile.color };
    }
  }
  return { typeName: 'バランス型', color: '#8b949e' };
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

  const title = data.title ?? '私のお気に入りリスト';
  const topTags = data.tags.slice(0, 3).map((t) => t.tag_name).join('・');
  const description = topTags
    ? `好きなジャンル: ${topTags}。${data.videos.length}作品のいいねリスト。`
    : `${data.videos.length}作品のいいねリスト。`;

  return {
    title: `${title} | 性癖ラボ`,
    description,
    openGraph: {
      title: `${title} | 性癖ラボ`,
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

  const title = data.title ?? 'お気に入りリスト';
  const pageUrl = `${SITE_URL}/list/${token}`;
  const typeProfile = deriveType(data.tags);
  const totalLikes = data.tags.reduce((sum, t) => sum + t.cnt, 0);

  return (
    <div className="min-h-screen bg-[#0d1117] text-[#e6edf3]">
      <div className="max-w-4xl mx-auto px-4 py-10">

        {/* ナビ */}
        <Link href="/" className="text-xs text-[#656d76] hover:text-[#8b949e] transition-colors mb-6 inline-block">
          ← 性癖ラボ
        </Link>

        {/* 性癖ステータスカード */}
        <div
          className="mb-8 rounded-2xl overflow-hidden border border-white/10"
          style={{ background: 'linear-gradient(135deg, #1a1040 0%, #0d1117 60%, #1a0d24 100%)' }}
        >
          {/* カードヘッダー */}
          <div className="px-6 pt-6 pb-4 border-b border-white/10 flex items-center justify-between gap-4">
            <div>
              <p className="text-[10px] font-bold tracking-[0.35em] uppercase mb-1" style={{ color: typeProfile.color }}>
                性癖ステータス
              </p>
              {data.display_name && (
                <p className="text-sm text-[#8b949e] mb-0.5">{data.display_name}</p>
              )}
              <h1 className="text-2xl font-black text-[#e6edf3]">{title}</h1>
            </div>
            <div
              className="shrink-0 px-4 py-2 rounded-xl text-sm font-black border"
              style={{ color: typeProfile.color, borderColor: `${typeProfile.color}40`, background: `${typeProfile.color}15` }}
            >
              {typeProfile.typeName}
            </div>
          </div>

          {/* タグステータス */}
          {data.tags.length > 0 && (
            <div className="px-6 py-5">
              <p className="text-[10px] font-bold tracking-[0.3em] uppercase text-[#656d76] mb-4">好きなジャンル — TOP {Math.min(data.tags.length, 8)}</p>
              <div className="flex flex-col gap-3">
                {data.tags.slice(0, 8).map((t, i) => {
                  const ratio = totalLikes > 0 ? (t.cnt / totalLikes) * 100 : 0;
                  const rankColors = ['#f9a8d4', '#fbbf24', '#a78bfa', '#60a5fa', '#34d399'];
                  const barColor = rankColors[i] ?? '#8b949e';
                  return (
                    <div key={t.tag_name} className="flex items-center gap-3">
                      <span className="w-5 text-right text-[11px] font-bold shrink-0" style={{ color: i < 3 ? barColor : '#656d76' }}>
                        {i + 1}
                      </span>
                      <span className="w-24 text-sm font-semibold text-[#c9d1d9] truncate shrink-0">{t.tag_name}</span>
                      <div className="flex-1 h-2 rounded-full bg-white/5 overflow-hidden">
                        <div
                          className="h-full rounded-full transition-all"
                          style={{ width: `${Math.max(ratio, 2)}%`, background: `linear-gradient(90deg, ${barColor}cc, ${barColor}66)` }}
                        />
                      </div>
                      <span className="text-xs text-[#656d76] shrink-0 w-10 text-right">{t.cnt}件</span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* フッター情報 */}
          <div className="px-6 pb-5 flex items-center justify-between gap-4">
            <p className="text-xs text-[#484f58]">いいね作品 {data.videos.length}本</p>
            <CopyLinkButton url={pageUrl} />
          </div>
        </div>

        {/* 作品グリッド */}
        {data.videos.length === 0 ? (
          <p className="text-center text-[#656d76] py-20">まだいいねした作品がありません。</p>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
            {data.videos.map((video) => {
              const affiliateUrl = toAffiliateUrl(video.product_url);
              const thumb = toLgThumb(video.thumbnail_vertical_url) ?? toLgThumb(video.thumbnail_url);
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
                      className="w-full aspect-[3/4] object-cover group-hover:opacity-90 transition-opacity"
                      loading="lazy"
                    />
                  ) : (
                    <div className="w-full aspect-[3/4] bg-[#21262d] flex items-center justify-center">
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
