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

  return (
    <div className="min-h-screen bg-[#0d1117] text-[#e6edf3]">
      <div className="max-w-4xl mx-auto px-4 py-10">

        {/* ヘッダー */}
        <div className="mb-8">
          <Link href="/" className="text-xs text-[#656d76] hover:text-[#8b949e] transition-colors mb-4 inline-block">
            ← 性癖ラボ
          </Link>
          {data.display_name && (
            <p className="text-sm text-[#656d76] mb-1">{data.display_name} のリスト</p>
          )}
          <h1 className="text-2xl font-black text-[#e6edf3] mb-1">{title}</h1>
          <p className="text-sm text-[#8b949e]">{data.videos.length}作品</p>
          <div className="mt-3">
            <CopyLinkButton url={pageUrl} />
          </div>
        </div>

        {/* タグまとめ */}
        {data.tags.length > 0 && (
          <div className="mb-8 p-4 rounded-xl border border-[#21262d] bg-[#161b22]">
            <p className="text-xs font-semibold text-[#656d76] uppercase tracking-wider mb-3">好きなジャンル</p>
            <div className="flex flex-wrap gap-2">
              {data.tags.map((t) => (
                <span
                  key={t.tag_name}
                  className="px-3 py-1 rounded-full text-sm bg-violet-900/40 text-violet-300 border border-violet-500/30"
                >
                  {t.tag_name}
                  <span className="ml-1.5 text-xs text-violet-400/60">{t.cnt}</span>
                </span>
              ))}
            </div>
          </div>
        )}

        {/* 作品グリッド */}
        {data.videos.length === 0 ? (
          <p className="text-center text-[#656d76] py-20">まだいいねした作品がありません。</p>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
            {data.videos.map((video) => {
              const affiliateUrl = toAffiliateUrl(video.product_url);
              const thumb = video.thumbnail_vertical_url ?? video.thumbnail_url;
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
