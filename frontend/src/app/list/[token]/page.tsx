import { Metadata } from 'next';
import { notFound } from 'next/navigation';
import Link from 'next/link';
import { supabase } from '@/lib/supabase';
import CopyLinkButton from './CopyLinkButton';
import ListEditMode, { VideoDeleteButton } from './ListEditMode';

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
  list_id: string;
  user_id: string;
  display_name: string | null;
  username: string | null;
  title: string | null;
  list_type: 'liked' | 'custom';
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

// ランクに応じたチップサイズ
const CHIP_SIZES = [
  { text: 'text-sm',  px: 'px-3.5', py: 'py-1.5' },
  { text: 'text-sm',  px: 'px-3',   py: 'py-1.5' },
  { text: 'text-xs',  px: 'px-3',   py: 'py-1' },
  { text: 'text-xs',  px: 'px-2.5', py: 'py-1' },
  { text: 'text-xs',  px: 'px-2',   py: 'py-0.5' },
];
function chipSize(i: number) { return CHIP_SIZES[Math.min(i, CHIP_SIZES.length - 1)]; }

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
  const topTags = filterTags(data.tags).slice(0, 3).map((t) => t.tag_name).join('・');
  const description = topTags
    ? `好きなジャンル: ${topTags}。${data.videos.length}作品のリスト。`
    : `${data.videos.length}作品のリスト。`;
  const listTitle = data.title ?? `${name}のお気に入りリスト`;

  return {
    title: `${listTitle} | 性癖ラボ`,
    description,
    openGraph: {
      title: `${listTitle} | 性癖ラボ`,
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
    <div className="min-h-screen bg-[#f5f5f7]">

      {/* ━━━ HERO ━━━ */}
      <div className="relative overflow-hidden bg-gradient-to-br from-indigo-500 via-violet-500 to-pink-400 text-white">
        {/* 背景の光彩 */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <div className="absolute -top-24 -left-24 w-96 h-96 bg-white/10 rounded-full blur-3xl" />
          <div className="absolute -bottom-16 -right-16 w-80 h-80 bg-pink-300/20 rounded-full blur-3xl" />
        </div>

        <div className="relative max-w-4xl mx-auto px-4 pt-8 pb-10">
          {/* ナビ */}
          <Link
            href="/grid"
            className="inline-flex items-center gap-1 text-xs text-white/60 hover:text-white transition-colors mb-6"
          >
            ← 性癖ラボへ戻る
          </Link>

          {/* タイトル＋作者＋アクション */}
          <div className="flex items-start justify-between gap-4 mb-6">
            <div className="space-y-2">
              <h1 className="text-3xl sm:text-4xl font-black leading-tight drop-shadow-sm">
                {data.title ?? (name ? `${name}のお気に入りリスト` : 'お気に入りリスト')}
              </h1>
              {name && (
                <p className="text-sm text-white/70">
                  by{' '}
                  {data.username ? (
                    <Link
                      href={`/u/${data.username}`}
                      className="text-white font-semibold hover:underline underline-offset-2"
                    >
                      {name}
                    </Link>
                  ) : (
                    <span className="text-white font-semibold">{name}</span>
                  )}
                </p>
              )}
              <p className="text-sm text-white/50">{data.videos.length}作品</p>
            </div>
            <CopyLinkButton url={pageUrl} variant="light" />
          </div>

          {/* タグランキング */}
          {filteredTags.length > 0 && (
            <div className="mb-4">
              <p className="text-[10px] font-bold text-white/50 uppercase tracking-widest mb-2">好きなジャンル</p>
              <div className="flex flex-wrap gap-2 items-center">
                {filteredTags.slice(0, 8).map((t, i) => {
                  const sz = chipSize(i);
                  return (
                    <span
                      key={t.tag_name}
                      className={`inline-flex items-center gap-1 ${sz.px} ${sz.py} rounded-full ${sz.text} font-semibold bg-white/15 backdrop-blur border border-white/20 text-white`}
                    >
                      <span className="text-white/50 font-black text-[9px]">{i + 1}</span>
                      {t.tag_name}
                    </span>
                  );
                })}
              </div>
            </div>
          )}

          {/* 女優ランキング */}
          {(data.performers ?? []).length > 0 && (
            <div>
              <p className="text-[10px] font-bold text-white/50 uppercase tracking-widest mb-2">推し女優</p>
              <div className="flex flex-wrap gap-2 items-center">
                {(data.performers ?? []).slice(0, 6).map((p, i) => {
                  const sz = chipSize(i);
                  return (
                    <span
                      key={p.performer_name}
                      className={`inline-flex items-center gap-1 ${sz.px} ${sz.py} rounded-full ${sz.text} font-semibold bg-pink-500/30 backdrop-blur border border-pink-300/30 text-white`}
                    >
                      <span className="text-pink-200/60 font-black text-[9px]">{i + 1}</span>
                      {p.performer_name}
                    </span>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* ━━━ BODY ━━━ */}
      <div className="max-w-4xl mx-auto px-4 py-8">

        {/* 編集モード（オーナーのみ） */}
        <ListEditMode
          ownerUserId={data.user_id}
          listId={data.list_id}
          listType={data.list_type}
          listTitle={data.title}
          token={token}
        />

        {/* 作品グリッド */}
        {data.videos.length === 0 ? (
          <p className="text-center text-gray-400 py-20">まだ作品がありません。</p>
        ) : (
          <div className="[column-count:2] sm:[column-count:3] [column-gap:12px]">
            {data.videos.map((video) => {
              const affiliateUrl = toAffiliateUrl(video.product_url);
              const thumb = toLgThumb(video.thumbnail_url) ?? toLgThumb(video.thumbnail_vertical_url);
              return (
                <div key={video.id} className="group relative mb-3 break-inside-avoid">
                  <a
                    href={affiliateUrl || '#'}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="block rounded-xl overflow-hidden bg-white shadow-sm hover:shadow-md transition-shadow border border-black/5"
                  >
                    {thumb ? (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img
                        src={thumb}
                        alt={video.title ?? ''}
                        className="w-full h-auto group-hover:scale-[1.02] transition-transform duration-300"
                        loading="lazy"
                      />
                    ) : (
                      <div className="w-full aspect-video bg-gray-100 flex items-center justify-center">
                        <span className="text-gray-400 text-xs">No Image</span>
                      </div>
                    )}
                    <div className="px-2.5 py-2">
                      <p className="text-[11px] text-gray-500 leading-tight line-clamp-2">
                        {video.title ?? ''}
                      </p>
                    </div>
                  </a>
                  {/* 削除ボタン（オーナー＆カスタムリストのみ） */}
                  {data.list_type === 'custom' && (
                    <VideoDeleteButton
                      ownerUserId={data.user_id}
                      listId={data.list_id}
                      videoId={video.id}
                    />
                  )}
                </div>
              );
            })}
          </div>
        )}

        {/* フッター */}
        <div className="mt-12 pt-6 border-t border-black/10 text-center">
          <p className="text-sm text-gray-400 mb-3">このリストは性癖ラボで作成されました</p>
          <Link
            href="/grid"
            className="inline-block px-6 py-2.5 rounded-full bg-gradient-to-r from-violet-500 to-pink-500 text-white text-sm font-bold hover:opacity-90 transition-opacity shadow-md"
          >
            自分のリストを作る
          </Link>
        </div>
      </div>
    </div>
  );
}
