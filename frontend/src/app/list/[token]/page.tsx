import { Metadata } from 'next';
import { notFound } from 'next/navigation';
import Link from 'next/link';
import { supabase } from '@/lib/supabase';
import CopyLinkButton from './CopyLinkButton';
import ListEditMode, { VideoDeleteButton } from './ListEditMode';

const SITE_URL = process.env.NEXT_PUBLIC_SITE_URL ?? 'https://www.seihekilab.com';
const AF_ID = process.env.NEXT_PUBLIC_FANZA_AFFILIATE_ID ?? 'yotadata2-001';

// スワイプ画面と同じグラデーション
const BG = 'linear-gradient(135deg, #C4C8E3 0%, #D7D1E3 30%, #F0D5E8 65%, #F9C9D6 100%)';

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
    <div className="min-h-screen" style={{ background: BG }}>
      <div className="max-w-4xl mx-auto px-4 py-8 sm:py-12">

        {/* ナビ */}
        <Link
          href="/grid"
          className="inline-flex items-center gap-1 text-xs text-white/60 hover:text-white/90 transition-colors mb-6 drop-shadow"
        >
          ← 性癖ラボへ戻る
        </Link>

        {/* ━━━ ヒーローカード（グラスモーフィズム） ━━━ */}
        <div className="rounded-3xl bg-white/30 backdrop-blur-xl border border-white/50 shadow-[0_8px_32px_rgba(0,0,0,0.12)] px-6 sm:px-10 py-8 mb-6">
          <div className="flex items-start justify-between gap-4 mb-5">
            <div className="space-y-1.5">
              <h1 className="text-2xl sm:text-3xl font-black text-gray-800 leading-tight">
                {data.title ?? (name ? `${name}のお気に入りリスト` : 'お気に入りリスト')}
              </h1>
              {name && (
                <p className="text-sm text-gray-500">
                  by{' '}
                  {data.username ? (
                    <Link
                      href={`/u/${data.username}`}
                      className="text-violet-600 font-semibold hover:underline underline-offset-2"
                    >
                      {name}
                    </Link>
                  ) : (
                    <span className="text-violet-600 font-semibold">{name}</span>
                  )}
                </p>
              )}
              <p className="text-xs text-gray-400">{data.videos.length}作品</p>
            </div>
            <CopyLinkButton url={pageUrl} variant="glass" />
          </div>

          {/* タグランキング */}
          {filteredTags.length > 0 && (
            <div className="mb-4">
              <p className="text-[10px] font-bold text-gray-400 uppercase tracking-widest mb-2">好きなジャンル</p>
              <div className="flex flex-wrap gap-2 items-center">
                {filteredTags.slice(0, 8).map((t, i) => {
                  const sz = chipSize(i);
                  return (
                    <span
                      key={t.tag_name}
                      className={`inline-flex items-center gap-1 ${sz.px} ${sz.py} rounded-full ${sz.text} font-semibold bg-white/50 border border-white/70 text-violet-700 shadow-sm`}
                    >
                      <span className="text-violet-400 font-black text-[9px]">{i + 1}</span>
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
              <p className="text-[10px] font-bold text-gray-400 uppercase tracking-widest mb-2">推し女優</p>
              <div className="flex flex-wrap gap-2 items-center">
                {(data.performers ?? []).slice(0, 6).map((p, i) => {
                  const sz = chipSize(i);
                  return (
                    <span
                      key={p.performer_name}
                      className={`inline-flex items-center gap-1 ${sz.px} ${sz.py} rounded-full ${sz.text} font-semibold bg-pink-100/70 border border-pink-200/80 text-pink-700 shadow-sm`}
                    >
                      <span className="text-pink-400 font-black text-[9px]">{i + 1}</span>
                      {p.performer_name}
                    </span>
                  );
                })}
              </div>
            </div>
          )}
        </div>

        {/* 編集モード（オーナーのみ） */}
        <ListEditMode
          ownerUserId={data.user_id}
          listId={data.list_id}
          listType={data.list_type}
          listTitle={data.title}
          token={token}
        />

        {/* ━━━ 動画グリッド ━━━ */}
        {data.videos.length === 0 ? (
          <div className="rounded-3xl bg-white/30 backdrop-blur-xl border border-white/50 p-16 text-center text-gray-400">
            まだ作品がありません。
          </div>
        ) : (
          <div className="[column-count:2] sm:[column-count:3] [column-gap:10px]">
            {data.videos.map((video) => {
              const affiliateUrl = toAffiliateUrl(video.product_url);
              const thumb = toLgThumb(video.thumbnail_url) ?? toLgThumb(video.thumbnail_vertical_url);
              return (
                <div key={video.id} className="group relative mb-2.5 break-inside-avoid">
                  <a
                    href={affiliateUrl || '#'}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="block rounded-2xl overflow-hidden bg-white/40 backdrop-blur-sm border border-white/60 shadow-sm hover:shadow-lg hover:bg-white/60 hover:-translate-y-0.5 transition-all duration-200"
                  >
                    {thumb ? (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img
                        src={thumb}
                        alt={video.title ?? ''}
                        className="w-full h-auto"
                        loading="lazy"
                      />
                    ) : (
                      <div className="w-full aspect-video bg-white/20 flex items-center justify-center">
                        <span className="text-gray-400 text-xs">No Image</span>
                      </div>
                    )}
                    <div className="px-2.5 py-2">
                      <p className="text-[11px] text-gray-600 leading-tight line-clamp-2">
                        {video.title ?? ''}
                      </p>
                    </div>
                  </a>
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
        <div className="mt-12 text-center">
          <div className="inline-block rounded-2xl bg-white/30 backdrop-blur-xl border border-white/50 px-8 py-6 shadow-sm">
            <p className="text-sm text-gray-500 mb-3">このリストは性癖ラボで作成されました</p>
            <Link
              href="/grid"
              className="inline-block px-6 py-2.5 rounded-full bg-white/60 hover:bg-white/80 text-violet-700 text-sm font-bold border border-white/70 shadow-sm transition-all"
            >
              自分のリストを作る →
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
