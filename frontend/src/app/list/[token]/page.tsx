import { Metadata } from 'next';
import { notFound } from 'next/navigation';
import Link from 'next/link';
import { supabase } from '@/lib/supabase';
import CopyLinkButton from './CopyLinkButton';
import ListEditMode, { VideoDeleteButton } from './ListEditMode';
import ListStats from './ListStats';
import { resolveThumbnail } from '@/utils/thumbnail';


const SITE_URL = process.env.NEXT_PUBLIC_SITE_URL ?? 'https://www.seihekilab.com';
const SYS_AF_ID     = process.env.NEXT_PUBLIC_FANZA_AFFILIATE_ID ?? 'yotadata2-001';
const SYS_MGS_AF_ID = process.env.NEXT_PUBLIC_MGS_AFFILIATE_ID   ?? 'HU3ADNBETQPYWHO8EFF88GY3NH';

type Video = {
  id: string;
  title: string | null;
  external_id: string | null;
  thumbnail_url: string | null;
  thumbnail_vertical_url: string | null;
  product_url: string | null;
  liked_at: string;
  source?: string | null;
  image_urls?: string[] | null;
};

type TagStat = { tag_name: string; cnt: number };
type PerformerStat = { performer_name: string; cnt: number };

type ListData = {
  list_id: string;
  user_id: string;
  display_name: string | null;
  username: string | null;
  affiliate_fanza_id: string | null;
  affiliate_mgs_id: string | null;
  title: string | null;
  list_type: 'liked' | 'custom';
  view_count: number;
  like_count: number;
  videos: Video[];
  tags: TagStat[];
  performers: PerformerStat[];
};

function toAffiliateUrl(raw?: string | null, source?: string | null, curatorFanzaId?: string | null, curatorMgsId?: string | null): string {
  if (!raw) return '';
  if (source === 'mgs') {
    const mgsId = curatorMgsId ?? SYS_MGS_AF_ID;
    try {
      const u = new URL(raw);
      u.searchParams.set('agef', '1');
      u.searchParams.set('utm_medium', 'mgs_affiliate');
      u.searchParams.set('utm_source', 'mgs_affiliate_linktool');
      u.searchParams.set('utm_campaign', 'mgs_affiliate_linktool');
      u.searchParams.set('utm_content', mgsId);
      u.searchParams.set('form', `mgs_asp_linktool_${mgsId}`);
      return u.toString();
    } catch { /* ignore */ }
    return raw;
  }
  const afId = curatorFanzaId ?? SYS_AF_ID;
  if (raw.startsWith('https://al.fanza.co.jp/')) {
    try {
      const u = new URL(raw);
      u.searchParams.set('af_id', afId);
      return u.toString();
    } catch { /* ignore */ }
  }
  return `https://al.fanza.co.jp/?lurl=${encodeURIComponent(raw)}&af_id=${encodeURIComponent(afId)}&ch=link_tool&ch_id=link`;
}

function toLgThumb(url: string | null | undefined): string | null {
  if (!url) return null;
  return url.replace('ps.jpg', 'pl.jpg');
}

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
    <div className="min-h-screen bg-[#0d1117] text-[#e6edf3]">
      <div className="max-w-4xl mx-auto px-4 py-10">

        {/* ナビ */}
        <Link href="/grid" className="text-xs text-[#656d76] hover:text-[#8b949e] transition-colors mb-6 inline-block">
          ← 性癖ラボへ戻る
        </Link>

        {/* ヘッダー */}
        <div className="mb-6 flex items-start justify-between gap-4">
          <div className="space-y-1">
            <h1 className="text-2xl font-black text-[#e6edf3]">
              {data.title ?? (name ? `${name}のお気に入りリスト` : 'お気に入りリスト')}
            </h1>
            {name && (
              <p className="text-sm text-[#656d76]">
                by <span className="text-violet-400">{name}</span>
              </p>
            )}
            <p className="text-xs text-[#484f58]">{data.videos.length}作品</p>
            <ListStats
              token={token}
              initialViewCount={data.view_count}
              initialLikeCount={data.like_count}
            />
          </div>
          <CopyLinkButton url={pageUrl} />
        </div>

        {/* 作者のリスト一覧へ */}
        {data.username && (
          <Link
            href={`/u/${data.username}`}
            className="flex items-center justify-between gap-3 mb-8 px-4 py-3 rounded-xl bg-[#161b22] border border-[#30363d] hover:border-violet-500/50 hover:bg-[#1c2128] transition-all group"
          >
            <div className="flex items-center gap-2.5 min-w-0">
              <div className="w-8 h-8 rounded-full bg-violet-500/20 border border-violet-500/30 flex items-center justify-center shrink-0 text-sm">
                {name?.charAt(0) ?? '?'}
              </div>
              <div className="min-w-0">
                <p className="text-xs text-[#656d76]">作成者</p>
                <p className="text-sm font-bold text-[#e6edf3] truncate">{name}</p>
              </div>
            </div>
            <span className="text-xs text-[#656d76] group-hover:text-violet-400 transition-colors shrink-0">
              他のリストを見る →
            </span>
          </Link>
        )}

        {/* 編集モードバナー＋動画追加ボタン（オーナーのみ表示） */}
        <ListEditMode
          ownerUserId={data.user_id}
          listId={data.list_id}
          listType={data.list_type}
          listTitle={data.title}
          token={token}
        />

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
          <p className="text-center text-[#656d76] py-20">まだ作品がありません。</p>
        ) : (
          <div className="[column-count:2] sm:[column-count:3] [column-gap:12px]">
            {data.videos.map((video) => {
              const affiliateUrl = toAffiliateUrl(video.product_url, video.source, data.affiliate_fanza_id, data.affiliate_mgs_id);
              const { primary: resolvedThumb } = resolveThumbnail({ source: video.source, thumbnail_url: video.thumbnail_url, image_urls: video.image_urls });
              const thumb = resolvedThumb ?? toLgThumb(video.thumbnail_url) ?? toLgThumb(video.thumbnail_vertical_url);
              return (
                <div key={video.id} className="group relative mb-3 break-inside-avoid">
                  <a
                    href={affiliateUrl || '#'}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="block rounded-lg overflow-hidden border border-[#21262d] hover:border-violet-500/50 transition-colors bg-[#161b22]"
                  >
                    {thumb ? (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img
                        src={thumb}
                        alt={video.title ?? ''}
                        className="w-full h-auto group-hover:opacity-90 transition-opacity"
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
