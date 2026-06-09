import { Metadata } from 'next';
import { notFound } from 'next/navigation';
import Link from 'next/link';
import { supabase } from '@/lib/supabase';

const SITE_URL = process.env.NEXT_PUBLIC_SITE_URL ?? 'https://www.seihekilab.com';

type ListItem = {
  id: string;
  token: string;
  title: string | null;
  description: string | null;
  list_type: 'liked' | 'custom';
  created_at: string;
  video_count: number;
  thumbnails: string[] | null;
};

type UserListsData = {
  display_name: string;
  lists: ListItem[];
};

function toLgThumb(url: string | null | undefined): string | null {
  if (!url) return null;
  return url.replace('ps.jpg', 'pl.jpg');
}

async function fetchUserLists(username: string): Promise<UserListsData | null> {
  const { data, error } = await supabase.rpc('get_user_public_lists', {
    p_username: username,
  });
  if (error || !data || data.error === 'not_found') return null;
  return data as UserListsData;
}

export async function generateMetadata(
  { params }: { params: Promise<{ username: string }> }
): Promise<Metadata> {
  const { username } = await params;
  const data = await fetchUserLists(username);
  if (!data) return { title: 'ユーザーが見つかりません | 性癖ラボ' };

  return {
    title: `${data.display_name}のリスト一覧 | 性癖ラボ`,
    description: `${data.display_name}が作成した${data.lists.length}件のリスト`,
    openGraph: {
      title: `${data.display_name}のリスト一覧 | 性癖ラボ`,
      description: `${data.display_name}が作成した${data.lists.length}件のリスト`,
      url: `${SITE_URL}/u/${username}`,
      siteName: '性癖ラボ',
      type: 'website',
    },
    robots: { index: false, follow: false },
  };
}

function ListCard({ list }: { list: ListItem }) {
  const thumbs = (list.thumbnails ?? []).slice(0, 3).map(toLgThumb).filter(Boolean) as string[];
  const label = list.list_type === 'liked' ? 'お気に入りリスト' : list.title ?? '無題のリスト';

  return (
    <Link
      href={`/list/${list.token}`}
      className="group block rounded-xl border border-[#21262d] hover:border-violet-500/50 bg-[#161b22] overflow-hidden transition-all hover:shadow-lg hover:shadow-violet-500/10"
    >
      {/* サムネイルプレビュー */}
      <div className="flex h-28 overflow-hidden">
        {thumbs.length > 0 ? (
          thumbs.map((url, i) => (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              key={i}
              src={url}
              alt=""
              className={`object-cover flex-1 min-w-0 ${i > 0 ? 'border-l border-[#0d1117]' : ''} group-hover:opacity-90 transition-opacity`}
              loading="lazy"
            />
          ))
        ) : (
          <div className="w-full h-full bg-[#21262d] flex items-center justify-center">
            <span className="text-[#484f58] text-xs">No Image</span>
          </div>
        )}
      </div>

      {/* 情報 */}
      <div className="p-3">
        <p className="text-sm font-bold text-[#e6edf3] group-hover:text-violet-400 transition-colors line-clamp-1">
          {label}
        </p>
        {list.description && (
          <p className="text-xs text-[#656d76] mt-0.5 line-clamp-2">{list.description}</p>
        )}
        <p className="text-xs text-[#484f58] mt-1.5">{list.video_count.toLocaleString('ja-JP')}作品</p>
      </div>
    </Link>
  );
}

export default async function UserListsPage(
  { params }: { params: Promise<{ username: string }> }
) {
  const { username } = await params;
  const data = await fetchUserLists(username);
  if (!data) notFound();

  return (
    <div className="min-h-screen bg-[#0d1117] text-[#e6edf3]">
      <div className="max-w-4xl mx-auto px-4 py-10">

        {/* ナビ */}
        <Link href="/grid" className="text-xs text-[#656d76] hover:text-[#8b949e] transition-colors mb-6 inline-block">
          ← 性癖ラボへ戻る
        </Link>

        {/* ヘッダー */}
        <div className="mb-8">
          <h1 className="text-2xl font-black text-[#e6edf3] mb-1">
            <span className="text-violet-400">{data.display_name}</span>のリスト
          </h1>
          <p className="text-sm text-[#656d76]">{data.lists.length}件のリスト</p>
        </div>

        {/* リスト一覧 */}
        {data.lists.length === 0 ? (
          <p className="text-center text-[#656d76] py-20">まだ公開リストがありません。</p>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {data.lists.map((list) => (
              <ListCard key={list.id} list={list} />
            ))}
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
