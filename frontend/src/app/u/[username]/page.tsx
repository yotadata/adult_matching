import { Metadata } from 'next';
import { notFound } from 'next/navigation';
import Link from 'next/link';
import { supabase } from '@/lib/supabase';
import UserListsClient, { type ListItem } from './UserListsEditMode';

const SITE_URL = process.env.NEXT_PUBLIC_SITE_URL ?? 'https://www.seihekilab.com';

type UserListsData = {
  user_id: string;
  display_name: string;
  username: string;
  avatar_url: string | null;
  bio: string | null;
  x_url: string | null;
  lists: ListItem[];
};

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

  const description = data.bio
    ? `${data.bio} | ${data.lists.length}件のリスト`
    : `${data.display_name}が作成した${data.lists.length}件のリスト`;
  const ogImage = data.avatar_url ?? data.lists[0]?.thumbnails?.[0] ?? null;

  return {
    title: `${data.display_name}のリスト一覧 | 性癖ラボ`,
    description,
    openGraph: {
      title: `${data.display_name}のリスト一覧 | 性癖ラボ`,
      description,
      url: `${SITE_URL}/u/${username}`,
      siteName: '性癖ラボ',
      type: 'profile',
      ...(ogImage ? { images: [{ url: ogImage, width: 400, height: 400, alt: data.display_name }] } : {}),
    },
    twitter: {
      card: ogImage ? 'summary' : 'summary',
      title: `${data.display_name}のリスト一覧 | 性癖ラボ`,
      description,
      ...(ogImage ? { images: [ogImage] } : {}),
    },
    robots: { index: true, follow: true },
  };
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

        {/* ヘッダー（キュレータープロフィール） */}
        <div className="mb-8 flex items-start gap-4">
          {/* アバター */}
          <div className="shrink-0">
            {data.avatar_url ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={data.avatar_url}
                alt={data.display_name}
                className="w-16 h-16 rounded-full object-cover border-2 border-violet-500/40"
              />
            ) : (
              <div className="w-16 h-16 rounded-full bg-violet-500/20 border-2 border-violet-500/30 flex items-center justify-center text-2xl font-black text-violet-400">
                {data.display_name?.charAt(0) ?? '?'}
              </div>
            )}
          </div>
          <div className="min-w-0">
            <h1 className="text-2xl font-black text-[#e6edf3] mb-0.5">
              <span className="text-violet-400">{data.display_name}</span>
            </h1>
            <p className="text-xs text-[#484f58] mb-1">@{data.username}</p>
            {data.bio && (
              <p className="text-sm text-[#8b949e] leading-relaxed mb-2 whitespace-pre-wrap">{data.bio}</p>
            )}
            <div className="flex items-center gap-3 flex-wrap">
              <p className="text-xs text-[#656d76]">{data.lists.length}件のリスト</p>
              {data.x_url && (
                <a
                  href={data.x_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-1 text-xs text-[#656d76] hover:text-[#8b949e] transition-colors"
                >
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-4.714-6.231-5.401 6.231H2.744l7.73-8.835L1.254 2.25H8.08l4.258 5.63 5.906-5.63zm-1.161 17.52h1.833L7.084 4.126H5.117z"/>
                  </svg>
                  X（旧Twitter）
                </a>
              )}
            </div>
          </div>
        </div>

        {/* リスト一覧（編集モード対応クライアントコンポーネント） */}
        <UserListsClient
          ownerUserId={data.user_id}
          username={username}
          displayName={data.display_name}
          initialLists={data.lists}
        />

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
