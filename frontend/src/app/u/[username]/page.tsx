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
