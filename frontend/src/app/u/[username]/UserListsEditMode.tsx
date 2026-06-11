'use client';

import { useEffect, useState, useCallback, useTransition } from 'react';
import Link from 'next/link';
import { Plus, Trash2, Loader2, Search } from 'lucide-react';
import { supabase } from '@/lib/supabase';
import VideoSearchModal from '@/components/lists/VideoSearchModal';

export type ListItem = {
  id: string;
  token: string;
  title: string | null;
  description: string | null;
  list_type: 'liked' | 'custom';
  created_at: string;
  video_count: number;
  thumbnails: string[] | null;
};

function toLgThumb(url: string | null | undefined): string | null {
  if (!url) return null;
  return url.replace('ps.jpg', 'pl.jpg');
}

interface Props {
  ownerUserId: string;
  username: string;
  displayName: string;
  initialLists: ListItem[];
}

export default function UserListsClient({ ownerUserId, username, displayName, initialLists }: Props) {
  const [isOwner, setIsOwner] = useState(false);
  const [lists, setLists] = useState<ListItem[]>(initialLists);

  // 新規リスト作成
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [newTitle, setNewTitle] = useState('');
  const [newDesc, setNewDesc] = useState('');
  const [creating, setCreating] = useState(false);

  // 動画追加モーダル
  const [searchTarget, setSearchTarget] = useState<ListItem | null>(null);

  const [, startTransition] = useTransition();

  useEffect(() => {
    supabase.auth.getUser().then(({ data: { user } }) => {
      setIsOwner(!!user && user.id === ownerUserId);
    });
  }, [ownerUserId]);

  // リスト一覧を再取得
  const reloadLists = useCallback(async () => {
    const { data } = await supabase.rpc('get_user_public_lists', { p_username: username });
    if (data && !data.error) {
      startTransition(() => setLists((data as { lists: ListItem[] }).lists ?? []));
    }
  }, [username]);

  const handleCreate = useCallback(async () => {
    if (!newTitle.trim()) return;
    setCreating(true);
    const { data: { user } } = await supabase.auth.getUser();
    if (!user) return;
    const { error } = await supabase.from('public_lists').insert({
      user_id: user.id,
      title: newTitle.trim(),
      description: newDesc.trim() || null,
      list_type: 'custom',
    });
    if (error) { alert('作成に失敗しました: ' + error.message); }
    setNewTitle('');
    setNewDesc('');
    setShowCreateForm(false);
    setCreating(false);
    await reloadLists();
  }, [newTitle, newDesc, reloadLists]);

  const handleDelete = useCallback(async (id: string) => {
    if (!confirm('このリストを削除しますか？')) return;
    const { error } = await supabase.from('public_lists').delete().eq('id', id);
    if (error) { alert('削除に失敗しました: ' + error.message); return; }
    await reloadLists();
  }, [reloadLists]);

  return (
    <>
      {/* 編集モードバナー＋新規作成ボタン */}
      {isOwner && (
        <div className="mb-6 flex items-center justify-between gap-3 px-4 py-2.5 rounded-xl border border-violet-500/40 bg-violet-500/10">
          <span className="text-xs text-violet-300 font-semibold">編集モード</span>
          <button
            onClick={() => setShowCreateForm(true)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-violet-600 hover:bg-violet-500 text-white text-xs font-bold transition-colors"
          >
            <Plus size={13} />
            新しいリスト
          </button>
        </div>
      )}

      {/* 新規作成フォーム */}
      {isOwner && showCreateForm && (
        <div className="mb-6 p-4 rounded-xl border border-violet-500/40 bg-violet-500/10">
          <p className="text-sm font-bold text-[#e6edf3] mb-3">新しいリストを作成</p>
          <input
            type="text"
            value={newTitle}
            onChange={(e) => setNewTitle(e.target.value)}
            placeholder="リストのタイトル（例: NTR厳選リスト）"
            className="w-full bg-[#0d1117] border border-[#30363d] rounded-lg px-3 py-2 text-sm text-[#e6edf3] placeholder-[#484f58] outline-none focus:border-violet-500/50 mb-2"
          />
          <textarea
            value={newDesc}
            onChange={(e) => setNewDesc(e.target.value)}
            placeholder="説明（省略可）"
            rows={2}
            className="w-full bg-[#0d1117] border border-[#30363d] rounded-lg px-3 py-2 text-sm text-[#e6edf3] placeholder-[#484f58] outline-none focus:border-violet-500/50 resize-none mb-3"
          />
          <div className="flex gap-2">
            <button
              onClick={handleCreate}
              disabled={creating || !newTitle.trim()}
              className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-violet-600 hover:bg-violet-500 text-white text-xs font-bold transition-colors disabled:opacity-50"
            >
              {creating ? <Loader2 size={12} className="animate-spin" /> : <Plus size={12} />}
              作成
            </button>
            <button
              onClick={() => setShowCreateForm(false)}
              className="px-4 py-2 rounded-lg bg-white/5 text-[#8b949e] text-xs font-bold hover:bg-white/10 transition-colors"
            >
              キャンセル
            </button>
          </div>
        </div>
      )}

      {/* リスト一覧 */}
      {lists.length === 0 ? (
        <p className="text-center text-[#656d76] py-20">
          {isOwner ? 'まだリストがありません。「新しいリスト」から作成できます。' : 'まだ公開リストがありません。'}
        </p>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {lists.map((list) => {
            const thumbs = (list.thumbnails ?? []).slice(0, 3).map(toLgThumb).filter(Boolean) as string[];
            const label = list.list_type === 'liked' ? 'お気に入りリスト' : list.title ?? '無題のリスト';
            return (
              <div key={list.id} className="group relative">
                <Link
                  href={`/list/${list.token}`}
                  className="block rounded-xl border border-[#21262d] hover:border-violet-500/50 bg-[#161b22] overflow-hidden transition-all hover:shadow-lg hover:shadow-violet-500/10"
                >
                  {/* サムネイル */}
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

                {/* 編集ボタン（オーナーのみ、カード右上に重ねる） */}
                {isOwner && list.list_type === 'custom' && (
                  <div className="absolute top-2 right-2 flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button
                      onClick={() => setSearchTarget(list)}
                      className="flex items-center gap-1 px-2 py-1 rounded-lg bg-violet-600/90 backdrop-blur text-white text-[10px] font-bold hover:bg-violet-500 transition-colors shadow"
                    >
                      <Search size={10} />
                      動画追加
                    </button>
                    <button
                      onClick={() => handleDelete(list.id)}
                      className="flex items-center justify-center w-6 h-6 rounded-lg bg-red-500/80 backdrop-blur hover:bg-red-500 text-white transition-colors shadow"
                    >
                      <Trash2 size={11} />
                    </button>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* 動画追加モーダル */}
      {searchTarget && (
        <VideoSearchModal
          listId={searchTarget.id}
          onClose={() => setSearchTarget(null)}
          onAdded={reloadLists}
        />
      )}
    </>
  );
}
