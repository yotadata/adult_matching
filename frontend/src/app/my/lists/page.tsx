'use client';

import { useEffect, useState, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { Plus, ExternalLink, Loader2, Search, Trash2, ListVideo } from 'lucide-react';
import { supabase } from '@/lib/supabase';
import VideoSearchModal from '@/components/lists/VideoSearchModal';
import ListVideosModal from '@/components/lists/ListVideosModal';

type PublicList = {
  id: string;
  token: string;
  title: string | null;
  description: string | null;
  list_type: 'liked' | 'custom';
  is_active: boolean;
  created_at: string;
};

export default function MyListsPage() {
  const router = useRouter();
  const [lists, setLists] = useState<PublicList[]>([]);
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [newTitle, setNewTitle] = useState('');
  const [newDesc, setNewDesc] = useState('');
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [searchTarget, setSearchTarget] = useState<PublicList | null>(null);
  const [manageTarget, setManageTarget] = useState<PublicList | null>(null);

  const fetchLists = useCallback(async () => {
    const { data: { user } } = await supabase.auth.getUser();
    if (!user) { router.push('/'); return; }
    const { data } = await supabase
      .from('public_lists')
      .select('*')
      .eq('user_id', user.id)
      .eq('is_active', true)
      .order('created_at', { ascending: false });
    setLists((data as PublicList[]) ?? []);
    setLoading(false);
  }, [router]);

  useEffect(() => { fetchLists(); }, [fetchLists]);

  const handleCreate = async () => {
    if (!newTitle.trim()) return;
    setCreating(true);
    const { data: { user } } = await supabase.auth.getUser();
    if (!user) return;
    await supabase.from('public_lists').insert({
      user_id: user.id,
      title: newTitle.trim(),
      description: newDesc.trim() || null,
      list_type: 'custom',
    });
    setNewTitle('');
    setNewDesc('');
    setShowCreateForm(false);
    setCreating(false);
    await fetchLists();
  };

  const handleDelete = async (id: string) => {
    if (!confirm('このリストを削除しますか？')) return;
    const { error } = await supabase
      .from('public_lists')
      .delete()
      .eq('id', id);
    if (error) {
      alert('削除に失敗しました: ' + error.message);
      return;
    }
    await fetchLists();
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-[#0d1117] flex items-center justify-center">
        <Loader2 className="text-violet-400 animate-spin" size={24} />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#0d1117] text-[#e6edf3]">
      <div className="max-w-2xl mx-auto px-4 py-10">

        {/* ナビ */}
        <Link href="/grid" className="text-xs text-[#656d76] hover:text-[#8b949e] transition-colors mb-6 inline-block">
          ← 性癖ラボへ戻る
        </Link>

        {/* ヘッダー */}
        <div className="flex items-center justify-between mb-6">
          <h1 className="text-xl font-black text-[#e6edf3]">マイリスト</h1>
          <button
            onClick={() => setShowCreateForm(true)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-violet-600 hover:bg-violet-500 text-white text-xs font-bold transition-colors"
          >
            <Plus size={14} />
            新しいリスト
          </button>
        </div>

        {/* 新規作成フォーム */}
        {showCreateForm && (
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
          <p className="text-center text-[#656d76] py-20 text-sm">まだリストがありません</p>
        ) : (
          <div className="space-y-3">
            {lists.map((list) => (
              <div
                key={list.id}
                className="flex items-center gap-3 p-4 rounded-xl border border-[#21262d] bg-[#161b22] hover:border-[#30363d] transition-colors"
              >
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-bold text-[#e6edf3] line-clamp-1">
                    {list.list_type === 'liked' ? 'お気に入りリスト' : (list.title ?? '無題のリスト')}
                  </p>
                  {list.description && (
                    <p className="text-xs text-[#656d76] mt-0.5 line-clamp-1">{list.description}</p>
                  )}
                  <span className="inline-block mt-1 text-[10px] px-2 py-0.5 rounded-full bg-white/5 text-[#484f58]">
                    {list.list_type === 'liked' ? 'いいねリスト' : 'カスタム'}
                  </span>
                </div>

                <div className="flex items-center gap-1.5 shrink-0">
                  {list.list_type === 'custom' && (
                    <>
                      <button
                        onClick={() => setSearchTarget(list)}
                        className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg bg-violet-600/20 hover:bg-violet-600/40 text-violet-400 text-xs font-bold transition-colors border border-violet-500/30"
                      >
                        <Search size={11} />
                        動画追加
                      </button>
                      <button
                        onClick={() => setManageTarget(list)}
                        className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 text-[#8b949e] text-xs font-bold transition-colors border border-[#30363d]"
                      >
                        <ListVideo size={11} />
                        動画管理
                      </button>
                    </>
                  )}
                  <a
                    href={`/list/${list.token}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center justify-center w-8 h-8 rounded-lg bg-white/5 hover:bg-white/10 text-[#8b949e] transition-colors"
                  >
                    <ExternalLink size={14} />
                  </a>
                  {list.list_type === 'custom' && (
                    <button
                      onClick={() => handleDelete(list.id)}
                      className="flex items-center justify-center w-8 h-8 rounded-lg bg-red-500/10 hover:bg-red-500/20 text-red-400 transition-colors"
                    >
                      <Trash2 size={14} />
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* 動画検索モーダル */}
      {searchTarget && (
        <VideoSearchModal
          listId={searchTarget.id}
          onClose={() => setSearchTarget(null)}
          onAdded={fetchLists}
        />
      )}

      {/* 動画管理モーダル */}
      {manageTarget && (
        <ListVideosModal
          listId={manageTarget.id}
          listTitle={manageTarget.title ?? '無題のリスト'}
          onClose={() => setManageTarget(null)}
          onChanged={fetchLists}
        />
      )}
    </div>
  );
}
