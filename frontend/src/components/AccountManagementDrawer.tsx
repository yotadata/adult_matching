'use client';

import { Dialog, Transition } from '@headlessui/react';
import { Fragment, useEffect, useRef, useState, useCallback } from 'react';
import { X, Heart, Tag, Users, TrendingUp, Link2, Check, Loader2, ExternalLink, ListVideo, ShieldOff, Search, Plus } from 'lucide-react';
import { supabase } from '@/lib/supabase';

interface AccountManagementDrawerProps {
  isOpen: boolean;
  onClose: () => void;
}

type TagItem = { id: string; name: string; cnt: number };
type PerformerItem = { id: string; name: string; cnt: number };
type ExcludedTag = { tag_id: string; name: string };

const AccountManagementDrawer: React.FC<AccountManagementDrawerProps> = ({ isOpen, onClose }) => {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [username, setUsername] = useState<string | null>(null);
  const [likeCount, setLikeCount] = useState<number | null>(null);
  const [totalCount, setTotalCount] = useState<number | null>(null);
  const [tags, setTags] = useState<TagItem[]>([]);
  const [performers, setPerformers] = useState<PerformerItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [publicToken, setPublicToken] = useState<string | null>(null);
  const [tokenLoading, setTokenLoading] = useState(false);
  const [copied, setCopied] = useState(false);

  // 除外タグ
  const [excludedTags, setExcludedTags] = useState<ExcludedTag[]>([]);
  const [tagSearchQuery, setTagSearchQuery] = useState('');
  const [tagSearchResults, setTagSearchResults] = useState<{ id: string; name: string }[]>([]);
  const [tagSearchOpen, setTagSearchOpen] = useState(false);
  const tagSearchRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const checkLoginStatus = async () => {
      const { data: { session } } = await supabase.auth.getSession();
      setIsLoggedIn(!!session);
    };
    checkLoginStatus();

    const { data: authListener } = supabase.auth.onAuthStateChange((_event, session) => {
      setIsLoggedIn(!!session);
    });

    return () => { authListener.subscription.unsubscribe(); };
  }, []);

  const fetchInsights = useCallback(async () => {
    setLoading(true);
    const [
      { count: likes },
      { count: total },
      { data: tagData },
      { data: perfData },
    ] = await Promise.all([
      supabase.from('user_video_decisions').select('*', { count: 'exact', head: true }).in('decision_type', ['swipe_like', 'grid_like']),
      supabase.from('user_video_decisions').select('*', { count: 'exact', head: true }),
      supabase.rpc('get_user_liked_tags'),
      supabase.rpc('get_user_liked_performers'),
    ]);
    setLikeCount(likes ?? 0);
    setTotalCount(total ?? 0);
    setTags(((tagData as TagItem[] | null) ?? []).slice(0, 3));
    setPerformers(((perfData as PerformerItem[] | null) ?? []).slice(0, 3));
    setLoading(false);
  }, []);

  // 除外タグ読み込み
  useEffect(() => {
    if (isOpen && isLoggedIn) {
      supabase.rpc('get_user_excluded_tags').then(({ data }) => {
        if (data) setExcludedTags(data as ExcludedTag[]);
      });
    }
  }, [isOpen, isLoggedIn]);

  // タグ検索 debounce
  useEffect(() => {
    if (!tagSearchQuery.trim()) { setTagSearchResults([]); return; }
    const timer = setTimeout(async () => {
      const { data } = await supabase.from('tags').select('id, name').ilike('name', `%${tagSearchQuery}%`).limit(10);
      setTagSearchResults((data as { id: string; name: string }[]) ?? []);
    }, 300);
    return () => clearTimeout(timer);
  }, [tagSearchQuery]);

  // ドロップダウン外クリックで閉じる
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (tagSearchRef.current && !tagSearchRef.current.contains(e.target as Node)) setTagSearchOpen(false);
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  const handleAddExcludedTag = async (tag: { id: string; name: string }) => {
    if (excludedTags.some((t) => t.tag_id === tag.id)) return;
    await supabase.rpc('add_user_excluded_tag', { p_tag_id: tag.id });
    setExcludedTags((prev) => [...prev, { tag_id: tag.id, name: tag.name }]);
    setTagSearchQuery('');
    setTagSearchResults([]);
    setTagSearchOpen(false);
  };

  const handleRemoveExcludedTag = async (tagId: string) => {
    await supabase.rpc('remove_user_excluded_tag', { p_tag_id: tagId });
    setExcludedTags((prev) => prev.filter((t) => t.tag_id !== tagId));
  };

  useEffect(() => {
    if (isOpen && isLoggedIn) {
      fetchInsights();
      // 自分の公開トークンを取得（user_id フィルタ必須）
      supabase.auth.getUser().then(({ data: { user } }) => {
        if (!user) return;
        setUsername((user.user_metadata?.username as string) ?? null);
        supabase.from('public_lists').select('token')
          .eq('user_id', user.id).eq('is_active', true).eq('list_type', 'liked').maybeSingle()
          .then(({ data }) => { if (data) setPublicToken(data.token); });
      });
    }
  }, [isOpen, isLoggedIn, fetchInsights]);

  const handleCreatePublicList = async () => {
    setTokenLoading(true);
    try {
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) return;
      const { data: existing } = await supabase
        .from('public_lists').select('token')
        .eq('user_id', user.id).eq('is_active', true).eq('list_type', 'liked').maybeSingle();
      if (existing) {
        setPublicToken(existing.token);
      } else {
        const { data } = await supabase
          .from('public_lists').insert({ user_id: user.id, list_type: 'liked' }).select('token').single();
        if (data) setPublicToken(data.token);
      }
    } finally {
      setTokenLoading(false);
    }
  };

  const handleCopyLink = async () => {
    if (!publicToken) return;
    await navigator.clipboard.writeText(`${window.location.origin}/list/${publicToken}`);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDeletePublicList = async () => {
    if (!publicToken) return;
    await supabase.from('public_lists').delete().eq('token', publicToken);
    setPublicToken(null);
  };

  const handleLogout = async () => {
    try {
      const { error } = await supabase.auth.signOut({ scope: 'global' });
      if (error) console.error('signOut error:', error.message);
    } catch (e) {
      console.error('logout exception', e);
    }
    onClose();
    try { if (typeof window !== 'undefined') window.location.assign('/swipe'); } catch {}
  };

  const likeRate = likeCount !== null && totalCount ? Math.round((likeCount / totalCount) * 100) : null;

  return (
    <Transition appear show={isOpen} as={Fragment}>
      <Dialog as="div" className="relative z-50" onClose={onClose}>
        <Transition.Child
          as={Fragment}
          enter="ease-out duration-300" enterFrom="opacity-0" enterTo="opacity-100"
          leave="ease-in duration-200" leaveFrom="opacity-100" leaveTo="opacity-0"
        >
          <div className="fixed inset-0 bg-black/50 backdrop-blur-sm" />
        </Transition.Child>

        <div className="fixed inset-0 overflow-hidden">
          <div className="absolute inset-0 overflow-hidden">
            <div className="pointer-events-none fixed inset-y-0 right-0 flex max-w-full pl-10 sm:pl-16">
              <Transition.Child
                as={Fragment}
                enter="transform transition ease-in-out duration-500" enterFrom="translate-x-full" enterTo="translate-x-0"
                leave="transform transition ease-in-out duration-500" leaveFrom="translate-x-0" leaveTo="translate-x-full"
              >
                <Dialog.Panel className="pointer-events-auto w-screen max-w-sm">
                  <div className="flex h-full flex-col bg-[#0d1117] text-[#e6edf3] shadow-xl">
                    {/* ヘッダー */}
                    <div className="flex items-center justify-between px-5 py-4 border-b border-white/10">
                      <Dialog.Title className="text-base font-bold">設定</Dialog.Title>
                      <button onClick={onClose} className="text-[#8b949e] hover:text-white transition-colors">
                        <X size={18} />
                      </button>
                    </div>

                    <div className="flex-1 overflow-y-auto px-5 py-5 space-y-6">
                      {isLoggedIn && (
                        <>
                          {/* インサイト */}
                          <section>
                            <p className="text-[11px] text-[#8b949e] font-semibold uppercase tracking-wider mb-3">あなたの性癖</p>

                            {loading ? (
                              <div className="space-y-2">
                                {[1,2,3].map(i => <div key={i} className="h-12 rounded-xl bg-white/5 animate-pulse" />)}
                              </div>
                            ) : (
                              <div className="space-y-3">
                                {/* 統計 */}
                                <div className="grid grid-cols-2 gap-2">
                                  <div className="rounded-xl bg-white/5 border border-white/10 px-4 py-3">
                                    <div className="flex items-center gap-1.5 text-pink-400 mb-1">
                                      <Heart size={12} fill="currentColor" />
                                      <span className="text-[10px] font-semibold">累計いいね</span>
                                    </div>
                                    <div className="text-2xl font-extrabold text-white">{(likeCount ?? 0).toLocaleString('ja-JP')}</div>
                                  </div>
                                  <div className="rounded-xl bg-white/5 border border-white/10 px-4 py-3">
                                    <div className="flex items-center gap-1.5 text-violet-400 mb-1">
                                      <TrendingUp size={12} />
                                      <span className="text-[10px] font-semibold">いいね率</span>
                                    </div>
                                    <div className="text-2xl font-extrabold text-white">
                                      {likeRate !== null ? `${likeRate}%` : '—'}
                                    </div>
                                  </div>
                                </div>

                                {/* タグ */}
                                {tags.length > 0 && (
                                  <div className="rounded-xl bg-white/5 border border-white/10 px-4 py-3">
                                    <div className="flex items-center gap-1.5 text-[#8b949e] mb-2">
                                      <Tag size={11} />
                                      <span className="text-[10px] font-semibold">よく見るタグ</span>
                                    </div>
                                    <div className="flex flex-wrap gap-1.5">
                                      {tags.map(t => (
                                        <span key={t.id} className="px-2.5 py-1 rounded-full bg-white/10 text-white text-xs font-semibold">
                                          {t.name} <span className="text-[#8b949e] font-normal">{t.cnt}</span>
                                        </span>
                                      ))}
                                    </div>
                                  </div>
                                )}

                                {/* 女優 */}
                                {performers.length > 0 && (
                                  <div className="rounded-xl bg-white/5 border border-white/10 px-4 py-3">
                                    <div className="flex items-center gap-1.5 text-[#8b949e] mb-2">
                                      <Users size={11} />
                                      <span className="text-[10px] font-semibold">お気に入り女優</span>
                                    </div>
                                    <div className="flex flex-wrap gap-1.5">
                                      {performers.map(p => (
                                        <span key={p.id} className="px-2.5 py-1 rounded-full bg-pink-500/20 text-pink-300 text-xs font-semibold">
                                          {p.name} <span className="text-pink-400/60 font-normal">{p.cnt}</span>
                                        </span>
                                      ))}
                                    </div>
                                  </div>
                                )}
                              </div>
                            )}
                          </section>

                          <div className="border-t border-white/10" />
                        </>
                      )}

                      {/* マイリスト */}
                      {isLoggedIn && (
                        <>
                          <div className="border-t border-white/10" />
                          <section className="space-y-3">
                            <p className="text-[11px] text-[#8b949e] font-semibold uppercase tracking-wider">マイリスト</p>

                            {/* マイページへ */}
                            {username ? (
                              <a
                                href={`/u/${username}`}
                                className="w-full flex items-center gap-2 py-2.5 px-4 rounded-xl bg-[#161b22] border border-[#30363d] hover:border-violet-500/50 text-[#8b949e] hover:text-[#e6edf3] text-sm font-bold transition-all"
                              >
                                <ListVideo size={14} />
                                マイページを開く
                              </a>
                            ) : (
                              <a
                                href="/my/lists"
                                className="w-full flex items-center gap-2 py-2.5 px-4 rounded-xl bg-[#161b22] border border-[#30363d] hover:border-violet-500/50 text-[#8b949e] hover:text-[#e6edf3] text-sm font-bold transition-all"
                              >
                                <ListVideo size={14} />
                                マイリストを開く
                              </a>
                            )}

                            {/* いいねリストのURL共有（サブ機能） */}
                            <div>
                              <p className="text-[10px] text-[#484f58] mb-2">いいねリストを URL で共有</p>
                              {publicToken ? (
                                <div className="space-y-1.5">
                                  <div className="flex items-center gap-2 p-2 rounded-lg bg-[#161b22] border border-[#30363d] text-xs text-[#8b949e] truncate">
                                    <Link2 size={11} className="shrink-0" />
                                    <span className="truncate">{typeof window !== 'undefined' ? window.location.host : 'seihekilab.com'}/list/{publicToken}</span>
                                  </div>
                                  <div className="flex gap-2">
                                    <button
                                      onClick={handleCopyLink}
                                      className="flex-1 flex items-center justify-center gap-1.5 py-1.5 rounded-lg bg-violet-600 hover:bg-violet-500 text-white text-xs font-bold transition-colors"
                                    >
                                      {copied ? <><Check size={11} />コピー済み</> : <><Link2 size={11} />URLをコピー</>}
                                    </button>
                                    <button
                                      onClick={handleDeletePublicList}
                                      className="px-3 py-1.5 rounded-lg bg-red-500/20 text-red-400 border border-red-500/30 text-xs font-bold hover:bg-red-500/30 transition-colors"
                                    >
                                      削除
                                    </button>
                                  </div>
                                </div>
                              ) : (
                                <button
                                  onClick={handleCreatePublicList}
                                  disabled={tokenLoading}
                                  className="w-full flex items-center justify-center gap-2 py-2 rounded-lg bg-[#161b22] border border-[#30363d] hover:border-violet-500/50 text-[#8b949e] hover:text-[#e6edf3] text-xs font-bold transition-all disabled:opacity-50"
                                >
                                  {tokenLoading
                                    ? <><Loader2 size={12} className="animate-spin" />作成中…</>
                                    : <><Link2 size={12} />共有URLを作成</>}
                                </button>
                              )}
                            </div>
                          </section>
                        </>
                      )}

                      {/* 除外タグ設定 */}
                      {isLoggedIn && (
                        <>
                          <div className="border-t border-white/10" />
                          <section className="space-y-3">
                            <div className="flex items-center gap-1.5">
                              <ShieldOff size={11} className="text-[#8b949e]" />
                              <p className="text-[11px] text-[#8b949e] font-semibold uppercase tracking-wider">除外タグ</p>
                            </div>
                            <p className="text-[10px] text-[#484f58]">設定したタグの動画はフィードに表示されません</p>

                            {/* 除外中タグ一覧 */}
                            {excludedTags.length > 0 && (
                              <div className="flex flex-wrap gap-1.5">
                                {excludedTags.map((tag) => (
                                  <span key={tag.tag_id} className="inline-flex items-center gap-1 rounded-full bg-red-500/15 border border-red-500/30 px-2.5 py-0.5 text-xs text-red-400">
                                    {tag.name}
                                    <button type="button" onClick={() => handleRemoveExcludedTag(tag.tag_id)} className="ml-0.5 hover:text-red-200">
                                      <X size={10} />
                                    </button>
                                  </span>
                                ))}
                              </div>
                            )}

                            {/* タグ検索 */}
                            <div ref={tagSearchRef} className="relative">
                              <div className="flex items-center gap-2 rounded-lg bg-[#161b22] border border-[#30363d] px-3 py-2 focus-within:border-violet-500/50">
                                <Search size={12} className="text-[#484f58] shrink-0" />
                                <input
                                  type="text"
                                  value={tagSearchQuery}
                                  onChange={(e) => { setTagSearchQuery(e.target.value); setTagSearchOpen(true); }}
                                  onFocus={() => setTagSearchOpen(true)}
                                  className="flex-1 bg-transparent text-xs text-[#e6edf3] placeholder:text-[#484f58] focus:outline-none"
                                  placeholder="タグ名で検索して追加..."
                                />
                              </div>
                              {tagSearchOpen && tagSearchResults.length > 0 && (
                                <ul className="absolute z-10 mt-1 w-full rounded-lg border border-[#30363d] bg-[#161b22] shadow-xl overflow-hidden">
                                  {tagSearchResults.map((tag) => {
                                    const already = excludedTags.some((t) => t.tag_id === tag.id);
                                    return (
                                      <li key={tag.id}>
                                        <button
                                          type="button"
                                          onClick={() => !already && handleAddExcludedTag(tag)}
                                          disabled={already}
                                          className={`w-full flex items-center justify-between px-3 py-2 text-xs text-left transition ${already ? 'text-[#484f58] cursor-default' : 'text-[#e6edf3] hover:bg-white/5'}`}
                                        >
                                          <span>{tag.name}</span>
                                          {already ? <span className="text-[#484f58]">追加済み</span> : <Plus size={12} className="text-violet-400" />}
                                        </button>
                                      </li>
                                    );
                                  })}
                                </ul>
                              )}
                            </div>
                          </section>
                        </>
                      )}

                      {/* アカウント操作 */}
                      <section className="space-y-2">
                        <p className="text-[11px] text-[#8b949e] font-semibold uppercase tracking-wider mb-3">アカウント</p>
                        {isLoggedIn ? (
                          <button
                            onClick={handleLogout}
                            className="w-full py-2.5 px-4 rounded-xl bg-red-500/20 text-red-400 border border-red-500/30 font-bold text-sm hover:bg-red-500/30 transition-colors"
                          >
                            ログアウト
                          </button>
                        ) : (
                          <button
                            onClick={() => { window.dispatchEvent(new Event('open-auth-modal')); onClose(); }}
                            className="w-full py-2.5 px-4 rounded-xl bg-violet-600 hover:bg-violet-500 text-white font-bold text-sm transition-colors"
                          >
                            ログイン
                          </button>
                        )}
                      </section>
                    </div>
                  </div>
                </Dialog.Panel>
              </Transition.Child>
            </div>
          </div>
        </div>
      </Dialog>
    </Transition>
  );
};

export default AccountManagementDrawer;
