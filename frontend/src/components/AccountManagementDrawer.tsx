'use client';

import { Dialog, Transition } from '@headlessui/react';
import { Fragment, useEffect, useRef, useState, useCallback } from 'react';
import {
  X, Heart, Tag, Users, TrendingUp, Link2, Check, Loader2, ListVideo,
  ShieldOff, Search, Plus, ChevronDown, ChevronUp, Edit3, Key, Trash2, User,
} from 'lucide-react';
import Link from 'next/link';
import toast from 'react-hot-toast';
import { supabase } from '@/lib/supabase';

interface AccountManagementDrawerProps {
  isOpen: boolean;
  onClose: () => void;
}

type TagItem = { id: string; name: string; cnt: number };
type PerformerItem = { id: string; name: string; cnt: number };
type ExcludedTag = { tag_id: string; name: string };
type CuratorProfile = { bio: string; x_url: string; affiliate_fanza_id: string; affiliate_fc2_id: string; affiliate_mgs_id: string };

const AccountManagementDrawer: React.FC<AccountManagementDrawerProps> = ({ isOpen, onClose }) => {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [username, setUsername] = useState<string | null>(null);
  const [displayName, setDisplayName] = useState<string>('');
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

  // アカウント編集
  const [accountOpen, setAccountOpen] = useState(false);
  const [editSection, setEditSection] = useState<'displayName' | 'password' | 'curator' | null>(null);
  const [newDisplayName, setNewDisplayName] = useState('');
  const [passwords, setPasswords] = useState({ next: '', confirm: '' });
  const [accountSaving, setAccountSaving] = useState(false);
  const [curatorProfile, setCuratorProfile] = useState<CuratorProfile>({ bio: '', x_url: '', affiliate_fanza_id: '', affiliate_fc2_id: '', affiliate_mgs_id: '' });
  const [curatorSaving, setCuratorSaving] = useState(false);

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

  useEffect(() => {
    if (isOpen && isLoggedIn) {
      fetchInsights();
      supabase.auth.getUser().then(({ data: { user } }) => {
        if (!user) return;
        const uname = (user.user_metadata?.username as string) ?? null;
        const dname = (user.user_metadata?.display_name as string) || (user.user_metadata?.displayName as string) || '';
        setUsername(uname);
        setDisplayName(dname);
        supabase.from('public_lists').select('token')
          .eq('user_id', user.id).eq('is_active', true).eq('list_type', 'liked').maybeSingle()
          .then(({ data }) => { if (data) setPublicToken(data.token); });
        supabase.from('user_profiles')
          .select('bio, x_url, affiliate_fanza_id, affiliate_fc2_id, affiliate_mgs_id')
          .eq('user_id', user.id).maybeSingle()
          .then(({ data: up }) => {
            if (up) setCuratorProfile({
              bio: up.bio ?? '', x_url: up.x_url ?? '',
              affiliate_fanza_id: up.affiliate_fanza_id ?? '',
              affiliate_fc2_id: up.affiliate_fc2_id ?? '',
              affiliate_mgs_id: up.affiliate_mgs_id ?? '',
            });
          });
      });
      supabase.rpc('get_user_excluded_tags').then(({ data }) => {
        if (data) setExcludedTags(data as ExcludedTag[]);
      });
    }
  }, [isOpen, isLoggedIn, fetchInsights]);

  // タグ検索 debounce
  useEffect(() => {
    if (!tagSearchQuery.trim()) { setTagSearchResults([]); return; }
    const timer = setTimeout(async () => {
      const { data } = await supabase.from('tags').select('id, name').ilike('name', `%${tagSearchQuery}%`).limit(10);
      setTagSearchResults((data as { id: string; name: string }[]) ?? []);
    }, 300);
    return () => clearTimeout(timer);
  }, [tagSearchQuery]);

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

  const handleCreatePublicList = async () => {
    setTokenLoading(true);
    try {
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) return;
      const { data: existing } = await supabase
        .from('public_lists').select('token')
        .eq('user_id', user.id).eq('is_active', true).eq('list_type', 'liked').maybeSingle();
      if (existing) { setPublicToken(existing.token); }
      else {
        const { data } = await supabase
          .from('public_lists').insert({ user_id: user.id, list_type: 'liked' }).select('token').single();
        if (data) setPublicToken(data.token);
      }
    } finally { setTokenLoading(false); }
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

  const handleUpdateDisplayName = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newDisplayName.trim()) return;
    setAccountSaving(true);
    try {
      const { error } = await supabase.auth.updateUser({ data: { display_name: newDisplayName.trim() } });
      if (error) throw error;
      setDisplayName(newDisplayName.trim());
      setEditSection(null);
      toast.success('表示名を更新しました');
    } catch (err) {
      toast.error(err instanceof Error ? err.message : '更新に失敗しました');
    } finally { setAccountSaving(false); }
  };

  const handleUpdatePassword = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!passwords.next || passwords.next.length < 8) { toast.error('パスワードは8文字以上で入力してください'); return; }
    if (passwords.next !== passwords.confirm) { toast.error('確認用パスワードが一致しません'); return; }
    setAccountSaving(true);
    try {
      const { error } = await supabase.auth.updateUser({ password: passwords.next });
      if (error) throw error;
      setPasswords({ next: '', confirm: '' });
      setEditSection(null);
      toast.success('パスワードを更新しました');
    } catch (err) {
      toast.error(err instanceof Error ? err.message : '更新に失敗しました');
    } finally { setAccountSaving(false); }
  };

  const handleCuratorSave = async (e: React.FormEvent) => {
    e.preventDefault();
    setCuratorSaving(true);
    try {
      const { error } = await supabase.rpc('upsert_user_profile', {
        p_bio: curatorProfile.bio || null,
        p_x_url: curatorProfile.x_url || null,
        p_affiliate_fanza_id: curatorProfile.affiliate_fanza_id || null,
        p_affiliate_fc2_id: curatorProfile.affiliate_fc2_id || null,
        p_affiliate_mgs_id: curatorProfile.affiliate_mgs_id || null,
      });
      if (error) throw error;
      setEditSection(null);
      toast.success('プロフィールを保存しました');
    } catch (err) {
      toast.error(err instanceof Error ? err.message : '保存に失敗しました');
    } finally { setCuratorSaving(false); }
  };

  const handleLogout = async () => {
    try {
      const { error } = await supabase.auth.signOut({ scope: 'global' });
      if (error) console.error('signOut error:', error.message);
    } catch (e) { console.error('logout exception', e); }
    onClose();
    try { if (typeof window !== 'undefined') window.location.assign('/swipe'); } catch {}
  };

  const likeRate = likeCount !== null && totalCount ? Math.round((likeCount / totalCount) * 100) : null;

  const inputCls = 'w-full bg-[#0d1117] border border-[#30363d] rounded-lg px-3 py-2 text-xs text-[#e6edf3] placeholder:text-[#484f58] focus:outline-none focus:border-violet-500/50';
  const labelCls = 'block text-[10px] text-[#8b949e] font-semibold mb-1';

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

                          {/* マイリスト */}
                          <section className="space-y-3">
                            <p className="text-[11px] text-[#8b949e] font-semibold uppercase tracking-wider">マイリスト</p>
                            {username ? (
                              <a href={`/u/${username}`} className="w-full flex items-center gap-2 py-2.5 px-4 rounded-xl bg-[#161b22] border border-[#30363d] hover:border-violet-500/50 text-[#8b949e] hover:text-[#e6edf3] text-sm font-bold transition-all">
                                <ListVideo size={14} />マイページを開く
                              </a>
                            ) : (
                              <a href="/my/lists" className="w-full flex items-center gap-2 py-2.5 px-4 rounded-xl bg-[#161b22] border border-[#30363d] hover:border-violet-500/50 text-[#8b949e] hover:text-[#e6edf3] text-sm font-bold transition-all">
                                <ListVideo size={14} />マイリストを開く
                              </a>
                            )}
                            <div>
                              <p className="text-[10px] text-[#484f58] mb-2">いいねリストを URL で共有</p>
                              {publicToken ? (
                                <div className="space-y-1.5">
                                  <div className="flex items-center gap-2 p-2 rounded-lg bg-[#161b22] border border-[#30363d] text-xs text-[#8b949e] truncate">
                                    <Link2 size={11} className="shrink-0" />
                                    <span className="truncate">{typeof window !== 'undefined' ? window.location.host : 'seihekilab.com'}/list/{publicToken}</span>
                                  </div>
                                  <div className="flex gap-2">
                                    <button onClick={handleCopyLink} className="flex-1 flex items-center justify-center gap-1.5 py-1.5 rounded-lg bg-violet-600 hover:bg-violet-500 text-white text-xs font-bold transition-colors">
                                      {copied ? <><Check size={11} />コピー済み</> : <><Link2 size={11} />URLをコピー</>}
                                    </button>
                                    <button onClick={handleDeletePublicList} className="px-3 py-1.5 rounded-lg bg-red-500/20 text-red-400 border border-red-500/30 text-xs font-bold hover:bg-red-500/30 transition-colors">削除</button>
                                  </div>
                                </div>
                              ) : (
                                <button onClick={handleCreatePublicList} disabled={tokenLoading} className="w-full flex items-center justify-center gap-2 py-2 rounded-lg bg-[#161b22] border border-[#30363d] hover:border-violet-500/50 text-[#8b949e] hover:text-[#e6edf3] text-xs font-bold transition-all disabled:opacity-50">
                                  {tokenLoading ? <><Loader2 size={12} className="animate-spin" />作成中…</> : <><Link2 size={12} />共有URLを作成</>}
                                </button>
                              )}
                            </div>
                          </section>

                          <div className="border-t border-white/10" />

                          {/* 除外タグ */}
                          <section className="space-y-3">
                            <div className="flex items-center gap-1.5">
                              <ShieldOff size={11} className="text-[#8b949e]" />
                              <p className="text-[11px] text-[#8b949e] font-semibold uppercase tracking-wider">除外タグ</p>
                            </div>
                            <p className="text-[10px] text-[#484f58]">設定したタグの動画はフィードに表示されません</p>
                            {excludedTags.length > 0 && (
                              <div className="flex flex-wrap gap-1.5">
                                {excludedTags.map((tag) => (
                                  <span key={tag.tag_id} className="inline-flex items-center gap-1 rounded-full bg-red-500/15 border border-red-500/30 px-2.5 py-0.5 text-xs text-red-400">
                                    {tag.name}
                                    <button type="button" onClick={() => handleRemoveExcludedTag(tag.tag_id)} className="ml-0.5 hover:text-red-200"><X size={10} /></button>
                                  </span>
                                ))}
                              </div>
                            )}
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

                          <div className="border-t border-white/10" />

                          {/* アカウント */}
                          <section className="space-y-3">
                            <button
                              type="button"
                              onClick={() => { setAccountOpen((v) => !v); setEditSection(null); }}
                              className="w-full flex items-center justify-between px-4 py-3 rounded-xl bg-[#161b22] border border-[#30363d] hover:border-violet-500/50 transition-colors"
                            >
                              <div className="flex items-center gap-2">
                                <User size={14} className="text-[#8b949e]" />
                                <span className="text-sm font-bold text-[#e6edf3]">アカウント設定</span>
                                {displayName && <span className="text-xs text-[#8b949e] truncate max-w-[120px]">{displayName}</span>}
                              </div>
                              {accountOpen ? <ChevronUp size={14} className="text-[#8b949e] shrink-0" /> : <ChevronDown size={14} className="text-[#8b949e] shrink-0" />}
                            </button>

                            {accountOpen && (
                              <div className="space-y-3">
                                {/* 基本情報 */}
                                <div className="rounded-xl bg-[#161b22] border border-[#30363d] px-4 py-3 text-xs space-y-1.5">
                                  <div className="flex justify-between text-[#8b949e]">
                                    <span>ユーザーID</span>
                                    <span className="font-mono text-[#e6edf3]">{username || '未設定'}</span>
                                  </div>
                                  <div className="flex justify-between text-[#8b949e]">
                                    <span>表示名</span>
                                    <span className="text-[#e6edf3]">{displayName || '未設定'}</span>
                                  </div>
                                  {username && (
                                    <div className="flex justify-between text-[#8b949e]">
                                      <span>公開ページ</span>
                                      <Link href={`/u/${username}`} className="text-violet-400 hover:underline" onClick={onClose}>/u/{username}</Link>
                                    </div>
                                  )}
                                </div>

                                {/* 表示名変更 */}
                                <div className="rounded-xl bg-[#161b22] border border-[#30363d] overflow-hidden">
                                  <button
                                    type="button"
                                    onClick={() => setEditSection(editSection === 'displayName' ? null : 'displayName')}
                                    className="w-full flex items-center gap-2 px-4 py-2.5 text-xs text-[#8b949e] hover:text-[#e6edf3] transition-colors"
                                  >
                                    <Edit3 size={12} /><span className="font-semibold">表示名の変更</span>
                                    {editSection === 'displayName' ? <ChevronUp size={12} className="ml-auto" /> : <ChevronDown size={12} className="ml-auto" />}
                                  </button>
                                  {editSection === 'displayName' && (
                                    <form onSubmit={handleUpdateDisplayName} className="px-4 pb-3 space-y-2 border-t border-[#30363d]">
                                      <div className="pt-2">
                                        <label className={labelCls}>新しい表示名</label>
                                        <input type="text" value={newDisplayName} onChange={(e) => setNewDisplayName(e.target.value)} className={inputCls} placeholder="表示名" />
                                      </div>
                                      <button type="submit" disabled={accountSaving || !newDisplayName.trim()} className="w-full py-1.5 rounded-lg bg-violet-600 hover:bg-violet-500 text-white text-xs font-bold disabled:opacity-50 transition-colors">
                                        {accountSaving ? '更新中...' : '更新する'}
                                      </button>
                                    </form>
                                  )}
                                </div>

                                {/* パスワード変更 */}
                                <div className="rounded-xl bg-[#161b22] border border-[#30363d] overflow-hidden">
                                  <button
                                    type="button"
                                    onClick={() => setEditSection(editSection === 'password' ? null : 'password')}
                                    className="w-full flex items-center gap-2 px-4 py-2.5 text-xs text-[#8b949e] hover:text-[#e6edf3] transition-colors"
                                  >
                                    <Key size={12} /><span className="font-semibold">パスワードの変更</span>
                                    {editSection === 'password' ? <ChevronUp size={12} className="ml-auto" /> : <ChevronDown size={12} className="ml-auto" />}
                                  </button>
                                  {editSection === 'password' && (
                                    <form onSubmit={handleUpdatePassword} className="px-4 pb-3 space-y-2 border-t border-[#30363d]">
                                      <div className="pt-2">
                                        <label className={labelCls}>新しいパスワード（8文字以上）</label>
                                        <input type="password" value={passwords.next} onChange={(e) => setPasswords((p) => ({ ...p, next: e.target.value }))} className={inputCls} placeholder="新しいパスワード" />
                                      </div>
                                      <div>
                                        <label className={labelCls}>確認用パスワード</label>
                                        <input type="password" value={passwords.confirm} onChange={(e) => setPasswords((p) => ({ ...p, confirm: e.target.value }))} className={inputCls} placeholder="同じパスワードを入力" />
                                      </div>
                                      <button type="submit" disabled={accountSaving} className="w-full py-1.5 rounded-lg bg-violet-600 hover:bg-violet-500 text-white text-xs font-bold disabled:opacity-50 transition-colors">
                                        {accountSaving ? '更新中...' : '更新する'}
                                      </button>
                                    </form>
                                  )}
                                </div>

                                {/* キュレータープロフィール */}
                                <div className="rounded-xl bg-[#161b22] border border-[#30363d] overflow-hidden">
                                  <button
                                    type="button"
                                    onClick={() => setEditSection(editSection === 'curator' ? null : 'curator')}
                                    className="w-full flex items-center gap-2 px-4 py-2.5 text-xs text-[#8b949e] hover:text-[#e6edf3] transition-colors"
                                  >
                                    <User size={12} /><span className="font-semibold">プロフィール編集</span>
                                    {editSection === 'curator' ? <ChevronUp size={12} className="ml-auto" /> : <ChevronDown size={12} className="ml-auto" />}
                                  </button>
                                  {editSection === 'curator' && (
                                    <form onSubmit={handleCuratorSave} className="px-4 pb-3 space-y-2 border-t border-[#30363d]">
                                      <div className="pt-2">
                                        <label className={labelCls}>自己紹介（200字以内）</label>
                                        <textarea value={curatorProfile.bio} onChange={(e) => setCuratorProfile((p) => ({ ...p, bio: e.target.value }))} rows={3} maxLength={200} className={`${inputCls} resize-none`} placeholder="自己紹介を書いてください" />
                                      </div>
                                      <div>
                                        <label className={labelCls}>X（旧Twitter）URL</label>
                                        <input type="url" value={curatorProfile.x_url} onChange={(e) => setCuratorProfile((p) => ({ ...p, x_url: e.target.value }))} className={inputCls} placeholder="https://x.com/yourname" />
                                      </div>
                                      <div>
                                        <label className={labelCls}>FANZA アフィリエイトID</label>
                                        <input type="text" value={curatorProfile.affiliate_fanza_id} onChange={(e) => setCuratorProfile((p) => ({ ...p, affiliate_fanza_id: e.target.value }))} className={inputCls} placeholder="例: yourname-001" />
                                      </div>
                                      <div>
                                        <label className={labelCls}>FC2 アフィリエイトID</label>
                                        <input type="text" value={curatorProfile.affiliate_fc2_id} onChange={(e) => setCuratorProfile((p) => ({ ...p, affiliate_fc2_id: e.target.value }))} className={inputCls} placeholder="FC2アフィリエイトID" />
                                      </div>
                                      <div>
                                        <label className={labelCls}>MGS アフィリエイトID</label>
                                        <input type="text" value={curatorProfile.affiliate_mgs_id} onChange={(e) => setCuratorProfile((p) => ({ ...p, affiliate_mgs_id: e.target.value }))} className={inputCls} placeholder="MGSアフィリエイトID" />
                                      </div>
                                      <button type="submit" disabled={curatorSaving} className="w-full py-1.5 rounded-lg bg-violet-600 hover:bg-violet-500 text-white text-xs font-bold disabled:opacity-50 transition-colors">
                                        {curatorSaving ? '保存中...' : '保存する'}
                                      </button>
                                    </form>
                                  )}
                                </div>

                                {/* ログアウト・削除 */}
                                <button onClick={handleLogout} className="w-full py-2.5 px-4 rounded-xl bg-red-500/20 text-red-400 border border-red-500/30 font-bold text-sm hover:bg-red-500/30 transition-colors">
                                  ログアウト
                                </button>
                                <div className="flex items-center gap-1.5 justify-center">
                                  <Trash2 size={11} className="text-[#484f58]" />
                                  <Link href="/contact" onClick={onClose} className="text-[10px] text-[#484f58] hover:text-[#8b949e] underline">
                                    アカウント削除はこちら
                                  </Link>
                                </div>
                              </div>
                            )}

                            {/* アカウント折りたたみ時はログアウトのみ表示 */}
                            {!accountOpen && (
                              <button onClick={handleLogout} className="w-full py-2.5 px-4 rounded-xl bg-red-500/20 text-red-400 border border-red-500/30 font-bold text-sm hover:bg-red-500/30 transition-colors">
                                ログアウト
                              </button>
                            )}
                          </section>
                        </>
                      )}

                      {!isLoggedIn && (
                        <section>
                          <button
                            onClick={() => { window.dispatchEvent(new Event('open-auth-modal')); onClose(); }}
                            className="w-full py-2.5 px-4 rounded-xl bg-violet-600 hover:bg-violet-500 text-white font-bold text-sm transition-colors"
                          >
                            ログイン / 新規登録
                          </button>
                        </section>
                      )}
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
