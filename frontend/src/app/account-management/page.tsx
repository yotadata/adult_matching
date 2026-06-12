'use client';

import { Fragment, useEffect, useMemo, useRef, useState } from 'react';
import { Edit3, Key, LogOut, Trash2, X, User, ShieldOff, Search, Plus } from 'lucide-react';
import { Dialog, Transition } from '@headlessui/react';
import { supabase } from '@/lib/supabase';
import { useRouter } from 'next/navigation';
import toast from 'react-hot-toast';
import Link from 'next/link';

type CardKey = 'displayName' | 'password' | 'logout' | 'delete';

const cards: Array<{ key: CardKey; title: string; icon: React.ComponentType<{ size?: number }>; description: string }> = [
  { key: 'displayName', title: '表示名の変更', icon: Edit3, description: 'プロフィールに表示される名前を更新します。' },
  { key: 'password', title: 'パスワードの変更', icon: Key, description: '現在のパスワードと新しいパスワードを入力して更新します。' },
];

type Tag = { id: string; name: string };

type CuratorProfile = {
  bio: string;
  x_url: string;
  affiliate_fanza_id: string;
  affiliate_fc2_id: string;
  affiliate_mgs_id: string;
};

export default function AccountManagementPage() {
  const [activeModal, setActiveModal] = useState<CardKey | null>(null);
  const [displayName, setDisplayName] = useState('');
  const [passwords, setPasswords] = useState({ current: '', next: '', confirm: '' });
  const [loading, setLoading] = useState(false);
  const [userProfile, setUserProfile] = useState<{ displayName: string; username: string; userId: string } | null>(null);
  const [curatorProfile, setCuratorProfile] = useState<CuratorProfile>({
    bio: '', x_url: '', affiliate_fanza_id: '', affiliate_fc2_id: '', affiliate_mgs_id: '',
  });
  const [curatorSaving, setCuratorSaving] = useState(false);

  // 除外タグ
  const [excludedTags, setExcludedTags] = useState<Tag[]>([]);
  const [tagSearchQuery, setTagSearchQuery] = useState('');
  const [tagSearchResults, setTagSearchResults] = useState<Tag[]>([]);
  const [tagSearchOpen, setTagSearchOpen] = useState(false);
  const tagSearchRef = useRef<HTMLDivElement>(null);

  const pseudoDomain = useMemo(() => process.env.NEXT_PUBLIC_PSEUDO_EMAIL_DOMAIN || 'anon.seihekilab.com', []);
  const router = useRouter();

  useEffect(() => {
    const loadUser = async () => {
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) { setUserProfile(null); return; }

      const username = (() => {
        const metaUsername = (user.user_metadata?.username as string) || '';
        if (metaUsername) return metaUsername;
        const email = user.email ?? '';
        const suffix = `@${pseudoDomain}`;
        if (email && email.endsWith(suffix)) return email.slice(0, -suffix.length);
        return email || '';
      })();

      setUserProfile({
        displayName: (user.user_metadata?.display_name as string) || user.user_metadata?.displayName || '',
        username,
        userId: user.id,
      });

      const { data: up } = await supabase
        .from('user_profiles')
        .select('bio, x_url, affiliate_fanza_id, affiliate_fc2_id, affiliate_mgs_id')
        .eq('user_id', user.id)
        .maybeSingle();

      if (up) {
        setCuratorProfile({
          bio: up.bio ?? '',
          x_url: up.x_url ?? '',
          affiliate_fanza_id: up.affiliate_fanza_id ?? '',
          affiliate_fc2_id: up.affiliate_fc2_id ?? '',
          affiliate_mgs_id: up.affiliate_mgs_id ?? '',
        });
      }
    };
    loadUser();
  }, [pseudoDomain]);

  // 除外タグ読み込み
  useEffect(() => {
    supabase.rpc('get_user_excluded_tags').then(({ data }) => {
      if (data) setExcludedTags(data as Tag[]);
    });
  }, []);

  // タグ検索（300ms debounce）
  useEffect(() => {
    if (!tagSearchQuery.trim()) { setTagSearchResults([]); return; }
    const timer = setTimeout(async () => {
      const { data } = await supabase
        .from('tags')
        .select('id, name')
        .ilike('name', `%${tagSearchQuery}%`)
        .limit(10);
      setTagSearchResults((data as Tag[]) ?? []);
    }, 300);
    return () => clearTimeout(timer);
  }, [tagSearchQuery]);

  // 検索ドロップダウン外クリックで閉じる
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (tagSearchRef.current && !tagSearchRef.current.contains(e.target as Node)) {
        setTagSearchOpen(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  const handleAddExcludedTag = async (tag: Tag) => {
    if (excludedTags.some((t) => t.id === tag.id)) return;
    const { error } = await supabase.rpc('add_user_excluded_tag', { p_tag_id: tag.id });
    if (error) { toast.error('追加に失敗しました'); return; }
    setExcludedTags((prev) => [...prev, tag]);
    setTagSearchQuery('');
    setTagSearchResults([]);
    setTagSearchOpen(false);
    toast.success(`「${tag.name}」を除外タグに追加しました`);
  };

  const handleRemoveExcludedTag = async (tag: Tag) => {
    const { error } = await supabase.rpc('remove_user_excluded_tag', { p_tag_id: tag.id });
    if (error) { toast.error('削除に失敗しました'); return; }
    setExcludedTags((prev) => prev.filter((t) => t.id !== tag.id));
    toast.success(`「${tag.name}」を除外タグから削除しました`);
  };

  const closeModal = () => {
    setActiveModal(null);
    setDisplayName('');
    setPasswords({ current: '', next: '', confirm: '' });
    setLoading(false);
  };

  const handleDisplayNameUpdate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!displayName.trim()) { toast.error('表示名を入力してください'); return; }
    setLoading(true);
    try {
      const { error } = await supabase.auth.updateUser({ data: { display_name: displayName.trim() } });
      if (error) throw error;
      toast.success('表示名を更新しました');
      setUserProfile((prev) => (prev ? { ...prev, displayName: displayName.trim() } : prev));
      closeModal();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : '更新に失敗しました');
    } finally {
      setLoading(false);
    }
  };

  const handlePasswordUpdate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!passwords.next || passwords.next.length < 8) { toast.error('パスワードは8文字以上で入力してください'); return; }
    if (passwords.next !== passwords.confirm) { toast.error('確認用パスワードが一致しません'); return; }
    setLoading(true);
    try {
      const { error } = await supabase.auth.updateUser({ password: passwords.next });
      if (error) throw error;
      toast.success('パスワードを更新しました');
      closeModal();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : '更新に失敗しました');
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = async () => {
    setLoading(true);
    try {
      await supabase.auth.signOut({ scope: 'global' });
      toast.success('ログアウトしました');
      closeModal();
      router.push('/swipe');
      router.refresh();
    } catch {
      toast.error('ログアウトに失敗しました');
      setLoading(false);
    }
  };

  const handleCuratorProfileSave = async (e: React.FormEvent) => {
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
      toast.success('プロフィールを保存しました');
    } catch (err) {
      toast.error(err instanceof Error ? err.message : '保存に失敗しました');
    } finally {
      setCuratorSaving(false);
    }
  };

  const handleOpenCard = (key: CardKey) => {
    if (key === 'displayName') setDisplayName(userProfile?.displayName ?? '');
    setActiveModal(key);
  };

  return (
    <main className="w-full min-h-screen px-0 sm:px-4 py-8 text-white">
      <section className="w-full max-w-4xl mx-auto rounded-2xl bg-white/20 backdrop-blur-xl border border-white/30 shadow-[0_20px_60px_rgba(0,0,0,0.25)] p-4 sm:p-8 space-y-8">
        <header className="space-y-3">
          <p className="text-xs uppercase tracking-[0.35em] text-white/70">Account</p>
          <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-white">アカウント設定</h1>
          <p className="text-sm text-white/85">
            AI にあなたの好みを学習してもらうにはアカウントが必要です。表示名やパスワードの管理、ログアウトや削除リクエストはここから行えます。
          </p>
        </header>

        {/* 基本情報 */}
        <div className="rounded-2xl border border-white/60 bg-white/95 p-5 text-gray-900 shadow-lg">
          <h2 className="text-lg font-semibold">基本情報</h2>
          {userProfile ? (
            <dl className="mt-4 space-y-3 text-sm text-gray-700">
              <div className="flex items-start justify-between gap-3">
                <dt className="text-gray-500">ユーザーID</dt>
                <dd className="font-mono text-right break-all text-gray-900">{userProfile.username || '未設定'}</dd>
              </div>
              <div className="flex items-start justify-between gap-3">
                <dt className="text-gray-500">表示名</dt>
                <dd className="text-right break-words text-gray-900">{userProfile.displayName || '未設定'}</dd>
              </div>
              {userProfile.username && (
                <div className="flex items-start justify-between gap-3">
                  <dt className="text-gray-500">公開プロフィール</dt>
                  <dd>
                    <Link href={`/u/${userProfile.username}`} className="text-violet-600 hover:underline text-xs">
                      /u/{userProfile.username}
                    </Link>
                  </dd>
                </div>
              )}
              <p className="text-xs text-gray-500">
                ユーザーIDは登録後に変更できません。表示名は下の編集カードからいつでも更新できます。
              </p>
            </dl>
          ) : (
            <p className="mt-4 text-sm text-gray-500">ユーザー情報を取得できませんでした。再読み込みしてください。</p>
          )}
        </div>

        {/* キュレータープロフィール編集 */}
        <div className="rounded-2xl border border-white/60 bg-white/95 p-5 text-gray-900 shadow-lg">
          <div className="flex items-center gap-2 mb-4">
            <div className="w-8 h-8 rounded-full bg-violet-100 text-violet-600 flex items-center justify-center">
              <User size={16} />
            </div>
            <h2 className="text-lg font-semibold">キュレータープロフィール</h2>
          </div>
          <p className="text-xs text-gray-500 mb-4">
            公開プロフィールページに表示される情報です。リストにあなたの情報が表示されます。
          </p>
          <form onSubmit={handleCuratorProfileSave} className="space-y-4">
            <div className="space-y-1">
              <label className="text-sm font-semibold text-gray-700">自己紹介</label>
              <textarea
                value={curatorProfile.bio}
                onChange={(e) => setCuratorProfile((p) => ({ ...p, bio: e.target.value }))}
                rows={3}
                maxLength={200}
                className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm text-gray-900 placeholder:text-gray-400 focus:border-violet-400 focus:outline-none resize-none"
                placeholder="200文字以内で自己紹介を書いてください"
              />
              <p className="text-xs text-gray-400 text-right">{curatorProfile.bio.length}/200</p>
            </div>
            <div className="space-y-1">
              <label className="text-sm font-semibold text-gray-700">X（旧Twitter）URL</label>
              <input
                type="url"
                value={curatorProfile.x_url}
                onChange={(e) => setCuratorProfile((p) => ({ ...p, x_url: e.target.value }))}
                className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm text-gray-900 placeholder:text-gray-400 focus:border-violet-400 focus:outline-none"
                placeholder="https://x.com/yourname"
              />
            </div>

            <div className="border-t border-gray-100 pt-4">
              <p className="text-sm font-semibold text-gray-700 mb-1">アフィリエイトID設定</p>
              <p className="text-xs text-gray-500 mb-3">
                設定すると、あなたのリスト経由の購入であなたのアフィリエイトIDが適用されます。
              </p>
              <div className="space-y-3">
                <div className="space-y-1">
                  <label className="text-xs font-semibold text-gray-600">FANZA アフィリエイトID</label>
                  <input
                    type="text"
                    value={curatorProfile.affiliate_fanza_id}
                    onChange={(e) => setCuratorProfile((p) => ({ ...p, affiliate_fanza_id: e.target.value }))}
                    className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm text-gray-900 placeholder:text-gray-400 focus:border-violet-400 focus:outline-none"
                    placeholder="例: yourname-001"
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-xs font-semibold text-gray-600">FC2 アフィリエイトID</label>
                  <input
                    type="text"
                    value={curatorProfile.affiliate_fc2_id}
                    onChange={(e) => setCuratorProfile((p) => ({ ...p, affiliate_fc2_id: e.target.value }))}
                    className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm text-gray-900 placeholder:text-gray-400 focus:border-violet-400 focus:outline-none"
                    placeholder="FC2アフィリエイトID"
                  />
                </div>
                <div className="space-y-1">
                  <label className="text-xs font-semibold text-gray-600">MGS アフィリエイトID</label>
                  <input
                    type="text"
                    value={curatorProfile.affiliate_mgs_id}
                    onChange={(e) => setCuratorProfile((p) => ({ ...p, affiliate_mgs_id: e.target.value }))}
                    className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm text-gray-900 placeholder:text-gray-400 focus:border-violet-400 focus:outline-none"
                    placeholder="MGSアフィリエイトID"
                  />
                </div>
              </div>
            </div>

            <div className="flex justify-end">
              <button
                type="submit"
                disabled={curatorSaving}
                className="px-5 py-2 rounded-lg bg-violet-600 text-white text-sm font-semibold hover:bg-violet-700 disabled:opacity-50 transition"
              >
                {curatorSaving ? '保存中...' : '保存する'}
              </button>
            </div>
          </form>
        </div>

        {/* 除外タグ設定 */}
        <div className="rounded-2xl border border-white/60 bg-white/95 p-5 text-gray-900 shadow-lg">
          <div className="flex items-center gap-2 mb-2">
            <div className="w-8 h-8 rounded-full bg-red-100 text-red-500 flex items-center justify-center">
              <ShieldOff size={16} />
            </div>
            <h2 className="text-lg font-semibold">除外タグ設定</h2>
          </div>
          <p className="text-xs text-gray-500 mb-4">
            設定したタグの動画はフィード・おすすめ・人気・探索ページに表示されなくなります。
          </p>

          {/* 現在の除外タグ */}
          <div className="flex flex-wrap gap-2 mb-4 min-h-[32px]">
            {excludedTags.length === 0 ? (
              <p className="text-xs text-gray-400">除外タグはありません</p>
            ) : (
              excludedTags.map((tag) => (
                <span
                  key={tag.id}
                  className="inline-flex items-center gap-1 rounded-full bg-red-50 border border-red-200 px-3 py-1 text-xs text-red-700"
                >
                  {tag.name}
                  <button
                    type="button"
                    onClick={() => handleRemoveExcludedTag(tag)}
                    className="ml-1 text-red-400 hover:text-red-600"
                    aria-label={`${tag.name}を除外から外す`}
                  >
                    <X size={12} />
                  </button>
                </span>
              ))
            )}
          </div>

          {/* タグ検索・追加 */}
          <div ref={tagSearchRef} className="relative">
            <div className="flex items-center gap-2 rounded-lg border border-gray-300 px-3 py-2 focus-within:border-violet-400">
              <Search size={14} className="text-gray-400 shrink-0" />
              <input
                type="text"
                value={tagSearchQuery}
                onChange={(e) => { setTagSearchQuery(e.target.value); setTagSearchOpen(true); }}
                onFocus={() => setTagSearchOpen(true)}
                className="flex-1 text-sm text-gray-900 placeholder:text-gray-400 focus:outline-none bg-transparent"
                placeholder="タグ名で検索して追加..."
              />
            </div>
            {tagSearchOpen && tagSearchResults.length > 0 && (
              <ul className="absolute z-10 mt-1 w-full rounded-lg border border-gray-200 bg-white shadow-lg overflow-hidden">
                {tagSearchResults.map((tag) => {
                  const already = excludedTags.some((t) => t.id === tag.id);
                  return (
                    <li key={tag.id}>
                      <button
                        type="button"
                        onClick={() => !already && handleAddExcludedTag(tag)}
                        disabled={already}
                        className={`w-full flex items-center justify-between px-4 py-2 text-sm text-left transition ${already ? 'text-gray-400 cursor-default' : 'text-gray-800 hover:bg-violet-50'}`}
                      >
                        <span>{tag.name}</span>
                        {already ? (
                          <span className="text-xs text-gray-400">追加済み</span>
                        ) : (
                          <Plus size={14} className="text-violet-500" />
                        )}
                      </button>
                    </li>
                  );
                })}
              </ul>
            )}
          </div>
        </div>

        {/* 表示名・パスワード変更カード */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {cards.map((card) => {
            const Icon = card.icon;
            return (
              <button
                type="button"
                key={card.key}
                onClick={() => handleOpenCard(card.key)}
                className="text-left rounded-xl border border-white/60 bg-white/95 p-5 shadow-lg hover:shadow-xl transition text-gray-900"
              >
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-full bg-rose-50 text-rose-500 flex items-center justify-center">
                    <Icon size={18} />
                  </div>
                  <h2 className="text-lg font-semibold">{card.title}</h2>
                </div>
                <p className="mt-3 text-sm text-gray-600 leading-relaxed">{card.description}</p>
              </button>
            );
          })}
        </div>

        {/* ログアウト・削除 */}
        <div className="rounded-xl border border-white/60 bg-white/95 p-5 space-y-4 text-gray-900 shadow-lg">
          <h2 className="text-lg font-semibold">その他のアクション</h2>
          <div className="space-y-3 text-sm text-gray-600">
            <button
              type="button"
              onClick={() => setActiveModal('logout')}
              className="w-full flex items-start gap-3 rounded-lg border border-gray-100 px-4 py-3 text-left hover:bg-gray-50 transition"
            >
              <div className="w-9 h-9 rounded-full bg-gray-100 text-gray-600 flex items-center justify-center">
                <LogOut size={16} />
              </div>
              <div>
                <p className="font-semibold text-gray-900">ログアウト</p>
                <p className="text-xs text-gray-500 mt-1">端末のセッションを安全に終了します。</p>
              </div>
            </button>
            <button
              type="button"
              onClick={() => setActiveModal('delete')}
              className="w-full flex items-start gap-3 rounded-lg border border-gray-100 px-4 py-3 text-left hover:bg-gray-50 transition"
            >
              <div className="w-9 h-9 rounded-full bg-rose-100 text-rose-500 flex items-center justify-center">
                <Trash2 size={16} />
              </div>
              <div>
                <p className="font-semibold text-gray-900">アカウント削除</p>
                <p className="text-xs text-gray-500 mt-1">全データを削除したい場合の手続きを確認します。</p>
              </div>
            </button>
          </div>
        </div>
      </section>
      <ActionModal
        active={activeModal}
        loading={loading}
        displayName={displayName}
        setDisplayName={setDisplayName}
        passwords={passwords}
        setPasswords={setPasswords}
        onSubmitDisplayName={handleDisplayNameUpdate}
        onSubmitPassword={handlePasswordUpdate}
        onLogout={handleLogout}
        onClose={closeModal}
      />
    </main>
  );
}

type ActionModalProps = {
  active: CardKey | null;
  loading: boolean;
  displayName: string;
  setDisplayName: (value: string) => void;
  passwords: { current: string; next: string; confirm: string };
  setPasswords: React.Dispatch<React.SetStateAction<{ current: string; next: string; confirm: string }>>;
  onSubmitDisplayName: (e: React.FormEvent) => void;
  onSubmitPassword: (e: React.FormEvent) => void;
  onLogout: () => Promise<void>;
  onClose: () => void;
};

const modalTitles: Record<CardKey, string> = {
  displayName: '表示名の変更',
  password: 'パスワードの変更',
  logout: 'ログアウト',
  delete: 'アカウント削除',
};

function ActionModal({
  active, loading, displayName, setDisplayName, passwords, setPasswords,
  onSubmitDisplayName, onSubmitPassword, onLogout, onClose,
}: ActionModalProps) {
  if (!active) return null;

  const renderBody = () => {
    switch (active) {
      case 'displayName':
        return (
          <form onSubmit={onSubmitDisplayName} className="space-y-4">
            <div className="space-y-1">
              <label className="text-sm font-semibold text-gray-700">新しい表示名</label>
              <input
                type="text"
                value={displayName}
                onChange={(e) => setDisplayName(e.target.value)}
                className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm text-gray-900 placeholder:text-gray-400 focus:border-rose-400 focus:outline-none"
                placeholder="例: 新しい表示名"
              />
            </div>
            <div className="flex justify-end gap-2 text-sm">
              <button type="button" onClick={onClose} className="px-4 py-2 rounded-lg border border-gray-200 text-gray-600 hover:bg-gray-50">キャンセル</button>
              <button type="submit" disabled={loading} className="px-4 py-2 rounded-lg bg-rose-500 text-white font-semibold hover:bg-rose-600 disabled:opacity-50">
                {loading ? '更新中...' : '更新する'}
              </button>
            </div>
          </form>
        );
      case 'password':
        return (
          <form onSubmit={onSubmitPassword} className="space-y-4">
            <div className="space-y-1">
              <label className="text-sm font-semibold text-gray-700">新しいパスワード</label>
              <input
                type="password"
                value={passwords.next}
                onChange={(e) => setPasswords((prev) => ({ ...prev, next: e.target.value }))}
                className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm text-gray-900 placeholder:text-gray-400 focus:border-rose-400 focus:outline-none"
                placeholder="8文字以上"
              />
            </div>
            <div className="space-y-1">
              <label className="text-sm font-semibold text-gray-700">確認用パスワード</label>
              <input
                type="password"
                value={passwords.confirm}
                onChange={(e) => setPasswords((prev) => ({ ...prev, confirm: e.target.value }))}
                className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm text-gray-900 placeholder:text-gray-400 focus:border-rose-400 focus:outline-none"
                placeholder="同じパスワードを入力"
              />
            </div>
            <div className="flex justify-end gap-2 text-sm">
              <button type="button" onClick={onClose} className="px-4 py-2 rounded-lg border border-gray-200 text-gray-600 hover:bg-gray-50">キャンセル</button>
              <button type="submit" disabled={loading} className="px-4 py-2 rounded-lg bg-rose-500 text-white font-semibold hover:bg-rose-600 disabled:opacity-50">
                {loading ? '更新中...' : '更新する'}
              </button>
            </div>
          </form>
        );
      case 'logout':
        return (
          <div className="space-y-4 text-sm text-gray-700">
            <p>現在の端末からログアウトします。再度利用するにはログインが必要です。</p>
            <div className="flex justify-end gap-2">
              <button type="button" onClick={onClose} className="px-4 py-2 rounded-lg border border-gray-200 text-gray-600 hover:bg-gray-50">キャンセル</button>
              <button type="button" onClick={onLogout} disabled={loading} className="px-4 py-2 rounded-lg bg-rose-500 text-white font-semibold hover:bg-rose-600 disabled:opacity-50">
                {loading ? '処理中...' : 'ログアウト'}
              </button>
            </div>
          </div>
        );
      case 'delete':
        return (
          <div className="space-y-3 text-sm text-gray-700 leading-relaxed">
            <p>アカウントと全ての履歴を削除する場合は、本人確認のためお問い合わせが必要です。</p>
            <p>以下のフォームから「アカウント削除希望」とユーザーIDを記載してご連絡ください。確認後、スタッフが対応します。</p>
            <Link href="/contact" className="text-rose-500 underline inline-flex text-xs">お問い合わせフォームへ</Link>
          </div>
        );
      default:
        return null;
    }
  };

  return (
    <Transition appear show={Boolean(active)} as={Fragment}>
      <Dialog as="div" className="relative z-50" onClose={active === 'logout' ? () => {} : onClose}>
        <Transition.Child
          as={Fragment}
          enter="ease-out duration-200" enterFrom="opacity-0" enterTo="opacity-100"
          leave="ease-in duration-150" leaveFrom="opacity-100" leaveTo="opacity-0"
        >
          <div className="fixed inset-0 bg-black/40" />
        </Transition.Child>
        <div className="fixed inset-0 overflow-y-auto">
          <div className="flex min-h-full items-center justify-center p-4 text-center">
            <Transition.Child
              as={Fragment}
              enter="ease-out duration-200" enterFrom="opacity-0 scale-95" enterTo="opacity-100 scale-100"
              leave="ease-in duration-150" leaveFrom="opacity-100 scale-100" leaveTo="opacity-0 scale-95"
            >
              <Dialog.Panel className="w-full max-w-md transform overflow-hidden rounded-2xl bg-white p-6 text-left align-middle shadow-xl">
                <div className="flex items-start justify-between">
                  <Dialog.Title as="h3" className="text-lg font-bold text-gray-900">
                    {modalTitles[active]}
                  </Dialog.Title>
                  <button aria-label="閉じる" onClick={onClose} className="text-gray-400 hover:text-gray-600">
                    <X size={18} />
                  </button>
                </div>
                <div className="mt-4">{renderBody()}</div>
              </Dialog.Panel>
            </Transition.Child>
          </div>
        </div>
      </Dialog>
    </Transition>
  );
}
