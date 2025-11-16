'use client';

import { Fragment, useEffect, useMemo, useState } from 'react';
import { Edit3, Key, LogOut, Trash2, X } from 'lucide-react';
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

export default function AccountManagementPage() {
  const [activeModal, setActiveModal] = useState<CardKey | null>(null);
  const [displayName, setDisplayName] = useState('');
  const [passwords, setPasswords] = useState({ current: '', next: '', confirm: '' });
  const [loading, setLoading] = useState(false);
  const [userProfile, setUserProfile] = useState<{ displayName: string; username: string } | null>(null);
  const pseudoDomain = useMemo(() => process.env.NEXT_PUBLIC_PSEUDO_EMAIL_DOMAIN || 'anon.seihekilab.com', []);
  const router = useRouter();

  useEffect(() => {
    const loadUser = async () => {
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) {
        setUserProfile(null);
        return;
      }
      setUserProfile({
        displayName: (user.user_metadata?.display_name as string) || user.user_metadata?.displayName || '',
        username: (() => {
          const metaUsername = (user.user_metadata?.username as string) || '';
          if (metaUsername) return metaUsername;
          const email = user.email ?? '';
          const suffix = `@${pseudoDomain}`;
          if (email && email.endsWith(suffix)) {
            return email.slice(0, -suffix.length);
          }
          return email || '';
        })(),
      });
    };
    loadUser();
  }, [pseudoDomain]);

  const closeModal = () => {
    setActiveModal(null);
    setDisplayName('');
    setPasswords({ current: '', next: '', confirm: '' });
    setLoading(false);
  };

  const handleDisplayNameUpdate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!displayName.trim()) {
      toast.error('表示名を入力してください');
      return;
    }
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
    if (!passwords.next || passwords.next.length < 8) {
      toast.error('パスワードは8文字以上で入力してください');
      return;
    }
    if (passwords.next !== passwords.confirm) {
      toast.error('確認用パスワードが一致しません');
      return;
    }
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

  const handleOpenCard = (key: CardKey) => {
    if (key === 'displayName') {
      setDisplayName(userProfile?.displayName ?? '');
    }
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
              <p className="text-xs text-gray-500">
                ユーザーIDは登録後に変更できません。表示名は下の編集カードからいつでも更新できます。
              </p>
            </dl>
          ) : (
            <p className="mt-4 text-sm text-gray-500">ユーザー情報を取得できませんでした。再読み込みしてください。</p>
          )}
        </div>

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
  active,
  loading,
  displayName,
  setDisplayName,
  passwords,
  setPasswords,
  onSubmitDisplayName,
  onSubmitPassword,
  onLogout,
  onClose,
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
                className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-rose-400 focus:outline-none"
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
                className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-rose-400 focus:outline-none"
                placeholder="8文字以上"
              />
            </div>
            <div className="space-y-1">
              <label className="text-sm font-semibold text-gray-700">確認用パスワード</label>
              <input
                type="password"
                value={passwords.confirm}
                onChange={(e) => setPasswords((prev) => ({ ...prev, confirm: e.target.value }))}
                className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-rose-400 focus:outline-none"
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
          enter="ease-out duration-200"
          enterFrom="opacity-0"
          enterTo="opacity-100"
          leave="ease-in duration-150"
          leaveFrom="opacity-100"
          leaveTo="opacity-0"
        >
          <div className="fixed inset-0 bg-black/40" />
        </Transition.Child>

        <div className="fixed inset-0 overflow-y-auto">
          <div className="flex min-h-full items-center justify-center p-4 text-center">
            <Transition.Child
              as={Fragment}
              enter="ease-out duration-200"
              enterFrom="opacity-0 scale-95"
              enterTo="opacity-100 scale-100"
              leave="ease-in duration-150"
              leaveFrom="opacity-100 scale-100"
              leaveTo="opacity-0 scale-95"
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
