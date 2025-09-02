'use client';

import { useState, useEffect } from 'react';
<<<<<<< HEAD
import { useRouter } from 'next/navigation';
import { createClient } from '@supabase/supabase-js';
import { Dialog, Transition } from '@headlessui/react';
import { Fragment } from 'react';
import { Trash2, User, Mail, Calendar, AlertTriangle } from 'lucide-react';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!;
const supabase = createClient(supabaseUrl, supabaseKey);

export default function AccountManagementPage() {
  const router = useRouter();
  const [user, setUser] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [deleteLoading, setDeleteLoading] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [confirmText, setConfirmText] = useState('');
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    checkUser();
  }, []);

  const checkUser = async () => {
    try {
      const { data: { session } } = await supabase.auth.getSession();
      if (!session) {
        router.push('/');
        return;
      }
      setUser(session.user);
    } catch (error) {
      console.error('Error checking user:', error);
      setError('ユーザー情報の取得に失敗しました');
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteAccount = async () => {
    if (confirmText !== 'アカウント削除') {
      setError('確認テキストが正しくありません');
      return;
    }

    try {
      setDeleteLoading(true);
      setError(null);

      const { data, error } = await supabase.functions.invoke('delete_account');
      
      if (error) {
        console.error('Delete account error:', error);
        setError('アカウント削除に失敗しました');
        return;
      }

      if (data?.success) {
        // ログアウト処理
        await supabase.auth.signOut();
        // ホームページにリダイレクト
        router.push('/');
      } else {
        setError('アカウント削除に失敗しました');
      }
    } catch (err) {
      console.error('Unexpected error:', err);
      setError('予期しないエラーが発生しました');
    } finally {
      setDeleteLoading(false);
    }
  };

  const handleLogout = async () => {
    try {
      await supabase.auth.signOut();
      router.push('/');
    } catch (error) {
      console.error('Logout error:', error);
    }
=======
import { supabase } from '@/lib/supabase';
import { useRouter } from 'next/navigation';
import Header from '@/components/Header';

export default function AccountManagementPage() {
  const [email, setEmail] = useState<string | null>(null);
  const [displayName, setDisplayName] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [message, setMessage] = useState<string | null>(null);
  const router = useRouter();

  useEffect(() => {
    const fetchUserData = async () => {
      setLoading(true);
      const { data: { user }, error: userError } = await supabase.auth.getUser();

      if (userError) {
        console.error('Error fetching user:', userError);
        setMessage('ユーザー情報の取得に失敗しました。');
        setLoading(false);
        return;
      }

      if (user) {
        setEmail(user.email ?? null);
        const { data: profile, error: profileError } = await supabase
          .from('profiles')
          .select('display_name')
          .eq('user_id', user.id)
          .single();

        if (profileError && profileError.code !== 'PGRST116') { // PGRST116はデータが見つからない場合
          console.error('Error fetching profile:', profileError);
          setMessage('プロフィール情報の取得に失敗しました。');
        } else if (profile) {
          setDisplayName(profile.display_name);
        }
      } else {
        router.push('/'); // ユーザーがログインしていない場合はトップページへリダイレクト
      }
      setLoading(false);
    };

    fetchUserData();
  }, [router]);

  const handleUpdateProfile = async () => {
    if (!displayName) {
      setMessage('表示名を入力してください。');
      return;
    }
    setLoading(true);
    const { data: { user } } = await supabase.auth.getUser();

    if (user) {
      const { error } = await supabase
        .from('profiles')
        .upsert({ user_id: user.id, display_name: displayName }, { onConflict: 'user_id' });

      if (error) {
        console.error('Error updating profile:', error);
        setMessage('プロフィールの更新に失敗しました。');
      } else {
        setMessage('プロフィールを更新しました！');
      }
    } else {
      setMessage('ユーザーがログインしていません。');
    }
    setLoading(false);
  };

  const handleLogout = async () => {
    setLoading(true);
    const { error } = await supabase.auth.signOut();

    if (error) {
      console.error('Error logging out:', error);
      setMessage('ログアウトに失敗しました。');
    } else {
      router.push('/'); // ログアウト後にトップページへリダイレクト
    }
    setLoading(false);
>>>>>>> origin/main
  };

  if (loading) {
    return (
<<<<<<< HEAD
      <div className="min-h-screen bg-gradient-to-br from-purple-100 to-pink-100 flex items-center justify-center">
        <p className="text-lg text-gray-600">読み込み中...</p>
=======
      <div className="min-h-screen flex items-center justify-center bg-gray-100">
        <p className="text-gray-600">読み込み中...</p>
>>>>>>> origin/main
      </div>
    );
  }

  return (
<<<<<<< HEAD
    <div className="min-h-screen bg-gradient-to-br from-purple-100 to-pink-100">
      <div className="max-w-2xl mx-auto py-8 px-4">
        <div className="bg-white rounded-2xl shadow-xl overflow-hidden">
          {/* Header */}
          <div className="bg-gradient-to-r from-purple-500 to-pink-500 px-6 py-8">
            <h1 className="text-2xl font-bold text-white flex items-center gap-2">
              <User className="h-6 w-6" />
              アカウント管理
            </h1>
          </div>

          {/* User Info */}
          <div className="p-6 border-b">
            <h2 className="text-lg font-semibold text-gray-800 mb-4">アカウント情報</h2>
            <div className="space-y-3">
              <div className="flex items-center gap-3 text-gray-600">
                <Mail className="h-5 w-5" />
                <span>{user?.email}</span>
              </div>
              <div className="flex items-center gap-3 text-gray-600">
                <Calendar className="h-5 w-5" />
                <span>登録日: {new Date(user?.created_at).toLocaleDateString('ja-JP')}</span>
              </div>
            </div>
          </div>

          {/* Actions */}
          <div className="p-6">
            <div className="space-y-4">
              {/* Logout Button */}
              <button
                onClick={handleLogout}
                className="w-full bg-gray-100 hover:bg-gray-200 text-gray-800 font-medium py-3 px-4 rounded-lg transition-colors"
              >
                ログアウト
              </button>

              {/* Delete Account Button */}
              <button
                onClick={() => setShowDeleteModal(true)}
                className="w-full bg-red-50 hover:bg-red-100 text-red-600 font-medium py-3 px-4 rounded-lg transition-colors flex items-center justify-center gap-2 border border-red-200"
              >
                <Trash2 className="h-4 w-4" />
                アカウント削除
              </button>
            </div>

            {/* Back to Home */}
            <div className="mt-6 pt-6 border-t">
              <button
                onClick={() => router.push('/')}
                className="text-purple-600 hover:text-purple-800 font-medium"
              >
                ← ホームに戻る
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Delete Confirmation Modal */}
      <Transition appear show={showDeleteModal} as={Fragment}>
        <Dialog as="div" className="relative z-50" onClose={() => setShowDeleteModal(false)}>
          <Transition.Child
            as={Fragment}
            enter="ease-out duration-300"
            enterFrom="opacity-0"
            enterTo="opacity-100"
            leave="ease-in duration-200"
            leaveFrom="opacity-100"
            leaveTo="opacity-0"
          >
            <div className="fixed inset-0 bg-black/30 backdrop-blur-sm" />
          </Transition.Child>

          <div className="fixed inset-0 overflow-y-auto">
            <div className="flex min-h-full items-center justify-center p-4">
              <Transition.Child
                as={Fragment}
                enter="ease-out duration-300"
                enterFrom="opacity-0 scale-95"
                enterTo="opacity-100 scale-100"
                leave="ease-in duration-200"
                leaveFrom="opacity-100 scale-100"
                leaveTo="opacity-0 scale-95"
              >
                <Dialog.Panel className="w-full max-w-md transform overflow-hidden rounded-2xl bg-white p-6 shadow-xl transition-all">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="flex-shrink-0 w-10 h-10 bg-red-100 rounded-full flex items-center justify-center">
                      <AlertTriangle className="h-5 w-5 text-red-600" />
                    </div>
                    <Dialog.Title as="h3" className="text-lg font-bold text-gray-900">
                      アカウント削除の確認
                    </Dialog.Title>
                  </div>

                  <div className="mb-6">
                    <p className="text-gray-600 mb-4">
                      この操作は取り消すことができません。以下のデータがすべて削除されます：
                    </p>
                    <ul className="text-sm text-gray-600 space-y-1 mb-4 pl-4">
                      <li>• アカウント情報</li>
                      <li>• いいねした動画</li>
                      <li>• 利用履歴</li>
                      <li>• その他すべてのデータ</li>
                    </ul>
                    <p className="text-red-600 font-medium mb-4">
                      続行するには「アカウント削除」と入力してください：
                    </p>
                    <input
                      type="text"
                      value={confirmText}
                      onChange={(e) => setConfirmText(e.target.value)}
                      placeholder="アカウント削除"
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-red-500 focus:border-transparent"
                      disabled={deleteLoading}
                    />
                    {error && (
                      <p className="text-red-600 text-sm mt-2">{error}</p>
                    )}
                  </div>

                  <div className="flex gap-3">
                    <button
                      type="button"
                      className="flex-1 bg-gray-100 hover:bg-gray-200 text-gray-800 font-medium py-2 px-4 rounded-lg transition-colors"
                      onClick={() => {
                        setShowDeleteModal(false);
                        setConfirmText('');
                        setError(null);
                      }}
                      disabled={deleteLoading}
                    >
                      キャンセル
                    </button>
                    <button
                      type="button"
                      className="flex-1 bg-red-600 hover:bg-red-700 text-white font-medium py-2 px-4 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                      onClick={handleDeleteAccount}
                      disabled={deleteLoading || confirmText !== 'アカウント削除'}
                    >
                      {deleteLoading ? '削除中...' : '削除'}
                    </button>
                  </div>
                </Dialog.Panel>
              </Transition.Child>
            </div>
          </div>
        </Dialog>
      </Transition>
=======
    <div className="min-h-screen bg-gray-100 flex flex-col">
      <Header cardWidth={undefined} />
      <main className="flex-grow container mx-auto px-4 py-8">
        <h1 className="text-3xl font-bold text-gray-800 mb-8">アカウント管理</h1>
        <div className="bg-white p-6 rounded-lg shadow-md max-w-md mx-auto">
          {message && (
            <div className="bg-blue-100 border border-blue-400 text-blue-700 px-4 py-3 rounded relative mb-4" role="alert">
              <span className="block sm:inline">{message}</span>
            </div>
          )}
          <div className="mb-4">
            <label className="block text-gray-700 text-sm font-bold mb-2">メールアドレス:</label>
            <p className="text-gray-900 text-lg font-semibold">{email}</p>
          </div>
          <div className="mb-6">
            <label htmlFor="displayName" className="block text-gray-700 text-sm font-bold mb-2">表示名:</label>
            <input
              type="text"
              id="displayName"
              className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
              value={displayName || ''}
              onChange={(e) => setDisplayName(e.target.value)}
              placeholder="表示名を入力してください"
            />
          </div>
          <div className="flex items-center justify-between">
            <button
              onClick={handleUpdateProfile}
              className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline"
            >
              プロフィールを更新
            </button>
            <button
              onClick={handleLogout}
              className="bg-red-500 hover:bg-red-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline"
            >
              ログアウト
            </button>
          </div>
        </div>
      </main>
>>>>>>> origin/main
    </div>
  );
}