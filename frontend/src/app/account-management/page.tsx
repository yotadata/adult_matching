'use client';

import { useState, useEffect } from 'react';
import { supabase } from '@/lib/supabase';
import { useRouter } from 'next/navigation';

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
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-100">
        <p className="text-gray-600">読み込み中...</p>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col">
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
    </div>
  );
}
