'use client';

import Image from 'next/image';
import { useState, useEffect } from 'react'; // useEffect をインポート
import AuthModal from './auth/AuthModal';
import useMediaQuery from '@/hooks/useMediaQuery';
import { supabase } from '@/lib/supabase'; // supabase クライアントをインポート
import { User } from '@supabase/supabase-js'; // User 型をインポート

const Header = () => {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [user, setUser] = useState<User | null>(null); // ユーザー情報を保持するステート
  const isMobile = useMediaQuery('(max-width: 639px)');

  useEffect(() => {
    // 初期ロード時にセッションを取得
    const getSession = async () => {
      const { data: { session } } = await supabase.auth.getSession();
      setUser(session?.user || null);
    };
    getSession();

    // 認証状態の変更を購読
    const { data: authListener } = supabase.auth.onAuthStateChange((_event, session) => {
      setUser(session?.user || null);
    });

    // クリーンアップ
    return () => {
      authListener?.unsubscribe();
    };
  }, []);

  const handleOpenModal = () => {
    setIsModalOpen(true);
  };

  const handleCloseModal = () => {
    setIsModalOpen(false);
  };

  const handleLogout = async () => {
    await supabase.auth.signOut();
    // ログアウト後の処理（例: ページリロードやリダイレクト）
    // setUser(null) は onAuthStateChange で自動的に行われる
  };

  return (
    <header id="main-header" className={`w-full ${isMobile ? 'fixed top-0 left-0 right-0 z-40 p-4 bg-gradient-to-r from-[#C4C8E3] via-[#D7D1E3] to-[#F7D7E0] to-[#F8DBB9] shadow-md' : 'max-w-md mx-auto mt-4 mb-2'}`}>
      <div className="flex justify-between items-center text-white max-w-md mx-auto">
        <Image
          src="/seiheki_lab.png"
          alt="Seiheki Lab Logo"
          width={230}
          height={100}
          priority
          draggable="false"
          style={{ filter: 'drop-shadow(0 0 0.5rem rgba(0, 0, 0, 0.1))' }}
        />
        {user ? (
          // ログイン後の表示
          <button
            onClick={handleLogout} // 仮でログアウトボタン
            className="p-4 py-2 mx-2 text-sm font-bold text-gray-900 rounded-xl border border-gray-300 bg-white shadow-lg hover:bg-gray-100 transition-colors duration-200"
            style={{ filter: 'drop-shadow(0 0 0.5rem rgba(0, 0, 0, 0.1))' }}
          >
            ログアウト
          </button>
        ) : (
          // ログイン前の表示
          <button
            onClick={handleOpenModal}
            className="p-4 py-2 mx-2 text-sm font-bold text-gray-900 rounded-xl border border-gray-300 bg-white shadow-lg hover:bg-gray-100 transition-colors duration-200"
            style={{ filter: 'drop-shadow(0 0 0.5rem rgba(0, 0, 0, 0.1))' }}
          >
            ログイン
          </button>
        )}
      </div>
      <AuthModal isOpen={isModalOpen} onClose={handleCloseModal} />
    </header>
  );
};

export default Header;
