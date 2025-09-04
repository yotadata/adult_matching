'use client';

import Image from 'next/image';
import { useState, useEffect } from 'react';
import AuthModal from './auth/AuthModal';
import useMediaQuery from '@/hooks/useMediaQuery';
import { supabase } from '@/lib/supabase';
import { User as SupabaseUser } from '@supabase/supabase-js';
import { Menu } from '@headlessui/react';
import Link from 'next/link';
import { Heart, User, UserPlus } from 'lucide-react';
import LikedVideosDrawer from './LikedVideosDrawer'; // ドロワーコンポーネントをインポート

const Header = ({ cardWidth, mobileGauge }: { cardWidth: number | undefined; mobileGauge?: React.ReactNode }) => {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [authInitialTab, setAuthInitialTab] = useState<'login' | 'register'>('login');
  const [showRegisterNotice, setShowRegisterNotice] = useState(false);
  const [isDrawerOpen, setIsDrawerOpen] = useState(false); // ドロワー用のstate
  const [user, setUser] = useState<SupabaseUser | null>(null);
  const [authChecked, setAuthChecked] = useState(false);
  const isMobile = useMediaQuery('(max-width: 639px)');

  useEffect(() => {
    const getSession = async () => {
      const { data: { session } } = await supabase.auth.getSession();
      setUser(session?.user || null);
      setAuthChecked(true);
    };
    getSession();

    const { data: authListener } = supabase.auth.onAuthStateChange((_event, session) => {
      setUser(session?.user || null);
    });

    // Allow other components to request opening the auth modal
    const openHandler = () => {
      setAuthInitialTab('login');
      setShowRegisterNotice(false);
      setIsModalOpen(true);
    };
    const openRegisterHandler = () => {
      setAuthInitialTab('register');
      setShowRegisterNotice(true);
      setIsModalOpen(true);
    };
    window.addEventListener('open-auth-modal', openHandler as EventListener);
    window.addEventListener('open-register-modal', openRegisterHandler as EventListener);

    return () => {
      authListener.subscription.unsubscribe();
      window.removeEventListener('open-auth-modal', openHandler as EventListener);
      window.removeEventListener('open-register-modal', openRegisterHandler as EventListener);
    };
  }, []);

  const handleOpenModal = () => setIsModalOpen(true);
  const handleCloseModal = () => setIsModalOpen(false);
  const handleOpenDrawer = () => setIsDrawerOpen(true);
  const handleCloseDrawer = () => setIsDrawerOpen(false);

  const handleLogout = async () => {
    await supabase.auth.signOut();
  };

  return (
    <header id="main-header" className={`mx-auto ${isMobile ? 'sticky top-0 z-50 pt-2 pb-0 bg-gradient-to-r from-[#C4C8E3] via-[#D7D1E3] to-[#F7D7E0] to-[#F8DBB9] shadow-md w-full' : 'mt-4 mb-2'}`} style={!isMobile ? { width: cardWidth ? `${cardWidth}px` : 'auto' } : {}}>
      <div className={`relative z-10 flex justify-between items-center text-white mx-auto ${isMobile ? 'px-3 border-b border-white/40' : ''}`} style={!isMobile ? { width: cardWidth ? `${cardWidth}px` : 'auto' } : {}}>
        <Image
          src="/seiheki_lab.png"
          alt="Seiheki Lab Logo"
          width={isMobile ? 120 : 180}
          height={isMobile ? 50 : 78}
          priority
          draggable="false"
          style={{ filter: 'drop-shadow(0 0 0.5rem rgba(0, 0, 0, 0.1))' }}
        />
        {user ? (
          <div className="flex items-center space-x-2">
            <button onClick={handleOpenDrawer} className="p-2 rounded-full hover:bg-white/20 transition-colors">
              <Heart className="text-white drop-shadow-md" size={24} />
            </button>
            
            <Menu as="div" className="relative inline-block text-left">
              <div>
                <Menu.Button className="p-2 rounded-full hover:bg-white/20 transition-colors">
                  <User className="text-white drop-shadow-md" size={24} />
                </Menu.Button>
              </div>

              <Menu.Items className="absolute right-0 mt-2 w-56 origin-top-right divide-y divide-gray-100 rounded-2xl bg-white shadow-2xl ring-1 ring-black ring-opacity-5 focus:outline-none z-50">
                <div className="px-1 py-1 ">
                  <Menu.Item>
                    {({ active }) => (
                      <Link href="/account-management" passHref>
                        <button
                          className={`${
                            active ? 'bg-violet-500 text-white' : 'text-gray-900'
                          } group flex w-full items-center rounded-md px-2 py-2 text-sm`}
                        >
                          アカウント管理
                        </button>
                      </Link>
                    )}
                  </Menu.Item>
                  <Menu.Item>
                    {({ active }) => (
                      <Link href="/analysis-results" passHref>
                        <button
                          className={`${
                            active ? 'bg-violet-500 text-white' : 'text-gray-900'
                          } group flex w-full items-center rounded-md px-2 py-2 text-sm`}
                        >
                          性癖分析結果
                        </button>
                      </Link>
                    )}
                  </Menu.Item>
                </div>
                <div className="px-1 py-1">
                  <Menu.Item>
                    {({ active }) => (
                      <button
                        onClick={handleLogout}
                        className={`${
                          active ? 'bg-violet-500 text-white' : 'text-gray-900'
                        } group flex w-full items-center rounded-md px-2 py-2 text-sm`}
                      >
                        ログアウト
                      </button>
                    )}
                  </Menu.Item>
                </div>
              </Menu.Items>
            </Menu>
          </div>
        ) : (
          authChecked ? (
            <button
              onClick={handleOpenModal}
              className={`flex items-center gap-2 ${isMobile ? 'px-2 py-1.5' : 'px-4 py-2'} mx-2 text-sm font-bold text-white rounded-xl bg-transparent border border-white hover:bg-purple-500 hover:text-white shadow-lg transition-all duration-300`}
              style={{ filter: 'drop-shadow(0 0 0.5rem rgba(0, 0, 0, 0.1))' }}
              aria-label="ログインまたは新規登録"
            >
              <UserPlus size={18} className="opacity-90" />
              <span className="hidden sm:inline">ログイン / 新規登録</span>
              <span className="sm:hidden">ログイン</span>
            </button>
          ) : null
        )}
      </div>
      {isMobile && mobileGauge ? (
        <div className="relative z-0 mt-0">
          {mobileGauge}
        </div>
      ) : null}
      <AuthModal
        isOpen={isModalOpen}
        onClose={handleCloseModal}
        initialTab={authInitialTab}
        registerNotice={showRegisterNotice ? (
          <>
            <p className="mb-1">・捨てアドレスでの登録で大丈夫です。</p>
            <p>・個人情報やクレジットカード情報の取得意図は一切ありません。</p>
          </>
        ) : undefined}
      />
      <LikedVideosDrawer isOpen={isDrawerOpen} onClose={handleCloseDrawer} />
    </header>
  );
};

export default Header;
