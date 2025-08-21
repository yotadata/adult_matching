'use client';

import Image from 'next/image';
import { useState, useEffect, useRef } from 'react';
import AuthModal from './auth/AuthModal';
import useMediaQuery from '@/hooks/useMediaQuery';
import { supabase } from '@/lib/supabase';
import { User as SupabaseUser } from '@supabase/supabase-js';
import { Menu } from '@headlessui/react';
import Link from 'next/link';
import { Heart, User } from 'lucide-react'; // アイコンをインポート

const Header = () => {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [user, setUser] = useState<SupabaseUser | null>(null);
  const isMobile = useMediaQuery('(max-width: 639px)');

  useEffect(() => {
    const getSession = async () => {
      const { data: { session } } = await supabase.auth.getSession();
      setUser(session?.user || null);
    };
    getSession();

    const { data: authListener } = supabase.auth.onAuthStateChange((_event, session) => {
      setUser(session?.user || null);
    });

    return () => {
      authListener.subscription.unsubscribe();
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
          <div className="flex items-center space-x-2">
            <Link href="/liked-videos" passHref>
              <button className="p-2 rounded-full hover:bg-white/20 transition-colors">
                <Heart className="text-white drop-shadow-md" size={24} />
              </button>
            </Link>
            
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
