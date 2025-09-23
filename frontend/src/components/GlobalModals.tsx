'use client';

import { useEffect, useState } from 'react';
import AuthModal from './auth/AuthModal';
import LikedVideosDrawer from './LikedVideosDrawer';
import { supabase } from '@/lib/supabase';

export default function GlobalModals() {
  const [isAuthOpen, setIsAuthOpen] = useState(false);
  const [authInitialTab, setAuthInitialTab] = useState<'login' | 'register'>('login');
  const [showRegisterNotice, setShowRegisterNotice] = useState(false);

  const [isLikedOpen, setIsLikedOpen] = useState(false);

  useEffect(() => {
    const authHandler = () => {
      setAuthInitialTab('login');
      setShowRegisterNotice(false);
      setIsAuthOpen(true);
    };
    const registerHandler = () => {
      setAuthInitialTab('register');
      setShowRegisterNotice(true);
      setIsAuthOpen(true);
    };
    const likedHandler = () => setIsLikedOpen(true);

    window.addEventListener('open-auth-modal', authHandler as EventListener);
    window.addEventListener('open-register-modal', registerHandler as EventListener);
    window.addEventListener('open-liked-drawer', likedHandler as EventListener);

    const { data: listener } = supabase.auth.onAuthStateChange((_e, session) => {
      if (session?.user) setIsAuthOpen(false);
    });

    return () => {
      window.removeEventListener('open-auth-modal', authHandler as EventListener);
      window.removeEventListener('open-register-modal', registerHandler as EventListener);
      window.removeEventListener('open-liked-drawer', likedHandler as EventListener);
      listener.subscription.unsubscribe();
    };
  }, []);

  return (
    <>
      <AuthModal
        isOpen={isAuthOpen}
        onClose={() => setIsAuthOpen(false)}
        initialTab={authInitialTab}
        registerNotice={showRegisterNotice ? (
          <>
            <p className="mb-1">・捨てアドレスでの登録で大丈夫です。</p>
            <p>・個人情報やクレジットカード情報の取得意図は一切ありません。</p>
          </>
        ) : undefined}
      />
      <LikedVideosDrawer isOpen={isLikedOpen} onClose={() => setIsLikedOpen(false)} />
    </>
  );
}

