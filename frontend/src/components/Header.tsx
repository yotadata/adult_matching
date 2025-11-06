'use client';

import Image from 'next/image';
import Link from 'next/link';
import { useState, useEffect, useRef, Fragment } from 'react';
import AuthModal from './auth/AuthModal';
import useMediaQuery from '@/hooks/useMediaQuery';
import { supabase } from '@/lib/supabase';
import { User as SupabaseUser } from '@supabase/supabase-js';
import { Dialog, Transition } from '@headlessui/react';
import { UserPlus, Menu as MenuIcon, X, Home as HomeIcon, Sparkles, BarChart2, Brain } from 'lucide-react';
import { useRouter } from 'next/navigation';
import LikedVideosDrawer from './LikedVideosDrawer'; // ドロワーコンポーネントをインポート
import { toast } from 'react-hot-toast';
import { forceClearSupabaseAuth } from '@/lib/authUtils';

const Header = ({ cardWidth, mobileGauge }: { cardWidth: number | undefined; mobileGauge?: React.ReactNode }) => {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [authInitialTab, setAuthInitialTab] = useState<'login' | 'register'>('login');
  const [showRegisterNotice, setShowRegisterNotice] = useState(false);
  const [isDrawerOpen, setIsDrawerOpen] = useState(false); // ドロワー用のstate
  const [user, setUser] = useState<SupabaseUser | null>(null);
  const [authChecked, setAuthChecked] = useState(false);
  const isMobile = useMediaQuery('(max-width: 639px)');
  const [isMenuDrawerOpen, setIsMenuDrawerOpen] = useState(false);
  const mobileCloseBtnRef = useRef<HTMLButtonElement | null>(null);
  const router = useRouter();
  const [decisionCount, setDecisionCount] = useState<number>(0);
  const personalizeTarget = Number(process.env.NEXT_PUBLIC_PERSONALIZE_TARGET || 20);
  const diagnosisTarget = Number(process.env.NEXT_PUBLIC_DIAGNOSIS_TARGET || 30);

  useEffect(() => {
    const getSession = async () => {
      const { data: { session } } = await supabase.auth.getSession();
      setUser(session?.user || null);
      setAuthChecked(true);
    };
    getSession();

    const { data: authListener } = supabase.auth.onAuthStateChange((_event, session) => {
      setUser(session?.user || null);
      if (session?.user) {
        // ログイン完了時は認証モーダルを確実に閉じる
        setIsModalOpen(false);
      }
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
    const openLikedHandler = () => setIsDrawerOpen(true);
    window.addEventListener('open-liked-drawer', openLikedHandler as EventListener);

    // 初期の決定数を取得（ログイン時はDB、未ログイン時はlocalStorage）
    (async () => {
      const { data: { user: u } } = await supabase.auth.getUser();
      if (!u) {
        try {
          const raw = localStorage.getItem('guest_decisions_v1');
          const arr = raw ? JSON.parse(raw) : [];
          setDecisionCount(Array.isArray(arr) ? arr.length : 0);
        } catch {
          setDecisionCount(0);
        }
      } else {
        const { count } = await supabase
          .from('user_video_decisions')
          .select('*', { count: 'exact', head: true })
          .eq('user_id', u.id);
        setDecisionCount(count || 0);
      }
    })();

    return () => {
      authListener.subscription.unsubscribe();
      window.removeEventListener('open-auth-modal', openHandler as EventListener);
      window.removeEventListener('open-register-modal', openRegisterHandler as EventListener);
      window.removeEventListener('open-liked-drawer', openLikedHandler as EventListener);
    };
  }, []);

  const handleOpenModal = () => setIsModalOpen(true);
  const handleCloseModal = () => setIsModalOpen(false);
  // const handleOpenDrawer = () => setIsDrawerOpen(true);
  const handleCloseDrawer = () => setIsDrawerOpen(false);

  const handleLogout = async () => {
    try {
      // セッションを確実に初期化してからサインアウト
      await supabase.auth.getSession();
      const { error } = await supabase.auth.signOut({ scope: 'global' });
      if (error) console.error('signOut error:', error.message);
    } catch (e) {
      console.error('logout exception', e);
    }
    setIsMenuDrawerOpen(false);
    setIsDrawerOpen(false);
    // 即時にUIへ反映
    setUser(null);
    setAuthChecked(true);
    toast.success('ログアウトしました');
    try { forceClearSupabaseAuth(); } catch {}
    try { router.push('/swipe'); } catch {}
    try { router.refresh(); } catch {}
    try { setTimeout(() => { if (typeof window !== 'undefined') window.location.assign('/swipe'); }, 100); } catch {}
  };

  return (
    <header
      id="main-header"
      className={`mx-auto ${isMobile ? 'sticky top-0 z-50 pt-2 pb-0 w-full' : 'w-full py-2'}`}
      style={!isMobile ? { width: cardWidth ? `${cardWidth}px` : 'auto' } : {}}
    >
      {/* Glass bar over gradient bg */}
      <div
        className={`relative z-10 mx-auto ${isMobile ? 'px-3' : ''}`}
        style={!isMobile ? { width: cardWidth ? `${cardWidth}px` : 'auto' } : {}}
      >
        {/* モバイル: ログイン状態でUIを分岐 */}
        {isMobile ? (
          <>
            {user ? (
              // ログイン時: 中央ロゴ＋右ハンバーガー
              <div className="grid grid-cols-3 items-center text-white">
                <div />
                <div className="flex justify-center">
                  <Link href="/swipe" aria-label="ホームへ移動" className="inline-flex">
                    <Image
                      src="/seiheki_lab.png"
                      alt="Seiheki Lab Logo"
                      width={120}
                      height={50}
                      priority
                      draggable="false"
                      className="cursor-pointer"
                      style={{ filter: 'drop-shadow(0 0 0.5rem rgba(0, 0, 0, 0.1))' }}
                    />
                  </Link>
                </div>
                <div className="flex justify-end">
                  <button
                    type="button"
                    onClick={() => setIsMenuDrawerOpen(true)}
                    className="p-2 rounded-md hover:bg-white/20 transition-colors"
                    aria-label="メニュー"
                  >
                    <MenuIcon className="text-white" size={24} />
                  </button>
                </div>
              </div>
            ) : (
              // 未ログイン時: 左寄せロゴ＋右端ログインボタン
              <div className="grid grid-cols-3 items-center text-white">
                <div className="flex justify-start">
                  <Link href="/swipe" aria-label="ホームへ移動" className="inline-flex">
                    <Image
                      src="/seiheki_lab.png"
                      alt="Seiheki Lab Logo"
                      width={120}
                      height={50}
                      priority
                      draggable="false"
                      className="cursor-pointer"
                      style={{ filter: 'drop-shadow(0 0 0.5rem rgba(0, 0, 0, 0.1))' }}
                    />
                  </Link>
                </div>
                <div />
                <div className="flex justify-end">
                  {authChecked ? (
                    <button
                      onClick={handleOpenModal}
                      className={`flex items-center gap-2 px-3 py-1.5 text-xs font-bold text-white rounded-full backdrop-blur-md bg-white/10 border border-white/30 hover:bg-[#FF6B81] hover:text-white shadow-md transition-all duration-300`}
                      style={{ filter: 'drop-shadow(0 0 0.5rem rgba(0, 0, 0, 0.1))' }}
                      aria-label="ログインまたは新規登録"
                    >
                      <UserPlus size={16} className="opacity-90" />
                      <span>ログイン</span>
                    </button>
                  ) : null}
                </div>
              </div>
            )}
            {/* Mobile right-side drawer */}
            <Transition appear show={isMenuDrawerOpen} as={Fragment}>
              <Dialog as="div" className="relative z-50 sm:hidden" onClose={() => setIsMenuDrawerOpen(false)} initialFocus={mobileCloseBtnRef}>
                <Transition.Child as={Fragment} enter="ease-out duration-200" enterFrom="opacity-0" enterTo="opacity-100" leave="ease-in duration-150" leaveFrom="opacity-100" leaveTo="opacity-0">
                  <div className="fixed inset-0 bg-black/30" />
                </Transition.Child>
                <div className="fixed inset-0 overflow-hidden">
                  <div className="absolute inset-0 overflow-hidden">
                    <div className="pointer-events-none fixed inset-y-0 right-0 flex max-w-full">
                      <Transition.Child as={Fragment} enter="transform transition ease-in-out duration-300" enterFrom="translate-x-full" enterTo="translate-x-0" leave="transform transition ease-in-out duration-300" leaveFrom="translate-x-0" leaveTo="translate-x-full">
                        <Dialog.Panel className="pointer-events-auto w-screen max-w-xs h-full bg-white text-gray-800 shadow-xl flex flex-col">
                          <div className="p-4 border-b flex items-center justify-between">
                            <Dialog.Title className="text-base font-bold text-gray-900">メニュー</Dialog.Title>
                            <button ref={mobileCloseBtnRef} aria-label="閉じる" onClick={() => setIsMenuDrawerOpen(false)} className="p-1 text-gray-600 hover:text-gray-800">
                              <X size={20} />
                            </button>
                          </div>
                          <div className="flex-1 flex flex-col">
                            <div className="flex-1 flex flex-col divide-y divide-gray-200 overflow-y-auto">
                              <button className="w-full flex items-center gap-3 text-left px-4 py-3 text-gray-800 hover:bg-gray-100" onClick={() => { setIsMenuDrawerOpen(false); router.push('/swipe'); }}>
                                <HomeIcon size={18} />
                                <span>ホーム画面</span>
                              </button>
                              <button className="w-full text-left px-4 py-3 text-gray-800 hover:bg-gray-100" onClick={() => { setIsMenuDrawerOpen(false); router.push('/ai-recommend'); }}>
                                <div className="flex items-center gap-3 text-gray-800 mb-2">
                                  <Sparkles size={18} />
                                  <span>AIレコメンド</span>
                                </div>
                                <div className="w-full h-2 bg-gray-200 rounded-full overflow-hidden">
                                  <div className="h-full rounded-full" style={{ width: `${Math.min(decisionCount / personalizeTarget, 1) * 100}%`, background: 'linear-gradient(90deg, #ADB4E3 0%, #C8BAE3 33.333%, #F7BECE 66.666%, #F9B1C4 100%)' }} />
                                </div>
                                <div className="mt-1 text-xs text-gray-600 text-right">パーソナライズまであと{Math.max(personalizeTarget - decisionCount, 0)}枚</div>
                              </button>
                              <button className="w-full flex items-center gap-3 text-left px-4 py-3 text-gray-800 hover:bg-gray-100" onClick={() => { setIsMenuDrawerOpen(false); router.push('/analysis-results'); }}>
                                <BarChart2 size={18} />
                                <span>性癖分析</span>
                              </button>
                              <button className="w-full text-left px-4 py-3 text-gray-800 hover:bg-gray-100" onClick={() => { setIsMenuDrawerOpen(false); router.push('/personality'); }}>
                                <div className="flex items-center gap-3 text-gray-800 mb-2">
                                  <Brain size={18} />
                                  <span>性癖パーソナリティ診断</span>
                                </div>
                                <div className="w-full h-2 bg-gray-200 rounded-full overflow-hidden">
                                  <div className="h-full rounded-full" style={{ width: `${Math.min(decisionCount / diagnosisTarget, 1) * 100}%`, background: 'linear-gradient(90deg, #ADB4E3 0%, #C8BAE3 33.333%, #F7BECE 66.666%, #F9B1C4 100%)' }} />
                                </div>
                                <div className="mt-1 text-xs text-gray-600 text-right">診断まであと{Math.max(diagnosisTarget - decisionCount, 0)}枚</div>
                              </button>
                            </div>
                            <div className="border-t px-4 py-3">
                              {user ? (
                                <button onClick={handleLogout} className="w-full text-left text-gray-800 hover:bg-gray-100 rounded-md px-3 py-2">ログアウト</button>
                              ) : (
                                <button onClick={() => { setIsMenuDrawerOpen(false); handleOpenModal(); }} className="w-full text-left text-gray-800 hover:bg-gray-100 rounded-md px-3 py-2">ログイン / 新規登録</button>
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
          </>
        ) : (
          // デスクトップ: ロゴ中央。ハンバーガーは廃止（左固定ナビに移行）
          <div className="grid grid-cols-3 items-center text-white">
            <div />
            <div className="flex justify-center">
              <Link href="/swipe" aria-label="ホームへ移動" className="inline-flex">
                <Image
                  src="/seiheki_lab.png"
                  alt="Seiheki Lab Logo"
                  width={180}
                  height={78}
                  priority
                  draggable="false"
                  className="cursor-pointer"
                  style={{ filter: 'drop-shadow(0 0 0.5rem rgba(0, 0, 0, 0.1))' }}
                />
              </Link>
            </div>
            <div className="flex justify-end">
              {user ? (
                null
              ) : (
                authChecked ? (
                  <button
                    onClick={handleOpenModal}
                    className={`flex items-center gap-2 px-4 py-2 text-sm font-bold text-white rounded-full backdrop-blur-md bg-white/10 border border-white/30 hover:bg-[#FF6B81] hover:text-white shadow-md transition-all duration-300`}
                    style={{ filter: 'drop-shadow(0 0 0.5rem rgba(0, 0, 0, 0.1))' }}
                    aria-label="ログインまたは新規登録"
                  >
                    <UserPlus size={18} className="opacity-90" />
                    <span>ログイン / 新規登録</span>
                  </button>
                ) : null
              )}
            </div>
          </div>
        )}
      </div>
      {isMobile && mobileGauge ? (
        <div className="relative z-0 mt-1">
          {mobileGauge}
        </div>
      ) : null}
      {/* Desktop hamburger drawer removed in favor of fixed left sidebar */}
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
