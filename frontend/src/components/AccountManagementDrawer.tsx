'use client';

import { Dialog, Transition } from '@headlessui/react';
import { Fragment, useEffect, useState } from 'react';
import { X, Sparkles, BarChart2, List } from 'lucide-react';
import { supabase } from '@/lib/supabase';
import { useDecisionCount } from '@/hooks/useDecisionCount';
import { useRouter } from 'next/navigation';

interface AccountManagementDrawerProps {
  isOpen: boolean;
  onClose: () => void;
}

const AccountManagementDrawer: React.FC<AccountManagementDrawerProps> = ({ isOpen, onClose }) => {
  const router = useRouter();
  const { decisionCount } = useDecisionCount();
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const personalizeTarget = Number(process.env.NEXT_PUBLIC_PERSONALIZE_TARGET || 20);

  useEffect(() => {
    const checkLoginStatus = async () => {
      const { data: { session } } = await supabase.auth.getSession();
      setIsLoggedIn(!!session);
    };
    checkLoginStatus();

    const { data: authListener } = supabase.auth.onAuthStateChange((_event, session) => {
      setIsLoggedIn(!!session);
    });

    return () => {
      authListener.subscription.unsubscribe();
    };
  }, []);

  const handleLogout = async () => {
    try {
      await supabase.auth.getSession();
      const { error } = await supabase.auth.signOut({ scope: 'global' });
      if (error) console.error('signOut error:', error.message);
    } catch (e) {
      console.error('logout exception', e);
    }
    onClose();
    try { if (typeof window !== 'undefined') window.location.assign('/swipe'); } catch {}
  };

  const handleNavigate = (href: string) => {
    router.push(href);
    onClose();
  };

  const remainingSwipes = Math.max(personalizeTarget - decisionCount, 0);
  const caption = remainingSwipes === 0 ? 'パーソナライズ完了' : `パーソナライズまであと${remainingSwipes}枚`;
  const progress = decisionCount / personalizeTarget;

  return (
    <Transition appear show={isOpen} as={Fragment}>
      <Dialog as="div" className="relative z-50" onClose={onClose}>
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

        <div className="fixed inset-0 overflow-hidden">
          <div className="absolute inset-0 overflow-hidden">
            <div className="pointer-events-none fixed inset-y-0 right-0 flex max-w-full pl-10 sm:pl-16">
              <Transition.Child
                as={Fragment}
                enter="transform transition ease-in-out duration-500 sm:duration-700"
                enterFrom="translate-x-full"
                enterTo="translate-x-0"
                leave="transform transition ease-in-out duration-500 sm:duration-700"
                leaveFrom="translate-x-0"
                leaveTo="translate-x-full"
              >
                <Dialog.Panel className="pointer-events-auto w-screen max-w-md">
                  <div className="flex h-full flex-col bg-white shadow-xl">
                    <div className="p-6">
                      <div className="flex items-start justify-between">
                        <Dialog.Title className="text-xl font-bold text-gray-900">
                          メニュー
                        </Dialog.Title>
                        <div className="ml-3 flex h-7 items-center">
                          <button
                            type="button"
                            className="relative rounded-md bg-white text-gray-400 hover:text-gray-500 focus:outline-none focus:ring-2 focus:ring-amber-500"
                            onClick={onClose}
                          >
                            <span className="absolute -inset-2.5" />
                            <span className="sr-only">Close panel</span>
                            <X className="h-6 w-6" aria-hidden="true" />
                          </button>
                        </div>
                      </div>
                    </div>
                    
                    <div className="flex-1 px-4 sm:px-6 overflow-y-auto">
                       <div className="space-y-2">
                        {isLoggedIn && (
                          <>
                            <button
                              onClick={() => handleNavigate('/lists')}
                              className="w-full flex items-center gap-3 px-3 py-2 rounded-md text-sm transition-colors text-gray-800 hover:bg-gray-100"
                            >
                              <List size={18} className="shrink-0" />
                              <span className="truncate">気になるリスト</span>
                            </button>
                            <button
                              onClick={() => handleNavigate('/ai-recommend')}
                              className="w-full text-left min-w-0 px-3 py-2 rounded-md text-sm transition-colors text-gray-800 hover:bg-gray-100"
                            >
                              <div className="flex items-center gap-3">
                                <Sparkles size={18} className="shrink-0" />
                                <span className="truncate">AIで探す</span>
                              </div>
                              <div className="mt-2 pr-1">
                                <div className="w-full h-2 bg-gray-200 rounded-full overflow-hidden">
                                  <div className="h-full rounded-full" style={{ width: `${Math.min(Math.max(progress, 0), 1) * 100}%`, background: 'linear-gradient(90deg, #ADB4E3 0%, #C8BAE3 33.333%, #F7BECE 66.666%, #F9B1C4 100%)' }} />
                                </div>
                                <div className="mt-1 text-[10px] text-gray-600 text-right">{caption}</div>
                              </div>
                            </button>
                            <button
                              onClick={() => handleNavigate('/analysis-results')}
                              className="w-full flex items-center gap-3 px-3 py-2 rounded-md text-sm transition-colors text-gray-800 hover:bg-gray-100"
                            >
                              <BarChart2 size={18} className="shrink-0" />
                              <span className="truncate">あなたの性癖</span>
                            </button>
                          </>
                        )}
                         <button
                           onClick={handleLogout}
                           className="w-full mt-4 py-2 px-4 bg-red-500 text-white font-bold rounded-lg hover:bg-red-600 transition-colors duration-300"
                         >
                           ログアウト
                         </button>
                       </div>
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
