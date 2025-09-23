'use client';

import { Dialog, Transition } from '@headlessui/react';
import { Fragment } from 'react';
import { X } from 'lucide-react';
import { supabase } from '@/lib/supabase'; // supabaseをインポート

interface AccountManagementDrawerProps {
  isOpen: boolean;
  onClose: () => void;
}

const AccountManagementDrawer: React.FC<AccountManagementDrawerProps> = ({ isOpen, onClose }) => {
  const handleLogout = async () => {
    try { await supabase.auth.signOut({ scope: 'global' }); } catch {}
    onClose();
  };

  return (
    <Transition appear show={isOpen} as={Fragment}>
      <Dialog as="div" className="relative z-50" onClose={onClose}>
        {/* Backdrop */}
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

        {/* Drawer Panel */}
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
                          アカウント管理
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
                    
                    {/* Content */}
                    <div className="flex-1 px-4 sm:px-6 overflow-y-auto">
                       <div className="space-y-4">
                         <p className="text-gray-700">ここにアカウント管理の機能が表示されます。</p>
                         <button
                           onClick={handleLogout}
                           className="w-full py-2 px-4 bg-red-500 text-white font-bold rounded-lg hover:bg-red-600 transition-colors duration-300"
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
