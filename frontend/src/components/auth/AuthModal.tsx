'use client';

import { Dialog, Tab } from '@headlessui/react';
import { Fragment } from 'react';
import LoginForm from './LoginForm';
import RegisterForm from './RegisterForm';
import { X } from 'lucide-react';

interface AuthModalProps {
  isOpen: boolean;
  onClose: () => void;
  initialTab?: 'login' | 'register';
  registerNotice?: React.ReactNode;
}

const AuthModal: React.FC<AuthModalProps> = ({ isOpen, onClose, initialTab = 'login', registerNotice }) => {
  return (
    <Dialog as="div" className="relative z-50" open={isOpen} onClose={onClose}>
      {/* Backdrop */}
      <div className="fixed inset-0 bg-black/50 backdrop-blur-sm" aria-hidden="true" />

      {/* Modal Content */}
      <div className="fixed inset-0 flex items-center justify-center p-4">
                  <Dialog.Panel className="w-full max-w-md rounded-2xl bg-white p-8 shadow-2xl">
            <button onClick={onClose} className='absolute top-4 right-4 text-gray-600 hover:text-gray-900 transition-colors'>
              <X size={24} />
            </button>
            <Tab.Group defaultIndex={initialTab === 'register' ? 1 : 0}>
              <Tab.List className="flex space-x-1 rounded-xl bg-gray-100 p-1 mb-6">
                {['ログイン', '新規登録'].map((tab) => (
                  <Tab as={Fragment} key={tab}>
                    {({ selected }) => (
                      <button
                        className={`w-full rounded-lg py-2.5 text-sm font-bold leading-5 transition-all duration-300 focus:outline-none ${selected
                            ? 'bg-white shadow-md text-gray-900'
                            : 'text-gray-600 hover:bg-gray-200'
                          }`}
                      >
                        {tab}
                      </button>
                    )}
                  </Tab>
                ))}
              </Tab.List>
              <Tab.Panels>
                <Tab.Panel className="focus:outline-none">
                  <LoginForm onClose={onClose} />
                </Tab.Panel>
                <Tab.Panel className="focus:outline-none">
                  {registerNotice ? (
                    <div className="mb-4 p-3 rounded-lg bg-amber-50 text-amber-700 text-xs leading-relaxed">
                      {registerNotice}
                    </div>
                  ) : null}
                  <RegisterForm onClose={onClose} />
                </Tab.Panel>
              </Tab.Panels>
            </Tab.Group>
          </Dialog.Panel>
      </div>
    </Dialog>
  );
};

export default AuthModal;
