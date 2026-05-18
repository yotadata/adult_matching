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
      <div className="fixed inset-0 bg-black/70 backdrop-blur-sm" aria-hidden="true" />

      {/* Modal Content */}
      <div className="fixed inset-0 flex items-center justify-center p-4">
                  <Dialog.Panel className="w-full max-w-md rounded-2xl bg-[#0d1117] border border-[#30363d] p-8 shadow-2xl">
            <button onClick={onClose} className='absolute top-4 right-4 text-[#8b949e] hover:text-[#e6edf3] transition-colors'>
              <X size={24} />
            </button>
            <Tab.Group defaultIndex={initialTab === 'register' ? 1 : 0}>
              <Tab.List className="flex space-x-1 rounded-xl bg-[#161b22] p-1 mb-6">
                {['ログイン', '新規登録'].map((tab) => (
                  <Tab as={Fragment} key={tab}>
                    {({ selected }) => (
                      <button
                        className={`w-full rounded-lg py-2.5 text-sm font-bold leading-5 transition-all duration-300 focus:outline-none ${selected
                            ? 'bg-violet-600 text-white'
                            : 'text-[#8b949e] hover:bg-[#30363d]'
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
                    <div className="mb-4 p-3 rounded-lg bg-amber-500/10 border border-amber-500/30 text-amber-300 text-xs leading-relaxed">
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
