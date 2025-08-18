'use client';

import { Dialog, Tab } from '@headlessui/react';
import { Fragment } from 'react';
import LoginForm from './LoginForm';
import RegisterForm from './RegisterForm';
import { X } from 'lucide-react';

interface AuthModalProps {
  isOpen: boolean;
  onClose: () => void;
}

const AuthModal: React.FC<AuthModalProps> = ({ isOpen, onClose }) => {
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
            <Tab.Group>
              <Tab.List className="flex space-x-1 rounded-xl bg-gray-100 p-1 mb-6">
                {['ログイン', '新規登録'].map((category) => (
                  <Tab as={Fragment} key={category}>
                    {({ selected }) => (
                      <button
                        className={`w-full rounded-lg py-2.5 text-sm font-bold leading-5 transition-all duration-300 focus:outline-none ${selected
                            ? 'bg-white shadow-md text-gray-900'
                            : 'text-gray-600 hover:bg-gray-200'
                          }`}
                      >
                        {category}
                      </button>
                    )}
                  </Tab>
                ))}
              </Tab.List>
              <Tab.Panels>
                <Tab.Panel className="focus:outline-none">
                  <LoginForm />
                </Tab.Panel>
                <Tab.Panel className="focus:outline-none">
                  <RegisterForm />
                </Tab.Panel>
              </Tab.Panels>
            </Tab.Group>
          </Dialog.Panel>
      </div>
    </Dialog>
  );
};

export default AuthModal;
