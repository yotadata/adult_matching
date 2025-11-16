'use client';

import { Dialog, Transition } from '@headlessui/react';
import { Fragment } from 'react';
import { X } from 'lucide-react';

interface TermsModalProps {
  open: boolean;
  onClose: () => void;
}

const TermsModal: React.FC<TermsModalProps> = ({ open, onClose }) => (
  <Transition appear show={open} as={Fragment}>
    <Dialog as="div" className="relative z-50" onClose={onClose}>
      <Transition.Child
        as={Fragment}
        enter="ease-out duration-200"
        enterFrom="opacity-0"
        enterTo="opacity-100"
        leave="ease-in duration-150"
        leaveFrom="opacity-100"
        leaveTo="opacity-0"
      >
        <div className="fixed inset-0 bg-black/40" />
      </Transition.Child>

      <div className="fixed inset-0 overflow-y-auto">
        <div className="flex min-h-full items-center justify-center p-4 text-center">
          <Transition.Child
            as={Fragment}
            enter="ease-out duration-200"
            enterFrom="opacity-0 scale-95"
            enterTo="opacity-100 scale-100"
            leave="ease-in duration-150"
            leaveFrom="opacity-100 scale-100"
            leaveTo="opacity-0 scale-95"
          >
            <Dialog.Panel className="w-full max-w-2xl transform overflow-hidden rounded-2xl bg-white p-6 text-left align-middle shadow-xl">
              <div className="flex items-start justify-between">
                <Dialog.Title as="h3" className="text-lg font-bold text-gray-900">
                  利用規約
                </Dialog.Title>
                <button aria-label="閉じる" onClick={onClose} className="text-gray-400 hover:text-gray-600">
                  <X size={20} />
                </button>
              </div>
              <div className="mt-4 space-y-4 text-sm text-gray-700 max-h-[60vh] overflow-y-auto">
                <p>
                  性癖ラボ（以下「当サービス」）は、成人向けコンテンツのレコメンド機能を提供します。18歳未満の方や各地域で成人年齢に達していない方はご利用いただけません。
                </p>
                <p>
                  利用者は当サービスを個人的な目的でのみ利用し、著作権や肖像権など第三者の権利を侵害しないものとします。また、当サービスが案内するコンテンツにアクセスする際は、各プラットフォームの利用規約および法令を遵守してください。
                </p>
                <p>
                  当サービスは正確で安全な情報提供に努めますが、表示されるコンテンツの完全性・合法性・最新性について保証するものではありません。利用者自身の判断により視聴・購入してください。
                </p>
                <p>
                  ご利用を継続することで、本規約ならびにプライバシーポリシーに同意いただいたものとみなします。本規約は予告なく改定される場合があります。最新版は本モーダルまたは公式ドキュメントをご確認ください。
                </p>
                <p>
                  当サービスの画面やAPIに対するスクレイピング、データ抽出、自動化ツールの利用は禁止です。収集行為が判明した場合、即時にアカウント停止や法的措置を行うことがあります。
                </p>
              </div>
            </Dialog.Panel>
          </Transition.Child>
        </div>
      </div>
    </Dialog>
  </Transition>
);

export default TermsModal;
