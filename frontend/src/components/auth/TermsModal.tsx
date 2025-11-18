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
                  性癖ラボは、成人向けコンテンツのレコメンド機能を提供する 18 歳以上限定のサービスです。AI が好みを学習し、スワイプや検索結果をパーソナライズする目的でアカウント登録をお願いしています。
                </p>
                <ul className="list-disc pl-5 space-y-2">
                  <li>18 歳未満、もしくは居住地域で成人年齢に達していない方は利用できません。登録時の年齢確認チェックに必ず同意してください。</li>
                  <li>登録にはメールアドレス形式ではないユーザー ID を使用します。個人を特定できる氏名・住所・クレジットカード情報などは収集しません。</li>
                  <li>当サービスの画面・API・ストレージに対するスクレイピング、自動取得、BOT 利用を禁止します。検知した場合はアカウント停止や法的措置の対象となります。</li>
                  <li>表示される外部プラットフォームの作品は、各プラットフォームの規約・法令に従って視聴・購入してください。当サービスは内容の完全性や合法性を保証しません。</li>
                </ul>
                <p>
                  利用停止やアカウント削除を希望する場合は、アカウント設定ページから案内しているフォーム経由でユーザー ID を添えてご連絡ください。確認後スタッフが削除を行います。
                </p>
                <p>
                  本規約およびプライバシーポリシーは予告なく改定される場合があります。重要な変更はこのモーダルとサイト内のお知らせで周知します。
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
