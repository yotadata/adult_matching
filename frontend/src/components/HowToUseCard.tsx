
'use client';

import { X } from 'lucide-react';

type HowToUseCardProps = {
  onClose: () => void;
};

const HowToUseCard: React.FC<HowToUseCardProps> = ({ onClose }) => {
  return (
    <div className="fixed bottom-6 left-6 w-80 max-w-[calc(100%-3rem)] bg-white rounded-2xl shadow-lg p-6 text-gray-800 z-50">
      <button
        onClick={onClose}
        className="absolute top-3 right-3 text-gray-600 hover:text-gray-900 transition-colors"
        aria-label="閉じる"
      >
        <X size={20} />
      </button>
      <div className="flex items-center mb-3">
        <span className="text-2xl mr-3">✨</span>
        <h3 className="font-bold text-lg">使い方</h3>
      </div>
      <p className="text-sm leading-relaxed">
        カードを<span className="font-bold text-amber-500">右にスワイプ</span>すると「いいね」、
        <span className="font-bold text-violet-500">左にスワイプ</span>すると「スキップ」できます。
      </p>
    </div>
  );
};

export default HowToUseCard;
