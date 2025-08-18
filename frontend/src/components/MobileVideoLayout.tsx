'use client';

import { CardData } from '@/components/SwipeCard';
import ActionButtons from '@/components/ActionButtons';

interface MobileVideoLayoutProps {
  cardData: CardData;
  onSkip: () => void;
  onLike: () => void;
}

const MobileVideoLayout: React.FC<MobileVideoLayoutProps> = ({ cardData, onSkip, onLike }) => {
  return (
    <div className="flex flex-col w-full h-full">
      {/* 動画表示エリア */}
      <div className="w-full h-[200px] overflow-hidden relative">
        <iframe
          src={cardData.videoUrl} // width と height 属性を削除
          title="YouTube video player"
          frameBorder="0"
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
          allowFullScreen
          className="absolute inset-0"
        ></iframe>
      </div>

      {/* テキスト情報エリア */}
      <div className="p-4 text-gray-700 flex-grow overflow-y-auto">
        <h2 className="text-xl font-bold">{cardData.title}</h2>
        <div className="flex flex-wrap gap-2 my-2">
          {cardData.category.split(' ').map(tag => (
            <span key={tag} className="bg-purple-600 bg-opacity-60 text-white text-xs font-bold px-2 py-1 rounded-full">{tag}</span>
          ))}
        </div>
        <p className="text-sm">{cardData.description}</p>
      </div>

      {/* フッターボタンエリア */}
      <div className="w-full">
        <ActionButtons
          onSkip={onSkip}
          onLike={onLike}
          nopeColor="#A78BFA"
          likeColor="#FBBF24"
        isMobileLayout={true} // 追加
        />
      </div>
    </div>
  );
};

export default MobileVideoLayout;
