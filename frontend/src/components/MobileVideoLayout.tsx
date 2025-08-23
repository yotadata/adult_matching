'use client';

import { CardData } from '@/components/SwipeCard';
import ActionButtons from '@/components/ActionButtons';
import { useState } from 'react';
import { Play } from 'lucide-react';

interface MobileVideoLayoutProps {
  cardData: CardData;
  onSkip: () => void;
  onLike: () => void;
}

const MobileVideoLayout: React.FC<MobileVideoLayoutProps> = ({ cardData, onSkip, onLike }) => {
  const [showVideo, setShowVideo] = useState(false);

  const handlePlayClick = () => {
    setShowVideo(true);
  };

  return (
    <div className="flex flex-col w-full h-full">
      {/* 動画表示エリア */}
      <div className="w-full overflow-hidden relative aspect-[494/370] bg-black flex items-center justify-center">
        {!showVideo && cardData.thumbnail_url ? (
          <div
            className="absolute inset-0 w-full h-full bg-cover bg-center cursor-pointer flex items-center justify-center"
            style={{ backgroundImage: `url(${cardData.thumbnail_url})` }}
            onClick={handlePlayClick}
          >
            <div className="absolute inset-0 bg-black bg-opacity-40 flex items-center justify-center">
              <Play className="text-white w-16 h-16 opacity-80" fill="white" />
            </div>
          </div>
        ) : !showVideo && !cardData.thumbnail_url ? (
          <div
            className="absolute inset-0 w-full h-full bg-gray-800 flex items-center justify-center text-white text-lg"
            onClick={handlePlayClick}
          >
            <Play className="text-white w-16 h-16 opacity-80" fill="white" />
          </div>
        ) : null}

        {showVideo && (
          <iframe
            src={cardData.videoUrl}
            title="YouTube video player"
            frameBorder="0"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
            allowFullScreen
            className="absolute inset-0 w-full h-full"
          ></iframe>
        )}
      </div>

      {/* テキスト情報エリア */}
      <div className="p-4 text-gray-700 flex-grow overflow-y-auto">
        <h2 className="text-xl font-bold">{cardData.title}</h2>
        <div className="flex flex-wrap gap-2 my-2">
          {Array.isArray(cardData.genre) && cardData.genre.map((tag) => (
            <span
              key={tag}
              className="bg-purple-600 bg-opacity-60 text-white text-xs font-bold px-2 py-1 rounded-full"
            >
              {tag}
            </span>
          ))}
        </div>
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
