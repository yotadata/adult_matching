'use client';

import { motion, useAnimation, PanInfo } from 'framer-motion';
import { forwardRef, useImperativeHandle, useState } from 'react';
import useWindowSize from '../hooks/useWindowSize';
import { Play, User, Tag } from 'lucide-react'; // Userアイコンをインポート

// カードデータの型定義
export interface CardData {
  id: number;
  title: string;
  genre: string[];
  description: string;
  videoUrl: string;
  thumbnail_url: string; // 追加
  performers?: { id: string; name: string; }[]; // 追加
  tags?: { id: string; name: string; }[]; // 追加
}

export interface SwipeCardHandle {
  swipe: (direction: 'left' | 'right') => void;
  getCardWidth: () => number | undefined;
}

interface SwipeCardProps {
  cardData: CardData;
  onSwipe: (direction: 'left' | 'right') => void;
  onDrag?: (event: MouseEvent | TouchEvent | PointerEvent, info: PanInfo) => void; // 追加
  onDragEnd?: (event: MouseEvent | TouchEvent | PointerEvent, info: PanInfo) => void; // 追加
}

const SwipeCard = forwardRef<SwipeCardHandle, SwipeCardProps>(({ cardData, onSwipe, onDrag, onDragEnd }, ref) => {
  const controls = useAnimation();
  const { height: windowHeight } = useWindowSize();
  const [showVideo, setShowVideo] = useState(false);

  // 動画のアスペクト比を21:20と仮定（プレーヤーUIを考慮）
  const videoAspectRatio = 21 / 20;
  // カードの高さをウィンドウの高さとし、その半分の高さの動画がアスペクト比を維持するのに必要な横幅を計算
  const cardWidth = windowHeight ? (windowHeight / 2) * videoAspectRatio : undefined;

  const swipe = async (direction: 'left' | 'right') => {
    const swipeWidth = cardWidth || 448; // cardWidthが未定義の場合のフォールバック
    const x = direction === 'right' ? `calc(100vw + ${swipeWidth}px)` : `calc(-100vw - ${swipeWidth}px)`;
    await controls.start({ x, opacity: 0, transition: { duration: 0.6 } }); 
    onSwipe(direction);
  };

  useImperativeHandle(ref, () => ({
    swipe,
    getCardWidth: () => cardWidth,
  }));

  const handleDrag = (event: MouseEvent | TouchEvent | PointerEvent, info: PanInfo) => {
    onDrag?.(event, info); // 親の onDrag を呼び出す
  };

  const handleDragEnd = (event: MouseEvent | TouchEvent | PointerEvent, info: PanInfo) => {
    if (info.offset.x > 100) {
      swipe('right');
    } else if (info.offset.x < -100) {
      swipe('left');
    } else {
      controls.start({ x: 0 });
    }
    onDragEnd?.(event, info); // 親の onDragEnd を呼び出す
  };

  const handlePlayClick = () => {
    setShowVideo(true);
  };

  return (
    <motion.div 
      className="absolute h-full rounded-2xl bg-white backdrop-blur-lg border border-white/60 shadow-2xl flex flex-col p-4 cursor-grab overflow-hidden"
      style={{ width: cardWidth ? `${cardWidth}px` : 'auto' }}
      drag="x"
      dragConstraints={{ left: 0, right: 0 }}
      onDragStart={(event, info) => onDrag?.(event, info)} // onDragStart も追加
      onDrag={handleDrag}
      onDragEnd={handleDragEnd}
      animate={controls}
      initial={false}
      whileTap={{ cursor: "grabbing" }}
    >
      {/* 上部: 動画エリア */}
      <div className="relative w-full h-1/2 bg-black flex items-center justify-center">
        {!showVideo && cardData.thumbnail_url ? (
          <div
            className="absolute inset-0 w-full h-full bg-contain bg-no-repeat bg-center cursor-pointer flex items-center justify-center"
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
            src={cardData.videoUrl} // FANZA埋め込みURLがここに渡されることを想定
            title="FANZA Video Player" // タイトルを修正
            frameBorder="0"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" // 必要に応じてallow属性を調整
            allowFullScreen
            className="absolute top-0 left-0 w-full h-full" // classNameに変更
          ></iframe>
        )}
      </div>
      
      {/* 下部: テキスト情報エリア */}
      <div className="flex flex-col text-gray-700 p-4 overflow-y-auto h-1/2">
        <h2 className="text-xl font-bold">{cardData.title}</h2>
        {cardData.performers && cardData.performers.length > 0 && (
          <div className="grid grid-cols-[auto_1fr] items-baseline gap-x-2 mt-2">
            <div className="flex flex-shrink-0 items-center text-sm text-gray-500">
              <User className="mr-1 h-4 w-4" />
              <span>出演:</span>
            </div>
            <div className="flex flex-wrap gap-2">
              {cardData.performers.map((performer) => (
                <span key={performer.id} className="rounded-full bg-pink-300 px-2 py-1 text-xs font-bold text-white">
                  {performer.name}
                </span>
              ))}
            </div>
          </div>
        )}
        <div className="grid grid-cols-[auto_1fr] items-baseline gap-x-2 mt-2">
            {/* Grid Column 1: Label */}
            <div className="flex items-center text-sm text-gray-500">
              <Tag className="mr-1 h-4 w-4" />
              <span>タグ:</span>
            </div>
            {/* Grid Column 2: Tags container */}
            <div className="flex flex-wrap gap-2">
              {cardData.tags.map((tag) => (
                <span key={tag.id} className="rounded-full bg-purple-300 px-2 py-1 text-xs font-bold text-white">
                  {tag.name}
                </span>
              ))}
            </div>
          </div>
      </div>
    </motion.div>
  );
});

SwipeCard.displayName = "SwipeCard";

export default SwipeCard;