'use client';

import { motion, useAnimation, PanInfo } from 'framer-motion';
import { forwardRef, useImperativeHandle } from 'react';
import useWindowSize from '../hooks/useWindowSize';

// カードデータの型定義
export interface CardData {
  id: number;
  title: string;
  genre: string[];
  videoUrl: string;
}

export interface SwipeCardHandle {
  swipe: (direction: 'left' | 'right') => void;
}

interface SwipeCardProps {
  cardData: CardData;
  onSwipe: () => void;
  onDrag?: (event: MouseEvent | TouchEvent | PointerEvent, info: PanInfo) => void; // 追加
  onDragEnd?: (event: MouseEvent | TouchEvent | PointerEvent, info: PanInfo) => void; // 追加
}

const SwipeCard = forwardRef<SwipeCardHandle, SwipeCardProps>(({ cardData, onSwipe, onDrag, onDragEnd }, ref) => {
  const controls = useAnimation();
  const { height: windowHeight } = useWindowSize();

  // 動画のアスペクト比を11:10と仮定（プレーヤーUIを考慮）
  const videoAspectRatio = 11 / 10;
  // カードの高さをウィンドウの高さとし、その半分の高さの動画がアスペクト比を維持するのに必要な横幅を計算
  const cardWidth = windowHeight ? (windowHeight / 2) * videoAspectRatio : undefined;

  const swipe = async (direction: 'left' | 'right') => {
    const swipeWidth = cardWidth || 448; // cardWidthが未定義の場合のフォールバック
    const x = direction === 'right' ? `calc(100vw + ${swipeWidth}px)` : `calc(-100vw - ${swipeWidth}px)`;
    await controls.start({ x, opacity: 0, transition: { duration: 0.6 } }); 
    onSwipe();
  };

  useImperativeHandle(ref, () => ({
    swipe,
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
      {/* 上部: YouTube動画エリア */}
      <div className="relative w-full h-1/2">
        <iframe
          src={cardData.videoUrl} // FANZA埋め込みURLがここに渡されることを想定
          title="FANZA Video Player" // タイトルを修正
          frameBorder="0"
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" // 必要に応じてallow属性を調整
          allowFullScreen
          className="absolute top-0 left-0 w-full h-full" // classNameに変更
        ></iframe>
      </div>
      
      {/* 下部: テキスト情報エリア */}
      <div className="flex flex-col text-gray-700 p-4 overflow-y-auto h-1/2">
        <h2 className="text-xl font-bold">{cardData.title}</h2>
        <div className="flex flex-wrap gap-2 my-2">
          {Array.isArray(cardData.genre) && cardData.genre.map((tag) => (
            <span key={tag} className="bg-purple-300 text-white text-xs font-bold px-2 py-1 rounded-full">
              {tag}
            </span>
          ))}
        </div>
      </div>
    </motion.div>
  );
});

SwipeCard.displayName = "SwipeCard";

export default SwipeCard;