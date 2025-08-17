'use client';

import { motion, useAnimation, PanInfo } from 'framer-motion';
import { forwardRef, useImperativeHandle } from 'react';

// カードデータの型定義
export interface CardData {
  id: number;
  title: string;
  category: string;
  description: string;
  videoUrl: string;
}

export interface SwipeCardHandle {
  swipe: (direction: 'left' | 'right') => void;
}

interface SwipeCardProps {
  cardData: CardData;
  onSwipe: () => void;
}

const SwipeCard = forwardRef<SwipeCardHandle, SwipeCardProps>(({ cardData, onSwipe }, ref) => {
  const controls = useAnimation();

  // カードの幅を考慮して画面外に完全に移動させる
  // max-w-md は 28rem = 448px
  const CARD_WIDTH_PX = 448; 

  const swipe = async (direction: 'left' | 'right') => {
    const x = direction === 'right' ? `calc(100vw + ${CARD_WIDTH_PX}px)` : `calc(-100vw - ${CARD_WIDTH_PX}px)`;
    await controls.start({ x, opacity: 0, transition: { duration: 0.4 } });
    onSwipe();
  };

  useImperativeHandle(ref, () => ({
    swipe,
  }));

  const handleDragEnd = (event: MouseEvent | TouchEvent | PointerEvent, info: PanInfo) => {
    // スワイプの閾値はそのまま
    if (info.offset.x > 100) {
      swipe('right');
    } else if (info.offset.x < -100) {
      swipe('left');
    } else {
      controls.start({ x: 0 });
    }
  };

  return (
    <motion.div 
      className="absolute w-full max-w-md h-[70vh] rounded-2xl bg-white/10 backdrop-blur-lg border border-white/30 shadow-2xl flex flex-col p-4 cursor-grab overflow-hidden"
      drag="x"
      dragConstraints={{ left: 0, right: 0 }}
      onDragEnd={handleDragEnd}
      animate={controls}
      initial={{ scale: 0.95, opacity: 0 }}
      animate={{ scale: 1, opacity: 1, transition: { duration: 0.3 } }}
      whileTap={{ cursor: "grabbing" }}
    >
      {/* 上部: YouTube動画エリア */}
      <div className="w-full h-3/5 rounded-lg overflow-hidden">
        <iframe
          width="100%"
          height="100%"
          src={cardData.videoUrl}
          title="YouTube video player"
          frameBorder="0"
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
          allowFullScreen
        ></iframe>
      </div>
      
      {/* 下部: テキスト情報エリア */}
      <div className="flex flex-col text-white p-4 flex-grow">
        <h2 className="text-xl font-bold line-clamp-2">{cardData.title}</h2>
        <div className="flex flex-wrap gap-2 my-2">
          {cardData.category.split(' ').map(tag => (
            <span key={tag} className="bg-white/20 text-xs font-semibold px-2 py-1 rounded-full">{tag}</span>
          ))}
        </div>
        <p className="text-sm line-clamp-3">{cardData.description}</p>
      </div>
    </motion.div>
  );
});

SwipeCard.displayName = "SwipeCard";

export default SwipeCard;