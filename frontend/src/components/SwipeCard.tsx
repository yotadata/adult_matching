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
  onDrag?: (event: MouseEvent | TouchEvent | PointerEvent, info: PanInfo) => void; // 追加
  onDragEnd?: (event: MouseEvent | TouchEvent | PointerEvent, info: PanInfo) => void; // 追加
}

const SwipeCard = forwardRef<SwipeCardHandle, SwipeCardProps>(({ cardData, onSwipe, onDrag, onDragEnd }, ref) => {
  const controls = useAnimation();

  const CARD_WIDTH_PX = 448; 

  const swipe = async (direction: 'left' | 'right') => {
    const x = direction === 'right' ? `calc(100vw + ${CARD_WIDTH_PX}px)` : `calc(-100vw - ${CARD_WIDTH_PX}px)`;
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
      className="absolute w-full max-w-md h-full rounded-2xl bg-white backdrop-blur-lg border border-white/60 shadow-2xl flex flex-col p-4 cursor-grab overflow-hidden"
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
      <div className="w-full aspect-w-4 aspect-h-3 rounded-lg overflow-hidden">
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
      <div className="flex flex-col text-gray-700 p-4 flex-grow overflow-y-auto">
        <h2 className="text-xl font-bold">{cardData.title}</h2>
        <div className="flex flex-wrap gap-2 my-2">
          {cardData.category.split(' ').map(tag => (
            <span key={tag} className="bg-purple-300 text-white text-xs font-bold px-2 py-1 rounded-full">{tag}</span>
          ))}
        </div>
        <p className="text-sm">{cardData.description}</p>
      </div>
    </motion.div>
  );
});

SwipeCard.displayName = "SwipeCard";

export default SwipeCard;