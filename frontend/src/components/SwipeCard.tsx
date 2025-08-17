'use client';

import { motion, useAnimation, PanInfo } from 'framer-motion';
import { forwardRef, useImperativeHandle } from 'react';

// カードデータの型定義
export interface CardData {
  id: number;
  title: string;
  category: string;
  description: string;
}

export interface SwipeCardHandle {
  swipe: (direction: 'left' | 'right') => void;
}

interface SwipeCardProps {
  cardData: CardData;
  onSwipe: () => void;
  index: number;
  isTop: boolean;
}

const SwipeCard = forwardRef<SwipeCardHandle, SwipeCardProps>(({ cardData, onSwipe, index, isTop }, ref) => {
  const controls = useAnimation();

  const swipe = async (direction: 'left' | 'right') => {
    const x = direction === 'right' ? '100vw' : '-100vw';
    // isTop が false の場合はアニメーションが完了する前に onSwipe を呼んで即座にカードを削除する
    if (!isTop) {
      onSwipe();
      return;
    }
    await controls.start({ x, opacity: 0 });
    onSwipe();
  };

  useImperativeHandle(ref, () => ({
    swipe,
  }));

  const handleDragEnd = (event: MouseEvent | TouchEvent | PointerEvent, info: PanInfo) => {
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
      className="absolute w-full max-w-md h-[60vh] rounded-2xl bg-white/10 backdrop-blur-lg border border-white/30 shadow-2xl flex flex-col justify-between p-6 cursor-grab"
      drag={isTop ? "x" : false}
      dragConstraints={{ left: 0, right: 0 }}
      onDragEnd={handleDragEnd}
      animate={controls}
      initial={{ 
        scale: 1 - (3 - index) * 0.1, 
        y: (3 - index) * -30, 
        zIndex: 10 - index 
      }}
      style={{ 
        zIndex: 10 - index,
        transformOrigin: 'bottom',
      }}
      transition={{ ease: "easeOut", duration: 0.3 }}
      whileTap={{ cursor: "grabbing" }}
    >
      {/* ... card content ... */}
      <div className="w-full h-1/2 bg-white/20 rounded-lg flex items-center justify-center">
        <p className="text-white/50">Sample Video</p>
      </div>
      <div className="flex flex-col text-white">
        <h2 className="text-xl font-bold">{cardData.title}</h2>
        <div className="flex flex-wrap gap-2 my-2">
          {cardData.category.split(' ').map(tag => (
            <span key={tag} className="bg-white/20 text-xs font-semibold px-2 py-1 rounded-full">{tag}</span>
          ))}
        </div>
        <p className="text-sm">{cardData.description}</p>
      </div>
    </motion.div>
  );
});

SwipeCard.displayName = "SwipeCard";

export default SwipeCard;
