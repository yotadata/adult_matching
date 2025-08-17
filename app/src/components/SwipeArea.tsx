'use client';

import React, { useState, useRef } from 'react';
import { motion, useMotionValue, useTransform, animate } from 'framer-motion';
import Card from './Card';
import ActionButton from './ActionButton';
import { dummyVideos } from '@/lib/data';

const SwipeArea = () => {
  const [videos, setVideos] = useState(dummyVideos);
  const x = useMotionValue(0);
  const constraintsRef = useRef(null);

  const handleEndSwipe = () => {
    setVideos((prev) => prev.slice(0, prev.length - 1));
    x.set(0);
  };

  const rotate = useTransform(x, [-200, 200], [-25, 25]);

  const handleSwipeComplete = (direction: 'left' | 'right') => {
    const flyAwayDistance = 500;
    animate(x, direction === 'right' ? flyAwayDistance : -flyAwayDistance, {
      type: 'spring',
      stiffness: 400,
      damping: 50,
      onComplete: handleEndSwipe,
    });
  };

  return (
    <div ref={constraintsRef} className="relative w-full h-full">
      {/* カードコンテナを中央に配置 */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[90vw] max-w-md h-[70vh]">
        {videos.length > 0 ? (
          videos.map((video, index) => {
            const isTopCard = index === videos.length - 1;
            return (
              <motion.div
                key={video.id}
                drag={isTopCard ? "x" : false}
                style={{
                  x: isTopCard ? x : 0,
                  rotate: isTopCard ? rotate : 0,
                  scale: 1 - (videos.length - 1 - index) * 0.05,
                  zIndex: index,
                }}
                dragConstraints={constraintsRef}
                onDragEnd={(event, info) => {
                  if (info.offset.x > 100) {
                    handleSwipeComplete('right');
                  } else if (info.offset.x < -100) {
                    handleSwipeComplete('left');
                  }
                }}
                className="absolute w-full h-full"
              >
                <Card video={video} />
              </motion.div>
            );
          })
        ) : (
          <p className="text-center text-white">本日のカードは以上です。</p>
        )}
      </div>

      {videos.length > 0 && (
        <div className="fixed bottom-10 left-1/2 -translate-x-1/2 flex space-x-8 z-20">
          <ActionButton onClick={() => handleSwipeComplete('left')} className="bg-white/20">
            <span className="text-3xl">❌</span>
          </ActionButton>
          <ActionButton onClick={() => handleSwipeComplete('right')} className="bg-pink-500/50">
            <span className="text-3xl">❤️</span>
          </ActionButton>
        </div>
      )}
    </div>
  );
};

export default SwipeArea;
