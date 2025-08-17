'use client';

import React, { useState } from 'react';
import { motion, useMotionValue, useTransform, animate } from 'framer-motion';
import Card from './Card';
import ActionButton from './ActionButton';
import { dummyVideos, Video } from '@/lib/data';

const SwipeArea = () => {
  const [videos, setVideos] = useState(dummyVideos);
  const x = useMotionValue(0);

  const handleSwipe = () => {
    setVideos((prevVideos) => prevVideos.slice(1));
    x.set(0);
  };

  const rotate = useTransform(x, [-200, 200], [-30, 30]);
  const opacity = useTransform(x, [-200, 200], [0.5, 1]); // Make it slightly visible when swiping

  const handleSwipeComplete = (direction: 'left' | 'right') => {
    const flyAwayDistance = 500;
    animate(x, direction === 'right' ? flyAwayDistance : -flyAwayDistance, {
      onComplete: handleSwipe,
    });
  };

  return (
    <div className="relative w-full h-screen flex flex-col items-center justify-center">
      <div className="relative w-full h-[70vh] flex items-center justify-center">
        {videos.length > 0 ? (
          videos.map((video, index) => {
            const isTopCard = index === videos.length - 1;
            if (isTopCard) {
              return (
                <motion.div
                  key={video.id}
                  drag="x"
                  style={{ x, rotate }}
                  dragConstraints={{ left: 0, right: 0, top: 0, bottom: 0 }}
                  onDragEnd={(event, info) => {
                    if (info.offset.x > 100) {
                      handleSwipeComplete('right');
                    } else if (info.offset.x < -100) {
                      handleSwipeComplete('left');
                    }
                  }}
                  className="absolute"
                >
                  <Card video={video} />
                </motion.div>
              );
            }
            // Render underlying cards for visual effect
            return (
              <motion.div
                key={video.id}
                className="absolute"
                style={{
                  scale: 1 - (videos.length - 1 - index) * 0.05,
                  top: (videos.length - 1 - index) * 10,
                  opacity: 1 - (videos.length - 1 - index) * 0.1,
                }}
              >
                <Card video={video} />
              </motion.div>
            );
          })
        ) : (
          <p>本日のカードは以上です。</p>
        )}
      </div>

      {videos.length > 0 && (
        <div className="fixed bottom-10 flex space-x-8 z-10">
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
