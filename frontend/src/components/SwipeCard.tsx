'use client';

import { motion, useAnimation, PanInfo } from 'framer-motion';
import { forwardRef, useImperativeHandle, useState, useEffect, useRef } from 'react';
import { Play, User, Tag, Calendar } from 'lucide-react'; // アイコンをインポート

// カードデータの型定義
export interface CardData {
  id: number;
  title: string;
  genre: string[];
  description: string;
  videoUrl: string;
  thumbnail_url: string; // 追加
  sampleVideoUrl?: string; // 追加: 直接再生用
  embedUrl?: string; // 追加: iframe用
  performers?: { id: string; name: string; }[]; // 追加
  tags?: { id: string; name: string; }[]; // 追加
  product_released_at?: string; // 追加: 発売日
  productUrl?: string; // 追加: 外部再生リンク
}

export interface SwipeCardHandle {
  swipe: (direction: 'left' | 'right') => void;
}

interface SwipeCardProps {
  cardData: CardData;
  onSwipe: (direction: 'left' | 'right') => void;
  onDrag?: (event: MouseEvent | TouchEvent | PointerEvent, info: PanInfo) => void;
  onDragEnd?: (event: MouseEvent | TouchEvent | PointerEvent, info: PanInfo) => void;
  cardWidth: number | undefined; // cardWidth propを追加
}

const SwipeCard = forwardRef<SwipeCardHandle, SwipeCardProps>(({ cardData, onSwipe, onDrag, onDragEnd, cardWidth }, ref) => {
  const controls = useAnimation();
  
  const [showVideo, setShowVideo] = useState(false);
  const [showOverlay, setShowOverlay] = useState(true);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const overlayHideTimer = useRef<number | null>(null);
  const overlayHideDelayMs = 700; // iframe読込後も少しサムネイルを維持

  // Reset iframe fallback when card changes
  useEffect(() => {
    setShowVideo(false);
    setShowOverlay(true);
    if (overlayHideTimer.current) {
      clearTimeout(overlayHideTimer.current);
      overlayHideTimer.current = null;
    }
  }, [cardData.id]);

  

  const swipe = async (direction: 'left' | 'right') => {
    const swipeWidth = cardWidth || 448; // cardWidthが未定義の場合のフォールバック
    const x = direction === 'right' ? `calc(100vw + ${swipeWidth}px)` : `calc(-100vw - ${swipeWidth}px)`;
    await controls.start({ x, opacity: 0, transition: { duration: 0.6 } }); 
    onSwipe(direction);
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

  const handlePlayClick = () => {
    // デバッグ: どの再生パスか確認
    if (cardData.sampleVideoUrl) {
      console.warn('[SwipeCard] Using <video> src:', cardData.sampleVideoUrl);
    } else {
      console.warn('[SwipeCard] Using <iframe> src:', cardData.embedUrl || cardData.videoUrl);
    }
    setShowVideo(true);
  };

  useEffect(() => {
    if (showVideo && videoRef.current) {
      const v = videoRef.current;
      // iOS/Android 対策: タップ直後に明示的に再生
      const playPromise = v.play();
      if (playPromise && typeof playPromise.then === 'function') {
        playPromise.catch((err) => {
          console.warn('[SwipeCard] Autoplay failed, waiting for user gesture.', err);
        });
      }
    }
  }, [showVideo]);

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
      {/* 上部: 動画エリア（PC表示では高さ60%） */}
      <div className="relative w-full h-[60%] bg-black flex items-center justify-center">
        {showOverlay && (
          <div
            className="absolute inset-0 w-full h-full bg-contain bg-no-repeat bg-center flex items-center justify-center z-10"
            style={{ backgroundImage: cardData.thumbnail_url ? `url(${cardData.thumbnail_url})` : undefined, backgroundColor: cardData.thumbnail_url ? undefined : '#1f2937' }}
            onClick={() => {
              // iframe再生に統一。オーバーレイを即時非表示
              setShowOverlay(false);
            }}
          >
            <div className="absolute inset-0 bg-black bg-opacity-40 flex items-center justify-center">
              <Play className="text-white w-16 h-16 opacity-80" fill="white" />
            </div>
            <div className="absolute bottom-2 left-1/2 -translate-x-1/2 text-[10px] sm:text-xs text-white/80 bg-black/40 px-2 py-0.5 rounded">
              注: 再生には最大2回のクリックが必要な場合があります
            </div>
          </div>
        )}
        {/* iframe 埋め込み（litevideo） */}
        <iframe
          src={cardData.embedUrl || cardData.videoUrl}
          title="Embedded Video Player"
          frameBorder="0"
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share; fullscreen"
          loading="eager"
          onLoad={() => {
            if (overlayHideTimer.current) clearTimeout(overlayHideTimer.current);
            overlayHideTimer.current = window.setTimeout(() => {
              setShowOverlay(false);
              overlayHideTimer.current = null;
            }, overlayHideDelayMs);
          }}
          className="absolute top-0 left-0 w-full h-full"
        />
        {cardData.productUrl && (
          <a
            href={cardData.productUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="absolute bottom-2 right-2 z-10 text-xs text-white bg-black/50 px-2 py-1 rounded"
          >
            新しいタブで再生
          </a>
        )}
      </div>
      
      {/* 下部: テキスト情報エリア（PC表示では高さ40%） */}
      <div className="flex flex-col text-gray-700 p-4 overflow-y-auto h-[40%]">
        <h2 className="text-lg font-bold">{cardData.title}</h2>
        {cardData.product_released_at && (
          <div className="grid grid-cols-[auto_1fr] items-start gap-x-2 mt-2">
            <div className="flex w-18 flex-shrink-0 items-center text-sm text-gray-500">
              <Calendar className="mr-1 h-4 w-4" />
              <span>発売日:</span>
            </div>
            <div className="flex flex-wrap gap-2">
              <span className="rounded-full bg-blue-300 px-2 py-1 text-xs font-bold text-white">
                {new Date(cardData.product_released_at).toLocaleDateString('ja-JP')}
              </span>
            </div>
          </div>
        )}
        {cardData.performers && cardData.performers.length > 0 && (
          <div className="grid grid-cols-[auto_1fr] items-start gap-x-2 mt-2">
            <div className="flex w-18 flex-shrink-0 items-center text-sm text-gray-500">
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
        {cardData.tags && cardData.tags.length > 0 && (
          <div className="grid grid-cols-[auto_1fr] items-start gap-x-2 mt-2">
            {/* Grid Column 1: Label */}
            <div className="flex w-18 flex-shrink-0 items-center text-sm text-gray-500">
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
        )}
      </div>
    </motion.div>
  );
});

SwipeCard.displayName = "SwipeCard";

export default SwipeCard;
