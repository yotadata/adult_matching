'use client';

import { motion, useAnimation, PanInfo } from 'framer-motion';
import { forwardRef, useImperativeHandle, useState, useEffect, useRef } from 'react';
import { Play, User, Tag, Calendar, Share2 } from 'lucide-react'; // アイコンをインポート

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
  canSwipe?: boolean; // 追加: ゲスト制限時にスワイプを抑制
}

const SwipeCard = forwardRef<SwipeCardHandle, SwipeCardProps>(({ cardData, onSwipe, onDrag, onDragEnd, cardWidth, canSwipe = true }, ref) => {
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
    if (!canSwipe) {
      await controls.start({ x: 0 });
      return;
    }
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
    if (!canSwipe) {
      controls.start({ x: 0 });
      onDragEnd?.(event, info);
      return;
    }
    if (info.offset.x > 100) {
      swipe('right');
    } else if (info.offset.x < -100) {
      swipe('left');
    } else {
      controls.start({ x: 0 });
    }
    onDragEnd?.(event, info); // 親の onDragEnd を呼び出す
  };

  

  const toAffiliateUrl = (raw?: string) => {
    const AF_ID = 'yotadata2-001';
    try {
      if (raw && raw.startsWith('https://al.fanza.co.jp/')) {
        const u = new URL(raw);
        u.searchParams.set('af_id', AF_ID);
        return u.toString();
      }
    } catch {}
    if (raw) {
      return `https://al.fanza.co.jp/?lurl=${encodeURIComponent(raw)}&af_id=${encodeURIComponent(AF_ID)}&ch=link_tool&ch_id=link`;
    }
    try { return window.location.href; } catch { return ''; }
  };

  useEffect(() => {
    if (showVideo && videoRef.current) {
      const v = videoRef.current;
      // iOS/Android 対策: タップ直後に明示的に再生
      const playPromise = v.play();
      if (playPromise && typeof playPromise.then === 'function') {
        playPromise.catch((err) => {
          if (process.env.NODE_ENV !== 'production') {
            console.warn('[SwipeCard] Autoplay failed, waiting for user gesture.', err);
          }
        });
      }
    }
  }, [showVideo]);

  return (
    <div className="absolute h-full" style={{ width: cardWidth ? `${cardWidth}px` : 'auto' }}>
      {/* 背面のグレー“カード”：白カードより少し小さい高さで、下側だけ少しはみ出す */}
      <div className="absolute inset-x-4 top-4 -bottom-2 bg-gray-200/90 rounded-2xl shadow-md pointer-events-none z-0" />
      <motion.div 
        className="relative z-10 h-full rounded-2xl bg-white border border-gray-200 shadow-xl flex flex-col p-4 cursor-grab overflow-hidden"
        drag="x"
        dragConstraints={{ left: 0, right: 0 }}
        onDragStart={(event, info) => onDrag?.(event, info)} // onDragStart も追加
        onDrag={handleDrag}
        onDragEnd={handleDragEnd}
        animate={controls}
        initial={false}
        whileTap={{ cursor: "grabbing" }}
      >
      {/* 上部: 動画エリア（PC版は4:3のアスペクト比） */}
      <div className="relative w-full aspect-[4/3] bg-black/90 flex items-center justify-center rounded-xl overflow-hidden">
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
        {/* 外部タブ再生リンクは非表示にする */}
      </div>

      {/* 下部: テキスト情報エリア（残り領域にフィット） */}
      <div className="flex flex-col text-gray-800 p-4 overflow-y-auto flex-1">
        <h2 className="text-lg font-extrabold tracking-tight">{cardData.title}</h2>
          {cardData.product_released_at && (
            <div className="grid grid-cols-[auto_1fr_auto] items-center gap-x-2 mt-2">
              <div className="flex w-18 flex-shrink-0 items-center text-sm text-gray-500">
                <Calendar className="mr-1 h-4 w-4" />
                <span>発売日:</span>
              </div>
              <div className="flex flex-wrap gap-2">
                <span className="rounded-full bg-blue-500/70 px-2 py-1 text-[11px] font-bold text-white">
                  {new Date(cardData.product_released_at).toLocaleDateString('ja-JP')}
                </span>
              </div>
              <div className="flex justify-end">
                <button
                  onClick={() => {
                    try {
                      const text = cardData.title || '';
                      const url = toAffiliateUrl(cardData.productUrl);
                      const shareUrl = `https://twitter.com/intent/tweet?text=${encodeURIComponent(text)}&url=${encodeURIComponent(url)}`;
                      window.open(shareUrl, '_blank', 'noopener,noreferrer');
                    } catch {}
                  }}
                  className="w-8 h-8 flex items-center justify-center rounded-full bg-white text-black border border-gray-300 shadow-sm hover:bg-gray-50"
                  aria-label="Xで共有"
                  title="Xで共有"
                >
                  <Share2 size={14} />
                </button>
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
                <span key={performer.id} className="rounded-full bg-pink-500/70 px-2 py-1 text-[11px] font-bold text-white">
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
                <span key={tag.id} className="rounded-full bg-purple-600/60 px-2 py-1 text-[11px] font-bold text-white">
                  {tag.name}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
      </motion.div>
    </div>
  );
});

SwipeCard.displayName = "SwipeCard";

export default SwipeCard;
