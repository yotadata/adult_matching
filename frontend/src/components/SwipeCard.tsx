'use client';

import TinderCard from 'react-tinder-card';
import { motion } from 'framer-motion';
import { forwardRef, useImperativeHandle, useState, useRef } from 'react';
import { BookOpen, User, Tag, Calendar, Share2, ChevronLeft, ChevronRight } from 'lucide-react';

export interface CardData {
  id: number;
  title: string;
  genre: string[];
  description: string;
  thumbnail_url: string;
  thumbnailVerticalUrl?: string;
  sampleImageUrls?: string[];   // サンプル画像URL配列（試し読みページ）
  author?: string;              // 著者名
  tags?: { id: string; name: string; }[];
  product_released_at?: string;
  productUrl?: string;
  affiliateUrl?: string;
  pageCount?: number;
  recommendationSource?: string | null;
  recommendationScore?: number | null;
  recommendationModelVersion?: string | null;
  recommendationParams?: Record<string, unknown> | null;
}

export interface SwipeCardHandle {
  swipe: (direction: 'left' | 'right') => void;
}

interface SwipeCardProps {
  cardData: CardData;
  onSwipe: (direction: 'left' | 'right') => void;
  onDrag?: (direction: 'left' | 'right' | 'reset') => void;
  onDragEnd?: () => void;
  cardWidth: number | undefined;
  canSwipe?: boolean;
}

type TinderApi = {
  swipe: (dir?: 'left' | 'right' | 'up' | 'down') => Promise<void>;
  restoreCard: () => Promise<void>;
};

const SwipeCard = forwardRef<SwipeCardHandle, SwipeCardProps>(({ cardData, onSwipe, onDrag, onDragEnd, cardWidth, canSwipe = true }, ref) => {
  const cardRef = useRef<TinderApi | null>(null);
  const [imageIndex, setImageIndex] = useState(0);

  const images = [
    ...(cardData.thumbnailVerticalUrl ? [cardData.thumbnailVerticalUrl] : []),
    ...(cardData.sampleImageUrls ?? []),
    ...(cardData.thumbnail_url && !cardData.thumbnailVerticalUrl ? [cardData.thumbnail_url] : []),
  ].filter(Boolean);

  const displayImages = images.length > 0 ? images : [cardData.thumbnail_url].filter(Boolean) as string[];

  const swipe = async (direction: 'left' | 'right') => {
    if (!canSwipe) return;
    try {
      await cardRef.current?.swipe(direction);
    } catch (err) {
      if (process.env.NODE_ENV !== 'production') {
        console.warn('TinderCard swipe failed:', err);
      }
    }
  };

  useImperativeHandle(ref, () => ({ swipe }));

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

  const handlePrevImage = (e: React.MouseEvent) => {
    e.stopPropagation();
    setImageIndex((prev) => Math.max(0, prev - 1));
  };

  const handleNextImage = (e: React.MouseEvent) => {
    e.stopPropagation();
    setImageIndex((prev) => Math.min(displayImages.length - 1, prev + 1));
  };

  return (
    <div className="absolute h-full" style={{ width: cardWidth ? `${cardWidth}px` : 'auto' }}>
      <div className="absolute inset-x-4 top-4 -bottom-2 bg-[#1c2128] rounded-2xl shadow-md pointer-events-none z-0" />
      <TinderCard
        ref={cardRef as unknown as React.RefObject<TinderApi>}
        className="relative z-10 h-full w-full"
        preventSwipe={canSwipe ? ['up', 'down'] : ['left', 'right', 'up', 'down']}
        swipeRequirementType="position"
        swipeThreshold={(() => {
          const base = cardWidth ?? 360;
          const computed = base * 0.35;
          return Math.max(120, Math.min(200, computed));
        })()}
        onSwipe={(dir) => {
          if (dir === 'left' || dir === 'right') onSwipe(dir);
        }}
        onSwipeRequirementFulfilled={(dir) => {
          if (dir === 'left' || dir === 'right') onDrag?.(dir);
        }}
        onSwipeRequirementUnfulfilled={() => {
          onDrag?.('reset');
          onDragEnd?.();
        }}
        onCardLeftScreen={() => {
          onDrag?.('reset');
          onDragEnd?.();
        }}
      >
        <motion.div
          className="h-full rounded-2xl bg-[#0d1117] border border-[#30363d] shadow-xl flex flex-col p-4 cursor-grab overflow-hidden"
          whileTap={{ scale: 0.995 }}
        >
          {/* サンプル画像エリア */}
          <div className="relative w-full aspect-[3/4] bg-black/90 flex items-center justify-center rounded-xl overflow-hidden">
            {displayImages.length > 0 ? (
              <>
                <img
                  src={displayImages[imageIndex]}
                  alt={cardData.title}
                  className="w-full h-full object-contain"
                  draggable={false}
                />
                {/* 画像ナビゲーション */}
                {displayImages.length > 1 && (
                  <>
                    <button
                      onClick={handlePrevImage}
                      disabled={imageIndex === 0}
                      className="absolute left-2 top-1/2 -translate-y-1/2 w-8 h-8 flex items-center justify-center rounded-full bg-black/50 text-white disabled:opacity-30"
                    >
                      <ChevronLeft size={18} />
                    </button>
                    <button
                      onClick={handleNextImage}
                      disabled={imageIndex === displayImages.length - 1}
                      className="absolute right-2 top-1/2 -translate-y-1/2 w-8 h-8 flex items-center justify-center rounded-full bg-black/50 text-white disabled:opacity-30"
                    >
                      <ChevronRight size={18} />
                    </button>
                    <div className="absolute bottom-2 left-1/2 -translate-x-1/2 flex gap-1">
                      {displayImages.map((_, i) => (
                        <div
                          key={i}
                          className={`w-1.5 h-1.5 rounded-full ${i === imageIndex ? 'bg-white' : 'bg-white/40'}`}
                        />
                      ))}
                    </div>
                  </>
                )}
                {/* 試し読みラベル */}
                {imageIndex > 0 && (
                  <div className="absolute top-2 left-2 bg-black/60 text-white text-[10px] px-2 py-0.5 rounded">
                    試し読み {imageIndex}/{displayImages.length - 1}
                  </div>
                )}
              </>
            ) : (
              <div className="flex flex-col items-center gap-2 text-[#8b949e]">
                <BookOpen size={48} />
                <span className="text-xs">画像なし</span>
              </div>
            )}
          </div>

          {/* 下部: テキスト情報 */}
          <div className="flex flex-col p-4 overflow-y-auto flex-1">
            <h2 className="text-base font-bold text-[#e6edf3] leading-snug">{cardData.title}</h2>

            {cardData.product_released_at && (
              <div className="grid grid-cols-[auto_1fr_auto] items-center gap-x-2 mt-2">
                <div className="flex items-center text-xs text-[#8b949e] gap-1 whitespace-nowrap">
                  <Calendar className="h-3.5 w-3.5" />
                  <span>発売日:</span>
                </div>
                <div className="flex flex-wrap gap-1.5">
                  <span className="rounded-full bg-blue-900/50 border border-blue-700/40 px-2 py-0.5 text-[11px] text-blue-300">
                    {new Date(cardData.product_released_at).toLocaleDateString('ja-JP')}
                  </span>
                </div>
                <button
                  onClick={() => {
                    try {
                      const text = cardData.title || '';
                      const url = toAffiliateUrl(cardData.affiliateUrl || cardData.productUrl);
                      const shareUrl = `https://twitter.com/intent/tweet?text=${encodeURIComponent(text)}&url=${encodeURIComponent(url)}`;
                      window.open(shareUrl, '_blank', 'noopener,noreferrer');
                    } catch {}
                  }}
                  className="w-7 h-7 flex items-center justify-center rounded-full bg-[#21262d] border border-[#30363d] text-[#8b949e] hover:text-[#e6edf3]"
                  aria-label="Xで共有"
                >
                  <Share2 size={13} />
                </button>
              </div>
            )}

            {cardData.author && (
              <div className="flex items-start gap-2 mt-2">
                <div className="flex items-center text-xs text-[#8b949e] gap-1 whitespace-nowrap pt-0.5">
                  <User className="h-3.5 w-3.5" />
                  <span>著者:</span>
                </div>
                <div className="flex flex-wrap gap-1.5">
                  <span className="rounded-full bg-pink-900/40 border border-pink-700/40 px-2 py-0.5 text-[11px] text-pink-300">
                    {cardData.author}
                  </span>
                </div>
              </div>
            )}

            {cardData.tags && cardData.tags.length > 0 && (
              <div className="flex items-start gap-2 mt-2">
                <div className="flex items-center text-xs text-[#8b949e] gap-1 whitespace-nowrap pt-0.5">
                  <Tag className="h-3.5 w-3.5" />
                  <span>タグ:</span>
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {cardData.tags.map((tag) => (
                    <span key={tag.id} className="rounded-full bg-violet-900/40 border border-violet-700/40 px-2 py-0.5 text-[11px] text-violet-300">
                      {tag.name}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        </motion.div>
      </TinderCard>
    </div>
  );
});

SwipeCard.displayName = "SwipeCard";

export default SwipeCard;
