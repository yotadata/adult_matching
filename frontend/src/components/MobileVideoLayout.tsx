'use client';

import { CardData } from '@/components/SwipeCard';
import ActionButtons from '@/components/ActionButtons';
import { useEffect, useRef, useState } from 'react';
import { Play, Calendar, User, Tag } from 'lucide-react';

interface MobileVideoLayoutProps {
  cardData: CardData;
  onSkip: () => void;
  onLike: () => void;
}

const MobileVideoLayout: React.FC<MobileVideoLayoutProps> = ({ cardData, onSkip, onLike }) => {
  const [showVideo, setShowVideo] = useState(false);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [showOverlay, setShowOverlay] = useState(true);
  const overlayHideTimer = useRef<number | null>(null);
  const overlayHideDelayMs = 700; // iframe読込後も少しサムネイルを維持

  // Reset iframe fallback when card changes
  useEffect(() => {
    setShowOverlay(true);
    setShowVideo(false);
    if (overlayHideTimer.current) {
      clearTimeout(overlayHideTimer.current);
      overlayHideTimer.current = null;
    }
  }, [cardData?.id]);

  const handlePlayClick = () => {
    // デバッグ: どの再生パスか確認
    if (cardData.sampleVideoUrl) {
      console.warn('[MobileVideoLayout] Using <video> src:', cardData.sampleVideoUrl);
    } else {
      console.warn('[MobileVideoLayout] Using <iframe> src:', cardData.embedUrl || cardData.videoUrl);
    }
    setShowVideo(true);
  };

  useEffect(() => {
    if (showVideo && videoRef.current) {
      const v = videoRef.current;
      const playPromise = v.play();
      if (playPromise && typeof playPromise.then === 'function') {
        playPromise.catch((err) => {
          console.warn('[MobileVideoLayout] Autoplay failed, waiting for user gesture.', err);
        });
      }
    }
  }, [showVideo]);

  return (
    <div className="flex flex-col w-full h-full">
      {/* 動画表示エリア */}
      <div className="w-full overflow-hidden relative aspect-[494/370] bg-black flex items-center justify-center">
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
          title="Embedded video"
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
          className="absolute inset-0 w-full h-full"
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

      {/* テキスト情報エリア */}
      <div className="p-4 text-gray-700 flex-grow overflow-y-auto">
        <h2 className="text-xl font-bold">{cardData.title}</h2>
        {cardData.product_released_at && (
          <div className="grid grid-cols-[auto_1fr] items-start gap-x-2 mt-2">
            <div className="flex w-24 flex-shrink-0 items-center text-sm text-gray-500">
              <Calendar className="mr-1 h-4 w-4" />
              <span>発売日:</span>
            </div>
            <div className="flex flex-wrap gap-2">
              <span className="rounded-full bg-blue-500 bg-opacity-70 px-2 py-1 text-xs font-bold text-white">
                {new Date(cardData.product_released_at).toLocaleDateString('ja-JP')}
              </span>
            </div>
          </div>
        )}
        {cardData.performers && cardData.performers.length > 0 && (
          <div className="grid grid-cols-[auto_1fr] items-start gap-x-2 mt-2">
            <div className="flex w-24 flex-shrink-0 items-center text-sm text-gray-500">
              <User className="mr-1 h-4 w-4" />
              <span>出演:</span>
            </div>
            <div className="flex flex-wrap gap-2">
              {cardData.performers.map((p) => (
                <span key={p.id} className="rounded-full bg-pink-400 bg-opacity-70 px-2 py-1 text-xs font-bold text-white">
                  {p.name}
                </span>
              ))}
            </div>
          </div>
        )}
        {Array.isArray(cardData.tags) && cardData.tags.length > 0 && (
          <div className="grid grid-cols-[auto_1fr] items-start gap-x-2 mt-2">
            <div className="flex w-24 flex-shrink-0 items-center text-sm text-gray-500">
              <Tag className="mr-1 h-4 w-4" />
              <span>タグ:</span>
            </div>
            <div className="flex flex-wrap gap-2 my-2">
              {cardData.tags.map((tag) => (
                <span
                  key={tag.id}
                  className="bg-purple-600 bg-opacity-60 text-white text-xs font-bold px-2 py-1 rounded-full"
                >
                  {tag.name}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* フッターボタンエリア */}
      <div className="w-full">
        <ActionButtons
          onSkip={onSkip}
          onLike={onLike}
          nopeColor="#A78BFA"
          likeColor="#FBBF24"
        isMobileLayout={true} // 追加
        />
      </div>
    </div>
  );
};

export default MobileVideoLayout;
