'use client';

import { CardData } from '@/components/SwipeCard';
import ActionButtons from '@/components/ActionButtons';
import { useEffect, useRef, useState } from 'react';
import { Play, Calendar, User, Tag, ThumbsDown, ThumbsUp, Heart } from 'lucide-react';

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
    // デバッグ: どの再生パスか確認（開発のみ）
    if (process.env.NODE_ENV !== 'production') {
      if (cardData.sampleVideoUrl) {
        console.warn('[MobileVideoLayout] Using <video> src:', cardData.sampleVideoUrl);
      } else {
        console.warn('[MobileVideoLayout] Using <iframe> src:', cardData.embedUrl || cardData.videoUrl);
      }
    }
    setShowVideo(true);
  };

  useEffect(() => {
    if (showVideo && videoRef.current) {
      const v = videoRef.current;
      const playPromise = v.play();
      if (playPromise && typeof playPromise.then === 'function') {
        playPromise.catch((err) => {
          if (process.env.NODE_ENV !== 'production') {
            console.warn('[MobileVideoLayout] Autoplay failed, waiting for user gesture.', err);
          }
        });
      }
    }
  }, [showVideo]);

  return (
    <div className="relative flex flex-col w-full h-full">
      {/* ラッパー（グレーの次カードを下に少しだけ覗かせる） */}
      <div className="relative mx-0 mt-2 mb-[112px]">
        {/* 次カードのチラ見せ（背面） */}
        <div className="absolute inset-x-4 -bottom-3 h-6 rounded-2xl bg-gray-200/90 shadow-md z-0 pointer-events-none" />

        {/* メインカード（白背景・角丸・影） */}
        <div className="relative z-10 bg-white rounded-2xl overflow-hidden shadow-xl">
          {/* 動画表示エリア（4:3） */}
          <div className="w-full p-3">
            <div className="w-full overflow-hidden relative aspect-[4/3] bg-black flex items-center justify-center rounded-xl">
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
              {/* 外部タブ再生リンクは非表示にする */}
            </div>
          </div>

        {/* テキスト情報エリア */}
          <div className="px-4 pb-4 text-gray-800">
            <h2 className="text-base sm:text-xl font-extrabold tracking-tight">{cardData.title}</h2>
        {cardData.product_released_at && (
          <div className="grid grid-cols-[auto_1fr] items-start gap-x-2 mt-2">
            <div className="flex w-24 flex-shrink-0 items-center text-sm text-gray-500">
              <Calendar className="mr-1 h-4 w-4" />
              <span>発売日:</span>
            </div>
            <div className="flex flex-wrap gap-2">
              <span className="rounded-full bg-blue-500/70 px-2 py-1 text-[11px] font-bold text-white">
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
                <span key={p.id} className="rounded-full bg-pink-500/70 px-2 py-1 text-[11px] font-bold text-white">
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
                  className="bg-purple-600/60 text-white text-[11px] font-bold px-2 py-1 rounded-full"
                >
                  {tag.name}
                </span>
              ))}
            </div>
          </div>
        )}
          </div>
        </div>
      </div>

      {/* フッター（透明）: 3つの色ボタン＋白アイコン */}
      <div className="fixed left-0 right-0 bottom-0 z-50 pb-[calc(8px+env(safe-area-inset-bottom,0px))]">
        <div className="mx-auto max-w-md w-full flex items-center justify-center gap-6 py-3">
          {/* NOPE (thumb_down, #6C757D) */}
          <button
            onClick={onSkip}
            className="w-20 h-20 rounded-full bg-[#6C757D] shadow-lg active:scale-95 transition flex items-center justify-center"
            aria-label="イマイチ"
            title="イマイチ"
          >
            <ThumbsDown size={36} className="text-white" />
          </button>
          {/* Liked list */}
          <button
            onClick={() => { try { window.dispatchEvent(new Event('open-liked-drawer')); } catch {} }}
            className="w-[60px] h-[60px] rounded-full bg-[#BEBEBE] shadow-lg active:scale-95 transition flex items-center justify-center"
            aria-label="お気に入りリスト"
            title="お気に入りリスト"
          >
            <Heart size={28} className="text-white" />
          </button>
          {/* GOOD (heart, #FF6B81) */}
          <button
            onClick={onLike}
            className="w-20 h-20 rounded-full bg-[#FF6B81] shadow-lg active:scale-95 transition flex items-center justify-center"
            aria-label="好み"
            title="好み"
          >
            <Heart size={36} className="text-white" />
          </button>
        </div>
      </div>
    </div>
  );
};

export default MobileVideoLayout;
