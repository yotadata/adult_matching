'use client';

import { CardData } from '@/components/SwipeCard';
import { useEffect, useRef, useState, RefObject } from 'react';
import { Play, Calendar, User, Tag, ChevronsLeft, Heart, List, Share2 } from 'lucide-react';

interface MobileVideoLayoutProps {
  cardData: CardData;
  onSkip: () => void;
  onLike: () => void;
  onSamplePlay?: (card: CardData) => void;
  skipButtonRef?: RefObject<HTMLButtonElement | null>;
  likeButtonRef?: RefObject<HTMLButtonElement | null>;
  likedListButtonRef?: RefObject<HTMLButtonElement | null>;
}

const MobileVideoLayout: React.FC<MobileVideoLayoutProps> = ({ cardData, onSkip, onLike, onSamplePlay, skipButtonRef, likeButtonRef, likedListButtonRef }) => {
  const [showVideo, setShowVideo] = useState(false);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [showOverlay, setShowOverlay] = useState(true);
  const overlayHideTimer = useRef<number | null>(null);
  const overlayHideDelayMs = 700;

  useEffect(() => {
    setShowOverlay(true);
    setShowVideo(false);
    if (overlayHideTimer.current) {
      clearTimeout(overlayHideTimer.current);
      overlayHideTimer.current = null;
    }
  }, [cardData?.id]);

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

  const bgThumbnail = cardData.thumbnailVerticalUrl || cardData.thumbnail_url;

  return (
    <div className="relative flex flex-col w-full h-full">
      {/* 次カードのチラ見せ（背面） */}
      <div className="absolute inset-x-3 -bottom-[108px] top-3 rounded-2xl bg-[#1c2128] shadow-md z-0 pointer-events-none" />

      {/* メインカード */}
      <div
        className="relative z-10 mx-0 mt-2 mb-[112px] rounded-2xl overflow-hidden shadow-2xl border border-[#30363d] flex flex-col"
        style={{ background: '#0d1117' }}
      >
        {/* 動画エリア: 縦サムネを背景に、動画は4:3 */}
        <div className="relative w-full">
          {/* 縦サムネ背景（ぼかし） */}
          {bgThumbnail && (
            <div
              className="absolute inset-0 bg-cover bg-center scale-110"
              style={{ backgroundImage: `url(${bgThumbnail})`, filter: 'blur(12px) brightness(0.35)' }}
            />
          )}
          <div className="relative w-full aspect-[4/3] flex items-center justify-center">
            <div className="relative w-full h-full overflow-hidden bg-black/60">
              {showOverlay && (
                <div
                  className="absolute inset-0 w-full h-full bg-contain bg-no-repeat bg-center flex items-center justify-center z-10 cursor-pointer"
                  style={{
                    backgroundImage: cardData.thumbnail_url ? `url(${cardData.thumbnail_url})` : undefined,
                    backgroundColor: cardData.thumbnail_url ? undefined : '#161b22',
                  }}
                  onClick={() => {
                    setShowOverlay(false);
                    onSamplePlay?.(cardData);
                  }}
                >
                  <div className="absolute inset-0 bg-black/40 flex items-center justify-center">
                    <Play className="text-white w-16 h-16 opacity-90" fill="white" />
                  </div>
                  <div className="absolute bottom-2 left-1/2 -translate-x-1/2 text-[10px] text-white/70 bg-black/50 px-2 py-0.5 rounded">
                    注: 再生には最大2回のクリックが必要な場合があります
                  </div>
                </div>
              )}
              <iframe
                scrolling="no"
                referrerPolicy="no-referrer"
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
                className="absolute inset-0 w-full h-full overflow-hidden"
              />
            </div>
          </div>
        </div>

        {/* テキスト情報エリア */}
        <div className="px-4 pt-3 pb-4 bg-[#0d1117]">
          <h2 className="text-sm font-bold text-[#e6edf3] leading-snug line-clamp-2">{cardData.title}</h2>

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
                    const url = toAffiliateUrl(cardData.productUrl);
                    const shareUrl = `https://twitter.com/intent/tweet?text=${encodeURIComponent(text)}&url=${encodeURIComponent(url)}`;
                    window.open(shareUrl, '_blank', 'noopener,noreferrer');
                  } catch {}
                }}
                className="w-7 h-7 flex items-center justify-center rounded-full bg-[#21262d] border border-[#30363d] text-[#8b949e] hover:text-[#e6edf3]"
                aria-label="Xで共有"
                title="Xで共有"
              >
                <Share2 size={13} />
              </button>
            </div>
          )}

          {cardData.performers && cardData.performers.length > 0 && (
            <div className="flex items-start gap-2 mt-2">
              <div className="flex items-center text-xs text-[#8b949e] gap-1 whitespace-nowrap pt-0.5">
                <User className="h-3.5 w-3.5" />
                <span>出演:</span>
              </div>
              <div className="flex flex-wrap gap-1.5">
                {cardData.performers.map((p) => (
                  <span key={p.id} className="rounded-full bg-pink-900/40 border border-pink-700/40 px-2 py-0.5 text-[11px] text-pink-300">
                    {p.name}
                  </span>
                ))}
              </div>
            </div>
          )}

          {Array.isArray(cardData.tags) && cardData.tags.length > 0 && (
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
      </div>

      {/* フッター: スキップ / リスト / 気になる */}
      <div className="fixed left-0 right-0 bottom-0 z-50 pb-[calc(8px+env(safe-area-inset-bottom,0px))]">
        <div className="mx-auto max-w-md w-full flex items-center justify-center gap-6 py-3">
          <button
            onClick={onSkip}
            ref={skipButtonRef}
            className="w-20 h-20 rounded-full bg-[#21262d] border border-[#30363d] shadow-2xl active:scale-95 transition flex items-center justify-center"
            aria-label="スキップ"
            title="スキップ"
          >
            <ChevronsLeft size={36} className="text-[#8b949e]" />
          </button>
          <button
            onClick={() => { try { window.dispatchEvent(new Event('open-liked-drawer')); } catch {} }}
            ref={likedListButtonRef}
            className="w-[60px] h-[60px] rounded-full bg-[#21262d] border border-[#30363d] shadow-2xl active:scale-95 transition flex items-center justify-center"
            aria-label="気になるリスト"
            title="気になるリスト"
          >
            <List size={28} className="text-[#8b949e]" />
          </button>
          <button
            onClick={onLike}
            ref={likeButtonRef}
            className="w-20 h-20 rounded-full bg-violet-600 hover:bg-violet-500 shadow-2xl active:scale-95 transition flex items-center justify-center"
            aria-label="気になる"
            title="気になる"
          >
            <Heart size={36} className="text-white" fill="white" />
          </button>
        </div>
      </div>
    </div>
  );
};

export default MobileVideoLayout;
