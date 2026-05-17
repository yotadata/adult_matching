'use client';

import { CardData } from '@/components/SwipeCard';
import { useState, RefObject } from 'react';
import { BookOpen, Calendar, User, Tag, ChevronsLeft, Heart, List, Share2 } from 'lucide-react';

interface MobileVideoLayoutProps {
  cardData: CardData;
  onSkip: () => void;
  onLike: () => void;
  skipButtonRef?: RefObject<HTMLButtonElement | null>;
  likeButtonRef?: RefObject<HTMLButtonElement | null>;
  likedListButtonRef?: RefObject<HTMLButtonElement | null>;
}

const MobileVideoLayout: React.FC<MobileVideoLayoutProps> = ({ cardData, onSkip, onLike, skipButtonRef, likeButtonRef, likedListButtonRef }) => {
  const [imageIndex, setImageIndex] = useState(0);

  const images = [
    ...(cardData.thumbnailVerticalUrl ? [cardData.thumbnailVerticalUrl] : []),
    ...(cardData.sampleImageUrls ?? []),
    ...(cardData.thumbnail_url && !cardData.thumbnailVerticalUrl ? [cardData.thumbnail_url] : []),
  ].filter(Boolean) as string[];
  const displayImages = images.length > 0 ? images : [cardData.thumbnail_url].filter(Boolean) as string[];

  const toAffiliateUrl = (raw?: string) => {
    const AF_ID = 'yotadata2-001';
    try {
      if (raw && raw.startsWith('https://al.fanza.co.jp/')) {
        const u = new URL(raw);
        u.searchParams.set('af_id', AF_ID);
        return u.toString();
      }
    } catch {}
    if (raw) return `https://al.fanza.co.jp/?lurl=${encodeURIComponent(raw)}&af_id=${encodeURIComponent(AF_ID)}&ch=link_tool&ch_id=link`;
    try { return window.location.href; } catch { return ''; }
  };

  return (
    <div className="relative flex flex-col w-full h-full">
      {/* 次カードのチラ見せ（背面） */}
      <div className="absolute inset-x-3 -bottom-[108px] top-3 rounded-2xl bg-[#1c2128] shadow-md z-0 pointer-events-none" />

      {/* メインカード */}
      <div
        className="relative z-10 mx-0 mt-2 mb-[112px] rounded-2xl overflow-hidden shadow-2xl border border-[#30363d] flex flex-col"
        style={{ background: '#0d1117' }}
      >
        {/* サンプル画像エリア */}
        <div className="relative w-full aspect-[3/4] bg-black flex items-center justify-center overflow-hidden select-none">
          {displayImages.length > 0 ? (
            <>
              <img
                src={displayImages[imageIndex]}
                alt={cardData.title}
                className="w-full h-full object-contain"
                draggable={false}
              />

              {/* タップ左半分で前へ / 右半分で次へ */}
              {displayImages.length > 1 && (
                <>
                  <button
                    onClick={() => setImageIndex((p) => Math.max(0, p - 1))}
                    className="absolute left-0 top-0 w-1/2 h-full z-10"
                    aria-label="前の画像"
                  />
                  <button
                    onClick={() => setImageIndex((p) => Math.min(displayImages.length - 1, p + 1))}
                    className="absolute right-0 top-0 w-1/2 h-full z-10"
                    aria-label="次の画像"
                  />

                  {/* ページインジケーター */}
                  <div className="absolute bottom-2 left-1/2 -translate-x-1/2 flex gap-1 z-20 pointer-events-none">
                    {displayImages.map((_, i) => (
                      <div key={i} className={`w-1.5 h-1.5 rounded-full ${i === imageIndex ? 'bg-white' : 'bg-white/40'}`} />
                    ))}
                  </div>

                  {/* タップヒント（最初の表示時のみ） */}
                  {imageIndex === 0 && displayImages.length > 1 && (
                    <div className="absolute bottom-7 left-1/2 -translate-x-1/2 z-20 pointer-events-none">
                      <span className="text-[10px] text-white/60 bg-black/40 px-2 py-0.5 rounded">タップでページめくり</span>
                    </div>
                  )}
                </>
              )}

              {/* ページ番号 */}
              <div className="absolute top-2 right-2 bg-black/60 text-white text-[10px] px-2 py-0.5 rounded z-20 pointer-events-none">
                {imageIndex + 1} / {displayImages.length}
              </div>
            </>
          ) : (
            <div className="flex flex-col items-center gap-2 text-[#8b949e]">
              <BookOpen size={48} />
              <span className="text-xs">画像なし</span>
            </div>
          )}
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
                    const url = toAffiliateUrl(cardData.affiliateUrl || cardData.productUrl);
                    window.open(`https://twitter.com/intent/tweet?text=${encodeURIComponent(text)}&url=${encodeURIComponent(url)}`, '_blank', 'noopener,noreferrer');
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
              <span className="rounded-full bg-pink-900/40 border border-pink-700/40 px-2 py-0.5 text-[11px] text-pink-300">
                {cardData.author}
              </span>
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

      {/* フッター */}
      <div className="fixed left-0 right-0 bottom-0 z-50 pb-[calc(8px+env(safe-area-inset-bottom,0px))]">
        <div className="mx-auto max-w-md w-full flex items-center justify-center gap-6 py-3">
          <button onClick={onSkip} ref={skipButtonRef} className="w-20 h-20 rounded-full bg-[#21262d] border border-[#30363d] shadow-2xl active:scale-95 transition flex items-center justify-center" aria-label="スキップ">
            <ChevronsLeft size={36} className="text-[#8b949e]" />
          </button>
          <button onClick={() => { try { window.dispatchEvent(new Event('open-liked-drawer')); } catch {} }} ref={likedListButtonRef} className="w-[60px] h-[60px] rounded-full bg-[#21262d] border border-[#30363d] shadow-2xl active:scale-95 transition flex items-center justify-center" aria-label="気になるリスト">
            <List size={28} className="text-[#8b949e]" />
          </button>
          <button onClick={onLike} ref={likeButtonRef} className="w-20 h-20 rounded-full bg-violet-600 hover:bg-violet-500 shadow-2xl active:scale-95 transition flex items-center justify-center" aria-label="気になる">
            <Heart size={36} className="text-white" fill="white" />
          </button>
        </div>
      </div>
    </div>
  );
};

export default MobileVideoLayout;
