import { useState } from 'react';
import { ThumbsDown, ThumbsUp } from 'lucide-react';

interface ActionButtonsProps {
  onSkip: () => void;
  onLike: () => void;
  nopeColor: string;
  likeColor: string;
  isMobileLayout?: boolean; // 追加
  cardWidth?: number; // 追加
  variant?: 'full' | 'icons'; // 表示バリエーション
  includeCenter?: boolean; // 中央の「お気に入りリスト」ボタンを含める（PC用）
}

const ActionButtons = ({ onSkip, onLike, nopeColor, likeColor, isMobileLayout = false, cardWidth, variant = 'full', includeCenter = false }: ActionButtonsProps) => {
  const [isNopeHovered, setIsNopeHovered] = useState(false);
  const [isLikeHovered, setIsLikeHovered] = useState(false);

  const nopeButtonStyle = (
    variant === 'full'
      ? ({
          background: `linear-gradient(135deg, ${nopeColor} 0%, rgba(167,139,250,0.85) 60%, rgba(167,139,250,0.7) 100%)`,
          borderColor: nopeColor,
          transition: 'transform 0.15s ease-in-out',
        } as React.CSSProperties)
      : ({} as React.CSSProperties)
  );

  const likeButtonStyle = (
    variant === 'full'
      ? ({
          background: `linear-gradient(135deg, ${likeColor} 0%, rgba(251,191,36,0.85) 60%, rgba(251,191,36,0.7) 100%)`,
          borderColor: likeColor,
          transition: 'transform 0.15s ease-in-out',
        } as React.CSSProperties)
      : ({} as React.CSSProperties)
  );

  const buttonBaseClasses = "flex items-center justify-center h-14 font-semibold tracking-wide text-base text-white border-2 transition-transform duration-150";

  if (variant === 'icons') {
    return (
      <div className={`flex items-center justify-center w-full gap-6 px-4 py-3`}>
        <button
          onClick={onSkip}
          onMouseEnter={() => setIsNopeHovered(true)}
          onMouseLeave={() => setIsNopeHovered(false)}
          className={`w-14 h-14 rounded-full bg-white border border-gray-200 shadow-lg active:scale-95 transition`}
          aria-label="イマイチ"
          title="イマイチ"
        >
          <ThumbsDown size={22} className={`mx-auto ${isNopeHovered ? 'text-gray-700' : 'text-gray-500'}`} />
        </button>
        {includeCenter && (
          <button
            onClick={() => { try { window.dispatchEvent(new Event('open-liked-drawer')); } catch {} }}
            className={`w-14 h-14 rounded-full bg-[#BEBEBE] border border-gray-200 shadow-lg active:scale-95 transition`}
            aria-label="お気に入りリスト"
            title="お気に入りリスト"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" className="mx-auto fill-white" aria-hidden="true">
              <path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 6 3.99 4 6.5 4 8.04 4 9.54 4.81 10.35 6.09 11.16 4.81 12.66 4 14.2 4 16.71 4 18.7 6 18.7 8.5c0 3.78-3.4 6.86-8.05 11.54L12 21.35z" />
            </svg>
          </button>
        )}
        <button
          onClick={onLike}
          onMouseEnter={() => setIsLikeHovered(true)}
          onMouseLeave={() => setIsLikeHovered(false)}
          className={`w-14 h-14 rounded-full bg-white border border-gray-200 shadow-lg active:scale-95 transition`}
          aria-label="好み"
          title="好み"
        >
          <ThumbsUp size={22} className={`mx-auto ${isLikeHovered ? 'text-rose-600' : 'text-rose-500'}`} />
        </button>
      </div>
    );
  }

  return (
    <div className={`flex items-center ${isMobileLayout ? 'w-full gap-3 px-4 py-3' : 'justify-between gap-6'} ${isMobileLayout ? '' : 'mx-auto'}`} style={!isMobileLayout && cardWidth ? { width: `${cardWidth}px` } : {}}>
      {/* NOPE Button */}
      <button 
        onClick={onSkip} 
        onMouseEnter={() => setIsNopeHovered(true)}
        onMouseLeave={() => setIsNopeHovered(false)}
        className={`${buttonBaseClasses} ${isMobileLayout ? 'flex-1 rounded-full shadow-btnPurple' : 'w-40 rounded-2xl hover:scale-105 active:scale-95 shadow-btnPurple'}`}
        style={nopeButtonStyle}
        aria-label="イマイチ"
        title="イマイチ"
      >
        <ThumbsDown size={22} className="mr-2" />
        イマイチ
      </button>
      {!isMobileLayout && includeCenter && (
        <button
          onClick={() => { try { window.dispatchEvent(new Event('open-liked-drawer')); } catch {} }}
          className={`${buttonBaseClasses} w-24 rounded-2xl bg-[#BEBEBE] border-2 border-[#BEBEBE] text-white hover:scale-105 active:scale-95 shadow-lg`}
          style={{ transition: 'transform 0.15s ease-in-out' }}
          aria-label="お気に入りリスト"
          title="お気に入りリスト"
        >
          ❤︎
        </button>
      )}
      {/* LIKE Button */}
      <button 
        onClick={onLike} 
        onMouseEnter={() => setIsLikeHovered(true)}
        onMouseLeave={() => setIsLikeHovered(false)}
        className={`${buttonBaseClasses} ${isMobileLayout ? 'flex-1 rounded-full shadow-btnAmber' : 'w-40 rounded-2xl hover:scale-105 active:scale-95 shadow-btnAmber'}`}
        style={likeButtonStyle}
        aria-label="好み"
        title="好み"
      >
        <ThumbsUp size={22} className="mr-2" />
        好み
      </button>
    </div>
  );
};

export default ActionButtons;
