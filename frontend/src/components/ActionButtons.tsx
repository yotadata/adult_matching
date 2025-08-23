import { useState } from 'react';

interface ActionButtonsProps {
  onSkip: () => void;
  onLike: () => void;
  nopeColor: string;
  likeColor: string;
  isMobileLayout?: boolean; // 追加
  cardWidth?: number; // 追加
}

const ActionButtons = ({ onSkip, onLike, nopeColor, likeColor, isMobileLayout = false, cardWidth }: ActionButtonsProps) => {
  const [isNopeHovered, setIsNopeHovered] = useState(false);
  const [isLikeHovered, setIsLikeHovered] = useState(false);

  const nopeButtonStyle = {
    background: isNopeHovered && !isMobileLayout ? nopeColor : `color-mix(in srgb, ${nopeColor} 80%, transparent)`,
    borderColor: nopeColor,
    transition: 'background 0.2s ease-in-out',
  };

  const likeButtonStyle = {
    background: isLikeHovered && !isMobileLayout ? likeColor : `color-mix(in srgb, ${likeColor} 80%, transparent)`,
    borderColor: likeColor,
    transition: 'background 0.2s ease-in-out',
  };

  const buttonBaseClasses = "flex items-center justify-center h-14 font-bold tracking-wider text-lg text-white border-2";

  return (
    <div className="flex items-center justify-between mx-auto" style={{ width: cardWidth ? `${cardWidth}px` : 'auto' }}>
      {/* NOPE Button */}
      <button 
        onClick={onSkip} 
        onMouseEnter={() => setIsNopeHovered(true)}
        onMouseLeave={() => setIsLikeHovered(false)}
        className={`${buttonBaseClasses} ${isMobileLayout ? 'w-1/2' : 'w-40'} ${isMobileLayout ? 'rounded-none' : 'rounded-2xl'} ${!isMobileLayout ? 'hover:scale-105 active:scale-95 shadow-lg' : ''}`}
        style={nopeButtonStyle}
      >
        NOPE
      </button>
      {/* LIKE Button */}
      <button 
        onClick={onLike} 
        onMouseEnter={() => setIsLikeHovered(true)}
        onMouseLeave={() => setIsLikeHovered(false)}
        className={`${buttonBaseClasses} ${isMobileLayout ? 'w-1/2' : 'w-40'} ${isMobileLayout ? 'rounded-none' : 'rounded-2xl'} ${!isMobileLayout ? 'hover:scale-105 active:scale-95 shadow-lg' : ''}`}
        style={likeButtonStyle}
      >
        LIKE
      </button>
    </div>
  );
};

export default ActionButtons;
