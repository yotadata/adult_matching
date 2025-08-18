import { useState } from 'react';

interface ActionButtonsProps {
  onSkip: () => void;
  onLike: () => void;
  nopeColor: string;
  likeColor: string;
}

const ActionButtons = ({ onSkip, onLike, nopeColor, likeColor }: ActionButtonsProps) => {
  const [isNopeHovered, setIsNopeHovered] = useState(false);
  const [isLikeHovered, setIsLikeHovered] = useState(false);

  const nopeButtonStyle = {
    background: isNopeHovered ? nopeColor : `color-mix(in srgb, ${nopeColor} 80%, transparent)`,
    borderColor: nopeColor,
    transition: 'background 0.2s ease-in-out',
  };

  const likeButtonStyle = {
    background: isLikeHovered ? likeColor : `color-mix(in srgb, ${likeColor} 80%, transparent)`,
    borderColor: likeColor,
    transition: 'background 0.2s ease-in-out',
  };

  return (
    <div className="flex items-center justify-between w-full">
      {/* NOPE Button */}
      <button 
        onClick={onSkip} 
        onMouseEnter={() => setIsNopeHovered(true)}
        onMouseLeave={() => setIsNopeHovered(false)}
        className="flex items-center justify-center w-40 h-14 font-bold tracking-wider text-lg text-white rounded-2xl hover:scale-105 active:scale-95 border-2 shadow-lg"
        style={nopeButtonStyle}
      >
        NOPE
      </button>
      {/* LIKE Button */}
      <button 
        onClick={onLike} 
        onMouseEnter={() => setIsLikeHovered(true)}
        onMouseLeave={() => setIsLikeHovered(false)}
        className="flex items-center justify-center w-40 h-14 font-bold tracking-wider text-lg text-white rounded-2xl hover:scale-105 active:scale-95 border-2 shadow-lg"
        style={likeButtonStyle}
      >
        LIKE
      </button>
    </div>
  );
};

export default ActionButtons;
