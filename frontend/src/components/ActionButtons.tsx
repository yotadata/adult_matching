interface ActionButtonsProps {
  onSkip: () => void;
  onLike: () => void;
  nopeColor: string;
  likeColor: string;
}

const ActionButtons = ({ onSkip, onLike, nopeColor, likeColor }: ActionButtonsProps) => (
  <div className="flex items-center justify-between w-full">
    {/* NOPE Button */}
    <button 
      onClick={onSkip} 
      className="flex items-center justify-center w-40 h-14 font-bold tracking-wider text-lg text-white rounded-2xl transition-all duration-200 ease-in-out hover:scale-105 active:scale-95 border"
      style={{ backgroundColor: nopeColor, textShadow: '0 2px 4px rgba(0, 0, 0, 0.25)', borderColor: '#C4B5FD' }}
    >
      NOPE
    </button>
    {/* LIKE Button */}
    <button 
      onClick={onLike} 
      className="flex items-center justify-center w-40 h-14 font-bold tracking-wider text-lg text-white rounded-2xl transition-all duration-200 ease-in-out hover:scale-105 active:scale-95 border-2"
      style={{ background: `color-mix(in srgb, ${likeColor} 20%, transparent)`, borderColor: likeColor }}
    >
      LIKE
    </button>
  </div>
);

export default ActionButtons;