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
      className="flex items-center justify-center w-40 h-14 font-bold tracking-wider text-lg text-white rounded-2xl backdrop-blur-md border border-white/20 shadow-lg transition-all duration-200 ease-in-out hover:border-purple-400/80 hover:scale-105 active:scale-95"
      style={{ backgroundColor: nopeColor }}
    >
      NOPE
    </button>
    {/* LIKE Button */}
    <button 
      onClick={onLike} 
      className="flex items-center justify-center w-40 h-14 font-bold tracking-wider text-lg text-white rounded-2xl backdrop-blur-md border border-white/20 shadow-lg transition-all duration-200 ease-in-out hover:border-orange-400/80 hover:scale-105 active:scale-95"
      style={{ backgroundColor: likeColor }}
    >
      LIKE
    </button>
  </div>
);

export default ActionButtons;