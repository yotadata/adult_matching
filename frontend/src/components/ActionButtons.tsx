interface ActionButtonsProps {
  onSkip: () => void;
  onLike: () => void;
}

const ActionButtons = ({ onSkip, onLike }: ActionButtonsProps) => (
  <div className="flex space-x-8">
    <button onClick={onSkip} className="bg-white/20 rounded-full p-4 text-4xl transition-transform hover:scale-110 active:scale-95">❌</button>
    <button onClick={onLike} className="bg-white/20 rounded-full p-4 text-4xl transition-transform hover:scale-110 active:scale-95">❤️</button>
  </div>
);

export default ActionButtons;