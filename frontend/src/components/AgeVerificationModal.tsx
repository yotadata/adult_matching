interface AgeVerificationModalProps {
  onConfirm: () => void;
  onCancel: () => void;
}

const AgeVerificationModal = ({ onConfirm, onCancel }: AgeVerificationModalProps) => {
  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
      <div className="bg-white/80 backdrop-blur-lg rounded-2xl p-8 max-w-sm w-full text-center text-gray-800 shadow-2xl">
        <h2 className="text-2xl font-bold mb-4">年齢確認</h2>
        <p className="mb-6">あなたは18歳以上ですか？</p>
        <div className="flex justify-around">
          <button 
            onClick={onCancel} 
            className="px-8 py-3 rounded-xl font-bold text-gray-700 bg-gray-300/80 hover:bg-gray-400/80 transition-colors duration-200"
          >
            いいえ
          </button>
          <button 
            onClick={onConfirm} 
            className="px-8 py-3 rounded-xl font-bold text-white bg-blue-500/90 hover:bg-blue-600/90 transition-colors duration-200"
          >
            はい
          </button>
        </div>
      </div>
    </div>
  );
};

export default AgeVerificationModal;
